import torch
import torch.optim as optim
import time
import threading
import os
import socket
import traceback
from datetime import timedelta

from config import (
    MASTER_IP, MASTER_PORT, MASTER_BOOTSTRAP_PORT,
    MAX_WORKERS,
    EPOCHS, BATCH_SIZE, LEARNING_RATE, MIN_WORKERS,
    HEARTBEAT_ENABLED, GLOO_SOCKET_IFNAME, USE_LIBUV,
    SOCKET_TIMEOUT
)
from comm.distributed_socket import (
    init_process_group, get_rank, get_world_size,
    barrier, destroy_process_group, send_tensor
)
from master.registry   import WorkerRegistry
from master.scheduler  import Scheduler
from utils.logger      import TrainingLogger
from comm.heartbeat    import HeartbeatMonitor
from comm.bootstrap    import MasterBootstrapServer
from comm.comm_utils   import (
    send_signal, recv_signal,
    recv_metrics
)
from comm.signals      import (
    SIGNAL_START, SIGNAL_STOP,
    SIGNAL_NEXT, SIGNAL_STEP,
    SIGNAL_READY, SIGNAL_RECONFIG
)
from dataset.custom_dataset import get_dataloaders

class Master:
    """
    Central coordinator for distributed CNN training.

    Responsibilities:
    - Accept worker connections
    - Assign pipeline stages
    - Drive the training loop
    - Handle worker failures
    - Log all progress
    """

    def __init__(self):
        self.registry  = WorkerRegistry()
        self.scheduler = Scheduler(self.registry)
        self.logger    = TrainingLogger()
        self.monitor   = None   # built after workers join
        self.bootstrap = None

        # Event used to pause/resume master sending during promotions
        self.resume_event = threading.Event()
        self.resume_event.set()

        # DataLoaders — only Master loads data
        self.train_loader = None
        self.test_loader  = None

        print("[Master] Initialised")
        print("[Master] Waiting for workers to connect...")

    def _on_worker_assigned(self, rank, peer_info):
        if self.registry.register(rank):
            hostname = peer_info.get("hostname", "unknown")
            local_ip = peer_info.get("local_ip", "unknown")
            self.logger.log_worker_joined(rank)
            self.logger.log_event(
                f"Bootstrap worker rank {rank} joined from {hostname} ({local_ip})"
            )
            # If this rank is beyond the initially expected active
            # workers, note that it joined as a standby.
            try:
                if self.bootstrap and rank > self.bootstrap.expected_workers:
                    print(f"[Master] Rank {rank} joined as STANDBY")
                    self.logger.log_event(f"Rank {rank} joined as STANDBY")
            except Exception:
                pass
    
    def setup_network(self, world_size):
        """Setup distributed communication using sockets (no Gloo)."""
        print(f"[Master] Setting up socket-based distributed communication...")
        print(f"  Master IP  : {MASTER_IP}")
        print(f"  Port       : {MASTER_PORT}")
        print(f"  World size : {world_size}")

        # Initialize socket-based process group (rank 0 = master)
        init_process_group(
            backend="socket",
            world_size=world_size,
            rank=0,
            timeout=SOCKET_TIMEOUT
        )

        print("[Master] Socket network ready [OK]")

    def wait_for_workers(self, expected_workers):
        """Wait until all expected workers are assigned by bootstrap."""
        if expected_workers <= 0:
            print("[Master] No workers expected (world_size=1).")
            return 0

        if self.bootstrap:
            print(
                f"[Master] Waiting for all workers to join "
                f"({expected_workers}/{expected_workers})..."
            )
            # Wait until the bootstrap server has assigned the expected
            # number of worker ranks. The bootstrap server continues
            # running afterwards so late joiners can register as standby.
            assigned = self.bootstrap.wait_for_expected(timeout=120)
            if not assigned:
                raise TimeoutError(
                    f"Timed out waiting for workers: "
                    f"{self.registry.count_total()}/{expected_workers} joined"
                )

            bootstrap_error = self.bootstrap.get_error()
            if bootstrap_error:
                raise RuntimeError(f"Bootstrap failed:\n{bootstrap_error}")

        total = self.registry.count_total()
        if total < expected_workers:
            raise RuntimeError(
                f"Workers joined mismatch: expected {expected_workers}, got {total}"
            )

        print(f"[Master] All workers ready: {total}/{expected_workers}")
        return total

    def handle_failure(self, dead_rank):
        """
        Called automatically by HeartbeatMonitor
        when a worker goes silent.

        Args:
            dead_rank (int): rank of dead worker
        """
        print(f"\n[Master] [WARN] Worker {dead_rank} failed!")
        self.logger.log_worker_failed(dead_rank)

        # Get dead worker's stage
        dead_stage = self.registry.get_stage(dead_rank)

        # Pause sending new batches while we reconfigure
        print("[Master] Pausing master send loop until replacement is ready...")
        self.resume_event.clear()

        # Ask scheduler to promote a standby and get replacement rank
        replacement = self.scheduler.handle_failure(dead_rank)

        # Log the promotion
        self.logger.log_event(
            f"Stage {dead_stage} reassigned after "
            f"rank {dead_rank} failure"
        )

        # Update monitor's active ranks list (monitor should validate new heartbeats)
        if self.monitor:
            self.monitor.update_active_ranks(
                self.registry.get_active_ranks()
            )

        # If scheduler could not find a replacement, resume and exit
        if replacement is None:
            print("[Master] No replacement available - cannot continue. Resuming to allow shutdown.")
            self.resume_event.set()
            return

        # Wait for replacement worker to signal READY
        print(f"[Master] Waiting for replacement rank {replacement} READY signal...")
        start = time.time()
        ready = False
        TIMEOUT = 120
        while time.time() - start < TIMEOUT:
            try:
                sig = recv_signal(src=replacement)
                if sig == SIGNAL_READY.item():
                    ready = True
                    print(f"[Master] Replacement rank {replacement} is READY")
                    break
                else:
                    print(f"[Master] Received unexpected signal {sig} from {replacement}; waiting for READY")
            except Exception:
                time.sleep(0.5)

        if not ready:
            print(f"[Master] Timeout waiting for replacement {replacement} READY")

        # Resume master send loop
        self.resume_event.set()

    def run_batch(self, images, labels, epoch, batch_idx):
        """
        Drives one batch through the full pipeline.

        Steps:
            1. Send START signal to Worker 1
            2. Send image batch to Worker 1
            3. Wait for loss/accuracy from last Worker

        Args:
            images    (tensor): batch of images (B,3,32,32)
            labels    (tensor): batch of labels (B,)
            epoch     (int)   : current epoch
            batch_idx (int)   : current batch number

        Returns:
            loss (float), accuracy (float)
        """

        # If a promotion/reconfiguration is in progress, wait until it's ready
        self.resume_event.wait()

        # Who is first and last in pipeline?
        first_worker = self.scheduler.get_first_worker()
        last_worker  = self.scheduler.get_last_worker()

        print(f"[Master] Batch {batch_idx}: first_worker={first_worker}, last_worker={last_worker}")

        # Step 1: Send START to every active stage worker.
        # Each worker's loop expects one control signal per batch.
        for rank in self.registry.get_active_ranks():
            print(f"[Master] Sending START signal to rank {rank}")
            send_signal(SIGNAL_START, dst=rank)

        # Larger delay to ensure workers have processed START signal
        time.sleep(0.1)

        # Step 2: Send image batch to first worker
        print(f"[Master] Sending {images.shape} tensor to first_worker {first_worker}")
        send_tensor(images, dst=first_worker)
        
        # Small delay between tensor sends
        time.sleep(0.05)

        # Also send labels to last worker
        # (it needs them to compute loss)
        print(f"[Master] Sending {labels.shape} labels tensor to last_worker {last_worker}")
        send_tensor(labels.float(), dst=last_worker)

        # Delay before expecting metrics
        time.sleep(0.05)

        # Step 3: Wait for results from last worker
        try:
            print(f"[Master] Waiting for metrics from rank {last_worker}")
            loss, accuracy = recv_metrics(src=last_worker)
            print(f"[Master] Received metrics: loss={loss:.4f}, accuracy={accuracy:.2f}%")
            return loss, accuracy
        except Exception as e:
            print(f"[Master] Error receiving metrics from rank {last_worker}: {e}")
            # Trigger failure handling for the last worker and wait for replacement
            try:
                self.handle_failure(last_worker)
            except Exception as he:
                print(f"[Master] handle_failure raised: {he}")

            # Wait until resume_event (replacement READY) before retrying
            print("[Master] Waiting for promotion/resume before retrying batch...")
            self.resume_event.wait()

            # After promotion, resend START and tensors for this batch
            print("[Master] Resending START and tensors for interrupted batch...")
            # Recompute first/last workers (may have changed)
            first_worker = self.scheduler.get_first_worker()
            last_worker = self.scheduler.get_last_worker()

            # Send START to every active worker again
            for rank in self.registry.get_active_ranks():
                send_signal(SIGNAL_START, dst=rank)

            time.sleep(0.1)

            # Re-send the image batch and labels
            try:
                send_tensor(images, dst=first_worker)
                time.sleep(0.05)
                send_tensor(labels.float(), dst=last_worker)
            except Exception as send_e:
                print(f"[Master] Error resending tensors: {send_e}")
                raise

            # Delay to let P2P connections settle
            time.sleep(0.15)

            # Retry once to receive metrics
            try:
                print(f"[Master] Retrying: waiting for metrics from rank {last_worker}")
                loss, accuracy = recv_metrics(src=last_worker)
                print(f"[Master] Retry succeeded: loss={loss:.4f}, accuracy={accuracy:.2f}%")
                return loss, accuracy
            except Exception as e2:
                print(f"[Master] Retry failed receiving metrics: {e2}")
                raise


    def run_epoch(self, epoch):
        """
        Runs all batches for one epoch.

        Args:
            epoch (int): current epoch number

        Returns:
            avg_loss (float), avg_accuracy (float)
        """

        total_batches = len(self.train_loader)
        total_loss    = 0.0
        total_acc     = 0.0
        successful_batches = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader, 1):

            print(f"[Master] Processing batch {batch_idx}/{total_batches}")

            try:
                loss, accuracy = self.run_batch(images, labels, epoch, batch_idx)

            except Exception as e:
                # Log error but continue with next batch to keep training visible
                print(f"[Master] Batch {batch_idx} failed: {e}")
                self.logger.log_event(f"Batch {batch_idx} failed: {e}")
                # Still send SIGNAL_NEXT to keep workers in sync (even though this batch failed)
                loss = None
                accuracy = None

            # If batch succeeded, aggregate metrics
            if loss is not None and accuracy is not None:
                total_loss += loss
                total_acc  += accuracy
                successful_batches += 1

                # Log every successful batch
                self.logger.log_batch(
                    epoch         = epoch,
                    batch         = batch_idx,
                    total_batches = total_batches,
                    loss          = loss,
                    accuracy      = accuracy
                )
                
                print(f"[Master] Batch {batch_idx} complete: loss={loss:.4f}, accuracy={accuracy:.2f}%")
            else:
                print(f"[Master] Batch {batch_idx} skipped (failed)")

            # ALWAYS send SIGNAL_NEXT to keep workers in sync with batch counter
            # even if this batch failed, we still move to next
            print(f"[Master] Sending SIGNAL_NEXT to all active workers")
            for rank in self.registry.get_active_ranks():
                try:
                    send_signal(SIGNAL_NEXT, dst=rank)
                except Exception as sync_err:
                    print(f"[Master] Warning: could not send SIGNAL_NEXT to rank {rank}: {sync_err}")

        # Compute averages using only successful batches to avoid skew
        if successful_batches > 0:
            avg_loss = total_loss / successful_batches
            avg_acc  = total_acc  / successful_batches
        else:
            avg_loss = 0.0
            avg_acc = 0.0

        return avg_loss, avg_acc

    def run_epoch_streaming(self, epoch):
        """
        Runs one epoch using TRUE STREAMING PIPELINE PARALLELISM.
        
        Strategy:
        - Batch 1 → Worker 1 (stage 1)
        - When Worker 1 done: Batch 1 → Worker 2, Batch 2 → Worker 1 (no idle!)
        - When Worker 2 done: Batch 1 → Worker 3, Batch 2 → Worker 2, Batch 3 → Worker 1
        - Continue until all batches flow through all stages
        - Accumulate gradients for all in-flight batches
        - Update weights (SIGNAL_STEP) after epoch completes
        
        This creates a continuous pipeline where workers are never idle.
        
        Args:
            epoch (int): current epoch
        
        Returns:
            avg_loss (float), avg_accuracy (float)
        """
        print(f"\n[Master] === EPOCH {epoch}/{EPOCHS} (Streaming Pipeline) ===")
        
        first_worker = self.scheduler.get_first_worker()
        last_worker = self.scheduler.get_last_worker()
        active_ranks = self.registry.get_active_ranks()
        num_stages = len(active_ranks)
        
        # Load all batches
        all_batches = []
        for images, labels in self.train_loader:
            all_batches.append((images, labels))
        
        total_batches = len(all_batches)
        print(f"[Master] Loaded {total_batches} batches, {num_stages} stages")
        
        total_loss = 0.0
        total_acc = 0.0
        completed_batches = 0
        
        # Track in-flight batches: batch_idx → current stage
        in_flight = {}  # batch_idx → worker_rank (where it currently is)
        batch_idx = 0
        
        print(f"\n[Master] Starting streaming pipeline...")
        
        # Send initial batches to fill the pipeline
        for i in range(min(num_stages, total_batches)):
            images, labels = all_batches[i]
            print(f"[Master] Sending batch {i+1}/{total_batches} to Worker {first_worker} (stage 1)")
            
            # Send SIGNAL_START to ALL workers (synchronization point)
            for rank in active_ranks:
                send_signal(SIGNAL_START, dst=rank)
            
            # Send data to specific workers
            send_tensor(images, dst=first_worker)
            send_tensor(labels.float(), dst=last_worker)
            
            in_flight[i] = first_worker
            batch_idx = i + 1
        
        # Continue feeding new batches as workers complete stages
        while completed_batches < total_batches:
            # Wait for a metric from the last worker (batch completion signal)
            try:
                loss, accuracy = recv_metrics(src=last_worker)
                completed_batches += 1
                total_loss += loss
                total_acc += accuracy
                print(f"[Master] Batch completed! Loss={loss:.4f}, Acc={accuracy:.2f}% ({completed_batches}/{total_batches})")
            except Exception as e:
                print(f"[Master] Error receiving metrics: {e}")
                continue
            
            # If there are more batches to send, send next batch to first worker
            if batch_idx < total_batches:
                images, labels = all_batches[batch_idx]
                print(f"[Master] Sending batch {batch_idx+1}/{total_batches} to Worker {first_worker} (stage 1)")
                
                # Send SIGNAL_START to ALL workers (synchronization point)
                for rank in active_ranks:
                    send_signal(SIGNAL_START, dst=rank)
                
                # Send data to specific workers
                send_tensor(images, dst=first_worker)
                send_tensor(labels.float(), dst=last_worker)
                
                in_flight[batch_idx] = first_worker
                batch_idx += 1
        
        # All batches completed - signal workers to update weights
        print(f"\n[Master] All {total_batches} batches completed. Sending SIGNAL_STEP to update weights...")
        for rank in active_ranks:
            try:
                send_signal(SIGNAL_STEP, dst=rank)
            except Exception as e:
                print(f"[Master] Warning: could not send SIGNAL_STEP to rank {rank}: {e}")
        
        # Send SIGNAL_NEXT to bump epoch counter for checkpoints
        for rank in active_ranks:
            try:
                send_signal(SIGNAL_NEXT, dst=rank)
            except Exception as e:
                print(f"[Master] Warning: could not send SIGNAL_NEXT to rank {rank}: {e}")
        
        if completed_batches > 0:
            avg_loss = total_loss / completed_batches
            avg_acc = total_acc / completed_batches
        else:
            avg_loss = 0.0
            avg_acc = 0.0
        
        print(f"\n[Master] Epoch {epoch} Summary:")
        print(f"  Total Batches: {completed_batches}")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Avg Accuracy: {avg_acc:.2f}%")
        
        return avg_loss, avg_acc

    def run(self, world_size):
        """
        Main entry point — runs the full training pipeline.

        Args:
            world_size (int): total machines (master + workers)
        """

        expected_workers = max(world_size - 1, 0)

        # ── Step 1: Start bootstrap server ────────────────
        # Bind to specific MASTER_IP, not 0.0.0.0, to avoid hostname resolution
        self.bootstrap = MasterBootstrapServer(
            expected_workers  = expected_workers,
            host             = MASTER_IP,
            port             = MASTER_BOOTSTRAP_PORT,
            on_worker_assigned = self._on_worker_assigned,
            timeout          = 120
        )
        self.bootstrap.start()

        # ── Step 2: Connect network ────────────────────────
        self.setup_network(world_size)

        # ── Step 3: Wait for workers ───────────────────────
        total_workers = self.wait_for_workers(expected_workers)

        # Start timing only after every expected worker is connected.
        self.logger.start_timing()

        # ── Step 4: Assign stages ──────────────────────────
        stages = self.scheduler.assign_stages()

        # ── Step 5: Start heartbeat monitor ───────────────
        active_ranks = self.registry.get_active_ranks()
        if HEARTBEAT_ENABLED:
            self.monitor = HeartbeatMonitor(
                active_ranks        = active_ranks,
                on_failure_callback = self.handle_failure
            )
            self.monitor.start()
        else:
            print("[Master] Heartbeat monitor disabled")

        # ── Step 6: Load dataset ───────────────────────────
        print("\n[Master] Loading CIFAR-10 dataset...")
        self.train_loader, self.test_loader = get_dataloaders()

        # ── Step 7: Log training start ─────────────────────
        self.logger.log_training_start(
            num_workers = total_workers,
            num_stages  = len(stages),
            epochs      = EPOCHS
        )

        # ── Step 8: Training loop ──────────────────────────
        print(f"\n[Master] Starting training "
              f"for {EPOCHS} epochs...\n")
        
        # Import config for pipeline mode
        from config import PIPELINE_ENABLED, PIPELINE_DEPTH

        for epoch in range(1, EPOCHS + 1):
            print(f"\n[Master] ── Epoch {epoch}/{EPOCHS} ──")

            # Use pipelined training if enabled
            if PIPELINE_ENABLED and len(stages) > 1:
                print(f"[Master] Using TRUE PIPELINE PARALLELISM (depth={PIPELINE_DEPTH})")
                avg_loss, avg_acc = self.run_epoch_streaming(epoch)
            else:
                print(f"[Master] Using sequential training")
                avg_loss, avg_acc = self.run_epoch(epoch)

            self.logger.log_epoch(
                epoch        = epoch,
                total_epochs = EPOCHS,
                avg_loss     = avg_loss,
                avg_accuracy = avg_acc
            )

            # Print registry status every epoch
            self.registry.print_status()

        # ── Step 9: Stop all workers ───────────────────────
        print("\n[Master] Training complete!")
        print("[Master] Sending STOP to all workers...")

        for rank in self.registry.get_active_ranks():
            send_signal(SIGNAL_STOP, dst=rank)

        for rank in self.registry.get_standby_ranks():
            send_signal(SIGNAL_STOP, dst=rank)

        # ── Step 10: Stop monitor ──────────────────────────
        if self.monitor:
            self.monitor.stop()

        if self.bootstrap:
            self.bootstrap.stop()

        # ── Step 11: Final summary ─────────────────────────
        self.logger.log_training_complete()
        self.logger.print_summary()

        print("[Master] Shutting down ✅")
        destroy_process_group()

    