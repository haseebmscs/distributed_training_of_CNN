import torch
import torch.distributed as dist
import torch.optim as optim
import time
import os
import socket
import traceback
from datetime import timedelta

from config import (
    MASTER_IP, MASTER_PORT, MASTER_BOOTSTRAP_PORT,
    MAX_WORKERS,
    EPOCHS, BATCH_SIZE, LEARNING_RATE, MIN_WORKERS,
    HEARTBEAT_ENABLED, GLOO_SOCKET_IFNAME, USE_LIBUV
)
from master.registry   import WorkerRegistry
from master.scheduler  import Scheduler
from utils.logger      import TrainingLogger
from comm.heartbeat    import HeartbeatMonitor
from comm.bootstrap    import MasterBootstrapServer
from comm.comm_utils   import (
    send_signal, recv_signal,
    send_tensor, recv_metrics
)
from comm.signals      import (
    SIGNAL_START, SIGNAL_STOP,
    SIGNAL_NEXT
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

        # DataLoaders — only Master loads data
        self.train_loader = None
        self.test_loader  = None

        print("[Master] Initialised")
        print(f"[Master] Waiting for minimum "
              f"{MIN_WORKERS} workers...")

    def _on_worker_assigned(self, rank, peer_info):
        if self.registry.register(rank):
            hostname = peer_info.get("hostname", "unknown")
            local_ip = peer_info.get("local_ip", "unknown")
            self.logger.log_worker_joined(rank)
            self.logger.log_event(
                f"Bootstrap worker rank {rank} joined from {hostname} ({local_ip})"
            )
    
    def setup_network(self, world_size):
        import socket

        os.environ["GLOO_SOCKET_IFNAME"] = GLOO_SOCKET_IFNAME
        os.environ["USE_LIBUV"]          = "0"

        print(f"[Master] Setting up network...")
        print(f"  Hostname   : {socket.gethostname()}")
        print(f"  MASTER_IP  : {MASTER_IP}")
        print(f"  PORT       : {MASTER_PORT}")
        print(f"  World size : {world_size}")

        dist.init_process_group(
            backend    = "gloo",
            init_method = f"tcp://{MASTER_IP}:{MASTER_PORT}",
            world_size = world_size,
            rank       = 0
        )

        print("[Master] Network ready ✅")

    def wait_for_workers(self):
        if self.bootstrap:
            print("[Master] Waiting for bootstrap workers to join...")
            self.bootstrap.wait_until_complete(timeout=120)

        total = self.registry.count_total()
        if total < MIN_WORKERS:
            raise RuntimeError(
                f"Only {total} workers joined; need at least {MIN_WORKERS}"
            )

        print(f"[Master] {total} workers ready!")
        return total

    def handle_failure(self, dead_rank):
        """
        Called automatically by HeartbeatMonitor
        when a worker goes silent.

        Args:
            dead_rank (int): rank of dead worker
        """
        print(f"\n[Master] ⚠️  Worker {dead_rank} failed!")
        self.logger.log_worker_failed(dead_rank)

        # Get dead worker's stage
        dead_stage = self.registry.get_stage(dead_rank)

        # Ask scheduler to promote a standby
        self.scheduler.handle_failure(dead_rank)

        # Log the promotion
        new_rank = self.registry.get_active_ranks()
        self.logger.log_event(
            f"Stage {dead_stage} reassigned after "
            f"rank {dead_rank} failure"
        )

        # Update monitor's active ranks list
        if self.monitor:
            self.monitor.update_active_ranks(
                self.registry.get_active_ranks()
            )

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

        # Who is first and last in pipeline?
        first_worker = self.scheduler.get_first_worker()
        last_worker  = self.scheduler.get_last_worker()

        # Step 1: Send START to every active stage worker.
        # Each worker's loop expects one control signal per batch.
        for rank in self.registry.get_active_ranks():
            send_signal(SIGNAL_START, dst=rank)

        # Step 2: Send image batch to first worker
        send_tensor(images, dst=first_worker)

        # Also send labels to last worker
        # (it needs them to compute loss)
        send_tensor(labels.float(), dst=last_worker)

        # Step 3: Wait for results from last worker
        loss, accuracy = recv_metrics(src=last_worker)

        return loss, accuracy


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

        for batch_idx, (images, labels) in \
                enumerate(self.train_loader, 1):

            loss, accuracy = self.run_batch(
                images, labels, epoch, batch_idx
            )

            total_loss += loss
            total_acc  += accuracy

            # Log every batch
            self.logger.log_batch(
                epoch         = epoch,
                batch         = batch_idx,
                total_batches = total_batches,
                loss          = loss,
                accuracy      = accuracy
            )

            # Send NEXT to all active workers so batch counters
            # and checkpoint cadence stay aligned across stages.
            for rank in self.registry.get_active_ranks():
                send_signal(SIGNAL_NEXT, dst=rank)

        avg_loss = total_loss / total_batches
        avg_acc  = total_acc  / total_batches

        return avg_loss, avg_acc

    def run(self, world_size):
        """
        Main entry point — runs the full training pipeline.

        Args:
            world_size (int): total machines (master + workers)
        """

        expected_workers = max(world_size - 1, 0)

        # ── Step 1: Start bootstrap server ────────────────
        self.bootstrap = MasterBootstrapServer(
            expected_workers  = expected_workers,
            host             = "0.0.0.0",
            port             = MASTER_BOOTSTRAP_PORT,
            on_worker_assigned = self._on_worker_assigned,
            timeout          = 120
        )
        self.bootstrap.start()

        # ── Step 2: Connect network ────────────────────────
        self.setup_network(world_size)

        # ── Step 3: Wait for workers ───────────────────────
        total_workers = self.wait_for_workers()

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

        for epoch in range(1, EPOCHS + 1):
            print(f"\n[Master] ── Epoch {epoch}/{EPOCHS} ──")

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
        dist.destroy_process_group()

    