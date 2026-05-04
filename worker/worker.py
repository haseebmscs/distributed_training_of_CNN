import torch
import torch.nn as nn
import torch.optim as optim
import socket
import traceback
import json

from config import (
    MASTER_IP, MASTER_PORT,
    EPOCHS, LEARNING_RATE, HEARTBEAT_ENABLED
)
from comm.distributed_socket import (
    init_process_group, get_rank, get_world_size,
    barrier, destroy_process_group, recv_tensor, send_tensor
)
from comm.p2p_data import P2PDataServer, P2PDataClient
from models.pipeline_model import split_model, PipelineStage
from comm.comm_utils import (
    send_signal, recv_signal,
    send_metrics
)
from comm.signals import (
    SIGNAL_START, SIGNAL_STOP,
    SIGNAL_NEXT,  SIGNAL_HELLO,
    SIGNAL_STANDBY, SIGNAL_PROMOTE,
    SIGNAL_DONE, SIGNAL_ASSIGN
)
from comm.heartbeat    import HeartbeatSender
from utils.checkpoint  import (
    save_checkpoint, load_checkpoint,
    find_latest_checkpoint, should_save
)
import os



class Worker:
    """
    Generic worker — runs on every worker machine.
    Behaviour depends on assignment received from Master.

    Each worker:
    - Holds ONE stage of the CNN pipeline
    - Receives data from previous worker (or Master)
    - Processes it through its stage
    - Sends result to next worker (or Master)
    - Sends/receives gradients for backward pass
    """

    def __init__(self, rank, world_size):
        """
        Args:
            rank       (int): this machine's rank (1, 2, 3...)
            world_size (int): total machines in system
        """
        self.rank        = rank
        self.world_size  = world_size
        self.stage       = None    # PipelineStage object
        self.stage_idx   = None    # 0-based index
        self.total_stages = None   # total pipeline stages
        self.optimizer   = None
        self.is_active   = False   # active or standby?
        self.heartbeat   = None    # HeartbeatSender
        
        # P2P data transfer
        self.p2p_server = None     # P2PDataServer
        self.p2p_client = None     # P2PDataClient
        self.prev_rank  = None     # previous worker in pipeline
        self.next_rank  = None     # next worker in pipeline

        print(f"[Worker {rank}] Initialised")
    
    def setup_network(self):
        """Setup distributed communication using sockets (no Gloo)."""
        print(f"[Worker {self.rank}] Connecting to master via sockets...")
        print(f"  Master IP  : {MASTER_IP}")
        print(f"  Port       : {MASTER_PORT}")
        print(f"  Rank       : {self.rank}")
        print(f"  World size : {self.world_size}")

        # Initialize socket-based process group (rank 1+ = workers)
        init_process_group(
            backend="socket",
            world_size=self.world_size,
            rank=self.rank,
            timeout=60
        )

        print(f"[Worker {self.rank}] Socket connection established ")

    def _run_with_error_reporting(self, step_name, callback):
        try:
            return callback()
        except Exception:
            print(f"\n[Worker {self.rank}] {step_name} failed:")
            print(traceback.format_exc())
            raise

    def receive_assignment(self):
        """
        Receives stage assignment from Master.
        Master sends two parts:
        1. Tensor: [stage_idx, total_stages]
        2. Tensor: P2P neighbor info (JSON serialized as uint8)

        Sets:
            self.stage_idx    → which stage this worker runs
            self.total_stages → total pipeline stages
            self.prev_rank    → rank of previous worker (None if first)
            self.next_rank    → rank of next worker (None if last)
        """
        # Receive assignment tensor
        assignment = recv_tensor(src=0)
        self.stage_idx    = assignment[0].item()
        self.total_stages = assignment[1].item()

        print(f"[Worker {self.rank}] Assigned Stage "
              f"{self.stage_idx + 1}/{self.total_stages}")

        # Receive P2P neighbor info
        p2p_data = recv_tensor(src=0)
        p2p_payload = p2p_data.cpu().numpy().tobytes().decode("utf-8")
        p2p_info = json.loads(p2p_payload)
        
        self.prev_rank = p2p_info.get("prev_rank")
        self.next_rank = p2p_info.get("next_rank")
        self.p2p_neighbors = {
            int(rank): info
            for rank, info in p2p_info.get("neighbors", {}).items()
        }
        
        print(f"[Worker {self.rank}] P2P neighbors:")
        print(f"  Prev: {self.prev_rank}, Next: {self.next_rank}")

    def build_stage(self):
        """
        Builds this worker's portion of the CNN.
        Splits the full model and takes only its stage.
        """
        # Get all stages
        all_stages = split_model(self.total_stages)

        # Take only this worker's stage
        self.stage = all_stages[self.stage_idx]

        # Build optimizer for this stage's parameters
        self.optimizer = optim.SGD(
            self.stage.parameters(),
            lr=LEARNING_RATE,
            momentum=0.9
        )

        # In distributed training, start fresh each run
        # (checkpoint compatibility changes with pipeline reconfig)
        print(f"[Worker {self.rank}] Starting fresh")

        print(f"[Worker {self.rank}] Stage "
              f"{self.stage_idx + 1} built ")

    def setup_p2p(self):
        """
        Setup P2P direct data connections to neighbors.
        
        Starts P2P server (listening for incoming tensors from neighbors)
        Establishes P2P client connections to prev/next workers.
        """
        from config import WORKER_P2P_BASE_PORT
        
        # Start P2P server (listen for tensors from neighbors)
        p2p_port = WORKER_P2P_BASE_PORT + self.rank
        self.p2p_server = P2PDataServer(self.rank, port=p2p_port)
        self.p2p_server.start()
        
        # Create P2P client (for sending tensors to neighbors)
        self.p2p_client = P2PDataClient(self.rank)
        
        # Connect to neighbors
        import time
        if self.next_rank is not None:
            neighbor_info = self.p2p_neighbors.get(self.next_rank, {})
            if neighbor_info:
                next_ip = neighbor_info.get("ip")
                next_port = neighbor_info.get("p2p_port")
                if next_ip and next_port:
                    try:
                        self.p2p_client.connect(self.next_rank, next_ip, next_port)
                    except Exception as e:
                        print(f"[Worker {self.rank}] Warning: Could not connect to next rank {self.next_rank}: {e}")
        
        if self.prev_rank is not None:
            neighbor_info = self.p2p_neighbors.get(self.prev_rank, {})
            if neighbor_info:
                prev_ip = neighbor_info.get("ip")
                prev_port = neighbor_info.get("p2p_port")
                if prev_ip and prev_port:
                    try:
                        self.p2p_client.connect(self.prev_rank, prev_ip, prev_port)
                    except Exception as e:
                        print(f"[Worker {self.rank}] Warning: Could not connect to prev rank {self.prev_rank}: {e}")
        
        print(f"[Worker {self.rank}] P2P setup complete")

    def forward_pass(self):
        """
        Receives input, runs it through this stage,
        sends output to next worker.

        If this is the LAST stage:
            - Also receives labels from Master
            - Computes loss
            - Sends loss/accuracy back to Master

        Data path (GFS-style, direct P2P):
        - First worker: Master → P2P to worker2
        - Middle workers: P2P from prevWorker → P2P to nextWorker
        - Last worker: P2P from prevWorker, computes loss, sends metrics to Master

        Returns:
            output (tensor): result of this stage
            loss   (tensor): loss value (last stage only)
        """

        is_first = (self.stage_idx == 0)
        is_last  = (self.stage_idx == self.total_stages - 1)

        print(f"\n[Worker {self.rank}] ═══ FORWARD PASS ═══")
        print(f"  Stage: {self.stage_idx + 1}/{self.total_stages}")

        # ── Receive input ──────────────────────────────────
        if is_first:
            # First worker receives from Master (rank 0)
            received = recv_tensor(src=0)
            print(f"   Received from Master (rank 0): {list(received.shape)}")
        else:
            # Other workers receive from previous worker via P2P
            received = self.p2p_server.recv_tensor(self.prev_rank)
            print(f"   Received from Rank {self.prev_rank} (P2P): {list(received.shape)}")

        # Track received tensor for backward pass
        received.requires_grad_(True)
        self.last_input = received

        # ── Forward through this stage ─────────────────────
        print(f"    Processing through Stage {self.stage_idx + 1}...")
        self.stage.train()
        output = self.stage(received)
        self.last_output = output
        print(f"   Output shape: {list(output.shape)}")

        # ── Send output or compute loss ────────────────────
        if is_last:
            # Receive labels from Master
            labels_float = recv_tensor(src=0)
            labels       = labels_float.long()
            print(f"   Received labels from Master: {list(labels.shape)}")

            # Compute loss
            criterion = nn.CrossEntropyLoss()
            loss      = criterion(output, labels)

            # Compute accuracy
            _, predicted = torch.max(output, 1)
            correct      = (predicted == labels).sum().item()
            accuracy     = 100.0 * correct / labels.size(0)

            print(f"   Loss: {loss.item():.4f}")
            print(f"   Accuracy: {accuracy:.2f}%")

            # Send metrics to Master
            send_metrics(loss.item(), accuracy, dst=0)
            print(f"   Sent metrics to Master (rank 0)")

            return output, loss

        else:
            # Send output to next worker via P2P
            self.p2p_client.send_tensor(output.detach(), self.next_rank)
            print(f"   Sent to Rank {self.next_rank} (P2P): {list(output.detach().shape)}")

            return output, None

    def backward_pass(self, loss=None):
        """
        Receives gradients from next worker,
        runs backward pass through this stage,
        sends gradients to previous worker.

        If this is the LAST stage:
            - loss.backward() starts the backward pass
            - Sends gradient to previous worker via P2P

        If this is the FIRST stage:
            - Receives gradient from next worker via P2P
            - Runs backward with that gradient
            - No need to send further back

        Gradient path (GFS-style, direct P2P):
        - Last worker: starts from loss.backward()
        - Middle workers: recv gradient from next (P2P) → backward() → send to prev (P2P)
        - First worker: recv gradient from next (P2P) → backward(), no further send

        Args:
            loss (tensor): loss value — only for last stage
        """

        is_first = (self.stage_idx == 0)
        is_last  = (self.stage_idx == self.total_stages - 1)

        print(f"\n[Worker {self.rank}] ═══ BACKWARD PASS ═══")
        print(f"  Stage: {self.stage_idx + 1}/{self.total_stages}")

        # Zero gradients before backward
        self.optimizer.zero_grad()

        if is_last:
            # Start backward pass from loss
            print(f"  🔄 Starting backward from loss: {loss.item():.4f}")
            loss.backward()
            print(f"   Backward pass computed")

            # Send gradient to previous worker
            if not is_first:
                grad = self.last_input.grad
                if grad is not None:
                    print(f"   Sending gradient to Rank {self.prev_rank} (P2P): {list(grad.shape)}")
                    self.p2p_client.send_tensor(grad, self.prev_rank)

        else:
            # Receive gradient from next worker via P2P
            print(f"   Waiting for gradient from Rank {self.next_rank} (P2P)...")
            received_grad = self.p2p_server.recv_tensor(self.next_rank)
            print(f"   Received gradient: {list(received_grad.shape)}")

            # Continue backward pass
            print(f"  🔄 Computing backward pass with received gradient...")
            self.last_output.backward(received_grad)
            print(f"   Backward pass computed")

            # Send gradient further back
            if not is_first:
                grad = self.last_input.grad
                if grad is not None:
                    print(f"   Sending gradient to Rank {self.prev_rank} (P2P): {list(grad.shape)}")
                    self.p2p_client.send_tensor(grad, self.prev_rank)

        # Update this stage's weights
        print(f"  🔧 Updating stage weights via optimizer...")
        self.optimizer.step()
        print(f"   Weights updated")

    def run_active(self):
        """
        Main loop for an ACTIVE worker.
        Keeps running until Master sends SIGNAL_STOP.
        """

        print(f"[Worker {self.rank}] Starting "
              f"active training loop...")

        epoch = 1

        while True:
            # Wait for signal from Master
            signal_val = recv_signal(src=0)

            # ── STOP signal ────────────────────────────────
            if signal_val == SIGNAL_STOP.item():
                print(f"[Worker {self.rank}] "
                      f"Received STOP — shutting down")
                break

            # ── START signal ───────────────────────────────
            elif signal_val == SIGNAL_START.item():
                is_last = (
                    self.stage_idx == self.total_stages - 1
                )

                # Forward pass
                output, loss = self.forward_pass()

                # Backward pass
                self.backward_pass(loss)

            # ── NEXT signal ────────────────────────────────
            elif signal_val == SIGNAL_NEXT.item():
                # New batch coming — increment counter
                epoch += 1

                # Save checkpoint if needed
                if should_save(epoch):
                    save_checkpoint(
                        self.stage,
                        self.stage_idx,
                        self.optimizer,
                        epoch
                    )

    def run_standby(self):
        """
        Loop for a STANDBY worker.
        Waits silently until promoted or stopped.
        """

        print(f"[Worker {self.rank}] On STANDBY — waiting...")

        while True:
            signal_val = recv_signal(src=0)

            # ── STOP signal ────────────────────────────────
            if signal_val == SIGNAL_STOP.item():
                print(f"[Worker {self.rank}] "
                      f"Received STOP — shutting down")
                break

            # ── PROMOTE signal ─────────────────────────────
            elif signal_val == SIGNAL_PROMOTE.item():
                print(f"[Worker {self.rank}] "
                      f"PROMOTED → becoming active!")

                # Receive new assignment
                self.receive_assignment()

                # Build stage and load latest checkpoint
                self.build_stage()

                # Setup P2P connections to neighbors
                self.setup_p2p()

                # Switch to active mode
                self.is_active = True
                self.run_active()
                break

    def run(self):
        """
        Main entry point for worker.
        Called from run.py on each worker machine.
        """

        try:
            # ── Step 1: Connect to network ─────────────────────
            self._run_with_error_reporting("network setup", self.setup_network)

            # ── Step 2: Announce to Master ─────────────────────
                # ── Step 2: Receive assignment ─────────────────────
            # Master sends either:
            #   assignment tensor → active worker
            #   SIGNAL_STANDBY   → standby worker
            signal_val = recv_signal(src=0)

            if signal_val == SIGNAL_STANDBY.item():
                # Put in standby mode
                self.is_active = False
                self.run_standby()

            elif signal_val == SIGNAL_ASSIGN.item():
                # Active worker receives assignment payload next.
                self.receive_assignment()
                self.is_active    = True

                # ── Step 4: Build CNN stage ────────────────────
                self.build_stage()

                # ── Step 5: Setup P2P connections to neighbors ──
                self.setup_p2p()

                # ── Step 6: Start heartbeat ────────────────────
                if HEARTBEAT_ENABLED:
                    self.heartbeat = HeartbeatSender(self.rank)
                    self.heartbeat.start()

                # ── Step 7: Run training loop ──────────────────
                self.run_active()

                # ── Step 7: Stop heartbeat ─────────────────────
                if self.heartbeat:
                    self.heartbeat.stop()

            else:
                raise RuntimeError(
                    f"[Worker {self.rank}] Unexpected startup signal: "
                    f"{signal_val}"
                )

        except Exception:
            print(f"\n[Worker {self.rank}] Fatal worker error:")
            print(traceback.format_exc())
            raise
        finally:
            # ── Step 8: Cleanup ────────────────────────────────
            print(f"[Worker {self.rank}] Cleaning up...")
            
            # Cleanup P2P connections
            if self.p2p_server:
                try:
                    self.p2p_server.stop()
                except Exception as e:
                    print(f"[Worker {self.rank}] P2P server cleanup failed: {e}")
            
            if self.p2p_client:
                try:
                    self.p2p_client.close_all()
                except Exception as e:
                    print(f"[Worker {self.rank}] P2P client cleanup failed: {e}")
            
            try:
                destroy_process_group()
            except Exception:
                print(f"[Worker {self.rank}] Process group cleanup failed:")
                print(traceback.format_exc())
            print(f"[Worker {self.rank}] Done ")