import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import socket
import traceback
from datetime import timedelta

from config import (
    MASTER_IP, MASTER_PORT,
    EPOCHS, LEARNING_RATE, HEARTBEAT_ENABLED,
    GLOO_SOCKET_IFNAME, USE_LIBUV
)
from models.pipeline_model import split_model, PipelineStage
from comm.comm_utils import (
    send_tensor, recv_tensor,
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

        print(f"[Worker {rank}] Initialised")
    
    def setup_network(self):
        import os
        import socket
        import subprocess

        # On Windows, Gloo needs the correct interface name
        # Find the interface that has MASTER_IP
        try:
            result = subprocess.run(
                ['ipconfig'],
                capture_output=True,
                text=True
            )
            output = result.stdout
            
            # Look for the interface with MASTER_IP or use the one with internet access
            lines = output.split('\n')
            current_adapter = None
            for line in lines:
                if 'adapter' in line.lower() and ':' in line:
                    current_adapter = line.split(':')[0].strip()
                if '192.168' in line and current_adapter:  # Look for LAN adapters
                    print(f"[Worker {self.rank}] Found LAN adapter: {current_adapter}")
                    if "Wi-Fi" in current_adapter or "Ethernet" in current_adapter:
                        os.environ["GLOO_SOCKET_IFNAME"] = current_adapter
                    break
        except Exception as e:
            print(f"[Worker {self.rank}] Warning: Could not auto-detect interface: {e}")

        os.environ["USE_LIBUV"] = "0"

        print(f"[Worker {self.rank}] Connecting to master...")
        print(f"  Hostname   : {socket.gethostname()}")
        print(f"  MASTER_IP  : {MASTER_IP}")
        print(f"  PORT       : {MASTER_PORT}")
        print(f"  Rank       : {self.rank}")
        print(f"  World size : {self.world_size}")

        dist.init_process_group(
            backend    = "gloo",
            init_method = f"tcp://{MASTER_IP}:{MASTER_PORT}",
            world_size = self.world_size,
            rank       = self.rank,
            timeout    = timedelta(seconds=30)
        )

        print(f"[Worker {self.rank}] Connected ✅")

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
        Master sends a tensor: [stage_idx, total_stages]

        Sets:
            self.stage_idx    → which stage this worker runs
            self.total_stages → total pipeline stages
        """
        assignment = torch.zeros(2, dtype=torch.long)
        dist.recv(assignment, src=0)

        self.stage_idx    = assignment[0].item()
        self.total_stages = assignment[1].item()

        print(f"[Worker {self.rank}] Assigned Stage "
              f"{self.stage_idx + 1}/{self.total_stages}")

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

        # Check if resuming from checkpoint
        latest = find_latest_checkpoint(self.stage_idx)
        if latest > 0:
            print(f"[Worker {self.rank}] Resuming from "
                  f"epoch {latest}")
            load_checkpoint(
                self.stage,
                self.stage_idx,
                self.optimizer,
                latest
            )
        else:
            print(f"[Worker {self.rank}] Starting fresh")

        print(f"[Worker {self.rank}] Stage "
              f"{self.stage_idx + 1} built ✅")

    def forward_pass(self):
        """
        Receives input, runs it through this stage,
        sends output to next worker.

        If this is the LAST stage:
            - Also receives labels from Master
            - Computes loss
            - Sends loss/accuracy back to Master

        Returns:
            output (tensor): result of this stage
            loss   (tensor): loss value (last stage only)
        """

        is_first = (self.stage_idx == 0)
        is_last  = (self.stage_idx == self.total_stages - 1)

        # ── Receive input ──────────────────────────────────
        if is_first:
            # First worker receives from Master (rank 0)
            received = recv_tensor(src=0)
        else:
            # Other workers receive from previous worker
            prev_rank = self.rank - 1
            received  = recv_tensor(src=prev_rank)

        # Track received tensor for backward pass
        received.requires_grad_(True)
        self.last_input = received

        # ── Forward through this stage ─────────────────────
        self.stage.train()
        output = self.stage(received)
        self.last_output = output

        # ── Send output or compute loss ────────────────────
        if is_last:
            # Receive labels from Master
            labels_float = recv_tensor(src=0)
            labels       = labels_float.long()

            # Compute loss
            criterion = nn.CrossEntropyLoss()
            loss      = criterion(output, labels)

            # Compute accuracy
            _, predicted = torch.max(output, 1)
            correct      = (predicted == labels).sum().item()
            accuracy     = 100.0 * correct / labels.size(0)

            # Send metrics to Master
            send_metrics(loss.item(), accuracy, dst=0)

            return output, loss

        else:
            # Send output to next worker
            next_rank = self.rank + 1
            send_tensor(output.detach(), dst=next_rank)

            return output, None

    def backward_pass(self, loss=None):
        """
        Receives gradients from next worker,
        runs backward pass through this stage,
        sends gradients to previous worker.

        If this is the LAST stage:
            - loss.backward() starts the backward pass
            - Sends gradient to previous worker

        If this is the FIRST stage:
            - Receives gradient from next worker
            - Runs backward with that gradient
            - No need to send further back

        Args:
            loss (tensor): loss value — only for last stage
        """

        is_first = (self.stage_idx == 0)
        is_last  = (self.stage_idx == self.total_stages - 1)

        # Zero gradients before backward
        self.optimizer.zero_grad()

        if is_last:
            # Start backward pass from loss
            loss.backward()

            # Send gradient to previous worker
            if not is_first:
                grad = self.last_input.grad
                if grad is not None:
                    prev_rank = self.rank - 1
                    send_tensor(grad, dst=prev_rank)

        else:
            # Receive gradient from next worker
            next_rank     = self.rank + 1
            received_grad = recv_tensor(src=next_rank)

            # Continue backward pass
            self.last_output.backward(received_grad)

            # Send gradient further back
            if not is_first:
                grad = self.last_input.grad
                if grad is not None:
                    prev_rank = self.rank - 1
                    send_tensor(grad, dst=prev_rank)

        # Update this stage's weights
        self.optimizer.step()

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

                # ── Step 5: Start heartbeat ────────────────────
                if HEARTBEAT_ENABLED:
                    self.heartbeat = HeartbeatSender(self.rank)
                    self.heartbeat.start()

                # ── Step 6: Run training loop ──────────────────
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
            try:
                dist.destroy_process_group()
            except Exception:
                print(f"[Worker {self.rank}] Process group cleanup failed:")
                print(traceback.format_exc())
            print(f"[Worker {self.rank}] Done ✅")