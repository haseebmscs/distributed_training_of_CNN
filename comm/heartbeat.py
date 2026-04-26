import time
import threading
import torch
import torch.distributed as dist
from config import HEARTBEAT_INTERVAL, HEARTBEAT_TIMEOUT

HEARTBEAT_TAG = 999


class HeartbeatSender:
    """
    Runs on each WORKER machine.
    Sends alive signal to Master every HEARTBEAT_INTERVAL seconds.
    """

    def __init__(self, rank):
        self.rank    = rank
        self.running = False
        self.thread  = None

    def _send_loop(self):
        while self.running:
            try:
                signal = torch.tensor([self.rank], dtype=torch.long)
                dist.send(signal, dst=0, tag=HEARTBEAT_TAG)
                print(f"[Heartbeat] Rank {self.rank} → alive")
            except Exception as e:
                print(f"[Heartbeat] Send failed: {e}")
                break
            time.sleep(HEARTBEAT_INTERVAL)

    def start(self):
        self.running = True
        self.thread  = threading.Thread(
            target=self._send_loop,
            daemon=True
        )
        self.thread.start()
        print(f"[Heartbeat] Sender started rank {self.rank}")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        print(f"[Heartbeat] Sender stopped rank {self.rank}")


class HeartbeatMonitor:
    """
    Runs on MASTER machine.
    Watches all workers for signs of life.
    """

    def __init__(self, active_ranks, on_failure_callback):
        self.active_ranks        = list(active_ranks)
        self.on_failure_callback = on_failure_callback
        self.running             = False
        self.thread              = None
        self.last_seen = {rank: time.time() for rank in active_ranks}

    def _monitor_loop(self):
        while self.running:
            try:
                signal = torch.zeros(1, dtype=torch.long)
                sender_rank = dist.recv(
                    signal, src=None, tag=HEARTBEAT_TAG
                )
                if sender_rank in self.last_seen:
                    self.last_seen[sender_rank] = time.time()
                    print(f"[Monitor] Heartbeat from rank {sender_rank}")
            except Exception:
                pass

            now = time.time()
            for rank in list(self.active_ranks):
                elapsed = now - self.last_seen.get(rank, now)
                if elapsed > HEARTBEAT_TIMEOUT:
                    print(f"[Monitor] ⚠️  Rank {rank} DEAD "
                          f"({elapsed:.1f}s silent)")
                    self.active_ranks.remove(rank)
                    self.on_failure_callback(rank)

            time.sleep(2)

    def start(self):
        self.running = True
        self.thread  = threading.Thread(
            target=self._monitor_loop, daemon=True
        )
        self.thread.start()
        print("[Monitor] Heartbeat monitor started")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        print("[Monitor] Heartbeat monitor stopped")

    def update_active_ranks(self, new_ranks):
        self.active_ranks = list(new_ranks)
        now = time.time()
        for rank in new_ranks:
            if rank not in self.last_seen:
                self.last_seen[rank] = now