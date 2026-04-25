import time
import threading
import torch
import torch.distributed as dist
from config import HEARTBEAT_INTERVAL, HEARTBEAT_TIMEOUT
from comm.signals import SIGNAL_HEARTBEAT

class HeartbeatSender:
    """
    Runs on each WORKER machine.
    Sends 'I am alive' signal to Master every
    HEARTBEAT_INTERVAL seconds in the background.

    Usage:
        sender = HeartbeatSender(my_rank)
        sender.start()    # starts background thread
        ... training ...
        sender.stop()     # stops when training ends
    """

    def __init__(self, rank):
        self.rank    = rank
        self.running = False
        self.thread  = None

    def _send_loop(self):
        """
        The actual loop that runs in the background thread.
        Sends heartbeat signal then sleeps, forever.
        """
        while self.running:
            try:
                # Send alive signal to Master (rank 0)
                signal = torch.tensor(
                    [self.rank],
                    dtype=torch.long
                )
                dist.send(signal, dst=0)
                print(f"[Heartbeat] Rank {self.rank} "
                      f"→ sent alive signal to Master")

            except Exception as e:
                print(f"[Heartbeat] Failed to send: {e}")
                break

            # Sleep until next heartbeat
            time.sleep(HEARTBEAT_INTERVAL)

    def start(self):
        """Starts the background heartbeat thread."""
        self.running = True
        self.thread  = threading.Thread(
            target=self._send_loop,
            daemon=True   # dies automatically when main program ends
        )
        self.thread.start()
        print(f"[Heartbeat] Sender started for rank {self.rank}")

    def stop(self):
        """Stops the heartbeat thread cleanly."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        print(f"[Heartbeat] Sender stopped for rank {self.rank}")


class HeartbeatMonitor:
    """
    Runs on MASTER machine.
    Listens for heartbeats from all workers.
    Tracks last seen time for each worker.
    Detects dead workers automatically.

    Usage:
        monitor = HeartbeatMonitor(active_ranks, on_failure)
        monitor.start()
        ... training ...
        monitor.stop()
    """

    def __init__(self, active_ranks, on_failure_callback):
        """
        Args:
            active_ranks        (list): ranks of all active workers
                                        e.g. [1, 2, 3]
            on_failure_callback (func): function to call when a
                                        worker dies
                                        e.g. handle_failure(rank)
        """
        self.active_ranks        = active_ranks
        self.on_failure_callback = on_failure_callback
        self.running             = False
        self.thread              = None

        # Dictionary: rank → last time we heard from them
        # Starts as current time for all workers
        self.last_seen = {
            rank: time.time()
            for rank in active_ranks
        }

    def _monitor_loop(self):
        """
        Background thread that:
        1. Listens for heartbeat signals from any worker
        2. Updates last_seen time when signal arrives
        3. Checks if any worker has gone silent too long
        """
        while self.running:
            # Listen for any incoming heartbeat
            try:
                signal = torch.zeros(1, dtype=torch.long)
                # dist.recv with src=None receives from ANY rank
                info = dist.recv(signal, src=None)
                sender_rank = info  # which worker sent this

                # Update last seen time for this worker
                if sender_rank in self.last_seen:
                    self.last_seen[sender_rank] = time.time()
                    print(f"[Monitor] Heartbeat received "
                          f"from rank {sender_rank}")

            except Exception:
                pass

            # Check all workers for timeout
            now = time.time()
            for rank in list(self.active_ranks):
                elapsed = now - self.last_seen.get(rank, now)
                if elapsed > HEARTBEAT_TIMEOUT:
                    print(f"[Monitor] ⚠️  Rank {rank} is DEAD "
                          f"(no heartbeat for {elapsed:.1f}s)")
                    # Remove from active list
                    self.active_ranks.remove(rank)
                    # Call the failure handler
                    self.on_failure_callback(rank)

            # Check every 2 seconds
            time.sleep(2)

    def start(self):
        """Starts the background monitor thread."""
        self.running = True
        self.thread  = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.thread.start()
        print("[Monitor] Heartbeat monitor started")

    def stop(self):
        """Stops the monitor thread cleanly."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        print("[Monitor] Heartbeat monitor stopped")

    def update_active_ranks(self, new_ranks):
        """
        Called when a new worker joins or a standby
        gets promoted — updates who we are watching.
        """
        self.active_ranks = new_ranks
        # Add new ranks to last_seen
        now = time.time()
        for rank in new_ranks:
            if rank not in self.last_seen:
                self.last_seen[rank] = now


# This function will be written in master.py
# We pass it to HeartbeatMonitor as on_failure_callback

def handle_failure(dead_rank):
    """
    Called automatically when a worker dies.
    Master uses registry to find a standby worker
    and promote it to replace the dead one.
    """
    print(f"[Master] Handling failure of rank {dead_rank}")

    # Find what stage the dead worker had
    dead_stage = registry.get_stage(dead_rank)

    # Find a standby worker to replace it
    replacement = registry.get_standby_worker()

    if replacement:
        print(f"[Master] Promoting rank {replacement} "
              f"to replace rank {dead_rank}")
        # Tell replacement worker to take over
        # (we build this in master.py)
    else:
        print("[Master] No standby workers available!")


