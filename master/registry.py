import time
import threading
from config import MAX_WORKERS, MAX_ACTIVE, MIN_WORKERS


# Worker status codes
STATUS_ACTIVE  = "ACTIVE"    # currently in pipeline, doing work
STATUS_STANDBY = "STANDBY"   # connected, waiting as backup
STATUS_FAILED  = "FAILED"    # dead, no longer responding

class WorkerRegistry:
    """
    Tracks ALL workers connected to the Master.

    Master uses this to:
    - Know who is active in the pipeline
    - Find standby workers for replacement
    - Check if enough workers have joined
    - Get the pipeline order (rank → stage mapping)
    """

    def __init__(self):
        # Main storage: rank → worker info dictionary
        # Example entry:
        # {
        #   1: {
        #       "status"    : "ACTIVE",
        #       "stage"     : 1,
        #       "joined_at" : 1234567890.0,
        #       "last_seen" : 1234567890.0
        #   }
        # }
        self.workers = {}

        # Lock prevents two threads from
        # modifying registry at the same time
        self.lock = threading.Lock()

        print("[Registry] Initialised — waiting for workers")

    # ── REGISTRATION ──────────────────────────────────────

    def register(self, rank):
        """
        Called when a new worker connects to Master.
        Worker starts as STANDBY until assigned a stage.

        Args:
            rank (int): the new worker's rank number
        """
        with self.lock:
            # Check if we are at capacity
            if len(self.workers) >= MAX_WORKERS:
                print(f"[Registry] ⚠️  Rank {rank} rejected "
                      f"— max workers ({MAX_WORKERS}) reached")
                return False

            # Register with STANDBY status
            self.workers[rank] = {
                "status"    : STATUS_STANDBY,
                "stage"     : None,
                "joined_at" : time.time(),
                "last_seen" : time.time()
            }

            total = len(self.workers)
            print(f"[Registry] Rank {rank} registered "
                  f"— total workers: {total}/{MAX_WORKERS}")
            print(f"[Registry] Rank {rank} initial status: STANDBY")
            return True

    # ── STATUS UPDATES ────────────────────────────────────

    def set_active(self, rank, stage_number):
        """
        Promotes a worker from STANDBY → ACTIVE.
        Called by Scheduler when assigning a stage.

        Args:
            rank         (int): worker rank
            stage_number (int): which pipeline stage (1, 2, 3...)
        """
        with self.lock:
            if rank in self.workers:
                self.workers[rank]["status"] = STATUS_ACTIVE
                self.workers[rank]["stage"]  = stage_number
                print(f"[Registry] Rank {rank} → ACTIVE "
                      f"(Stage {stage_number})")

    def set_failed(self, rank):
        """
        Marks a worker as FAILED.
        Called by HeartbeatMonitor when a worker goes silent.

        Args:
            rank (int): rank of the dead worker
        """
        with self.lock:
            if rank in self.workers:
                self.workers[rank]["status"] = STATUS_FAILED
                self.workers[rank]["stage"]  = None
                print(f"[Registry] Rank {rank} → FAILED ⚠️")

    def update_heartbeat(self, rank):
        """
        Updates the last seen time for a worker.
        Called by HeartbeatMonitor when heartbeat arrives.

        Args:
            rank (int): rank that just sent a heartbeat
        """
        with self.lock:
            if rank in self.workers:
                self.workers[rank]["last_seen"] = time.time()

    # ── QUERIES ───────────────────────────────────────────

    def get_active_ranks(self):
        """
        Returns list of ranks currently ACTIVE in pipeline.
        Sorted by stage number so pipeline order is correct.

        Returns:
            list of ranks e.g. [1, 3, 5]
        """
        with self.lock:
            active = [
                (info["stage"], rank)
                for rank, info in self.workers.items()
                if info["status"] == STATUS_ACTIVE
            ]
            # Sort by stage number → pipeline order
            active.sort()
            return [rank for _, rank in active]

    def get_standby_ranks(self):
        """
        Returns list of ranks currently on STANDBY.
        Used to find replacement workers.

        Returns:
            list of ranks e.g. [2, 4, 6]
        """
        with self.lock:
            return [
                rank for rank, info in self.workers.items()
                if info["status"] == STATUS_STANDBY
            ]

    def get_stage(self, rank):
        """
        Returns which stage a worker is assigned to.

        Args:
            rank (int): worker rank

        Returns:
            stage number or None
        """
        with self.lock:
            if rank in self.workers:
                return self.workers[rank]["stage"]
            return None

    def get_first_standby(self):
        """
        Returns the first available standby worker rank.
        Used when promoting a replacement for a dead worker.

        Returns:
            rank (int) or None if no standby available
        """
        standbys = self.get_standby_ranks()
        if standbys:
            print(f"[Registry] Standby candidates: {standbys}")
            return standbys[0]
        print("[Registry] No standby candidates available")
        return None

    def is_ready_to_train(self):
        """
        Checks if enough workers have joined to start training.
        Training needs at least MIN_WORKERS.

        Returns:
            True if ready, False if still waiting
        """
        total = len(self.workers)
        ready = total >= MIN_WORKERS
        if not ready:
            print(f"[Registry] Waiting for workers: "
                  f"{total}/{MIN_WORKERS} minimum joined")
        return ready

    def count_active(self):
        """Returns number of currently ACTIVE workers."""
        with self.lock:
            return sum(
                1 for info in self.workers.values()
                if info["status"] == STATUS_ACTIVE
            )

    def count_total(self):
        """Returns total number of registered workers."""
        with self.lock:
            return len(self.workers)

    # ── DISPLAY ───────────────────────────────────────────

    def print_status(self):
        """
        Prints a formatted table of all workers.
        Master calls this after each epoch.
        """
        print("\n[Registry] Current Worker Status:")
        print(f"  {'Rank':>6} {'Status':>10} "
              f"{'Stage':>7} {'Last Seen':>12}")
        print("  " + "─" * 40)

        with self.lock:
            for rank in sorted(self.workers.keys()):
                info    = self.workers[rank]
                elapsed = time.time() - info["last_seen"]
                stage   = str(info["stage"]) \
                          if info["stage"] else "None"
                print(f"  {rank:>6} {info['status']:>10} "
                      f"{stage:>7} {elapsed:>10.1f}s ago")
        print()

