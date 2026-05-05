import json

import torch

from config import MAX_ACTIVE, MIN_WORKERS
from models.pipeline_model import split_model
from comm.signals import SIGNAL_STANDBY, SIGNAL_PROMOTE, SIGNAL_ASSIGN
from comm.comm_utils import send_signal
from comm import distributed_socket as dist_socket

class Scheduler:
    """
    Decides how to split CNN across workers.
    Assigns stages to workers.
    Handles promotions when workers fail.

    Works closely with Registry — reads registry
    to know who is available, updates it after
    each assignment.
    """

    def __init__(self, registry):
        """
        Args:
            registry (WorkerRegistry): the live worker table
        """
        self.registry    = registry
        self.stages      = []   # list of PipelineStage objects
        self.stage_map   = {}   # rank → stage index (0-based)
        self.num_stages  = 0

        print("[Scheduler] Initialised")

    # ── INITIAL ASSIGNMENT ────────────────────────────────

    def assign_stages(self):
        """
        Called once when enough workers have joined.

        Steps:
            1. Count available workers
            2. Cap at MAX_ACTIVE
            3. Split CNN into that many stages
            4. Assign one stage per active worker
            5. Remaining workers → STANDBY

        Returns:
            stages (list): PipelineStage objects
                           index 0 = Stage1, index 1 = Stage2...
        """

        # Get all registered workers
        all_ranks    = list(self.registry.workers.keys())
        total        = len(all_ranks)

        # How many will be ACTIVE?
        num_active   = min(total, MAX_ACTIVE)
        num_active   = max(num_active, MIN_WORKERS)

        # Sort ranks so lowest ranks get active roles
        all_ranks.sort()
        active_ranks  = all_ranks[:num_active]
        standby_ranks = all_ranks[num_active:]

        print(f"\n[Scheduler] Assigning stages:")
        print(f"  Total workers : {total}")
        print(f"  Active stages : {num_active}")
        print(f"  Standby pool  : {len(standby_ranks)}")

        # Split CNN into num_active stages
        self.stages     = split_model(num_active)
        self.num_stages = num_active

        # Assign one stage per active worker
        for stage_idx, rank in enumerate(active_ranks):
            stage_number = stage_idx + 1   # 1-based for display

            # Update registry
            self.registry.set_active(rank, stage_number)

            # Store rank → stage index mapping
            self.stage_map[rank] = stage_idx

            # Send stage config to worker
            self._send_stage_assignment(
                rank,
                stage_idx,
                num_active,
                active_order=active_ranks,
            )

        # Tell standby workers to wait
        for rank in standby_ranks:
            print(f"[Scheduler] Rank {rank} → STANDBY")
            send_signal(SIGNAL_STANDBY, dst=rank)

        return self.stages

    def _send_stage_assignment(self, rank, stage_idx, total_stages,
                               send_assign_signal=True, active_order=None):
        """
        Sends stage assignment info to a worker.
        Worker needs to know:
            - Which stage index it has (0-based)
            - Total number of stages
            - Who to send data to (next rank) - with P2P address
            - Who to receive data from (prev rank) - with P2P address

        Args:
            rank         (int): worker rank
            stage_idx    (int): 0-based stage index
            total_stages (int): total pipeline stages
        """
        # Tell worker an assignment payload is about to arrive.
        if send_assign_signal:
            send_signal(SIGNAL_ASSIGN, dst=rank)

        # Pack assignment into a tensor:
        # [stage_idx, total_stages]
        assignment = torch.tensor(
            [stage_idx, total_stages],
            dtype=torch.long
        )
        dist_socket.send_tensor(assignment, dst=rank)
        print(f"[Scheduler] Sent Stage {stage_idx+1} "
              f"assignment -> rank {rank}")

        # Now send P2P neighbor info as separate message
        # Get active ranks in pipeline order
        if active_order is None:
            active_order = self.get_pipeline_order()
        
        # Determine prev/next ranks
        prev_rank = None
        next_rank = None
        
        if rank in active_order:
            idx = active_order.index(rank)
            if idx > 0:
                prev_rank = active_order[idx - 1]
            if idx < len(active_order) - 1:
                next_rank = active_order[idx + 1]
        
        # Build P2P neighbor info
        p2p_info = {
            "prev_rank": prev_rank,
            "next_rank": next_rank,
            "neighbors": {}
        }
        
        # Get P2P address for each neighbor
        if prev_rank:
            worker_info = dist_socket._default_group.get_worker_info(prev_rank)
            p2p_info["neighbors"][prev_rank] = {
                "ip": worker_info.get("ip"),
                "p2p_port": worker_info.get("p2p_port")
            }
        
        if next_rank:
            worker_info = dist_socket._default_group.get_worker_info(next_rank)
            p2p_info["neighbors"][next_rank] = {
                "ip": worker_info.get("ip"),
                "p2p_port": worker_info.get("p2p_port")
            }
        
        # Serialize and send as tensor
        p2p_payload = json.dumps(p2p_info).encode("utf-8")
        p2p_tensor = torch.tensor(list(p2p_payload), dtype=torch.uint8)
        dist_socket.send_tensor(p2p_tensor, dst=rank)
        
        print(f"[Scheduler] Sent P2P info -> rank {rank}:")
        print(f"  Prev: {prev_rank}, Next: {next_rank}")

    # ── PIPELINE ORDER ────────────────────────────────────

    def get_pipeline_order(self):
        """
        Returns active worker ranks in pipeline order.
        i.e. sorted by their stage number.

        Returns:
            list of ranks in order e.g. [1, 3, 2]
            meaning data flows: rank1 → rank3 → rank2
        """
        return self.registry.get_active_ranks()

    def get_first_worker(self):
        """
        Returns rank of worker running Stage 1.
        Master sends batches to this worker.
        """
        order = self.get_pipeline_order()
        if order:
            return order[0]
        return None

    def get_last_worker(self):
        """
        Returns rank of worker running last stage.
        This worker computes loss and sends metrics to Master.
        """
        order = self.get_pipeline_order()
        if order:
            return order[-1]
        return None

    def get_next_rank(self, current_rank):
        """
        Given a worker's rank, returns who comes
        next in the pipeline.

        Args:
            current_rank (int): asking worker's rank

        Returns:
            next rank (int) or None if last stage
        """
        order = self.get_pipeline_order()
        if current_rank in order:
            idx = order.index(current_rank)
            if idx + 1 < len(order):
                return order[idx + 1]
        return None

    def get_prev_rank(self, current_rank):
        """
        Given a worker's rank, returns who comes
        before it in the pipeline.

        Args:
            current_rank (int): asking worker's rank

        Returns:
            previous rank (int) or None if first stage
        """
        order = self.get_pipeline_order()
        if current_rank in order:
            idx = order.index(current_rank)
            if idx - 1 >= 0:
                return order[idx - 1]
        return None

    # ── FAILURE HANDLING ──────────────────────────────────

    def handle_failure(self, dead_rank):
        """
        Called automatically when a worker dies.
        Finds a standby worker and promotes it
        to take over the dead worker's stage.

        Args:
            dead_rank (int): rank of dead worker
        """
        print(f"\n[Scheduler] ⚠️  Handling failure "
              f"of rank {dead_rank}")

        # What stage did the dead worker have?
        dead_stage_num = self.registry.get_stage(dead_rank)
        if dead_stage_num is None:
            print(f"[Scheduler] Rank {dead_rank} had no stage")
            return

        dead_stage_idx = dead_stage_num - 1   # convert to 0-based

        # Mark worker as failed in registry
        self.registry.set_failed(dead_rank)

        # Find a standby replacement
        replacement = self.registry.get_first_standby()

        if replacement is None:
            print("[Scheduler] ❌ No standby workers available!")
            print("[Scheduler] Pipeline cannot continue")
            return

        print(f"[Scheduler] Promoting rank {replacement} "
              f"→ Stage {dead_stage_num}")
        print(f"[Scheduler] Stage map before promotion: {self.stage_map}")

        # Update registry
        self.registry.set_active(replacement, dead_stage_num)
        self.stage_map[replacement] = dead_stage_idx

        # Send promotion signal and assignment to replacement
        send_signal(SIGNAL_PROMOTE, dst=replacement)
        self._send_stage_assignment(
            replacement,
            dead_stage_idx,
            self.num_stages,
            send_assign_signal=False
        )
        print(f"[Scheduler] ✅ Rank {replacement} "
              f"now handling Stage {dead_stage_num}")

        # Print updated pipeline
        print(f"[Scheduler] New pipeline order: "
              f"{self.get_pipeline_order()}")

        # Return the replacement rank so caller can wait for readiness
        return replacement


