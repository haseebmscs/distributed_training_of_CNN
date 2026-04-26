import argparse
import sys
import os
import socket
import traceback

# Set env vars before torch import
from config import (
    MASTER_IP, MASTER_PORT, MASTER_BOOTSTRAP_PORT,
    GLOO_SOCKET_IFNAME
)

os.environ["MASTER_ADDR"]        = MASTER_IP
os.environ["MASTER_PORT"]        = str(MASTER_PORT)
os.environ["GLOO_SOCKET_IFNAME"] = GLOO_SOCKET_IFNAME
os.environ["USE_LIBUV"]          = "0"

import torch
import torch.distributed as dist


def _get_local_ip_candidates():
    candidates = {"127.0.0.1", "localhost"}

    try:
        host_ips = socket.gethostbyname_ex(socket.gethostname())[2]
        candidates.update(host_ips)
    except socket.gaierror:
        pass

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            candidates.add(sock.getsockname()[0])
    except OSError:
        pass

    return candidates


def _resolve_role(requested_role):
    if requested_role != "auto":
        return requested_role

    if MASTER_IP in _get_local_ip_candidates():
        return "master"
    return "worker"


def main():
    parser = argparse.ArgumentParser(
        description="Distributed CNN Pipeline Training"
    )
    parser.add_argument("--role", type=str, default="auto",
                        choices=["auto", "master", "worker"])
    parser.add_argument("--world-size", type=int, required=True)

    args = parser.parse_args()
    role = _resolve_role(args.role)

    print(f"\n{'='*50}")
    print(f"  Distributed CNN Pipeline Training")
    print(f"  Role       : {role} (requested: {args.role})")
    print(f"  World size : {args.world_size}")
    print(f"  Master IP  : {MASTER_IP}")
    print(f"  Bootstrap  : {MASTER_BOOTSTRAP_PORT}")
    print(f"  Dist Port  : {MASTER_PORT}")
    print(f"{'='*50}\n")

    try:
        if role == "master":
            from master.master import Master
            master = Master()
            master.run(world_size=args.world_size)

        elif role == "worker":
            from comm.bootstrap import request_worker_rank
            from worker.worker import Worker

            worker_rank = request_worker_rank(
                master_ip=MASTER_IP,
                bootstrap_port=MASTER_BOOTSTRAP_PORT,
                world_size=args.world_size,
            )

            if worker_rank == 0:
                print("ERROR: rank 0 is reserved for Master!")
                sys.exit(1)

            print(f"  Rank       : {worker_rank}")

            worker = Worker(
                rank       = worker_rank,
                world_size = args.world_size
            )
            worker.run()
    except Exception:
        print("\n[main] Fatal error:")
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()