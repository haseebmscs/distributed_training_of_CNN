import argparse
import sys
import os

# Set env vars before torch import
from config import MASTER_IP, MASTER_PORT, GLOO_SOCKET_IFNAME

os.environ["MASTER_ADDR"]        = MASTER_IP
os.environ["MASTER_PORT"]        = str(MASTER_PORT)
os.environ["GLOO_SOCKET_IFNAME"] = GLOO_SOCKET_IFNAME
os.environ["USE_LIBUV"]          = "0"

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser(
        description="Distributed CNN Pipeline Training"
    )
    parser.add_argument("--role", type=str, required=True,
                        choices=["master", "worker"])
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--world-size", type=int, required=True)

    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  Distributed CNN Pipeline Training")
    print(f"  Role       : {args.role}")
    print(f"  World size : {args.world_size}")
    if args.role == "worker":
        print(f"  Rank       : {args.rank}")
    print(f"{'='*50}\n")

    if args.role == "master":
        from master.master import Master
        master = Master()
        master.run(world_size=args.world_size)

    elif args.role == "worker":
        if args.rank == 0:
            print("ERROR: rank 0 is reserved for Master!")
            sys.exit(1)
        from worker.worker import Worker
        worker = Worker(
            rank       = args.rank,
            world_size = args.world_size
        )
        worker.run()


if __name__ == "__main__":
    main()