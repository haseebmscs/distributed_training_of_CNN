# run.py
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Distributed CNN Pipeline Training"
    )

    parser.add_argument(
        "--role",
        type=str,
        required=True,
        choices=["master", "worker"],
        help="Role of this machine: master or worker"
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="Rank of this worker (1, 2, 3...). "
             "Ignored for master."
    )

    parser.add_argument(
        "--world-size",
        type=int,
        required=True,
        help="Total number of machines "
             "(1 master + N workers)"
    )

    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  Distributed CNN Pipeline Training")
    print(f"  Role       : {args.role}")
    print(f"  World size : {args.world_size}")
    if args.role == "worker":
        print(f"  Rank       : {args.rank}")
    print(f"{'='*50}\n")

    # ── Run as Master ──────────────────────────────────────
    if args.role == "master":
        from master.master import Master
        master = Master()
        master.run(world_size=args.world_size)

    # ── Run as Worker ──────────────────────────────────────
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