# main.py
# ─────────────────────────────────────────────────────────
# IMPORTANT: Socket patch must happen FIRST before
# any other imports — especially before torch imports.
# Otherwise gloo caches the wrong hostname resolution.
# ─────────────────────────────────────────────────────────

import socket
import argparse
import sys

# ── Step 1: Read IPs from config BEFORE torch imports ────
# We manually read config here to avoid circular imports
import importlib.util, os, pathlib

_cfg_path = pathlib.Path(__file__).parent / "config.py"
_spec     = importlib.util.spec_from_file_location("config", _cfg_path)
_cfg      = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)

MASTER_IP  = _cfg.MASTER_IP
WORKER_IP  = _cfg.WORKER_IP
MASTER_PORT = _cfg.MASTER_PORT
GLOO_SOCKET_IFNAME = _cfg.GLOO_SOCKET_IFNAME

# ── Step 2: Detect which machine we are ──────────────────
MY_HOSTNAME = socket.gethostname()

if MY_HOSTNAME == "Haris":
    MY_IP = MASTER_IP             # Master machine
else:
    MY_IP = WORKER_IP             # Worker machine

print(f"[Network] Hostname  : {MY_HOSTNAME}")
print(f"[Network] My IP     : {MY_IP}")
print(f"[Network] Master IP : {MASTER_IP}")

# ── Step 3: Patch socket BEFORE torch is imported ────────
# This intercepts gloo's hostname resolution and forces
# it to return the correct WiFi IP instead of vEthernet
_real_getaddrinfo   = socket.getaddrinfo
_real_gethostbyname = socket.gethostbyname

def _patched_getaddrinfo(host, port, *args, **kwargs):
    if host == MY_HOSTNAME:
        print(f"[Network] getaddrinfo: {host} → {MY_IP}")
        host = MY_IP
    return _real_getaddrinfo(host, port, *args, **kwargs)

def _patched_gethostbyname(host):
    if host == MY_HOSTNAME:
        print(f"[Network] gethostbyname: {host} → {MY_IP}")
        return MY_IP
    return _real_gethostbyname(host)

socket.getaddrinfo   = _patched_getaddrinfo
socket.gethostbyname = _patched_gethostbyname

print(f"[Network] Socket patch applied ✅")

# ── Step 4: Set env vars BEFORE torch imports ─────────────
os.environ["MASTER_ADDR"]        = MASTER_IP
os.environ["MASTER_PORT"]        = str(MASTER_PORT)
os.environ["GLOO_SOCKET_IFNAME"] = GLOO_SOCKET_IFNAME
os.environ["USE_LIBUV"]          = "0"

print(f"[Network] Env vars set ✅")
print(f"[Network] Now importing torch...")

# ── Step 5: NOW import torch (after patch is applied) ─────
import torch
import torch.distributed as dist

print(f"[Network] Torch imported ✅")

# ── Step 6: Normal argument parsing ──────────────────────
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
        help="Rank of this worker (1, 2, 3...)"
    )
    parser.add_argument(
        "--world-size",
        type=int,
        required=True,
        help="Total machines (1 master + N workers)"
    )

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