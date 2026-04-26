# ── Network Settings ──────────────────────────
# For single-machine testing use 127.0.0.1.
# For multi-machine runs set MASTER_IP to the master's LAN IP.
MASTER_IP   = "10.236.188.44"
MASTER_PORT = 29601
# Optional network interface pinning for gloo backend.
# Example (Windows): GLOO_SOCKET_IFNAME=Wi-Fi
GLOO_SOCKET_IFNAME = "Wi-Fi"
# Disable libuv rendezvous path on Windows to reduce hostname
# resolution issues when multiple adapters are present.
USE_LIBUV = False

# ── Worker Limits ─────────────────────────────
MAX_WORKERS  = 50  # max machines that can connect
MAX_ACTIVE   = 5   # max active pipeline stages
MIN_WORKERS  = 1  # minimum to start training

# ── Training Settings ─────────────────────────
EPOCHS       = 10
BATCH_SIZE   = 64
LEARNING_RATE = 0.01
DATA_LOADER_WORKERS = 0  # keep 0 for Windows multiprocessing stability

# ── Heartbeat Settings ────────────────────────
HEARTBEAT_INTERVAL = 5
HEARTBEAT_TIMEOUT  = 15
HEARTBEAT_ENABLED  = False  # keep disabled for local testing stability

# ── Checkpoint Settings ───────────────────────
CHECKPOINT_EVERY = 2
CHECKPOINT_DIR   = "checkpoints/"

# ── Logging ───────────────────────────────────
LOG_DIR = "logs/"

# ── Dataset ───────────────────────────────────
DATA_ROOT  = "dataset/data"
NUM_CLASSES = 10
