# ── Network Settings ──────────────────────────
MASTER_IP   = "10.236.188.44"   # Master's WiFi IP
MASTER_PORT = 29601
WORKER_IP   = "10.236.188.11"   # Worker's WiFi IP

GLOO_SOCKET_IFNAME = "Wi-Fi"
USE_LIBUV          = False

# ── Worker Limits ─────────────────────────────
MAX_WORKERS  = 50
MAX_ACTIVE   = 5
MIN_WORKERS  = 1

# ── Training Settings ─────────────────────────
EPOCHS            = 10
BATCH_SIZE        = 64
LEARNING_RATE     = 0.01
DATA_LOADER_WORKERS = 0

# ── Heartbeat Settings ────────────────────────
HEARTBEAT_INTERVAL = 5
HEARTBEAT_TIMEOUT  = 15
HEARTBEAT_ENABLED  = False

# ── Checkpoint Settings ───────────────────────
CHECKPOINT_EVERY = 2
CHECKPOINT_DIR   = "checkpoints/"

# ── Logging ───────────────────────────────────
LOG_DIR = "logs/"

# ── Dataset ───────────────────────────────────
DATA_ROOT   = "dataset/data"
NUM_CLASSES = 10