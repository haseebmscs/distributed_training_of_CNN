# ── Network Settings ──────────────────────────
MASTER_IP   = "10.66.68.44"  #  master machine's IP
MASTER_PORT = 29500            # any free port above 1024

# ── Worker Limits ─────────────────────────────
MAX_WORKERS  = 50   # maximum machines that can connect
MAX_ACTIVE   = 5    # maximum pipeline stages (CNN has 5 splittable units)
MIN_WORKERS  = 2    # training won't start until at least this many join

# ── Training Settings ─────────────────────────
EPOCHS       = 10
BATCH_SIZE   = 64
LEARNING_RATE = 0.01

# ── Heartbeat Settings ────────────────────────
HEARTBEAT_INTERVAL = 5    # worker sends alive signal every 5 seconds
HEARTBEAT_TIMEOUT  = 15   # if no signal in 15s → worker declared dead

# ── Checkpoint Settings ───────────────────────
CHECKPOINT_EVERY = 2              # save weights every 2 epochs
CHECKPOINT_DIR   = "checkpoints/" # folder where weights are saved

# ── Logging ───────────────────────────────────
LOG_DIR = "logs/"   # master saves loss/accuracy logs here

# ── Dataset ───────────────────────────────────
DATA_ROOT  = "dataset/data"  # where CIFAR-10 is stored
NUM_CLASSES = 10
DATA_ROOT = "dataset/data"