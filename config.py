import os


def _env_int(name, default):
	val = os.getenv(name)
	if val is None:
		return default
	try:
		return int(val)
	except ValueError:
		return default


def _env_float(name, default):
	val = os.getenv(name)
	if val is None:
		return default
	try:
		return float(val)
	except ValueError:
		return default


def _env_bool(name, default):
	val = os.getenv(name)
	if val is None:
		return default
	return val.strip().lower() in ("1", "true", "yes", "on")


# ── Network Settings ──────────────────────────
# For single-machine testing use 127.0.0.1.
# For multi-machine runs set MASTER_IP to the master's LAN IP,
# or override with environment variable MASTER_IP.
MASTER_IP   = os.getenv("MASTER_IP", "127.0.0.1")
MASTER_PORT = _env_int("MASTER_PORT", 29601)

# ── Worker Limits ─────────────────────────────
MAX_WORKERS  = _env_int("MAX_WORKERS", 50)  # max machines that can connect
MAX_ACTIVE   = _env_int("MAX_ACTIVE", 5)    # max active pipeline stages
MIN_WORKERS  = _env_int("MIN_WORKERS", 2)   # minimum to start training

# ── Training Settings ─────────────────────────
EPOCHS       = _env_int("EPOCHS", 10)
BATCH_SIZE   = _env_int("BATCH_SIZE", 64)
LEARNING_RATE = _env_float("LEARNING_RATE", 0.01)
DATA_LOADER_WORKERS = _env_int(
	"DATA_LOADER_WORKERS", 0
)  # keep 0 for Windows multiprocessing stability

# ── Heartbeat Settings ────────────────────────
HEARTBEAT_INTERVAL = _env_int("HEARTBEAT_INTERVAL", 5)
HEARTBEAT_TIMEOUT  = _env_int("HEARTBEAT_TIMEOUT", 15)
HEARTBEAT_ENABLED  = _env_bool(
	"HEARTBEAT_ENABLED", False
)  # keep disabled for local testing stability

# ── Checkpoint Settings ───────────────────────
CHECKPOINT_EVERY = _env_int("CHECKPOINT_EVERY", 2)
CHECKPOINT_DIR   = os.getenv("CHECKPOINT_DIR", "checkpoints/")

# ── Logging ───────────────────────────────────
LOG_DIR = os.getenv("LOG_DIR", "logs/")

# ── Dataset ───────────────────────────────────
DATA_ROOT  = os.getenv("DATA_ROOT", "dataset/data")
NUM_CLASSES = _env_int("NUM_CLASSES", 10)
