import os


def _env(name, default):
	value = os.getenv(name)
	return default if value is None or value == "" else value


def _env_int(name, default):
	value = os.getenv(name)
	return default if value is None or value == "" else int(value)


def _env_float(name, default):
	value = os.getenv(name)
	return default if value is None or value == "" else float(value)


def _env_bool(name, default):
	value = os.getenv(name)
	if value is None or value == "":
		return default
	return value.strip().lower() in {"1", "true", "yes", "on"}


# ── Network Settings ──────────────────────────
MASTER_IP   = _env("MASTER_IP", "10.236.188.44")   # Master's WiFi IP
MASTER_BOOTSTRAP_PORT = _env_int("MASTER_BOOTSTRAP_PORT", 29600)
MASTER_PORT = _env_int("MASTER_PORT", 29601)
WORKER_P2P_BASE_PORT  = _env_int("WORKER_P2P_BASE_PORT", 30000)  # P2P data transfers
WORKER_IP   = _env("WORKER_IP", "10.236.188.11")   # Worker's WiFi IP

GLOO_SOCKET_IFNAME = _env("GLOO_SOCKET_IFNAME", "Wi-Fi")
USE_LIBUV          = _env_bool("USE_LIBUV", False)

# ── Worker Limits ─────────────────────────────
MAX_WORKERS  = _env_int("MAX_WORKERS", 50)
MAX_ACTIVE   = _env_int("MAX_ACTIVE", 5)
MIN_WORKERS  = _env_int("MIN_WORKERS", 1)

# ── Training Settings ─────────────────────────
EPOCHS            = _env_int("EPOCHS", 1)
BATCH_SIZE        = _env_int("BATCH_SIZE", 128)
LEARNING_RATE     = _env_float("LEARNING_RATE", 0.01)
DATA_LOADER_WORKERS = _env_int("DATA_LOADER_WORKERS", 0)

# ── Heartbeat Settings ────────────────────────
HEARTBEAT_INTERVAL = _env_int("HEARTBEAT_INTERVAL", 5)
HEARTBEAT_TIMEOUT  = _env_int("HEARTBEAT_TIMEOUT", 15)
HEARTBEAT_ENABLED  = _env_bool("HEARTBEAT_ENABLED", False)

# ── Checkpoint Settings ───────────────────────
CHECKPOINT_EVERY = _env_int("CHECKPOINT_EVERY", 2)
CHECKPOINT_DIR   = _env("CHECKPOINT_DIR", "checkpoints/")

# ── Logging ───────────────────────────────────
LOG_DIR = _env("LOG_DIR", "logs/")

# ── Dataset ───────────────────────────────────
DATA_ROOT   = _env("DATA_ROOT", "dataset/data")
NUM_CLASSES = _env_int("NUM_CLASSES", 10)