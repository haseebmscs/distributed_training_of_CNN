# Distributed CNN Training — Pipeline Parallelism

Train a CNN across multiple machines using pipeline parallelism.
The model is split into stages. Each machine handles one stage.

---

## How It Works

- Master    → coordinates everything, keeps records
- Workers   → each holds one stage of the CNN
- Pipeline  → data flows machine by machine like an assembly line

---

## Setup

### 1. Install dependencies
pip install -r requirements.txt

### 2. Find Master machine IP
Windows: ipconfig
Linux:   ifconfig

### 3. Set Master IP in config.py
MASTER_IP = "your.master.ip.here"

### 4. On MASTER machine run:
python run.py --role master

### 5. On each WORKER machine run:
python run.py --role worker

Workers connect automatically. Training starts when
minimum workers have joined.

---

## Key Features

- Dynamic worker joining (up to 50 machines)
- CNN auto-splits based on number of workers
- Max 5 active pipeline stages
- Extra machines go on STANDBY as backup
- Heartbeat monitoring detects dead workers
- Standby workers promoted automatically on failure
- FP16 compression for faster network transfer
- Checkpointing every N epochs
- Full training log saved by Master

---

## File Structure

distributed-cnn/
├── master/
│   ├── master.py       ← master coordination logic
│   ├── registry.py     ← tracks all workers
│   └── scheduler.py    ← assigns stages dynamically
├── worker/
│   └── worker.py       ← generic worker (any stage)
├── models/
│   └── pipeline_model.py ← CNN split into N stages
├── comm/
│   ├── comm_utils.py   ← send/recv helpers
│   ├── signals.py      ← control signal constants
│   └── heartbeat.py    ← heartbeat thread
├── utils/
│   ├── checkpoint.py   ← save/load weights
│   └── logger.py       ← training log
├── dataset/
│   └── custom_dataset.py ← CIFAR-10 loader
├── config.py           ← all settings
└── run.py              ← entry point for all machines

---

## Supported Workers

Minimum : 2 workers
Maximum : 50 workers (5 active + rest on standby)