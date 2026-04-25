# Multi-Machine Runbook

This runbook is for launching the project across multiple machines on the same network.

## 1. Prerequisites

- Same project code on every machine
- Same Python and package versions on every machine
- Dataset available at the same relative path used by config.py
- Network connectivity from each worker to master

Install dependencies on every machine:

pip install -r requirements.txt

## 2. Master address and port

Pick one machine as master.

Find its LAN IPv4:
- Windows: ipconfig
- Linux: ifconfig or ip addr

Use the same values on all machines:
- MASTER_IP = <master_lan_ip>
- MASTER_PORT = <shared_port> (default 29601)

Recommended on Windows PowerShell (per terminal session):

$env:MASTER_IP = "10.0.0.2"
$env:MASTER_PORT = "29601"

You can also hardcode these in config.py instead.

## 3. Firewall and connectivity

Allow inbound TCP on MASTER_PORT on the master machine.

Quick connectivity test from each worker:

Test-NetConnection 10.0.0.2 -Port 29601

## 4. Launch order

World size means: 1 master + N workers.

For 2 workers:
- world-size = 3
- worker ranks = 1 and 2

Start in this order.

On master machine:

python main.py --role master --world-size 3

On worker machine 1:

python main.py --role worker --rank 1 --world-size 3

On worker machine 2:

python main.py --role worker --rank 2 --world-size 3

## 5. Expected healthy logs

Master should show:
- Network ready
- Worker rank joined events
- Stage assignments
- Epoch and batch logs with loss and accuracy

Worker should show:
- Network connected
- Assigned Stage X/Y
- Starting active training loop

## 6. Environment profile options

The project now supports environment-variable overrides in config.py.

Useful variables:
- MASTER_IP
- MASTER_PORT
- MIN_WORKERS
- MAX_ACTIVE
- EPOCHS
- BATCH_SIZE
- LEARNING_RATE
- DATA_LOADER_WORKERS
- HEARTBEAT_ENABLED

Example for local testing:

$env:MASTER_IP = "127.0.0.1"
$env:MASTER_PORT = "29601"
$env:HEARTBEAT_ENABLED = "false"
$env:DATA_LOADER_WORKERS = "0"

## 7. Notes

- The c10d warning mentioning mccs.local can appear on Windows and is usually non-fatal if training continues.
- HEARTBEAT_ENABLED is currently best kept false for local testing stability.
- Stop runs gracefully with Ctrl+C in each terminal.
