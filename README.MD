# Distributed CNN Training

This project trains a CNN on CIFAR-10 with a distributed, stage-based pipeline.
The model is split into stages and assigned to workers. The master coordinates batch flow, worker assignment, checkpoints, and fault handling.

## What the system does

- Master process: coordinates workers, assigns stages, logs progress, and handles failures
- Worker processes: run one pipeline stage each
- Pipeline training: batches move stage-by-stage so multiple workers can stay busy at once
- Standby support: extra workers can remain idle and be promoted if an active worker fails

## Why this is useful

- Faster training is possible when the number of workers matches the pipeline shape well
- Too many workers can reduce performance because network and scheduling overhead increase
- Model quality should remain the same when the same architecture, data, and optimizer settings are used

## Requirements

Install the Python packages first:

```powershell
pip install -r requirements.txt
```

The code depends on:
- PyTorch for the model and training loop
- torchvision for CIFAR-10 and transforms
- Pillow for image loading
- numpy and matplotlib for logs and plots

## Dataset

The project expects an organized folder structure under `dataset/data`:

```text
dataset/data/
	train/
		airplane/
		automobile/
		...
	test/
		airplane/
		automobile/
		...
```

If the organized dataset is missing, the loader can download and arrange CIFAR-10 automatically.

## Configuration

Most settings come from environment variables in [config.py](config.py).

Important values:

- `MASTER_IP`: master machine IP address
- `MASTER_PORT`: distributed training port
- `MASTER_BOOTSTRAP_PORT`: worker registration port
- `WORLD_SIZE`: passed on the command line, total processes including master
- `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`: training settings
- `MAX_ACTIVE`: maximum number of active pipeline stages
- `PIPELINE_ENABLED` and `PIPELINE_DEPTH`: pipeline behavior controls

Default local testing uses `127.0.0.1`.

## How to run

The launcher uses only `--role` and `--world-size`.

### Master

```powershell
python main.py --role master --world-size 4
```

### Worker

Start one worker per machine:

```powershell
python main.py --role worker --world-size 4
```

Repeat the worker command on each worker machine until you reach the configured `world-size`.

## Multi-machine run

1. Set `MASTER_IP` on every machine to the master machine address.
2. Make sure `MASTER_PORT` and `MASTER_BOOTSTRAP_PORT` are identical everywhere.
3. Start the master first.
4. Start each worker after the master is waiting for connections.

See [MULTI_MACHINE_RUNBOOK.md](MULTI_MACHINE_RUNBOOK.md) for the full checklist.

## Performance comparison

The project now supports a pipeline style distributed run, so you can compare:

- Single-PC baseline: one machine runs the full model
- Distributed pipeline: multiple workers split the model into stages

What you should expect:

- A suitable worker count can reduce epoch time
- More workers is not always better, because communication overhead can dominate
- When the split is balanced, training speed improves without changing the model architecture or output behavior

Example measured results from this workspace:

- Single-PC: 33.25s per epoch
- Distributed 3-worker pipeline: 25.6s in the last epoch of a 5-epoch run

## Key features

- Dynamic worker joining
- Automatic model splitting into stages
- Active and standby worker roles
- Heartbeat monitoring and failure detection
- Automatic promotion of standby workers
- Checkpoint save and restore
- CSV and event logging

## Project layout

```text
distributed_training_of_CNN/
	main.py
	config.py
	master/
	worker/
	comm/
	models/
	dataset/
	utils/
	logs/
	checkpoints/
```

## Notes

- The code is designed to run across different PCs, not only on one machine.
- If you increase workers beyond the useful split point, training may slow down because the network and synchronization cost grows.
- For slides or reports, compare time per epoch, loss, and accuracy side-by-side to show both speed and model behavior.