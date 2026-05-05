# Distributed CNN Training

---

## Slide 1: Project Goal

- Train a CNN across multiple machines using pipeline parallelism.
- Each worker owns one model stage; the master coordinates the whole run.
- The design supports late joiners, standby workers, and automatic failover.
- Goal of the project: keep training moving even if an active worker dies.

---

## Slide 2: High-Level Architecture

- `main.py` selects `master` or `worker` mode.
- `master/master.py` manages startup, stage assignment, logging, and recovery.
- `worker/worker.py` runs one stage of the CNN and participates in forward/backward flow.
- `comm/` contains control signals, heartbeat monitoring, bootstrap, and socket transport.
- `models/pipeline_model.py` splits the CNN into stages based on the active worker count.

---

## Slide 3: Startup and Assignment Flow

- Master starts first and waits for workers to register.
- `comm/bootstrap.py` assigns ranks and keeps accepting late joiners.
- `master/scheduler.py` chooses active workers up to `MAX_ACTIVE` and puts the rest on STANDBY.
- Active workers receive stage assignment plus P2P neighbor info.
- Workers build the correct model stage and prepare direct worker-to-worker links.

---

## Slide 4: Control Flow and Data Flow

- Control flow starts at the master: it assigns ranks, stages, start signals, next signals, and stop signals.
- The master talks to workers over the control channel, while tensors move through the P2P path.
- Workers do not discover neighbors themselves; the master gives each worker `prev_rank` and `next_rank`.
- Forward data flows worker to worker as activations, and backward data flows in the opposite direction as gradients.
- The last worker sends loss and accuracy back to the master, which then logs the batch and advances the pipeline.

---

## Slide 5: Training Data Flow

- Master sends `SIGNAL_START` to active workers for each batch.
- First stage receives images from the master.
- Intermediate stages forward activations directly to the next worker.
- Last stage receives labels, computes loss and accuracy, and sends metrics back.
- Master logs batch loss/accuracy and sends `SIGNAL_NEXT` to keep the pipeline aligned.

---

## Slide 6: Failure Handling and Standby Promotion

- `comm/heartbeat.py` detects silent workers using heartbeat timeouts.
- When a worker fails, `master/master.py` pauses the batch loop and calls the scheduler.
- `master/scheduler.py` picks the first STANDBY worker and promotes it.
- The promoted worker rebuilds its stage, restores checkpointed weights, and sends `SIGNAL_READY`.
- Master resumes training after the replacement is ready.

---

## Slide 7: What the System Shows

- Master terminal prints worker joins, stage assignments, batch loss, and accuracy.
- Worker terminals show the assigned stage, P2P traffic, forward pass, backward pass, and checkpoint saves.
- Checkpoints are saved per stage so a promoted worker can continue from the latest saved state.
- The project currently runs correctly for uninterrupted training; end-to-end kill-and-promotion behavior is the next validation target.
