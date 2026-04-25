import os
import torch
from config import CHECKPOINT_DIR, CHECKPOINT_EVERY

def save_checkpoint(stage, stage_idx,
                    optimizer, epoch):
    """
    Saves a worker's stage weights to disk.
    Called by each worker every CHECKPOINT_EVERY epochs.

    Args:
        stage      (PipelineStage) : the model stage to save
        stage_idx  (int)           : 0-based stage index
        optimizer                  : the optimizer state to save
        epoch      (int)           : current epoch number

    Saves to:
        checkpoints/stage{N}_epoch{E}.pth
    """

    # Create checkpoints folder if needed
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Build filename
    # stage_idx is 0-based so we add 1 for display
    file_name = f"stage{stage_idx + 1}_epoch{epoch}.pth"
    file_path = os.path.join(CHECKPOINT_DIR, file_name)

    # Build checkpoint dictionary
    # We save both model weights AND optimizer state
    # Optimizer state contains momentum, learning rate etc.
    # Without it, training restarts differently after reload
    checkpoint = {
        "epoch"           : epoch,
        "stage_idx"       : stage_idx,
        "model_state"     : stage.state_dict(),
        "optimizer_state" : optimizer.state_dict()
    }

    # Save to disk
    torch.save(checkpoint, file_path)

    print(f"[Checkpoint] Stage {stage_idx+1} saved "
          f"→ {file_path}")

    # Also delete old checkpoints to save disk space
    # Keep only the last 2 checkpoints per stage
    _cleanup_old_checkpoints(stage_idx, epoch, keep=2)


def load_checkpoint(stage, stage_idx,
                    optimizer, epoch):
    """
    Loads saved weights back into a stage.
    Called when resuming training after a crash
    OR when a standby worker is promoted and
    needs the latest weights.

    Args:
        stage      (PipelineStage): model stage to load INTO
        stage_idx  (int)          : 0-based stage index
        optimizer                 : optimizer to restore
        epoch      (int)          : which epoch to load

    Returns:
        epoch (int): the epoch that was loaded
                     (so training resumes from correct point)
    """

    file_name = f"stage{stage_idx + 1}_epoch{epoch}.pth"
    file_path = os.path.join(CHECKPOINT_DIR, file_name)

    # Check if file exists
    if not os.path.isfile(file_path):
        print(f"[Checkpoint] No checkpoint found "
              f"at {file_path}")
        return 0   # start from epoch 0

    # Load the checkpoint
    checkpoint = torch.load(file_path)

    # Restore model weights
    stage.load_state_dict(checkpoint["model_state"])

    # Restore optimizer state
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    loaded_epoch = checkpoint["epoch"]
    print(f"[Checkpoint] Stage {stage_idx+1} loaded "
          f"← {file_path} (epoch {loaded_epoch})")

    return loaded_epoch

def find_latest_checkpoint(stage_idx):
    """
    Finds the most recent checkpoint for a given stage.
    Used when resuming after a crash — we don't know
    which epoch to load, so we find the latest one.

    Args:
        stage_idx (int): 0-based stage index

    Returns:
        latest epoch number (int) or 0 if none found
    """

    if not os.path.isdir(CHECKPOINT_DIR):
        return 0   # no checkpoints folder at all

    # List all files in checkpoints folder
    files = os.listdir(CHECKPOINT_DIR)

    # Filter files for this stage
    # Files look like: stage1_epoch4.pth
    prefix  = f"stage{stage_idx + 1}_epoch"
    matches = [f for f in files if f.startswith(prefix)]

    if not matches:
        return 0   # no checkpoints for this stage

    # Extract epoch numbers from filenames
    epochs = []
    for f in matches:
        try:
            # Remove prefix and .pth to get epoch number
            epoch_str = f.replace(prefix, "").replace(".pth", "")
            epochs.append(int(epoch_str))
        except ValueError:
            continue

    if not epochs:
        return 0

    latest = max(epochs)
    print(f"[Checkpoint] Latest checkpoint for "
          f"Stage {stage_idx+1}: epoch {latest}")
    return latest


def _cleanup_old_checkpoints(stage_idx, current_epoch, keep=2):
    """
    Deletes old checkpoints keeping only the most recent ones.
    Prevents disk from filling up during long training runs.

    Args:
        stage_idx     (int): 0-based stage index
        current_epoch (int): just-saved epoch
        keep          (int): how many recent checkpoints to keep
    """

    if not os.path.isdir(CHECKPOINT_DIR):
        return

    files  = os.listdir(CHECKPOINT_DIR)
    prefix = f"stage{stage_idx + 1}_epoch"

    # Find all checkpoints for this stage
    matches = []
    for f in files:
        if f.startswith(prefix):
            try:
                epoch_str = f.replace(prefix, "").replace(".pth","")
                epoch_num = int(epoch_str)
                matches.append((epoch_num, f))
            except ValueError:
                continue

    # Sort by epoch number oldest first
    matches.sort()

    # Delete oldest ones keeping only `keep` most recent
    while len(matches) > keep:
        old_epoch, old_file = matches.pop(0)
        old_path = os.path.join(CHECKPOINT_DIR, old_file)
        os.remove(old_path)
        print(f"[Checkpoint] Deleted old checkpoint: {old_file}")


def should_save(epoch):
    """
    Simple helper — returns True if this epoch
    should trigger a checkpoint save.

    Args:
        epoch (int): current epoch number

    Returns:
        True if epoch is a multiple of CHECKPOINT_EVERY
    """
    return epoch % CHECKPOINT_EVERY == 0

