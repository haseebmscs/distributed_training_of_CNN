import os
import csv
import time
from datetime import datetime
from config import LOG_DIR


class TrainingLogger:
    """
    Master's record keeper.
    Tracks loss, accuracy, and events across all epochs.

    Creates two files:
        logs/training_log.csv  → numbers (open in Excel)
        logs/events.txt        → human readable events
    """

    def __init__(self):
        # Create logs folder if it doesn't exist
        os.makedirs(LOG_DIR, exist_ok=True)

        # File paths
        timestamp        = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path    = os.path.join(
            LOG_DIR, f"training_{timestamp}.csv"
        )
        self.events_path = os.path.join(
            LOG_DIR, f"events_{timestamp}.txt"
        )

        # In-memory history for plotting later
        self.history = {
            "epoch"     : [],
            "batch"     : [],
            "loss"      : [],
            "accuracy"  : [],
            "timestamp" : []
        }

        # Training start time
        self.start_time  = time.time()
        self.epoch_start = time.time()

        # Write CSV header
        self._write_csv_header()

        print(f"[Logger] Logging to:")
        print(f"  CSV    → {self.csv_path}")
        print(f"  Events → {self.events_path}")

    # ── CSV SETUP ─────────────────────────────────────────

    def _write_csv_header(self):
        """Writes column names to CSV file."""
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "batch", "loss",
                "accuracy", "elapsed_time"
            ])

    # ── LOGGING METHODS ───────────────────────────────────

    def log_batch(self, epoch, batch,
                  total_batches, loss, accuracy):
        """
        Called after every batch completes.
        Prints progress and saves to CSV.

        Args:
            epoch         (int)  : current epoch number
            batch         (int)  : current batch number
            total_batches (int)  : total batches per epoch
            loss          (float): batch loss value
            accuracy      (float): batch accuracy (%)
        """
        elapsed = time.time() - self.start_time

        # Print to terminal
        print(f"[Epoch {epoch}] "
              f"Batch {batch}/{total_batches} | "
              f"Loss: {loss:.4f} | "
              f"Acc: {accuracy:.2f}% | "
              f"Time: {elapsed:.1f}s")

        # Save to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, batch,
                f"{loss:.4f}",
                f"{accuracy:.2f}",
                f"{elapsed:.1f}"
            ])

        # Save to memory
        self.history["epoch"].append(epoch)
        self.history["batch"].append(batch)
        self.history["loss"].append(loss)
        self.history["accuracy"].append(accuracy)
        self.history["timestamp"].append(elapsed)

    def log_epoch(self, epoch, total_epochs,
                  avg_loss, avg_accuracy):
        """
        Called after every epoch completes.
        Prints epoch summary.

        Args:
            epoch        (int)  : completed epoch number
            total_epochs (int)  : total epochs
            avg_loss     (float): average loss for this epoch
            avg_accuracy (float): average accuracy for epoch
        """
        epoch_time = time.time() - self.epoch_start
        total_time = time.time() - self.start_time

        print(f"\n{'='*55}")
        print(f"  Epoch {epoch}/{total_epochs} Complete")
        print(f"  Avg Loss    : {avg_loss:.4f}")
        print(f"  Avg Accuracy: {avg_accuracy:.2f}%")
        print(f"  Epoch Time  : {epoch_time:.1f}s")
        print(f"  Total Time  : {total_time:.1f}s")
        print(f"{'='*55}\n")

        # Log epoch summary as event
        self.log_event(
            f"Epoch {epoch}/{total_epochs} done | "
            f"Loss: {avg_loss:.4f} | "
            f"Acc: {avg_accuracy:.2f}%"
        )

        # Reset epoch timer
        self.epoch_start = time.time()

    def log_event(self, message):
        """
        Logs any important system event to events.txt.
        Used for worker joins, failures, promotions etc.

        Args:
            message (str): event description
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        line      = f"[{timestamp}] {message}"

        # Print to terminal
        print(f"[Event] {line}")

        # Append to events file
        with open(self.events_path, "a") as f:
            f.write(line + "\n")

    def log_worker_joined(self, rank):
        """Logs when a new worker connects."""
        self.log_event(f"Worker rank {rank} joined")

    def log_worker_failed(self, rank):


