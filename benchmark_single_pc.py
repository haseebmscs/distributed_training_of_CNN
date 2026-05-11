"""
Single PC Training Benchmark
============================

Trains the same CNN pipeline model on a single PC (no distribution).
Use this to compare timing vs distributed training.

Run: python -u benchmark_single_pc.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys

from config import (
    EPOCHS, BATCH_SIZE, LEARNING_RATE,
    CHECKPOINT_EVERY, DATA_ROOT, NUM_CLASSES,
    LOG_DIR
)
from models.pipeline_model import split_model
from dataset.custom_dataset import get_dataloaders
from utils.logger import TrainingLogger
from utils.checkpoint import save_checkpoint, should_save
import os


class SinglePCPipelineTrainer:
    """
    Single PC trainer using the same pipeline model as distributed system.
    Runs all stages sequentially on one machine for baseline comparison.
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SinglePC] Using device: {self.device}")
        
        self.logger = TrainingLogger()
        
        # Load data
        print(f"[SinglePC] Loading CIFAR-10 dataset...")
        self.train_loader, self.test_loader = get_dataloaders()
        print(f"[SinglePC] Dataset loaded. Batches: {len(self.train_loader)}")
        
        # Build full pipeline model
        print(f"[SinglePC] Building pipeline model ({NUM_CLASSES} classes)...")
        self.all_stages = split_model(3)
        
        # Move all stages to device
        for i, stage in enumerate(self.all_stages):
            stage = stage.to(self.device)
            self.all_stages[i] = stage
        
        # Create single optimizer for all stages
        all_params = []
        for stage in self.all_stages:
            all_params.extend(stage.parameters())
        self.optimizer = optim.SGD(all_params, lr=LEARNING_RATE, momentum=0.9)
        
        print(f"[SinglePC] Model ready with {len(self.all_stages)} stages")
        self.criterion = nn.CrossEntropyLoss()
        
        # Timing
        self.epoch_times = []
        self.batch_times = []
    
    def forward_pass(self, images):
        """
        Run forward pass through all stages sequentially.
        
        Args:
            images: (B, 3, 32, 32) CIFAR-10 images
        
        Returns:
            output: (B, num_classes) logits
        """
        # Stage 1
        output = self.all_stages[0](images)
        
        # Stage 2
        output = self.all_stages[1](output)
        
        # Stage 3 (final)
        output = self.all_stages[2](output)
        
        return output
    
    def backward_pass(self):
        """
        Backward pass through all stages (PyTorch handles automatically).
        """
        self.optimizer.zero_grad()
        # Backward is called by the loss.backward()
        # which will flow back through all stages automatically
    
    def train_batch(self, images, labels):
        """
        Train on one batch.
        
        Args:
            images: (B, 3, 32, 32)
            labels: (B,)
        
        Returns:
            loss (float), accuracy (float)
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass
        logits = self.forward_pass(images)
        
        # Compute loss
        loss = self.criterion(logits, labels)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Compute accuracy
        _, predicted = torch.max(logits, 1)
        correct = (predicted == labels).sum().item()
        accuracy = 100.0 * correct / labels.size(0)
        
        return loss.item(), accuracy
    
    def train_epoch(self, epoch):
        """
        Train for one epoch.
        
        Args:
            epoch (int): epoch number
        
        Returns:
            avg_loss (float), avg_accuracy (float)
        """
        print(f"\n[SinglePC] === EPOCH {epoch}/{EPOCHS} ===")
        
        epoch_start = time.time()
        total_loss = 0.0
        total_acc = 0.0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader, 1):
            batch_start = time.time()
            
            # Train on batch
            loss, accuracy = self.train_batch(images, labels)
            total_loss += loss
            total_acc += accuracy
            
            batch_time = time.time() - batch_start
            self.batch_times.append(batch_time)
            
            if batch_idx % 50 == 0 or batch_idx == 1:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}: "
                      f"loss={loss:.4f}, acc={accuracy:.2f}%, time={batch_time:.3f}s")
            
            # Save checkpoint on configured epoch cadence.
            if should_save(epoch):
                for i, stage in enumerate(self.all_stages):
                    save_checkpoint(
                        stage, i, self.optimizer, epoch
                    )
        
        epoch_time = time.time() - epoch_start
        self.epoch_times.append(epoch_time)
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = total_acc / len(self.train_loader)
        
        print(f"\n[SinglePC] Epoch {epoch} Summary:")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Avg Accuracy: {avg_acc:.2f}%")
        print(f"  Epoch Time: {epoch_time:.2f}s")
        
        # Log to CSV
        self.logger.log_batch(epoch, len(self.train_loader), 
                             len(self.train_loader), avg_loss, avg_acc)
        
        return avg_loss, avg_acc
    
    def train(self):
        """
        Train for all epochs.
        """
        print("\n" + "="*60)
        print("  SINGLE PC TRAINING BENCHMARK")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Epochs: {EPOCHS}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Learning rate: {LEARNING_RATE}")
        print("="*60 + "\n")
        
        train_start = time.time()
        
        for epoch in range(1, EPOCHS + 1):
            avg_loss, avg_acc = self.train_epoch(epoch)
        
        total_time = time.time() - train_start
        
        print("\n" + "="*60)
        print("  TRAINING COMPLETE")
        print("="*60)
        print(f"Total training time: {total_time:.2f}s ({total_time/60:.2f}m)")
        print(f"Average time per epoch: {total_time/EPOCHS:.2f}s")
        print(f"Average time per batch: {sum(self.batch_times)/len(self.batch_times):.3f}s")
        print("="*60 + "\n")
        
        # Save timing report
        report_path = os.path.join(LOG_DIR, "single_pc_benchmark.txt")
        with open(report_path, "w") as f:
            f.write("SINGLE PC TRAINING BENCHMARK\n")
            f.write("="*60 + "\n")
            f.write(f"Total training time: {total_time:.2f}s\n")
            f.write(f"Epochs: {EPOCHS}\n")
            f.write(f"Batches per epoch: {len(self.train_loader)}\n")
            f.write(f"Average time per epoch: {total_time/EPOCHS:.2f}s\n")
            f.write(f"Average time per batch: {sum(self.batch_times)/len(self.batch_times):.3f}s\n")
            f.write("="*60 + "\n")
        
        print(f"Benchmark report saved to: {report_path}")


def main():
    try:
        trainer = SinglePCPipelineTrainer()
        trainer.train()
        
        print("\n[SinglePC] Training finished successfully!")
        return 0
    
    except Exception as e:
        print(f"\n[SinglePC] Fatal error:")
        print(f"{e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
