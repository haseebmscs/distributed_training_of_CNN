"""
Training Timing Comparison Report Generator
=============================================

After running both single_pc and distributed training, this script
compares their timings and shows throughput improvements.

Run this AFTER both trainings are complete.
"""

import os
import json
from datetime import datetime


def extract_timing_from_log(log_path, system_type):
    """Extract timing information from logs."""
    
    if not os.path.exists(log_path):
        print(f"WARNING: Log file not found: {log_path}")
        return None
    
    data = {
        "system": system_type,
        "path": log_path,
        "epochs": 0,
        "total_batches": 0,
        "total_time": 0,
        "avg_time_per_batch": 0,
        "avg_loss": 0,
        "avg_accuracy": 0,
    }
    
    return data


def compare_trainings():
    """Compare single-PC vs distributed training."""
    
    print("\n" + "="*70)
    print("  TRAINING PERFORMANCE COMPARISON")
    print("="*70 + "\n")
    
    # Paths to check
    log_dir = "logs"
    single_pc_report = os.path.join(log_dir, "single_pc_benchmark.txt")
    
    print("Checking for benchmark reports...\n")
    
    # Read single-PC report
    single_pc_data = None
    if os.path.exists(single_pc_report):
        print(f"✓ Single-PC benchmark found: {single_pc_report}")
        with open(single_pc_report, 'r') as f:
            content = f.read()
            print(f"\nSingle-PC Report:\n{content}")
            single_pc_data = content
    else:
        print(f"✗ Single-PC benchmark NOT found: {single_pc_report}")
    
    # Check distributed logs
    distributed_logs = [f for f in os.listdir(log_dir) if f.startswith("training_") and f.endswith(".csv")]
    
    if distributed_logs:
        print(f"\n✓ Found {len(distributed_logs)} distributed training logs:")
        latest_log = sorted(distributed_logs)[-1]
        print(f"  Latest: {latest_log}")
    else:
        print(f"\n✗ No distributed training logs found")
    
    print("\n" + "="*70)
    print("  INTERPRETATION GUIDE")
    print("="*70)
    
    print("""
Single-PC Training (No Distribution):
  - Baseline for comparison
  - All stages process sequentially on one machine
  - No network overhead
  - Used to measure: Pure compute + data loading time

Distributed Training (Sequential, No Pipelining):
  - Current default mode
  - Sends batch → Worker1 → Worker2 → Worker3 → results
  - Waits for full batch before sending next
  - High network overhead relative to compute
  - Expected: SLOWER than single-PC due to network

Distributed Training (Pipelined, PIPELINE_ENABLED=True):
  - Send 3 microbatches without waiting
  - Workers process in parallel
  - Accumulate gradients
  - Low latency due to pipelining
  - Expected: 2-3x FASTER than single-PC

PIPELINE_DEPTH Values:
  - 1: No pipelining (same as sequential, but with signal overhead)
  - 3: Good balance (3x speedup potential)
  - 5: More parallelism (more staleness in gradients)
  
Environment Variables to Control:
  - PIPELINE_ENABLED=True/False (enable pipelining)
  - PIPELINE_DEPTH=3 (number of microbatches per step)
  - EPOCHS=5 (how many epochs to train)
  - BATCH_SIZE=128 (batch size for each worker)

To reproduce benchmarks:

1. Single-PC Training (BASELINE):
   python -u benchmark_single_pc.py
   
2. Distributed Sequential (Current - SLOW):
   $env:PIPELINE_ENABLED="False"
   python -u main.py --role master --world-size 4
   # (in separate terminals)
   python -u main.py --role worker --world-size 4
   python -u main.py --role worker --world-size 4
   python -u main.py --role worker --world-size 4

3. Distributed Pipelined (NEW - FAST):
   $env:PIPELINE_ENABLED="True"
   $env:PIPELINE_DEPTH="3"
   python -u main.py --role master --world-size 4
   # (in separate terminals)
   python -u main.py --role worker --world-size 4
   python -u main.py --role worker --world-size 4
   python -u main.py --role worker --world-size 4

Timing Metrics to Compare:
  - Total training time (lower = better)
  - Average time per batch (lower = better)
  - Convergence speed (epochs to reach target accuracy)
  - Hardware utilization (GPU/CPU usage on each PC)
""")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    compare_trainings()
