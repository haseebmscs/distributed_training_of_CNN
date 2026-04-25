import torch
import torch.nn as nn
from config import MAX_ACTIVE

def make_block1():
    # Input:  (B, 3, 32, 32)
    # Output: (B, 32, 16, 16)
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )

def make_block2():
    # Input:  (B, 32, 16, 16)
    # Output: (B, 64, 8, 8)
    return nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )

def make_block3():
    # Input:  (B, 64, 8, 8)
    # Output: (B, 128, 4, 4)
    return nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )

def make_block4():
    # Input:  (B, 128, 4, 4)
    # Output: (B, 256, 4, 4)
    return nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Dropout(0.25)
    )

def make_classifier():
    # Input:  (B, 256, 4, 4) → flattened to (B, 4096)
    # Output: (B, 10)
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(256 * 4 * 4, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 10)
    )

class PipelineStage(nn.Module):
    """
    Represents ONE stage of the pipeline.
    A worker receives this and runs it.

    It contains one or more CNN units chained together.
    Example: Stage for Worker 1 might contain block1 + block2
    """

    def __init__(self, layers):
        super(PipelineStage, self).__init__()
        # nn.ModuleList tells PyTorch to track
        # these layers for gradient computation
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # Pass input through each layer in order
        for layer in self.layers:
            x = layer(x)
        return x


def split_model(num_workers):
    """
    Splits the CNN into num_workers stages automatically.

    Args:
        num_workers (int): number of active workers (2 to 5)

    Returns:
        stages (list of PipelineStage): one stage per worker

    Example:
        split_model(3) returns [Stage1, Stage2, Stage3]
        Stage1 has block1+block2
        Stage2 has block3+block4
        Stage3 has classifier
    """

    # Safety check — cap at MAX_ACTIVE
    num_workers = min(num_workers, MAX_ACTIVE)
    num_workers = max(num_workers, 2)

    # All 5 building blocks in order
    all_units = [
        make_block1(),      # index 0
        make_block2(),      # index 1
        make_block3(),      # index 2
        make_block4(),      # index 3
        make_classifier()   # index 4
    ]

    # Splitting rules
    # Key = num_workers
    # Value = list of lists (which unit indices go to which stage)
    split_plan = {
        2: [[0, 1, 2], [3, 4]],
        3: [[0, 1],    [2, 3], [4]],
        4: [[0, 1],    [2],    [3], [4]],
        5: [[0],       [1],    [2], [3], [4]],
    }

    plan = split_plan[num_workers]

    # Build one PipelineStage per worker
    stages = []
    for unit_indices in plan:
        layers = [all_units[i] for i in unit_indices]
        stage  = PipelineStage(layers)
        stages.append(stage)

    print(f"[split_model] CNN split into {num_workers} stages")
    for i, indices in enumerate(plan):
        names = ["Block1","Block2","Block3","Block4","Classifier"]
        unit_names = [names[j] for j in indices]
        print(f"  Stage {i+1} → {' + '.join(unit_names)}")

    return stages



if __name__ == "__main__":
    print("Testing 2 workers:")
    stages = split_model(2)
    x = torch.zeros(4, 3, 32, 32)   # fake batch of 4 images
    for i, stage in enumerate(stages):
        x = stage(x)
        print(f"  After Stage {i+1}: {x.shape}")

    print("\nTesting 3 workers:")
    stages = split_model(3)
    x = torch.zeros(4, 3, 32, 32)
    for i, stage in enumerate(stages):
        x = stage(x)
        print(f"  After Stage {i+1}: {x.shape}")