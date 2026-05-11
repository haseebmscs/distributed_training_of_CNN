import torch

# All signals explicitly torch.long to prevent
# type mismatches on Windows Python 3.14

SIGNAL_START     = torch.tensor([1], dtype=torch.long)
SIGNAL_NEXT      = torch.tensor([2], dtype=torch.long)
SIGNAL_STOP      = torch.tensor([0], dtype=torch.long)
SIGNAL_STANDBY   = torch.tensor([3], dtype=torch.long)
SIGNAL_PROMOTE   = torch.tensor([4], dtype=torch.long)
SIGNAL_HEARTBEAT = torch.tensor([5], dtype=torch.long)
SIGNAL_DONE      = torch.tensor([6], dtype=torch.long)
SIGNAL_HELLO     = torch.tensor([7], dtype=torch.long)
SIGNAL_ASSIGN    = torch.tensor([8], dtype=torch.long)
SIGNAL_READY     = torch.tensor([9], dtype=torch.long)
SIGNAL_RECONFIG  = torch.tensor([10], dtype=torch.long)
SIGNAL_STEP      = torch.tensor([11], dtype=torch.long)