# comm/signals.py
# ─────────────────────────────────────────────
# Control signals sent between Master and Workers.
# These are just integer codes wrapped in tensors.
# Master sends these to tell workers what to do.
# ─────────────────────────────────────────────

import torch

# ── Signal Codes ──────────────────────────────
SIGNAL_START     = torch.tensor([1])  # start processing this batch
SIGNAL_NEXT      = torch.tensor([2])  # move to next batch
SIGNAL_STOP      = torch.tensor([0])  # training complete, shut down
SIGNAL_STANDBY   = torch.tensor([3])  # you are a backup, wait
SIGNAL_PROMOTE   = torch.tensor([4])  # standby → you are now active!
SIGNAL_HEARTBEAT = torch.tensor([5])  # I am alive
SIGNAL_DONE      = torch.tensor([6])  # I finished my work for this batch
SIGNAL_HELLO     = torch.tensor([7])  # worker announcing itself to master
SIGNAL_ASSIGN    = torch.tensor([8])  # master will send stage assignment next