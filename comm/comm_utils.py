import torch
import torch.distributed as dist

def send_tensor(tensor, dst):
    """
    Sends a tensor to another machine.
    Steps:
        1. Convert to FP16 (half bandwidth)
        2. Send shape info so receiver can prepare buffer
        3. Send actual tensor data asynchronously

    Args:
        tensor (torch.Tensor): tensor to send
        dst    (int)          : rank of destination machine
    """

    # Step 1: Convert to FP16 to save bandwidth
    tensor_fp16 = tensor.detach().half()

    # Step 2: Send shape so receiver knows buffer size
    # shape is sent as a 1D tensor of integers
    shape_tensor = torch.tensor(tensor_fp16.shape,
                                dtype=torch.long)

    # Send number of dimensions first
    ndim = torch.tensor([len(shape_tensor)], dtype=torch.long)
    dist.send(ndim, dst=dst)

    # Send the actual shape
    dist.send(shape_tensor, dst=dst)

    # Step 3: Send tensor data asynchronously
    # .contiguous() ensures memory layout is correct for sending
    handle = dist.isend(tensor_fp16.contiguous(), dst=dst)

    # Wait for send to complete before returning
    handle.wait()

    print(f"[comm] Sent tensor {tuple(tensor.shape)} "
          f"→ rank {dst} (FP16)")

def recv_tensor(src):
    """
    Receives a tensor from another machine.
    Steps:
        1. Receive shape info
        2. Create empty buffer of that shape
        3. Receive actual tensor data
        4. Convert back to FP32

    Args:
        src (int): rank of sending machine

    Returns:
        tensor (torch.Tensor): received tensor in FP32
    """

    # Step 1: Receive number of dimensions
    ndim = torch.zeros(1, dtype=torch.long)
    dist.recv(ndim, src=src)

    # Step 2: Receive shape
    shape_tensor = torch.zeros(ndim.item(), dtype=torch.long)
    dist.recv(shape_tensor, src=src)
    shape = tuple(shape_tensor.tolist())

    # Step 3: Create empty buffer and receive tensor
    buffer = torch.zeros(shape, dtype=torch.float16)
    handle = dist.irecv(buffer, src=src)
    handle.wait()

    # Step 4: Convert back to FP32
    tensor_fp32 = buffer.float()

    print(f"[comm] Received tensor {shape} "
          f"← rank {src} (FP32)")

    return tensor_fp32

def send_signal(signal, dst):
    """
    Sends a control signal to another machine.
    Signals are small integer tensors.

    Args:
        signal (torch.Tensor): signal constant from signals.py
        dst    (int)          : destination rank
    """
    dist.send(signal.clone(), dst=dst)

def recv_signal(src):
    """
    Receives a control signal from another machine.

    Args:
        src (int): rank of sender

    Returns:
        signal value as integer
    """
    buffer = torch.zeros(1, dtype=torch.long)
    dist.recv(buffer, src=src)
    return buffer.item()

def broadcast_signal(signal, src=0):
    """
    Master sends same signal to ALL workers at once.
    Used when Master wants everyone to stop or start.

    Args:
        signal (torch.Tensor): signal to broadcast
        src    (int)          : who is sending (always 0 = Master)
    """
    data = signal.clone().long()
    dist.broadcast(data, src=src)
    return data.item()

def send_metrics(loss, accuracy, dst=0):
    """
    Worker sends loss and accuracy to Master after each batch.
    Master uses this to log training progress.

    Args:
        loss     (float): batch loss value
        accuracy (float): batch accuracy value
        dst      (int)  : destination rank (always 0 = Master)
    """
    metrics = torch.tensor([loss, accuracy], dtype=torch.float32)
    dist.send(metrics, dst=dst)

def recv_metrics(src):
    """
    Master receives loss and accuracy from last worker.

    Args:
        src (int): rank of last worker

    Returns:
        loss (float), accuracy (float)
    """
    metrics = torch.zeros(2, dtype=torch.float32)
    dist.recv(metrics, src=src)
    loss     = metrics[0].item()
    accuracy = metrics[1].item()
    return loss, accuracy

