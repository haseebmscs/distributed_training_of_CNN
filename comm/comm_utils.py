import torch
from comm.distributed_socket import recv_tensor, send_tensor


def send_signal(signal, dst):
    """Send a control signal tensor to a destination rank."""
    send_tensor(signal.clone(), dst=dst)


def recv_signal(src):
    """Receive a control signal tensor from a source rank."""
    signal_tensor = recv_tensor(src=src)
    return signal_tensor.item()


def send_metrics(loss, accuracy, dst=0):
    """Send loss/accuracy metrics tensor to destination rank."""
    metrics = torch.tensor([loss, accuracy], dtype=torch.float32)
    send_tensor(metrics, dst=dst)


def recv_metrics(src):
    """Receive loss/accuracy metrics tensor from source rank."""
    metrics = recv_tensor(src=src)
    loss = metrics[0].item()
    accuracy = metrics[1].item()
    return loss, accuracy

