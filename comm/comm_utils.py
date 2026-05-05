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
    """Send loss/accuracy metrics by wrapping in special 'metrics' tensor message."""
    metrics = torch.tensor([loss, accuracy], dtype=torch.float32)
    print(f"[send_metrics] Sending metrics: loss={loss:.4f}, accuracy={accuracy:.2f}% (shape: {metrics.shape}, values: {metrics.tolist()})")
    try:
        # Wrap metrics in a special format that includes the type
        import pickle
        from comm.distributed_socket import _default_group
        
        payload = pickle.dumps(metrics.detach().cpu()).hex()
        msg = {
            "type": "metrics_response",  # Special type to distinguish from regular tensors
            "src": _default_group.rank,
            "dst": int(dst),
            "data": payload,
            "loss": float(loss),
            "accuracy": float(accuracy),
        }
        
        master_conn = _default_group.peer_sockets.get(0)
        if master_conn is None:
            raise RuntimeError("No connection to master (rank 0)")
        _default_group._send_json(master_conn, msg, timeout=_default_group.timeout)
        print(f"[send_metrics] Successfully sent metrics to rank {dst}")
    except Exception as e:
        print(f"[send_metrics] ERROR: Failed to send metrics: {e}")
        raise


def recv_metrics(src):
    """Receive loss/accuracy metrics from source rank, filtering out signal messages."""
    print(f"[recv_metrics] Waiting for metrics from rank {src}...")
    try:
        import select
        import time
        from comm.distributed_socket import _default_group
        
        deadline = time.time() + _default_group.timeout
        
        # Keep reading messages until we get metrics
        while time.time() < deadline:
            conns = list(_default_group.peer_sockets.values())
            if not conns:
                raise RuntimeError("Master has no worker connections")
            
            wait_s = max(0.1, min(1.0, deadline - time.time()))
            readable, _, _ = select.select(conns, [], [], wait_s)
            
            for conn in readable:
                try:
                    msg = _default_group._recv_json(conn, timeout=5)
                    
                    # Check if this is a metrics message
                    if msg.get("type") == "metrics_response":
                        loss = float(msg.get("loss", 0.0))
                        accuracy = float(msg.get("accuracy", 0.0))
                        print(f"[recv_metrics] Received metrics: loss={loss:.4f}, accuracy={accuracy:.2f}%")
                        return loss, accuracy
                    elif msg.get("type") == "tensor":
                        # This is a tensor message (signal or data), queue it
                        msg_src = int(msg.get("src", -1))
                        import pickle
                        tensor = pickle.loads(bytes.fromhex(msg["data"]))
                        _default_group._queue_tensor(msg_src, tensor)
                        print(f"[recv_metrics] Queued tensor from rank {msg_src}, waiting for metrics")
                        continue
                    else:
                        # Unknown message type, skip it
                        print(f"[recv_metrics] Skipping unknown message type: {msg.get('type')}")
                        continue
                        
                except Exception as e:
                    print(f"[recv_metrics] Error reading message: {e}")
                    continue
        
        raise TimeoutError(f"Timeout waiting for metrics from rank {src}")
        
    except Exception as e:
        print(f"[recv_metrics] ERROR: Failed to receive metrics from rank {src}: {e}")
        raise

