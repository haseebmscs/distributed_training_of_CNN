"""
Socket-based distributed process group (replaces Gloo).

Provides:
- Rank initialization and validation
- Barrier synchronization
- Simple coordination protocol
- Works on Windows without hostname resolution issues
"""

import json
import select
import socket
import threading
import time
import traceback
from typing import Dict, Optional
from datetime import timedelta


class DistributedSocketGroup:
    """
    Manages distributed coordination via raw TCP sockets.
    Replaces torch.distributed.init_process_group for training coordination.
    """

    def __init__(self, backend="socket", world_size=1, rank=0, 
                 master_ip="127.0.0.1", master_port=29601, 
                 timeout=60):
        """
        Args:
            backend: ignored (for API compatibility)
            world_size: total processes
            rank: this process's rank (0=master, 1+=workers)
            master_ip: IP where master listens
            master_port: port for distributed communication
            timeout: seconds to wait for connections
        """
        self.backend = backend
        self.world_size = world_size
        self.rank = rank
        self.master_ip = master_ip
        self.master_port = master_port
        self.timeout = timeout
        
        # State
        self.is_master = (rank == 0)
        self.initialized = False
        self.peer_sockets = {}  # rank -> socket
        self.server_socket = None
        self.server_thread = None
        self._lock = threading.Lock()
        self._ready_event = threading.Event()
        self._error = None
        self._pending_tensors = {}
        
    def init_process_group(self):
        """Initialize the distributed group."""
        if self.is_master:
            self._init_master()
        else:
            self._init_worker()
            
        self.initialized = True
        print(f"[DistSocket] Rank {self.rank} initialized ({self.world_size} total)")

    def _init_master(self):
        """Master: listen for worker connections."""
        print(f"[DistSocket] Master starting on {self.master_ip}:{self.master_port}")
        
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.server_socket.bind((self.master_ip, self.master_port))
            self.server_socket.listen(self.world_size - 1)  # expect N-1 workers
            print(f"[DistSocket] Master listening for {self.world_size - 1} workers")
        except OSError as e:
            raise RuntimeError(f"Failed to bind master socket: {e}")
        
        # Start accepting worker connections in background
        self.server_thread = threading.Thread(
            target=self._master_accept_workers,
            daemon=True
        )
        self.server_thread.start()
        
        # Wait for all workers to connect
        if not self._ready_event.wait(timeout=self.timeout):
            raise TimeoutError(
                f"Master timed out waiting for {self.world_size - 1} workers "
                f"(only {len(self.peer_sockets)} connected)"
            )
        
        if self._error:
            raise RuntimeError(f"Master initialization failed: {self._error}")
        
        print(f"[DistSocket] Master ready with {len(self.peer_sockets)} workers")

    def _master_accept_workers(self):
        """Accept incoming worker connections."""
        try:
            start_time = time.time()
            expected = self.world_size - 1
            
            while len(self.peer_sockets) < expected:
                # Check timeout
                if time.time() - start_time > self.timeout:
                    with self._lock:
                        self._error = f"Timeout: only {len(self.peer_sockets)}/{expected} workers connected"
                    self._ready_event.set()
                    return
                
                self.server_socket.settimeout(1.0)
                try:
                    conn, addr = self.server_socket.accept()
                except socket.timeout:
                    continue
                except Exception as e:
                    with self._lock:
                        self._error = f"Accept failed: {e}"
                    self._ready_event.set()
                    return
                
                # Receive worker info
                try:
                    worker_info = self._recv_json(conn, timeout=5)
                    worker_rank = worker_info.get("rank")
                    
                    # Validate rank
                    if not worker_rank or worker_rank <= 0 or worker_rank >= self.world_size:
                        conn.close()
                        print(f"[DistSocket] Invalid rank {worker_rank} from {addr}")
                        continue
                    
                    if worker_rank in self.peer_sockets:
                        conn.close()
                        print(f"[DistSocket] Duplicate rank {worker_rank}")
                        continue
                    
                    # Store socket
                    with self._lock:
                        self.peer_sockets[worker_rank] = conn
                    
                    print(f"[DistSocket] Accepted rank {worker_rank} from {addr}")
                    
                    # Send acknowledgment
                    ack = {"status": "ok", "rank": worker_rank, "world_size": self.world_size}
                    self._send_json(conn, ack)
                    
                except Exception as e:
                    conn.close()
                    print(f"[DistSocket] Failed to handshake worker: {e}")
                    continue
            
            # All workers connected
            with self._lock:
                self._ready_event.set()
                
        except Exception as e:
            with self._lock:
                self._error = traceback.format_exc()
            self._ready_event.set()

    def _init_worker(self):
        """Worker: connect to master."""
        print(f"[DistSocket] Worker rank {self.rank} connecting to {self.master_ip}:{self.master_port}")
        
        start_time = time.time()
        last_error = None
        
        while time.time() - start_time < self.timeout:
            try:
                conn = socket.create_connection(
                    (self.master_ip, self.master_port),
                    timeout=5
                )
                
                # Send rank info
                info = {"rank": self.rank, "world_size": self.world_size}
                self._send_json(conn, info)
                
                # Receive acknowledgment
                ack = self._recv_json(conn, timeout=5)
                if ack.get("status") != "ok":
                    conn.close()
                    raise RuntimeError(f"Master rejected: {ack}")
                
                master_rank = ack.get("rank")
                if master_rank != self.rank:
                    conn.close()
                    raise RuntimeError(f"Rank mismatch: sent {self.rank}, got {master_rank}")
                
                # Store connection to master (rank 0)
                self.peer_sockets[0] = conn
                print(f"[DistSocket] Worker rank {self.rank} connected to master")
                return
                
            except Exception as e:
                last_error = e
                elapsed = time.time() - start_time
                print(f"[DistSocket] Connection attempt failed (elapsed {elapsed:.1f}s): {e}")
                time.sleep(0.5)
        
        raise ConnectionError(
            f"Worker rank {self.rank} failed to connect to master "
            f"at {self.master_ip}:{self.master_port} after {self.timeout}s: {last_error}"
        )

    def barrier(self, tag=0):
        """
        Synchronize all ranks (wait for all to reach this point).
        
        Args:
            tag: optional barrier ID (for multiple barriers)
        """
        if not self.initialized:
            raise RuntimeError("Process group not initialized")
        
        if self.is_master:
            self._barrier_master(tag)
        else:
            self._barrier_worker(tag)

    def _barrier_master(self, tag):
        """Master side: wait for all workers to signal."""
        signal = {"tag": tag, "type": "barrier_signal"}
        ack = {"tag": tag, "type": "barrier_ack"}
        
        # Tell all workers to wait
        for rank, conn in self.peer_sockets.items():
            try:
                self._send_json(conn, signal)
            except Exception as e:
                raise RuntimeError(f"Failed to signal rank {rank}: {e}")
        
        # Wait for all workers to ack
        for rank, conn in self.peer_sockets.items():
            try:
                msg = self._recv_json(conn, timeout=self.timeout)
                if msg.get("type") != "barrier_ack":
                    raise RuntimeError(f"Unexpected message from rank {rank}: {msg}")
            except Exception as e:
                raise RuntimeError(f"Failed barrier ack from rank {rank}: {e}")
        
        # Send final ready
        ready = {"tag": tag, "type": "barrier_ready"}
        for rank, conn in self.peer_sockets.items():
            try:
                self._send_json(conn, ready)
            except Exception as e:
                raise RuntimeError(f"Failed to send ready to rank {rank}: {e}")

    def _barrier_worker(self, tag):
        """Worker side: signal and wait for master."""
        master_conn = self.peer_sockets.get(0)
        if not master_conn:
            raise RuntimeError("Not connected to master")
        
        # Wait for signal from master
        try:
            msg = self._recv_json(master_conn, timeout=self.timeout)
            if msg.get("type") != "barrier_signal":
                raise RuntimeError(f"Unexpected barrier message: {msg}")
        except Exception as e:
            raise RuntimeError(f"Failed to receive barrier signal: {e}")
        
        # Send ack
        try:
            ack = {"tag": tag, "type": "barrier_ack"}
            self._send_json(master_conn, ack)
        except Exception as e:
            raise RuntimeError(f"Failed to send barrier ack: {e}")
        
        # Wait for ready
        try:
            msg = self._recv_json(master_conn, timeout=self.timeout)
            if msg.get("type") != "barrier_ready":
                raise RuntimeError(f"Unexpected barrier message: {msg}")
        except Exception as e:
            raise RuntimeError(f"Failed to receive barrier ready: {e}")

    def get_rank(self):
        """Return this process's rank."""
        return self.rank

    def get_world_size(self):
        """Return total number of processes."""
        return self.world_size

    def is_available(self):
        """Check if this backend is available."""
        return True

    @staticmethod
    def _send_json(conn, obj, timeout=30):
        """Send JSON object over socket with length prefix."""
        try:
            conn.settimeout(timeout)
            data = json.dumps(obj).encode("utf-8")
            length = len(data).to_bytes(4, "big")
            conn.sendall(length + data)
        except Exception as e:
            raise RuntimeError(f"JSON send failed: {e}")

    @staticmethod
    def _recv_json(conn, timeout=30):
        """Receive JSON object from socket with length prefix."""
        try:
            conn.settimeout(timeout)
            
            # Read length (4 bytes)
            length_bytes = b""
            while len(length_bytes) < 4:
                chunk = conn.recv(4 - len(length_bytes))
                if not chunk:
                    raise ConnectionError("Connection closed")
                length_bytes += chunk
            
            length = int.from_bytes(length_bytes, "big")
            
            # Read data
            data = b""
            while len(data) < length:
                chunk = conn.recv(length - len(data))
                if not chunk:
                    raise ConnectionError("Connection closed")
                data += chunk
            
            return json.loads(data.decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"JSON recv failed: {e}")

    def _queue_tensor(self, src, tensor):
        """Store out-of-order tensor until a matching recv_tensor(src)."""
        self._pending_tensors.setdefault(src, []).append(tensor)

    def _pop_queued_tensor(self, src):
        """Return queued tensor for src if available."""
        queued = self._pending_tensors.get(src)
        if queued:
            tensor = queued.pop(0)
            if not queued:
                self._pending_tensors.pop(src, None)
            return tensor
        return None

    def send_tensor(self, tensor, dst):
        """Send tensor to destination rank. Non-master sends are relayed via master."""
        import pickle

        if not self.initialized:
            raise RuntimeError("Process group not initialized")

        payload = pickle.dumps(tensor.detach().cpu()).hex()
        msg = {
            "type": "tensor",
            "src": self.rank,
            "dst": int(dst),
            "data": payload,
        }

        if self.rank == 0:
            if dst == 0:
                raise RuntimeError("Rank 0 cannot send tensor to itself")
            conn = self.peer_sockets.get(dst)
            if conn is None:
                raise RuntimeError(f"No connection to destination rank {dst}")
            self._send_json(conn, msg, timeout=self.timeout)
        else:
            master_conn = self.peer_sockets.get(0)
            if master_conn is None:
                raise RuntimeError("No connection to master (rank 0)")
            self._send_json(master_conn, msg, timeout=self.timeout)

    def recv_tensor(self, src=0):
        """Receive tensor from source rank. Master can also receive from workers."""
        import pickle

        if not self.initialized:
            raise RuntimeError("Process group not initialized")

        src = int(src)
        queued = self._pop_queued_tensor(src)
        if queued is not None:
            return queued

        deadline = time.time() + self.timeout

        if self.rank == 0:
            while time.time() < deadline:
                conns = list(self.peer_sockets.values())
                if not conns:
                    raise RuntimeError("Master has no worker connections")

                wait_s = max(0.1, min(1.0, deadline - time.time()))
                readable, _, _ = select.select(conns, [], [], wait_s)
                if not readable:
                    continue

                for conn in readable:
                    msg = self._recv_json(conn, timeout=self.timeout)
                    if msg.get("type") != "tensor":
                        raise RuntimeError(f"Unexpected message type on master: {msg}")

                    msg_src = int(msg.get("src", -1))
                    msg_dst = int(msg.get("dst", -1))
                    tensor = pickle.loads(bytes.fromhex(msg["data"]))

                    if msg_dst == 0:
                        if msg_src == src:
                            return tensor
                        self._queue_tensor(msg_src, tensor)
                    else:
                        dst_conn = self.peer_sockets.get(msg_dst)
                        if dst_conn is None:
                            raise RuntimeError(f"Cannot relay tensor to missing rank {msg_dst}")
                        self._send_json(dst_conn, msg, timeout=self.timeout)

            raise TimeoutError(f"Timed out waiting for tensor from rank {src}")

        master_conn = self.peer_sockets.get(0)
        if master_conn is None:
            raise RuntimeError("No connection to master (rank 0)")

        while time.time() < deadline:
            msg = self._recv_json(master_conn, timeout=self.timeout)
            if msg.get("type") != "tensor":
                raise RuntimeError(f"Unexpected message type on worker: {msg}")

            msg_src = int(msg.get("src", -1))
            msg_dst = int(msg.get("dst", -1))
            if msg_dst != self.rank:
                raise RuntimeError(f"Received tensor for rank {msg_dst} on rank {self.rank}")

            tensor = pickle.loads(bytes.fromhex(msg["data"]))
            if msg_src == src:
                return tensor
            self._queue_tensor(msg_src, tensor)

        raise TimeoutError(f"Timed out waiting for tensor from rank {src}")

    def destroy_process_group(self):
        """Clean up and close all connections."""
        for rank, conn in self.peer_sockets.items():
            try:
                conn.close()
            except:
                pass
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        self.peer_sockets.clear()
        self.initialized = False
        print(f"[DistSocket] Rank {self.rank} destroyed")


# Global singleton for compatibility with torch.distributed API
_default_group = None


def init_process_group(backend="socket", init_method=None, world_size=1, rank=0, timeout=60, **kwargs):
    """
    Initialize distributed process group (socket-based replacement for torch.distributed).
    
    Args:
        backend: ignored (always "socket")
        init_method: ignored (not needed for socket backend)
        world_size: total processes
        rank: this process's rank
        timeout: seconds to wait for connections
    """
    global _default_group
    
    # Parse rank and world_size from environment if available
    import os
    rank = int(os.getenv("RANK", rank))
    world_size = int(os.getenv("WORLD_SIZE", world_size))
    master_ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.getenv("MASTER_PORT", "29601"))
    
    _default_group = DistributedSocketGroup(
        backend=backend,
        world_size=world_size,
        rank=rank,
        master_ip=master_ip,
        master_port=master_port,
        timeout=timeout
    )
    _default_group.init_process_group()


def get_rank():
    """Get current process rank."""
    if _default_group is None:
        return 0
    return _default_group.get_rank()


def get_world_size():
    """Get total process count."""
    if _default_group is None:
        return 1
    return _default_group.get_world_size()


def send_tensor(tensor, dst):
    """
    Send a tensor to a destination rank (master to workers).
    Can only be called from rank 0.
    """
    if _default_group is None:
        raise RuntimeError("Process group not initialized")
    _default_group.send_tensor(tensor, dst)


def recv_tensor(src=0):
    """
    Receive a tensor from source rank (workers receive from master).
    Can only be called from workers (rank > 0).
    """
    if _default_group is None:
        raise RuntimeError("Process group not initialized")
    return _default_group.recv_tensor(src=src)


def barrier(async_op=False):
    """Barrier synchronization."""
    if _default_group is None:
        return
    _default_group.barrier()


def is_available():
    """Check if backend is available."""
    return True


def destroy_process_group():
    """Destroy the process group."""
    global _default_group
    if _default_group:
        _default_group.destroy_process_group()
        _default_group = None
