"""
P2P Data Transfer Module (GFS-style optimization).

Instead of relaying tensor data through master, workers establish
direct point-to-point connections for data transfer.

Master only handles control signals (START, STOP, etc.).
"""

import socket
import json
import pickle
import threading
import time
import traceback
from typing import Dict, Optional


class P2PDataServer:
    """
    Runs on each WORKER machine.
    Listens for incoming tensor data from other workers.
    """

    def __init__(self, rank, host="0.0.0.0", port=None):
        """
        Args:
            rank (int): worker rank
            host (str): bind address
            port (int): listening port (derived from rank if not provided)
        """
        self.rank = rank
        self.host = host
        self.port = port if port else (30000 + rank)  # Default: 30000 + rank
        self.server_socket = None
        self.running = False
        self.thread = None
        self._lock = threading.Lock()
        self._received_tensors = {}  # src_rank -> list of tensors
        self._tensor_event = threading.Event()

    def _send_json(self, conn, obj, timeout=30):
        """Send JSON with 4-byte length prefix."""
        try:
            conn.settimeout(timeout)
            data = json.dumps(obj).encode("utf-8")
            length = len(data).to_bytes(4, "big")
            conn.sendall(length + data)
        except Exception as e:
            raise RuntimeError(f"JSON send failed: {e}")

    def _recv_json(self, conn, timeout=30):
        """Receive JSON with 4-byte length prefix."""
        try:
            conn.settimeout(timeout)
            length_bytes = b""
            while len(length_bytes) < 4:
                chunk = conn.recv(4 - len(length_bytes))
                if not chunk:
                    raise ConnectionError("Connection closed")
                length_bytes += chunk
            
            length = int.from_bytes(length_bytes, "big")
            data = b""
            while len(data) < length:
                chunk = conn.recv(length - len(data))
                if not chunk:
                    raise ConnectionError("Connection closed")
                data += chunk
            
            return json.loads(data.decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"JSON recv failed: {e}")

    def _accept_connections(self):
        """Accept incoming tensor connections."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            try:
                self.server_socket.bind((self.host, self.port))
                self.server_socket.listen(10)
                print(f"[P2P-Server] Rank {self.rank} listening on {self.host}:{self.port}")
            except OSError as e:
                raise RuntimeError(f"Failed to bind P2P server: {e}")
            
            while self.running:
                self.server_socket.settimeout(1.0)
                try:
                    conn, addr = self.server_socket.accept()
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"[P2P-Server] Accept failed: {e}")
                    break
                
                # Handle connection in background thread
                handler_thread = threading.Thread(
                    target=self._handle_incoming,
                    args=(conn, addr),
                    daemon=True
                )
                handler_thread.start()
        
        except Exception as e:
            print(f"[P2P-Server] Fatal error: {e}")
            traceback.print_exc()

    def _handle_incoming(self, conn, addr):
        """Handle incoming tensor from another worker."""
        try:
            # Receive header with sender rank
            header = self._recv_json(conn, timeout=30)
            src_rank = header.get("src")
            
            # Receive tensor data
            data = self._recv_json(conn, timeout=30)
            payload = data.get("data")
            
            # Deserialize tensor
            tensor = pickle.loads(bytes.fromhex(payload))
            
            # Store tensor
            with self._lock:
                if src_rank not in self._received_tensors:
                    self._received_tensors[src_rank] = []
                self._received_tensors[src_rank].append(tensor)
                self._tensor_event.set()
            
            print(f"[P2P-Server] Rank {self.rank} received tensor from rank {src_rank}: {tensor.shape}")
            
            # Send acknowledgment
            ack = {"status": "ok"}
            self._send_json(conn, ack)
        
        except Exception as e:
            print(f"[P2P-Server] Error handling connection from {addr}: {e}")
        finally:
            try:
                conn.close()
            except:
                pass

    def start(self):
        """Start the P2P server."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(
            target=self._accept_connections,
            daemon=True
        )
        self.thread.start()
        print(f"[P2P-Server] Rank {self.rank} started on port {self.port}")

    def stop(self):
        """Stop the P2P server."""
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        if self.thread:
            self.thread.join(timeout=2)
        print(f"[P2P-Server] Rank {self.rank} stopped")

    def recv_tensor(self, src, timeout=60):
        """
        Receive a tensor from source rank.
        
        Args:
            src (int): source rank
            timeout (float): max seconds to wait (default: 60 seconds, increased from 30)
        
        Returns:
            tensor: the received tensor
        """
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            with self._lock:
                if src in self._received_tensors and self._received_tensors[src]:
                    tensor = self._received_tensors[src].pop(0)
                    if not self._received_tensors[src]:
                        self._received_tensors.pop(src)
                    return tensor
            
            # Wait for new tensor
            wait_time = min(1.0, deadline - time.time())
            self._tensor_event.clear()
            self._tensor_event.wait(timeout=wait_time)
        
        raise TimeoutError(f"Timed out waiting for tensor from rank {src}")


class P2PDataClient:
    """
    Used by each WORKER to send tensor data to other workers directly.
    Maintains persistent connections to neighbors.
    """

    def __init__(self, rank):
        """
        Args:
            rank (int): this worker's rank
        """
        self.rank = rank
        self._peer_connections = {}  # dst_rank -> socket
        self._neighbor_info = {}  # dst_rank -> (ip, port) for reconnection
        self._lock = threading.Lock()

    def _send_json(self, conn, obj, timeout=30):
        """Send JSON with 4-byte length prefix."""
        try:
            conn.settimeout(timeout)
            data = json.dumps(obj).encode("utf-8")
            length = len(data).to_bytes(4, "big")
            conn.sendall(length + data)
        except Exception as e:
            raise RuntimeError(f"JSON send failed: {e}")

    def _recv_json(self, conn, timeout=30):
        """Receive JSON with 4-byte length prefix."""
        try:
            conn.settimeout(timeout)
            length_bytes = b""
            while len(length_bytes) < 4:
                chunk = conn.recv(4 - len(length_bytes))
                if not chunk:
                    raise ConnectionError("Connection closed")
                length_bytes += chunk
            
            length = int.from_bytes(length_bytes, "big")
            data = b""
            while len(data) < length:
                chunk = conn.recv(length - len(data))
                if not chunk:
                    raise ConnectionError("Connection closed")
                data += chunk
            
            return json.loads(data.decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"JSON recv failed: {e}")

    def connect(self, dst_rank, dst_ip, dst_port, timeout=30, force_reconnect=False):
        """
        Establish connection to destination rank.
        
        Args:
            dst_rank (int): destination rank
            dst_ip (str): destination IP
            dst_port (int): destination port
            timeout (float): connection timeout
            force_reconnect (bool): force close and reconnect even if already connected
        """
        # Store neighbor info for reconnection attempts
        self._neighbor_info[dst_rank] = (dst_ip, dst_port)
        
        # Check if already connected
        if dst_rank in self._peer_connections and not force_reconnect:
            return  # Already connected
        
        # Close existing connection if force_reconnect
        if force_reconnect and dst_rank in self._peer_connections:
            try:
                self._peer_connections[dst_rank].close()
            except:
                pass
            self._peer_connections.pop(dst_rank, None)
        
        try:
            conn = socket.create_connection(
                (dst_ip, dst_port),
                timeout=timeout
            )
            
            with self._lock:
                self._peer_connections[dst_rank] = conn
            
            print(f"[P2P-Client] Rank {self.rank} connected to rank {dst_rank} at {dst_ip}:{dst_port}")
        
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to rank {dst_rank} at {dst_ip}:{dst_port}: {e}"
            )

    def send_tensor(self, tensor, dst_rank, timeout=30):
        """
        Send tensor to destination rank via direct P2P connection.
        Auto-reconnects if connection is stale.
        
        Args:
            tensor (torch.Tensor): tensor to send
            dst_rank (int): destination rank
            timeout (float): max seconds to wait
        """
        if dst_rank not in self._peer_connections:
            raise RuntimeError(
                f"Not connected to rank {dst_rank}. Call connect() first."
            )
        
        def _do_send():
            """Internal send attempt."""
            conn = self._peer_connections[dst_rank]
            
            # Serialize tensor
            payload = pickle.dumps(tensor.detach().cpu()).hex()
            
            # Send header
            header = {"src": self.rank, "dst": dst_rank}
            self._send_json(conn, header, timeout=timeout)
            
            # Send tensor data
            data = {"data": payload}
            self._send_json(conn, data, timeout=timeout)
            
            # Wait for ack
            ack = self._recv_json(conn, timeout=timeout)
            
            print(f"[P2P-Client] Rank {self.rank} sent tensor to rank {dst_rank}: {tensor.shape}")
        
        try:
            _do_send()
        except RuntimeError as e:
            # Check if it's a connection-related error wrapped in RuntimeError
            error_str = str(e).lower()
            if any(x in error_str for x in ['connection', 'aborted', 'reset', 'broken', 'closed', 'winerror 10053']):
                # Connection is dead, try to reconnect and retry once
                print(f"[P2P-Client] Rank {self.rank} connection to rank {dst_rank} died: {e}. Reconnecting...")
                
                try:
                    # Close dead socket
                    with self._lock:
                        if dst_rank in self._peer_connections:
                            try:
                                self._peer_connections[dst_rank].close()
                            except:
                                pass
                            self._peer_connections.pop(dst_rank, None)
                    
                    # Reconnect with stored neighbor info
                    if dst_rank not in self._neighbor_info:
                        raise RuntimeError(f"No stored neighbor info for rank {dst_rank}")
                    
                    dst_ip, dst_port = self._neighbor_info[dst_rank]
                    self.connect(dst_rank, dst_ip, dst_port, timeout=timeout, force_reconnect=True)
                    
                    # Retry send
                    print(f"[P2P-Client] Rank {self.rank} retrying send to rank {dst_rank}...")
                    _do_send()
                    
                except Exception as retry_err:
                    raise RuntimeError(f"Failed to send tensor to rank {dst_rank} (retry failed): {retry_err}")
            else:
                # Not a connection error
                with self._lock:
                    if dst_rank in self._peer_connections:
                        try:
                            self._peer_connections[dst_rank].close()
                        except:
                            pass
                        self._peer_connections.pop(dst_rank, None)
                
                raise RuntimeError(f"Failed to send tensor to rank {dst_rank}: {e}")
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError) as e:
            # Connection is dead, try to reconnect and retry once
            print(f"[P2P-Client] Rank {self.rank} connection to rank {dst_rank} died: {e.__class__.__name__}. Reconnecting...")
            
            try:
                # Close dead socket
                with self._lock:
                    if dst_rank in self._peer_connections:
                        try:
                            self._peer_connections[dst_rank].close()
                        except:
                            pass
                        self._peer_connections.pop(dst_rank, None)
                
                # Reconnect with stored neighbor info
                if dst_rank not in self._neighbor_info:
                    raise RuntimeError(f"No stored neighbor info for rank {dst_rank}")
                
                dst_ip, dst_port = self._neighbor_info[dst_rank]
                self.connect(dst_rank, dst_ip, dst_port, timeout=timeout, force_reconnect=True)
                
                # Retry send
                print(f"[P2P-Client] Rank {self.rank} retrying send to rank {dst_rank}...")
                _do_send()
                
            except Exception as retry_err:
                raise RuntimeError(f"Failed to send tensor to rank {dst_rank} (retry failed): {retry_err}")
        except Exception as e:
            # Other errors
            with self._lock:
                if dst_rank in self._peer_connections:
                    try:
                        self._peer_connections[dst_rank].close()
                    except:
                        pass
                    self._peer_connections.pop(dst_rank, None)
            
            raise RuntimeError(f"Failed to send tensor to rank {dst_rank}: {e}")

    def close_all(self):
        """Close all peer connections."""
        with self._lock:
            for conn in self._peer_connections.values():
                try:
                    conn.close()
                except:
                    pass
            self._peer_connections.clear()
        
        print(f"[P2P-Client] Rank {self.rank} closed all connections")
