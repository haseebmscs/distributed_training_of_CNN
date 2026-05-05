import json
import socket
import select
import threading
import time

from config import HEARTBEAT_INTERVAL, HEARTBEAT_TIMEOUT, HEARTBEAT_PORT, MASTER_IP


class HeartbeatSender:
    """
    Runs on each WORKER machine.
    Sends alive signal to Master every HEARTBEAT_INTERVAL seconds.
    """

    def __init__(self, rank, master_ip=MASTER_IP, heartbeat_port=HEARTBEAT_PORT):
        self.rank    = rank
        self.master_ip = master_ip
        self.heartbeat_port = heartbeat_port
        self.running = False
        self.thread  = None
        self._conn = None

    def _connect(self):
        if self._conn is not None:
            return self._conn
        self._conn = socket.create_connection((self.master_ip, self.heartbeat_port), timeout=5)
        self._conn.settimeout(5)
        print(f"[Heartbeat] Rank {self.rank} connected to {self.master_ip}:{self.heartbeat_port}")
        return self._conn

    def _send_json(self, payload):
        data = json.dumps(payload).encode("utf-8")
        conn = self._connect()
        conn.sendall(len(data).to_bytes(4, "big") + data)

    def _send_loop(self):
        while self.running:
            try:
                payload = {
                    "type": "heartbeat",
                    "rank": self.rank,
                    "timestamp": time.time(),
                }
                self._send_json(payload)
                print(f"[Heartbeat] Rank {self.rank} -> alive")
            except Exception as e:
                print(f"[Heartbeat] Send failed: {e}; reconnecting...")
                try:
                    if self._conn is not None:
                        self._conn.close()
                except Exception:
                    pass
                self._conn = None
            time.sleep(HEARTBEAT_INTERVAL)

    def start(self):
        self.running = True
        self.thread  = threading.Thread(
            target=self._send_loop,
            daemon=True
        )
        self.thread.start()
        print(f"[Heartbeat] Sender started rank {self.rank}")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        print(f"[Heartbeat] Sender stopped rank {self.rank}")


class HeartbeatMonitor:
    """
    Runs on MASTER machine.
    Watches all workers for signs of life.
    """

    def __init__(self, active_ranks, on_failure_callback,
                 host="0.0.0.0", port=HEARTBEAT_PORT):
        self.active_ranks        = list(active_ranks)
        self.on_failure_callback = on_failure_callback
        self.running             = False
        self.thread              = None
        self.last_seen = {rank: time.time() for rank in active_ranks}
        self.host = host
        self.port = port
        self.server_socket = None
        self.connections = {}

    def _recv_exact(self, conn, size):
        data = bytearray()
        while len(data) < size:
            chunk = conn.recv(size - len(data))
            if not chunk:
                raise ConnectionError("Connection closed")
            data.extend(chunk)
        return bytes(data)

    def _recv_json(self, conn):
        size = int.from_bytes(self._recv_exact(conn, 4), "big")
        raw = self._recv_exact(conn, size)
        return json.loads(raw.decode("utf-8"))

    def _monitor_loop(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen()
        self.server_socket.settimeout(1.0)
        print(f"[Monitor] Heartbeat listening on {self.host}:{self.port}")

        while self.running:
            try:
                conn, _ = self.server_socket.accept()
                conn.settimeout(1.0)
                try:
                    msg = self._recv_json(conn)
                    sender_rank = int(msg.get("rank", -1))
                    if sender_rank > 0:
                        self.connections[sender_rank] = conn
                        self.last_seen[sender_rank] = time.time()
                        print(f"[Monitor] Heartbeat from rank {sender_rank}")
                    else:
                        conn.close()
                except Exception:
                    conn.close()
            except Exception:
                pass

            if self.connections:
                try:
                    readable, _, _ = select.select(list(self.connections.values()), [], [], 0)
                except Exception:
                    readable = []

                for conn in readable:
                    try:
                        msg = self._recv_json(conn)
                        sender_rank = int(msg.get("rank", -1))
                        if sender_rank in self.last_seen:
                            self.last_seen[sender_rank] = time.time()
                            print(f"[Monitor] Heartbeat from rank {sender_rank}")
                    except Exception:
                        dead_rank = None
                        for rank, stored in list(self.connections.items()):
                            if stored is conn:
                                dead_rank = rank
                                break
                        if dead_rank is not None:
                            self.connections.pop(dead_rank, None)

            now = time.time()
            for rank in list(self.active_ranks):
                elapsed = now - self.last_seen.get(rank, now)
                if elapsed > HEARTBEAT_TIMEOUT:
                    print(f"[Monitor] ⚠️  Rank {rank} DEAD "
                          f"({elapsed:.1f}s silent)")
                    self.active_ranks.remove(rank)
                    self.on_failure_callback(rank)

            time.sleep(2)

    def start(self):
        self.running = True
        self.thread  = threading.Thread(
            target=self._monitor_loop, daemon=True
        )
        self.thread.start()
        print("[Monitor] Heartbeat monitor started")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        for conn in self.connections.values():
            try:
                conn.close()
            except Exception:
                pass
        self.connections.clear()
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
        print("[Monitor] Heartbeat monitor stopped")

    def update_active_ranks(self, new_ranks):
        self.active_ranks = list(new_ranks)
        now = time.time()
        for rank in new_ranks:
            if rank not in self.last_seen:
                self.last_seen[rank] = now