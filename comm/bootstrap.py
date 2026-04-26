import json
import os
import socket
import threading
import traceback

from config import MASTER_BOOTSTRAP_PORT


def _recv_exact(conn, size):
    data = bytearray()
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Connection closed during bootstrap message")
        data.extend(chunk)
    return bytes(data)


def _send_json(conn, payload):
    encoded = json.dumps(payload).encode("utf-8")
    conn.sendall(len(encoded).to_bytes(4, "big") + encoded)


def _recv_json(conn):
    size = int.from_bytes(_recv_exact(conn, 4), "big")
    raw = _recv_exact(conn, size)
    return json.loads(raw.decode("utf-8"))


class MasterBootstrapServer:
    def __init__(self, expected_workers, on_worker_assigned,
                 host="0.0.0.0", port=MASTER_BOOTSTRAP_PORT,
                 timeout=120):
        self.expected_workers = expected_workers
        self.on_worker_assigned = on_worker_assigned
        self.host = host
        self.port = port
        self.timeout = timeout
        self.running = False
        self.thread = None
        self._ready = threading.Event()
        self._complete = threading.Event()
        self._lock = threading.Lock()
        self._next_rank = 1
        self._server_error = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()
        self._ready.wait(timeout=5)
        if self._server_error:
            raise RuntimeError(self._server_error)

    def _serve(self):
        started_at = None
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind((self.host, self.port))
                server.listen()
                server.settimeout(1.0)
                started_at = __import__("time").time()
                self._ready.set()
                print(f"[Bootstrap] Listening on {self.host}:{self.port}")

                while self.running and self._next_rank <= self.expected_workers:
                    if self.timeout is not None and started_at is not None:
                        elapsed = __import__("time").time() - started_at
                        if elapsed > self.timeout:
                            raise TimeoutError(
                                f"Bootstrap timed out after {self.timeout}s; "
                                f"assigned {self._next_rank - 1}/{self.expected_workers} workers"
                            )

                    try:
                        conn, addr = server.accept()
                    except socket.timeout:
                        continue
                    with conn:
                        try:
                            peer_info = _recv_json(conn)
                            with self._lock:
                                rank = self._next_rank
                                self._next_rank += 1

                            response = {
                                "status": "ok",
                                "rank": rank,
                                "master_ip": peer_info.get("master_ip"),
                                "master_port": peer_info.get("master_port"),
                                "assigned_by": socket.gethostname(),
                            }
                            _send_json(conn, response)

                            if self.on_worker_assigned:
                                self.on_worker_assigned(rank, peer_info)

                            print(
                                f"[Bootstrap] Assigned rank {rank} to "
                                f"{peer_info.get('hostname', addr[0])}"
                            )
                        except Exception:
                            print("[Bootstrap] Worker bootstrap failed:")
                            print(traceback.format_exc())

                self._complete.set()
        except Exception:
            self._server_error = traceback.format_exc()
            self._ready.set()
            self._complete.set()
            print("[Bootstrap] Server failed:")
            print(self._server_error)

    def wait_until_complete(self, timeout=None):
        return self._complete.wait(timeout=timeout)

    def stop(self):
        self.running = False
        self._complete.set()
        if self.thread:
            self.thread.join(timeout=2)


def request_worker_rank(master_ip, bootstrap_port=MASTER_BOOTSTRAP_PORT,
                        world_size=None, timeout=15, retries=30,
                        retry_delay=1.0):
    peer_info = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "master_ip": master_ip,
        "master_port": bootstrap_port,
        "world_size": world_size,
    }

    last_error = None

    for attempt in range(1, retries + 1):
        try:
            with socket.create_connection((master_ip, bootstrap_port), timeout=timeout) as conn:
                _send_json(conn, peer_info)
                response = _recv_json(conn)

            if response.get("status") != "ok":
                raise RuntimeError(f"Bootstrap rejected worker: {response}")

            rank = response.get("rank")
            if rank is None:
                raise RuntimeError(f"Bootstrap response missing rank: {response}")

            print(f"[Bootstrap] Received rank {rank} from {master_ip}:{bootstrap_port}")
            return rank
        except Exception as exc:
            last_error = exc
            print(
                f"[Bootstrap] Attempt {attempt}/{retries} failed "
                f"for {master_ip}:{bootstrap_port}: {exc.__class__.__name__}: {exc}"
            )
            if attempt < retries:
                import time
                time.sleep(retry_delay)

    print("[Bootstrap] Failed to obtain worker rank:")
    print(traceback.format_exc())
    raise ConnectionError(
        f"Could not connect to bootstrap server at {master_ip}:{bootstrap_port}"
    ) from last_error