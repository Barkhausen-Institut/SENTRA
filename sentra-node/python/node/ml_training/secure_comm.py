"""
Secure Communication Protocol for Multi-Node MPC
Enables secure channels between nodes and share reconstruction
"""

import os
from typing import List, Dict, Optional, Callable, Any, Union
from ml_training.secret_sharing import Share, ShamirSecretSharing
import socket
import sys
import ssl
import json
import threading
import time
from dataclasses import dataclass, asdict
from enum import Enum
import struct
import base64
from array import array
from typing import Sequence


class MessageType(Enum):
    """Types of messages in the secure communication protocol"""
    SHARE_EXCHANGE = "share_exchange"
    RECONSTRUCTION_REQUEST = "reconstruction_request"
    RECONSTRUCTION_RESPONSE = "reconstruction_response"
    HEARTBEAT = "heartbeat"
    ACK = "ack"
    SYNC = "sync"
    BATCH_SHARE_EXCHANGE = "batch_share_exchange"
    VECTOR_SHARE_EXCHANGE = "vector_share_exchange"
    VECTOR_SHARE_EXCHANGE_BIN = "vector_share_exchange_bin"
    VECTOR_PAIR_EXCHANGE_BIN = "vector_pair_exchange_bin"
    TRIPLE_REQUEST = "triple_request"


@dataclass
class Message:
    """Message structure for secure communication"""
    msg_type: str
    sender_id: int
    receiver_id: int
    data: Dict[str, Any]
    timestamp: float
    message_id: str = ""
    signature: Optional[str] = None  # For authentication


class SecureChannel:
    """
    Secure communication channel between nodes
    Provides authenticated and encrypted communication
    """
    
    def __init__(self, node_id: int, port: int = 8000, use_tls: bool = False, host: str = '0.0.0.0',
                 expected_membership_epoch: Optional[int] = None):
        """
        Initialize secure channel
        Args:
            node_id: ID of this node
            port: Port to listen on
            use_tls: Whether to use TLS encryption (default: False for testing)
            host: Host interface to bind to
            expected_membership_epoch: If set, reject/ignore SYNC messages with different e (stale epoch).
        """
        self.node_id = node_id
        self.port = port
        self.use_tls = use_tls
        self.host = host
        self.expected_membership_epoch = expected_membership_epoch
        self.connections: Dict[int, socket.socket] = {}
        self.server_socket: Optional[socket.socket] = None
        self.running = False
        self.message_handlers: Dict[str, Callable] = {}
        self.received_shares: Dict[str, List[Share]] = {}
        self.received_sync: Dict[str, set] = {}
        # context -> sender_id -> {'x': int, 'values': List[int]}
        self.received_vectors: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self.lock = threading.Lock()
        self.message_counter = 0
        # Optional callback(peer_id) when a send to peer_id fails (e.g. connection reset).
        self.on_send_failure: Optional[Callable[[int], None]] = None

    @staticmethod
    def _socket_recv_timeout_s() -> float:
        raw = os.environ.get("SENTRA_SOCKET_RECV_TIMEOUT", "120")
        try:
            return max(1.0, float(raw))
        except ValueError:
            return 120.0

    def _recv_exact(self, sock: socket.socket, nbytes: int, deadline: float) -> bytes:
        """Read exactly nbytes before deadline (large MPC vector payloads)."""
        buf = b""
        while len(buf) < nbytes:
            if time.time() >= deadline:
                raise socket.timeout(f"timed out reading {nbytes} bytes (got {len(buf)})")
            try:
                sock.settimeout(max(0.1, deadline - time.time()))
                chunk = sock.recv(nbytes - len(buf))
            except socket.timeout:
                continue
            if not chunk:
                raise ConnectionError("peer closed connection during recv")
            buf += chunk
        return buf

    def start_server(self):
        """Start listening server for incoming connections"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if sys.platform != 'win32':
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(10)
        self.running = True
        
        def accept_connections():
            # print(f"DEBUG: Node {self.node_id} accept loop started on port {self.port}")
            while self.running:
                try:
                    client_socket, addr = self.server_socket.accept()
                    # print(f"DEBUG: Node {self.node_id} server.accept() returned address: {addr}")
                    try:
                        client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    except (OSError, socket.error) as e:
                        print(f"DEBUG: Node {self.node_id} error setting TCP_NODELAY: {e}")
                        pass
                    if self.use_tls:
                        # Wrap with TLS (simplified - would use proper certificates)
                        try:
                            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                            # In production, would load proper certificates
                            context.check_hostname = False
                            context.verify_mode = ssl.CERT_NONE
                            client_socket = context.wrap_socket(client_socket, server_side=True)
                        except Exception as e:
                            print(f"DEBUG: Node {self.node_id} TLS wrap failed: {e}")
                            # Continue with plain socket if TLS fails (silently)
                            pass
                    
                    t = threading.Thread(target=self._handle_client, args=(client_socket,), daemon=True)
                    t.start()
                    print(f"DEBUG: Node {self.node_id} spawned handle_client thread: {t.name}")

                except socket.error as e:
                    if self.running:
                        # Ignore errors when shutting down
                        if e.errno != 10038:  # WSAENOTSOCK
                            pass
                            # print(f"DEBUG: Node {self.node_id} Error accepting connection sc: {e}")
                except Exception as e:
                    if self.running:
                        print(f"DEBUG: Node {self.node_id} Error accepting connection ex: {e}")
        
        threading.Thread(target=accept_connections, daemon=True).start()
        print(f"Node {self.node_id} server started on port {self.port}")
    
    def connect_to_node(self, target_node_id: int, host: str, port: int):
        """
        Connect to another node
        Args:
            target_node_id: ID of target node
            host: Host address
            port: Port number
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            with self.lock:
                existing = self.connections.get(target_node_id)
                if existing is not None:
                    try:
                        if existing.fileno() != -1:
                            return True
                    except Exception:
                        pass
                    try:
                        existing.close()
                    except Exception:
                        pass
                    self.connections.pop(target_node_id, None)

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            # Connect first, then wrap with TLS if needed
            sock.connect((host, port))
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except (OSError, socket.error):
                pass

            if self.use_tls:
                try:
                    context = ssl.create_default_context()
                    context.check_hostname = False
                    context.verify_mode = ssl.CERT_NONE
                    sock = context.wrap_socket(sock, server_hostname=host)
                except Exception:
                    # If TLS fails, use plain socket (already connected) - silently
                    pass
            
            with self.lock:
                self.connections[target_node_id] = sock
            print(f"[SUCCESS] Node {self.node_id} connected to node {target_node_id} at {host}:{port}")
            return True
        except Exception as e:
            # print(f"[ERROR] Error connecting to node {target_node_id}: {e}")
            return False
    
    def send_message(self, target_node_id: int, msg_type: Union[MessageType, str], data: Dict[str, Any]):
        """
        Send a message to another node
        Args:
            target_node_id: Target node ID
            msg_type: Message type
            data: Message data
        """
        if isinstance(msg_type, MessageType):
            msg_type = msg_type.value
            
        with self.lock:
            if target_node_id not in self.connections:
                raise ValueError(f"No connection to node {target_node_id}")
            sock = self.connections[target_node_id]
        
        self.message_counter += 1
        message = Message(
            msg_type=msg_type,
            sender_id=self.node_id,
            receiver_id=target_node_id,
            data=data,
            timestamp=time.time(),
            message_id=f"{self.node_id}_{self.message_counter}"
        )
        
        try:
            message_json = json.dumps(asdict(message))
            # Send length first, then message
            message_bytes = message_json.encode('utf-8')
            length = len(message_bytes)
            packet = struct.pack('>I', length) + message_bytes
            sock.sendall(packet)
            
            if msg_type == "sync":
                pass
                # print(f"DEBUG: Node {self.node_id} sent SYNC using send_message to {target_node_id}. Length: {length}")
                
        except Exception as e:
            with self.lock:
                stale = self.connections.get(target_node_id)
                if stale is sock:
                    self.connections.pop(target_node_id, None)
            print(f"Error sending message to node {target_node_id}: {e}")
            if self.on_send_failure is not None:
                try:
                    self.on_send_failure(int(target_node_id))
                except Exception:
                    pass
            raise

    def send_message_with_binary(
        self,
        target_node_id: int,
        msg_type: MessageType,
        data: Dict[str, Any],
        payload: bytes,
    ):
        """
        Send a message whose header is JSON (length-prefixed) followed by a raw binary payload.
        This avoids base64/JSON for large vectors.
        """
        with self.lock:
            if target_node_id not in self.connections:
                raise ValueError(f"No connection to node {target_node_id}")
            sock = self.connections[target_node_id]

        self.message_counter += 1
        # Include payload length in the JSON header so receiver knows how many bytes to read.
        header_data = dict(data)
        header_data["payload_len"] = len(payload)
        message = Message(
            msg_type=msg_type.value,
            sender_id=self.node_id,
            receiver_id=target_node_id,
            data=header_data,
            timestamp=time.time(),
            message_id=f"{self.node_id}_{self.message_counter}",
        )

        try:
            message_json = json.dumps(asdict(message))
            header_bytes = message_json.encode("utf-8")
            header_len = struct.pack(">I", len(header_bytes))
            sock.sendall(header_len + header_bytes + payload)
        except Exception as e:
            with self.lock:
                stale = self.connections.get(target_node_id)
                if stale is sock:
                    self.connections.pop(target_node_id, None)
            print(f"Error sending binary message to node {target_node_id}: {e}")
            if self.on_send_failure is not None:
                try:
                    self.on_send_failure(int(target_node_id))
                except Exception:
                    pass
            raise
    
    def send_share(self, target_node_id: int, share: Share, context: str = "default"):
        """
        Send a share to another node
        Args:
            target_node_id: Target node ID
            share: Share to send
            context: Context identifier for this share exchange
        """
        data = {
            'context': context,
            'share': {
                'x': share.x,
                'y': share.y,
                'node_id': share.node_id
            }
        }
        self.send_message(target_node_id, MessageType.SHARE_EXCHANGE, data)

    def send_shares_batch(self, target_node_id: int, shares: List[Share], contexts: List[str]):
        """
        Send a batch of shares to another node in a single message.
        This amortizes per-message overhead for high-throughput MPC (e.g. Beaver openings).
        """
        if len(shares) != len(contexts):
            raise ValueError("shares and contexts length mismatch")
        items = []
        for share, context in zip(shares, contexts):
            items.append({
                'context': context,
                'share': {'x': share.x, 'y': share.y, 'node_id': share.node_id}
            })
        self.send_message(target_node_id, MessageType.BATCH_SHARE_EXCHANGE, {'items': items})

    def send_vector(self, target_node_id: int, context: str, x: int, values: Sequence[int]):
        """
        Send a vector of share values for a single context.
        This is the SIMD-style primitive: open many secrets in one round.
        """
        # Accept lists/arrays/numpy without materializing Python int lists.
        # Use u64 transport whenever any value exceeds uint32 (or SENTRA_FORCE_U64_VECTORS=1).
        force_u64 = os.environ.get("SENTRA_FORCE_U64_VECTORS", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        buf: bytes
        enc = "u32le"
        try:
            import numpy as _np  # local import
            if isinstance(values, _np.ndarray):
                arr_i64 = _np.asarray(values, dtype=_np.int64)
                if force_u64 or (
                    arr_i64.size > 0
                    and ((_np.max(arr_i64) > 0xFFFFFFFF) or (_np.min(arr_i64) < 0))
                ):
                    arr_u64 = _np.asarray(values, dtype=_np.uint64)
                    buf = arr_u64.tobytes(order="C")
                    enc = "u64le"
                else:
                    arr_u32 = _np.asarray(values, dtype=_np.uint32)
                    buf = arr_u32.tobytes(order="C")
            elif isinstance(values, array) and values.typecode in ("I", "Q"):
                if values.typecode == "Q":
                    enc = "u64le"
                buf = values.tobytes()
            else:
                vals = [int(v) for v in values]
                need_u64 = force_u64 or any((v < 0 or v > 0xFFFFFFFF) for v in vals)
                if need_u64:
                    buf = array("Q", (v & 0xFFFFFFFFFFFFFFFF for v in vals)).tobytes()
                    enc = "u64le"
                else:
                    buf = array("I", (v & 0xFFFFFFFF for v in vals)).tobytes()
        except Exception:
            vals = [int(v) for v in values]
            need_u64 = force_u64 or any((v < 0 or v > 0xFFFFFFFF) for v in vals)
            if need_u64:
                buf = array("Q", (v & 0xFFFFFFFFFFFFFFFF for v in vals)).tobytes()
                enc = "u64le"
            else:
                buf = array("I", (v & 0xFFFFFFFF for v in vals)).tobytes()

        # Send as binary payload (preferred)
        self.send_message_with_binary(
            target_node_id,
            MessageType.VECTOR_SHARE_EXCHANGE_BIN,
            {"context": context, "x": x, "n": len(values), "enc": enc},
            payload=buf,
        )

    def send_vector_pair(
        self,
        target_node_id: int,
        context_d: str,
        context_e: str,
        x: int,
        values_d: Sequence[int],
        values_e: Sequence[int],
    ):
        """
        Send both d and e vectors in one message (one round-trip per chunk instead of two).
        """
        def _to_bytes(values: Sequence[int]) -> tuple[bytes, str]:
            try:
                import numpy as _np
                if isinstance(values, _np.ndarray):
                    arr_i64 = _np.asarray(values, dtype=_np.int64)
                    if arr_i64.size > 0 and ((_np.max(arr_i64) > 0xFFFFFFFF) or (_np.min(arr_i64) < 0)):
                        arr_u64 = _np.asarray(values, dtype=_np.uint64)
                        return arr_u64.tobytes(order="C"), "u64le"
                    arr_u32 = _np.asarray(values, dtype=_np.uint32)
                    return arr_u32.tobytes(order="C"), "u32le"
                if isinstance(values, array) and values.typecode in ("I", "Q"):
                    return values.tobytes(), ("u64le" if values.typecode == "Q" else "u32le")
                vals = [int(v) for v in values]
                need_u64 = any((v < 0 or v > 0xFFFFFFFF) for v in vals)
                if need_u64:
                    return array("Q", (v & 0xFFFFFFFFFFFFFFFF for v in vals)).tobytes(), "u64le"
                return array("I", (v & 0xFFFFFFFF for v in vals)).tobytes(), "u32le"
            except Exception:
                vals = [int(v) for v in values]
                need_u64 = any((v < 0 or v > 0xFFFFFFFF) for v in vals)
                if need_u64:
                    return array("Q", (v & 0xFFFFFFFFFFFFFFFF for v in vals)).tobytes(), "u64le"
                return array("I", (v & 0xFFFFFFFF for v in vals)).tobytes(), "u32le"

        d_buf, d_enc = _to_bytes(values_d)
        e_buf, e_enc = _to_bytes(values_e)
        if d_enc != e_enc:
            # Keep protocol simple: promote both to u64 when mixed.
            d_vals = [int(v) for v in values_d]
            e_vals = [int(v) for v in values_e]
            d_buf = array("Q", (v & 0xFFFFFFFFFFFFFFFF for v in d_vals)).tobytes()
            e_buf = array("Q", (v & 0xFFFFFFFFFFFFFFFF for v in e_vals)).tobytes()
            enc = "u64le"
        else:
            enc = d_enc
        payload = d_buf + e_buf
        n_d, n_e = len(values_d), len(values_e)
        self.send_message_with_binary(
            target_node_id,
            MessageType.VECTOR_PAIR_EXCHANGE_BIN,
            {
                "context_d": context_d,
                "context_e": context_e,
                "x": x,
                "n_d": n_d,
                "n_e": n_e,
                "enc": enc,
            },
            payload=payload,
        )

    def _handle_client(self, client_socket: socket.socket):
        """Handle incoming client connection"""
        try:
            # Verify it's actually a socket
            if not isinstance(client_socket, socket.socket) and not hasattr(client_socket, 'recv'):
                print(f"Error: Invalid socket object received")
                return
            
            recv_timeout = self._socket_recv_timeout_s()
            client_socket.settimeout(recv_timeout)
            print(f"DEBUG: Node {self.node_id} accepted connection from {client_socket.getpeername()}")
            
            while self.running:
                try:
                    deadline = time.time() + recv_timeout
                    length_data = self._recv_exact(client_socket, 4, deadline)
                    length = struct.unpack(">I", length_data)[0]
                    if length <= 0 or length > 64 * 1024 * 1024:
                        print(f"DEBUG: Node {self.node_id} invalid message length {length}")
                        break
                    message_data = self._recv_exact(client_socket, length, deadline)
                    msg_dict = json.loads(message_data.decode("utf-8"))
                    msg_type = msg_dict.get("msg_type")
                    if msg_type == "sync":
                        print(f"DEBUG: Node {self.node_id} received SYNC packet from {msg_dict.get('sender_id')}")

                    try:
                        if msg_dict.get("msg_type") == MessageType.VECTOR_SHARE_EXCHANGE_BIN.value:
                            payload_len = int(msg_dict.get("data", {}).get("payload_len", 0))
                            if payload_len > 0:
                                payload = self._recv_exact(client_socket, payload_len, deadline)
                                if len(payload) != payload_len:
                                    raise ValueError(
                                        f"vector payload size mismatch: want {payload_len}, got {len(payload)}"
                                    )
                                msg_dict.setdefault("data", {})["payload_bytes"] = payload
                        elif msg_dict.get("msg_type") == MessageType.VECTOR_PAIR_EXCHANGE_BIN.value:
                            payload_len = int(msg_dict.get("data", {}).get("payload_len", 0))
                            if payload_len > 0:
                                payload = self._recv_exact(client_socket, payload_len, deadline)
                                if len(payload) != payload_len:
                                    raise ValueError(
                                        f"vector_pair payload size mismatch: want {payload_len}, got {len(payload)}"
                                    )
                                msg_dict.setdefault("data", {})["payload_bytes"] = payload
                    except Exception as e:
                        print(f"Error reading binary payload: {e}")
                        break

                    self._process_message(msg_dict, client_socket)
                except socket.timeout:
                    continue
                except ConnectionError:
                    print(f"DEBUG: Node {self.node_id} client disconnected (no length data).")
                    break
                except (socket.error, OSError) as e:
                    if e.errno in (10035, 10037, 11):
                        continue
                    if self.running:
                        print(f"Error handling client: {e}")
                    break
                except Exception as e:
                    print(f"DEBUG: Node {self.node_id} error processing content: {e}")
                    break

        except (socket.error, OSError) as e:
            # Ignore socket errors when shutting down or client disconnects
            if self.running and e.errno not in (10038, 10054, 10053):  # WSAENOTSOCK, WSAECONNRESET, WSAEINTR
                print(f"Error handling client: {e}")
        except Exception as e:
            if self.running:
                print(f"Error handling client: {e}")
        finally:
            try:
                if hasattr(client_socket, 'close'):
                    client_socket.close()
            except:
                pass
    
    def register_handler(self, msg_type: str, handler: Callable):
        """Register a callback for specific message type"""
        print(f"DEBUG: Registering handler for {msg_type}")
        self.message_handlers[msg_type] = handler

    def _process_message(self, msg_dict: Dict, client_socket: Optional[socket.socket] = None):
        """Process incoming message"""
        msg_type = msg_dict.get('msg_type')
        # print(f"DEBUG: Node {self.node_id} received message type: {msg_type} from {msg_dict.get('sender_id')}")
        sender_id = msg_dict.get('sender_id')
        data = msg_dict.get('data', {})

        # Learn/refresh reverse connection on first inbound message from peer.
        # if isinstance(sender_id, int) and sender_id != self.node_id and client_socket is not None:
        #     with self.lock:
        #         existing = self.connections.get(sender_id)
        #         if existing is None:
        #             self.connections[sender_id] = client_socket

        if msg_type == MessageType.SHARE_EXCHANGE.value:
            self._handle_share_exchange(sender_id, data)
        elif msg_type == MessageType.BATCH_SHARE_EXCHANGE.value:
            self._handle_batch_share_exchange(sender_id, data)
        elif msg_type == MessageType.VECTOR_SHARE_EXCHANGE.value:
            self._handle_vector_share_exchange(sender_id, data)
        elif msg_type == MessageType.VECTOR_SHARE_EXCHANGE_BIN.value:
            self._handle_vector_share_exchange(sender_id, data)
        elif msg_type == MessageType.VECTOR_PAIR_EXCHANGE_BIN.value:
            self._handle_vector_pair_exchange(sender_id, data)
        elif msg_type == MessageType.SYNC.value:
            self._handle_sync(sender_id, data)
        elif msg_type == MessageType.RECONSTRUCTION_REQUEST.value:
            self._handle_reconstruction_request(sender_id, data)
        elif msg_type == MessageType.RECONSTRUCTION_RESPONSE.value:
            self._handle_reconstruction_response(sender_id, data)
        elif msg_type == MessageType.HEARTBEAT.value:
            self._handle_heartbeat(sender_id, data)
        elif msg_type in self.message_handlers:
            self.message_handlers[msg_type](sender_id, data)

    def _handle_sync(self, sender_id: int, data: Dict):
        """Handle barrier/sync message. Reject stale epoch if expected_membership_epoch is set."""
        if self.expected_membership_epoch is not None and "e" in data:
            if data["e"] != self.expected_membership_epoch:
                return  # Ignore stale epoch
        tag = data.get("tag", "default")
        with self.lock:
            if tag not in self.received_sync:
                self.received_sync[tag] = set()
            self.received_sync[tag].add(sender_id)
            print(f"DEBUG: Node {self.node_id} added sync from {sender_id} for tag '{tag}'. Current set: {self.received_sync[tag]}")

    def get_received_sync(self, tag: str) -> set:
        with self.lock:
            s = set(self.received_sync.get(tag, set()))
            # print(f"DEBUG: Node {self.node_id} checking sync for tag '{tag}'. Found: {s}")
            return s

    def clear_sync(self, tag: str):
        with self.lock:
            if tag in self.received_sync:
                del self.received_sync[tag]

    def _handle_batch_share_exchange(self, sender_id: int, data: Dict):
        """Handle a batch of received shares."""
        items = data.get('items', [])
        for item in items:
            context = item.get('context')
            share_dict = item.get('share')
            if not context or not share_dict:
                continue
            share = Share(x=share_dict['x'], y=share_dict['y'], node_id=share_dict['node_id'])
            with self.lock:
                if context not in self.received_shares:
                    self.received_shares[context] = []
                self.received_shares[context].append(share)

    def _handle_vector_share_exchange(self, sender_id: int, data: Dict):
        """Handle a received vector payload."""
        context = data.get("context")
        x = data.get("x")
        if not context or x is None:
            return

        values_obj = None
        # Preferred: raw bytes attached by _handle_client for VECTOR_SHARE_EXCHANGE_BIN
        enc = data.get("enc")
        if "payload_bytes" in data and enc in ("u32le", "u64le"):
            raw = data.get("payload_bytes")
            if not isinstance(raw, (bytes, bytearray)):
                return
            arr = array("Q" if enc == "u64le" else "I")
            try:
                arr.frombytes(raw)
            except Exception:
                return
            n = data.get("n")
            if isinstance(n, int) and n >= 0 and len(arr) != n:
                return
            values_obj = arr
        else:
            # Legacy JSON formats
            if "payload" in data:
                try:
                    enc = data.get("enc")
                    if enc not in ("u32le_b64", "u64le_b64"):
                        return
                    raw = base64.b64decode(data["payload"].encode("ascii"))
                    arr = array("Q" if enc == "u64le_b64" else "I")
                    arr.frombytes(raw)
                    n = data.get("n")
                    if isinstance(n, int) and n >= 0 and len(arr) != n:
                        return
                    values_obj = arr
                except Exception:
                    return
            else:
                values = data.get("values")
                if values is None:
                    return
                values_obj = values

        with self.lock:
            if context not in self.received_vectors:
                self.received_vectors[context] = {}
            self.received_vectors[context][sender_id] = {"x": x, "values": values_obj}

    def _handle_vector_pair_exchange(self, sender_id: int, data: Dict):
        """Handle one message containing both d and e vectors (one round-trip per chunk)."""
        raw = data.get("payload_bytes")
        if not isinstance(raw, (bytes, bytearray)):
            return
        context_d = data.get("context_d")
        context_e = data.get("context_e")
        x = data.get("x")
        n_d = data.get("n_d")
        n_e = data.get("n_e")
        if not all(isinstance(v, int) and v >= 0 for v in (n_d, n_e)) or context_d is None or context_e is None or x is None:
            return
        enc = data.get("enc", "u32le")
        elem_bytes = 8 if enc == "u64le" else 4
        need = n_d * elem_bytes + n_e * elem_bytes
        if len(raw) < need:
            return
        d_bytes = raw[: n_d * elem_bytes]
        e_bytes = raw[n_d * elem_bytes : need]
        arr_d = array("Q" if enc == "u64le" else "I")
        arr_e = array("Q" if enc == "u64le" else "I")
        try:
            arr_d.frombytes(d_bytes)
            arr_e.frombytes(e_bytes)
        except Exception:
            return
        if len(arr_d) != n_d or len(arr_e) != n_e:
            return
        with self.lock:
            if context_d not in self.received_vectors:
                self.received_vectors[context_d] = {}
            self.received_vectors[context_d][sender_id] = {"x": x, "values": arr_d}
            if context_e not in self.received_vectors:
                self.received_vectors[context_e] = {}
            self.received_vectors[context_e][sender_id] = {"x": x, "values": arr_e}

    def _handle_share_exchange(self, sender_id: int, data: Dict):
        """Handle received share"""
        context = data.get('context')
        share_dict = data.get('share')
        share = Share(
            x=share_dict['x'],
            y=share_dict['y'],
            node_id=share_dict['node_id']
        )
        
        with self.lock:
            if context not in self.received_shares:
                self.received_shares[context] = []
            self.received_shares[context].append(share)
    
    def _handle_reconstruction_request(self, sender_id: int, data: Dict):
        """Handle reconstruction request"""
        context = data.get('context')
        with self.lock:
            if context in self.received_shares:
                # Send back our share
                for share in self.received_shares[context]:
                    if share.node_id == self.node_id:
                        self.send_share(sender_id, share, context)
    
    def _handle_reconstruction_response(self, sender_id: int, data: Dict):
        """Handle reconstruction response"""
        self._handle_share_exchange(sender_id, data)
    
    def _handle_heartbeat(self, sender_id: int, data: Dict):
        """Handle heartbeat message"""
        # Send ACK
        self.send_message(sender_id, MessageType.ACK, {'ack': True})
    
    def register_handler(self, msg_type: str, handler: Callable):
        """Register a custom message handler"""
        self.message_handlers[msg_type] = handler
    
    def get_received_shares(self, context: str) -> List[Share]:
        """Get received shares for a context"""
        with self.lock:
            return self.received_shares.get(context, []).copy()
    
    def clear_context(self, context: str):
        """Clear shares for a context"""
        with self.lock:
            if context in self.received_shares:
                del self.received_shares[context]

    def get_received_vector(self, context: str) -> Dict[int, Dict[str, Any]]:
        with self.lock:
            return dict(self.received_vectors.get(context, {}))

    def clear_vector(self, context: str):
        with self.lock:
            if context in self.received_vectors:
                del self.received_vectors[context]
    
    def stop(self):
        """Stop the server and close connections"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        for sock in self.connections.values():
            try:
                sock.close()
            except:
                pass
        self.connections.clear()


class SecureMPCNetwork:
    """
    High-level interface for secure multi-party computation network
    Manages nodes, channels, and reconstruction
    """
    
    def __init__(self, node_id: int, node_configs: Dict[int, Dict[str, Any]],
                 port: int = 8000, use_tls: bool = False, membership_epoch: Optional[int] = None):
        """
        Initialize MPC network
        Args:
            node_id: ID of this node
            node_configs: Dictionary mapping node_id to {host, port}
            port: Port for this node
            use_tls: Whether to use TLS (default: False for testing)
            membership_epoch: If set, include e in SYNC messages and reject SYNCs with different e.
        """
        self.node_id = node_id
        self.node_configs = node_configs
        self.membership_epoch = membership_epoch
        self._active_peer_ids: Optional[set] = None  # If set, restrict sends to these (for post-dropout)
        # Get host for this node from config, default to 0.0.0.0 if not found
        host = node_configs.get(node_id, {}).get('host', '0.0.0.0')
        self.channel = SecureChannel(
            node_id, port, use_tls, host=host,
            expected_membership_epoch=membership_epoch,
        )
        self.sss = ShamirSecretSharing()
        
        # Start server
        self.channel.start_server()
        
        # Wait a bit for server to start
        time.sleep(0.5)

        # Connect to other nodes with retries so early-started nodes don't remain degraded.
        max_attempts = 20
        retry_delay = 0.5
        peers = [other_id for other_id in node_configs.keys() if other_id != node_id]
        connected_nodes = set()
        failed_nodes = set(peers)
        for _ in range(max_attempts):
            progress = False
            for other_id in peers:
                if other_id in connected_nodes:
                    continue
                config = node_configs[other_id]
                success = self.channel.connect_to_node(
                    other_id, config['host'], config['port']
                )
                if success:
                    connected_nodes.add(other_id)
                    failed_nodes.discard(other_id)
                    progress = True
                time.sleep(0.05)
            if not failed_nodes:
                break
            if not progress:
                time.sleep(retry_delay)
        
        # Print connection summary
        print(f"\n{'='*70}")
        print(f"Connection Summary for Node {node_id}:")
        print(f"{'='*70}")
        if connected_nodes:
            print(f"[SUCCESS] Connected to {len(connected_nodes)} node(s): {sorted(list(connected_nodes))}")
        if failed_nodes:
            print(f"[ERROR] Failed to connect to {len(failed_nodes)} node(s): {sorted(list(failed_nodes))}")
        if not connected_nodes and not failed_nodes:
            print("No other nodes to connect to (single-node mode)")
        print(f"{'='*70}\n")
    
    def send_share(self, target_node_id: int, share: Share, context: str):
        """Send a share to another node (skips if target not in active peers post-dropout)."""
        if self._active_peer_ids is not None and target_node_id not in self._active_peer_ids:
            return
        self.channel.send_share(target_node_id, share, context)
    
    def connect_to_others(self):
        """Connect to other nodes"""
        import time
        for node_id, config in self.node_configs.items():
            if node_id == self.node_id:
                continue
            
            # Retry loop for connection
            connected = False
            for attempt in range(5):
                try:
                    self.channel.connect_node(node_id, config['host'], config['port'])
                    connected = True
                    break
                except Exception as e:
                    print(f"Node {self.node_id} failed to connect to node {node_id} (attempt {attempt+1}/5): {e}")
                    time.sleep(1.0)
            
            if not connected:
                print(f"[ERROR] Could not connect to node {node_id} after retries.")

    def _get_peers_for_send(self) -> list:
        """Peers to send to (excludes self). Uses _active_peer_ids when set (post-dropout)."""
        if self._active_peer_ids is not None:
            return [i for i in self._active_peer_ids if i != self.node_id]
        return [i for i in self.node_configs.keys() if i != self.node_id]

    def set_active_peers(self, peer_ids) -> None:
        """
        Restrict sends to these peers (for post-dropout MPC reconfiguration).
        Call after successful dropout recovery so broadcasts skip dead nodes.
        Pass None to reset to all node_configs.
        """
        self._active_peer_ids = set(int(i) for i in peer_ids) if peer_ids else None

    def update_membership_epoch(self, new_e: int) -> None:
        """Update membership epoch (e.g. after join recovery bump)."""
        self.membership_epoch = int(new_e)
        self.channel.expected_membership_epoch = int(new_e)

    def broadcast_share(self, share: Share, context: str):
        """Broadcast a share to all nodes (or active peers if set_active_peers was called)."""
        for node_id in self._get_peers_for_send():
                try:
                    self.send_share(node_id, share, context)
                except Exception as e:
                    pass
                    # print(f"Warning: Could not broadcast to node {node_id}: {e}")

    def broadcast_shares_batch(self, shares: List[Share], contexts: List[str]):
        """Broadcast a batch of shares to all nodes (or active peers if set)."""
        for node_id in self._get_peers_for_send():
                try:
                    self.channel.send_shares_batch(node_id, shares, contexts)
                except Exception as e:
                    print(f"Warning: Could not batch-broadcast to node {node_id}: {e}")

    def broadcast_vector(self, context: str, x: int, values: List[int]):
        """Broadcast a vector payload to all nodes (or active peers if set)."""
        for node_id in self._get_peers_for_send():
                try:
                    self.channel.send_vector(node_id, context, x, values)
                except Exception as e:
                    pass
                    # print(f"Warning: Could not broadcast vector to node {node_id}: {e}")

    def broadcast_vector_pair(
        self, context_d: str, context_e: str, x: int, values_d: List[int], values_e: List[int]
    ):
        """Broadcast both d and e vectors (or active peers if set)."""
        for node_id in self._get_peers_for_send():
                try:
                    self.channel.send_vector_pair(
                        node_id, context_d, context_e, x, values_d, values_e
                    )
                except Exception as e:
                    pass
                    # print(f"Warning: Could not broadcast vector pair to node {node_id}: {e}")

    def get_received_shares(self, context: str) -> List[Share]:
        """Get received shares for a context"""
        return self.channel.get_received_shares(context)
    
    def stop(self):
        """Stop the network"""
        self.channel.stop()

    def barrier(self, tag: str, timeout: float = 120.0):
        """
        Simple multi-node barrier: each node broadcasts a SYNC(tag) and waits until
        it has received SYNC(tag) from all other nodes.
        """
        # Broadcast our presence for this tag (include e for stale-epoch rejection)
        payload: Dict[str, Any] = {"tag": tag}
        if self.membership_epoch is not None:
            payload["e"] = self.membership_epoch
        for other_id in self.node_configs.keys():
            if other_id != self.node_id:
                try:
                    self.channel.send_message(other_id, MessageType.SYNC, payload)
                except Exception as e:
                    raise RuntimeError(f"Failed to send SYNC to node {other_id}: {e}")

        start = time.time()
        expected_peers = {i for i in self.node_configs.keys() if i != self.node_id}
        resend_s = float(os.environ.get("SENTRA_BARRIER_RESEND_S", "0") or "0")
        last_resend = start
        last_wait_log = start
        while time.time() - start < timeout:
            got = self.channel.get_received_sync(tag)
            if expected_peers.issubset(got):
                self.channel.clear_sync(tag)
                print(f"DEBUG: Node {self.node_id} barrier '{tag}' passed. Got: {got}")
                return
            now = time.time()
            if resend_s > 0.0 and (now - last_resend) >= resend_s:
                for other_id in self.node_configs.keys():
                    if other_id != self.node_id:
                        try:
                            self.channel.send_message(other_id, MessageType.SYNC, payload)
                        except Exception as e:
                            raise RuntimeError(f"Failed to re-send SYNC to node {other_id}: {e}") from e
                last_resend = now
            if (now - last_wait_log) >= 30.0:
                print(
                    f"DEBUG: Node {self.node_id} barrier '{tag}' waiting. "
                    f"Got: {got}, Expected: {expected_peers}"
                )
                last_wait_log = now
            time.sleep(0.05)

        print(f"DEBUG: Node {self.node_id} barrier '{tag}' TIMED OUT. Got: {got}, Expected: {expected_peers}")
        raise RuntimeError(f"Barrier timed out for tag={tag}; got={sorted(list(got))}, expected={sorted(list(expected_peers))}")

    def barrier_with_peers(self, tag: str, participating_node_ids: set, timeout: float = 300.0):
        """
        Barrier among a subset of nodes (e.g. survivors after dropout).
        Each node in ``participating_node_ids`` sends SYNC to every other participant
        and waits until it has received SYNC from all other participants.
        """
        sid = int(self.node_id)
        peers = {int(i) for i in participating_node_ids}
        if sid not in peers:
            raise ValueError(
                f"barrier_with_peers: this node {sid} not in participating set {sorted(peers)}"
            )
        others = {i for i in peers if i != sid}
        payload: Dict[str, Any] = {"tag": tag}
        if self.membership_epoch is not None:
            payload["e"] = self.membership_epoch
        for other_id in others:
            try:
                self.channel.send_message(other_id, MessageType.SYNC, payload)
            except Exception as e:
                raise RuntimeError(f"Failed to send SYNC to node {other_id}: {e}") from e

        start = time.time()
        while time.time() - start < timeout:
            got = self.channel.get_received_sync(tag)
            if others.issubset(got):
                self.channel.clear_sync(tag)
                print(
                    f"DEBUG: Node {self.node_id} barrier_with_peers '{tag}' passed. "
                    f"Got: {got}, expected others: {others}"
                )
                return
            time.sleep(0.05)

        got = self.channel.get_received_sync(tag)
        raise RuntimeError(
            f"barrier_with_peers timed out tag={tag}; got={sorted(got)}, expected_others={sorted(others)}"
        )


def create_mpc_network(node_id: int, node_configs: Dict[int, Dict[str, Any]],
                       port: int = 8000, use_tls: bool = False,
                       membership_epoch: Optional[int] = None) -> SecureMPCNetwork:
    """
    Factory function to create an MPC network
    
    Args:
        node_id: ID of this node
        node_configs: Dictionary mapping node_id to {host, port}
        port: Port for this node
        use_tls: Whether to use TLS (default: False for testing)
    Returns:
        Configured SecureMPCNetwork instance
    """
    return SecureMPCNetwork(node_id, node_configs, port, use_tls, membership_epoch=membership_epoch)
