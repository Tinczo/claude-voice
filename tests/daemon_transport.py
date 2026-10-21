"""Test-side client for the daemon wire protocol on either transport.

Unix sockets where Python has AF_UNIX; otherwise (Windows) localhost TCP
with the port and token the daemon publishes next to the socket path.
"""
from __future__ import annotations

import json
import os
import socket

TCP = not hasattr(socket, "AF_UNIX")


def addr_path(sock_path: str) -> str:
    return os.path.join(os.path.dirname(sock_path), "daemon.addr")


def ready(sock_path: str) -> bool:
    return os.path.exists(addr_path(sock_path) if TCP else sock_path)


def connect(sock_path: str, timeout: float) -> tuple[socket.socket, str]:
    """Connected socket plus the token to send ("" on Unix sockets)."""
    if not TCP:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(sock_path)
        return s, ""
    with open(addr_path(sock_path), encoding="utf-8") as f:
        addr = json.load(f)
    s = socket.create_connection(("127.0.0.1", addr["port"]), timeout=timeout)
    s.settimeout(timeout)
    return s, addr["token"]


def send(sock_path: str, payload: dict, timeout: float = 2.0) -> dict | None:
    """Send one JSON request and return the parsed response line."""
    try:
        s, token = connect(sock_path, timeout)
        if token:
            payload = dict(payload, token=token)
        s.sendall(json.dumps(payload).encode() + b"\n")
        buf = b""
        while b"\n" not in buf and len(buf) < 8192:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
        s.close()
        line = buf.split(b"\n", 1)[0]
        return json.loads(line.decode("utf-8", errors="replace"))
    except (OSError, socket.timeout, json.JSONDecodeError, ValueError):
        return None
