"""The wire between ProSim and the CARLA sensor worker.

Two processes, because two interpreters: ProSim lives in prosim_v4.sif
(Python 3.8, torch 2.4, no `carla` module), and a vision policy such as TFv6
needs Python 3.10, torch 2.5 and a `carla` client matching the server. Neither
environment can host the other, so the rollout and the rendering loop talk over
a local socket.

Deliberately small. Only poses, a route and a control cross it -- camera frames,
LiDAR and radar never leave the worker -- so line-delimited JSON is plenty, and
this module is stdlib-only so both interpreters can import it from the same
file.

    request  {"op": "init"|"step"|"close", ...}
    reply    {"ok": true, ...}   or   {"error": "<traceback>"}
"""

from __future__ import annotations

import json
import socket
from typing import Any, Dict

PROTOCOL_VERSION = 1


class ProtocolError(RuntimeError):
    """The peer answered with an error, or broke the protocol."""


def send(stream, message: Dict[str, Any]) -> None:
    stream.write((json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8"))
    stream.flush()


def receive(stream) -> Dict[str, Any]:
    line = stream.readline()
    if not line:
        raise ConnectionError("peer closed the connection")
    return json.loads(line.decode("utf-8"))


class Client:
    """One connection to a sensor worker."""

    def __init__(self, host: str, port: int, timeout_s: float = 300.0):
        self.sock = socket.create_connection((host, int(port)), timeout=timeout_s)
        self.stream = self.sock.makefile("rwb")

    def request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        message = dict(message, protocol=PROTOCOL_VERSION)
        send(self.stream, message)
        reply = receive(self.stream)
        if "error" in reply:
            raise ProtocolError(f"sensor worker failed on {message.get('op')!r}:\n"
                                f"{reply['error']}")
        return reply

    def close(self) -> None:
        try:
            self.stream.close()
        finally:
            self.sock.close()


def parse_address(text: str):
    """'127.0.0.1:2100' -> ('127.0.0.1', 2100)."""
    host, _, port = str(text).rpartition(":")
    if not host or not port.isdigit():
        raise ValueError(f"expected HOST:PORT, got {text!r}")
    return host, int(port)
