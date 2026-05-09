from __future__ import annotations

import json
import sys
from typing import Protocol


class Transport(Protocol):
    def send(self, payload: dict) -> None: ...


class StdoutTransport:
    def send(self, payload: dict) -> None:
        json.dump(payload, sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
        sys.stdout.flush()


class MemoryTransport:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def send(self, payload: dict) -> None:
        self.events.append(payload)
