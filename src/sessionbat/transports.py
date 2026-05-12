from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_INGESTION_ENDPOINT = "http://ingest.sessionbat.com:3000/api/v1/ingestion/events"


class Transport(Protocol):
    def send(self, payload: dict) -> None: ...


class TransportError(RuntimeError):
    pass


@dataclass(slots=True)
class IngestionTransport:
    api_key: str
    endpoint: str = DEFAULT_INGESTION_ENDPOINT
    timeout: float = 10.0

    def send(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            self.endpoint,
            data=body,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                status = response.status
                if status < 200 or status >= 300:
                    raise TransportError(f"ingestion failed with HTTP {status}")
        except HTTPError as error:
            raise TransportError(f"ingestion failed with HTTP {error.code}") from error
        except URLError as error:
            raise TransportError(f"ingestion request failed: {error.reason}") from error


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
