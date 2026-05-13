from __future__ import annotations

import atexit
import json
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
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
    max_retries: int = 3
    base_backoff: float = 0.25
    max_backoff: float = 2.0
    queue_size: int = 1000
    shutdown_timeout: float = 2.0
    _queue: Queue[dict] = field(init=False, repr=False)
    _worker: threading.Thread | None = field(default=None, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self._queue = Queue(maxsize=self.queue_size)
        atexit.register(self.close, timeout=self.shutdown_timeout)

    def send(self, payload: dict) -> None:
        with self._lock:
            if self._closed:
                return
            self._ensure_worker_started()

            try:
                self._queue.put_nowait(payload)
            except Full:
                return

    def flush(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._queue.unfinished_tasks:
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return True

    def close(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._lock:
            worker = self._worker
            if not self._closed:
                self._closed = True

        flushed = self.flush(timeout=timeout)
        if worker is None:
            return flushed

        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        worker.join(timeout=remaining)
        return flushed and not worker.is_alive()

    def _ensure_worker_started(self) -> None:
        if self._worker is not None:
            return
        self._worker = threading.Thread(target=self._run, name="sessionbat-ingestion")
        self._worker.daemon = True
        self._worker.start()

    def _run(self) -> None:
        while True:
            try:
                payload = self._queue.get(timeout=0.1)
            except Empty:
                if self._closed:
                    return
                continue

            try:
                self._send_with_retries(payload)
            finally:
                self._queue.task_done()

    def _send_with_retries(self, payload: dict) -> None:
        attempt = 0
        while True:
            try:
                self._send_once(payload)
                return
            except TransportError as error:
                if attempt >= self.max_retries or not _is_retryable(error):
                    return
                time.sleep(_backoff(attempt, self.base_backoff, self.max_backoff))
                attempt += 1
            except Exception:
                return

    def _send_once(self, payload: dict) -> None:
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
                    raise TransportError(f"ingestion failed with HTTP {status}", status)
        except HTTPError as error:
            raise TransportError(f"ingestion failed with HTTP {error.code}", error.code) from error
        except URLError as error:
            raise TransportError(f"ingestion request failed: {error.reason}") from error
        except TimeoutError as error:
            raise TransportError("ingestion request timed out") from error
        except OSError as error:
            raise TransportError(f"ingestion request failed: {error}") from error


def _is_retryable(error: TransportError) -> bool:
    if len(error.args) > 1 and isinstance(error.args[1], int):
        status = error.args[1]
        return status == 408 or status == 429 or status >= 500
    cause = error.__cause__
    if isinstance(cause, HTTPError):
        return cause.code == 408 or cause.code == 429 or cause.code >= 500
    if isinstance(cause, URLError | TimeoutError | OSError):
        return True
    return False


def _backoff(attempt: int, base_backoff: float, max_backoff: float) -> float:
    delay = min(max_backoff, base_backoff * (2**attempt))
    return random.uniform(0, delay)


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
