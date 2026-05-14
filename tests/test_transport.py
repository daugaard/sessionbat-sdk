from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlsplit

import pytest

from sessionbat.transports import DEFAULT_INGESTION_ENDPOINT, IngestionTransport


class _IngestionHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        body = self.rfile.read(length)
        self.__class__.requests.append(
            {
                "path": self.path,
                "headers": self.headers,
                "body": json.loads(body),
            }
        )
        self.send_response(202)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def ingestion_endpoint() -> Iterator[str]:
    _IngestionHandler.requests = []
    server = HTTPServer(("127.0.0.1", 0), _IngestionHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        path = urlsplit(DEFAULT_INGESTION_ENDPOINT).path
        yield f"http://127.0.0.1:{server.server_port}{path}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


class TestIngestionTransport:
    def test_posts_payload_to_ingestion_api(self, ingestion_endpoint: str) -> None:
        payload = {
            "id": "evt_123",
            "type": "tool",
            "session_id": "thread_123",
            "observation": {
                "kind": "tool",
                "name": "lookup_account",
                "input": {"account_id": "acct_123"},
            },
        }
        transport = IngestionTransport(
            api_key="sbat_ingest_test",
            endpoint=ingestion_endpoint,
        )

        transport.send(payload)

        assert transport.flush(timeout=1.0)
        assert transport.close(timeout=1.0)
        request = _IngestionHandler.requests[0]
        assert request["path"] == urlsplit(DEFAULT_INGESTION_ENDPOINT).path
        assert request["headers"]["Accept"] == "application/json"
        assert request["headers"]["Authorization"] == "Bearer sbat_ingest_test"
        assert request["headers"]["Content-Type"] == "application/json"
        assert request["body"] == payload
