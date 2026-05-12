from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from sessionbat import SessionBat
from sessionbat.transports import IngestionTransport, MemoryTransport


class TestSessionBatClient:
    def setup_method(self) -> None:
        self.transport = MemoryTransport()
        self.client = SessionBat(
            transport=self.transport,
            app="support-bot",
            default_tags=["production", "support"],
            default_context={"environment": "prod", "workspace_id": "ws_default"},
        )
        self.session = self.client.session(
            session_id="thread_123",
            tags=["support", "password-reset"],
            context={"workspace_id": "ws_123", "user_id": "user_123"},
        )

    def test_records_user_message_payload(self) -> None:
        observation_id = self.session.user_message(
            "I am locked out",
            tags=["urgent", "support"],
            context={"locale": "en-US"},
            metadata={"channel": "chat"},
        )

        event = self.transport.events[0]
        observation = event["observation"]

        assert event["id"] == observation_id
        assert event["type"] == "message"
        assert event["session_id"] == "thread_123"
        assert event["tags"] == ["production", "support", "password-reset", "urgent"]
        assert event["context"] == {
            "environment": "prod",
            "workspace_id": "ws_123",
            "user_id": "user_123",
            "locale": "en-US",
        }
        assert observation["kind"] == "message"
        assert observation["name"] == "user_message"
        assert observation["input"] == {"content": "I am locked out"}
        assert observation["metadata"] == {"role": "user", "channel": "chat"}
        assert observation["output"] is None
        assert observation["error"] is None
        assert observation["metrics"] == {}
        datetime.fromisoformat(event["created_at"])
        datetime.fromisoformat(observation["recorded_at"])
        json.dumps(event)

    def test_records_assistant_response_payload(self) -> None:
        self.session.assistant_response(
            model="gpt-test",
            request={"messages": [{"role": "user", "content": "Help"}]},
            response={"text": "Use the reset link."},
            metrics={"input_tokens": 10, "output_tokens": 6},
            metadata={"provider": "openai"},
        )

        observation = self.transport.events[0]["observation"]

        assert observation["kind"] == "llm"
        assert observation["name"] == "assistant_response"
        assert observation["input"] == {"messages": [{"role": "user", "content": "Help"}]}
        assert observation["output"] == {"text": "Use the reset link."}
        assert observation["metadata"] == {"model": "gpt-test", "provider": "openai"}
        assert observation["metrics"] == {"input_tokens": 10, "output_tokens": 6}

    def test_records_tool_call_payload(self) -> None:
        self.session.tool_call(
            tool_name="lookup_account",
            input={"account_id": "acct_123"},
            output={"status": "locked"},
            metrics={"latency_ms": 117},
        )

        event = self.transport.events[0]
        observation = event["observation"]

        assert event["type"] == "tool"
        assert observation["kind"] == "tool"
        assert observation["name"] == "lookup_account"
        assert observation["input"] == {"account_id": "acct_123"}
        assert observation["output"] == {"status": "locked"}
        assert observation["metrics"] == {"latency_ms": 117}

    def test_records_retrieval_payload(self) -> None:
        self.session.retrieval(
            query="reset password",
            documents=[{"id": "doc_123", "score": 0.93}],
            metadata={"index": "support_articles"},
        )

        event = self.transport.events[0]
        observation = event["observation"]

        assert event["type"] == "retrieval"
        assert observation["kind"] == "retrieval"
        assert observation["name"] == "retrieval"
        assert observation["input"] == {"query": "reset password"}
        assert observation["output"] == {"documents": [{"id": "doc_123", "score": 0.93}]}
        assert observation["metadata"] == {"index": "support_articles"}

    def test_records_errors_on_failed_operations(self) -> None:
        self.session.tool_call(
            tool_name="send_email",
            input={"template": "reset"},
            error={"type": "TimeoutError", "message": "email service timed out"},
        )

        observation = self.transport.events[0]["observation"]

        assert observation["kind"] == "tool"
        assert observation["output"] is None
        assert observation["error"] == {
            "type": "TimeoutError",
            "message": "email service timed out",
        }


class _RecordingHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []
    response_status: int | list[int] = 202

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        body = self.rfile.read(length)
        response_status = self.__class__.response_status
        if isinstance(response_status, list):
            status = response_status.pop(0) if response_status else 202
        else:
            status = response_status
        self.__class__.requests.append(
            {
                "path": self.path,
                "headers": self.headers,
                "body": json.loads(body),
            }
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def ingestion_server() -> Iterator[str]:
    _RecordingHandler.requests = []
    _RecordingHandler.response_status = 202
    server = HTTPServer(("127.0.0.1", 0), _RecordingHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/api/v1/ingestion/events"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


class TestIngestionTransport:
    def test_default_client_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SESSIONBAT_API_KEY", raising=False)

        with pytest.raises(ValueError, match="requires api_key"):
            SessionBat()

    def test_explicit_memory_transport_does_not_require_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SESSIONBAT_API_KEY", raising=False)
        transport = MemoryTransport()

        client = SessionBat(transport=transport)

        assert client.transport is transport

    def test_uses_environment_api_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
        ingestion_server: str,
    ) -> None:
        monkeypatch.setenv("SESSIONBAT_API_KEY", "sbat_ingest_env")

        client = SessionBat(endpoint=ingestion_server)
        session = client.session(session_id="thread_123")
        session.tool_call(tool_name="lookup_account", input={"account_id": "acct_123"})

        assert client.flush(timeout=1.0)
        request = _RecordingHandler.requests[0]
        assert request["headers"]["Authorization"] == "Bearer sbat_ingest_env"

    def test_posts_sdk_payload_with_bearer_auth(self, ingestion_server: str) -> None:
        client = SessionBat(api_key="sbat_ingest_test", endpoint=ingestion_server)
        session = client.session(
            session_id="thread_123",
            tags=["support"],
            context={"user_id": "user_123"},
        )

        observation_id = session.tool_call(
            tool_name="lookup_account",
            input={"account_id": "acct_123"},
            output={"status": "locked"},
        )

        assert client.flush(timeout=1.0)
        request = _RecordingHandler.requests[0]
        payload = request["body"]
        assert request["path"] == "/api/v1/ingestion/events"
        assert request["headers"]["Accept"] == "application/json"
        assert request["headers"]["Authorization"] == "Bearer sbat_ingest_test"
        assert request["headers"]["Content-Type"] == "application/json"
        assert payload["id"] == observation_id
        assert payload["type"] == "tool"
        assert payload["session_id"] == "thread_123"
        assert payload["tags"] == ["support"]
        assert payload["context"] == {"user_id": "user_123"}
        assert payload["observation"]["kind"] == "tool"
        assert payload["observation"]["name"] == "lookup_account"
        assert payload["observation"]["input"] == {"account_id": "acct_123"}
        assert payload["observation"]["output"] == {"status": "locked"}

    def test_retries_transient_http_failures(self, ingestion_server: str) -> None:
        _RecordingHandler.response_status = [500, 500, 202]
        transport = IngestionTransport(
            api_key="sbat_ingest_test",
            endpoint=ingestion_server,
            base_backoff=0,
            max_backoff=0,
        )

        transport.send({"id": "evt_123"})

        assert transport.flush(timeout=1.0)
        assert [request["body"]["id"] for request in _RecordingHandler.requests] == [
            "evt_123",
            "evt_123",
            "evt_123",
        ]

    def test_does_not_retry_or_raise_for_non_retryable_http_failures(
        self, ingestion_server: str
    ) -> None:
        _RecordingHandler.response_status = 400
        transport = IngestionTransport(api_key="sbat_ingest_test", endpoint=ingestion_server)

        transport.send({"id": "evt_123"})

        assert transport.flush(timeout=1.0)
        assert len(_RecordingHandler.requests) == 1

    def test_send_does_not_raise_for_network_failures(self) -> None:
        transport = IngestionTransport(
            api_key="sbat_ingest_test",
            endpoint="http://127.0.0.1:1/api/v1/ingestion/events",
            base_backoff=0,
            max_backoff=0,
            timeout=0.01,
        )

        transport.send({"id": "evt_123"})

        assert transport.flush(timeout=1.0)

    def test_full_queue_drops_newest_without_blocking(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        started = threading.Event()
        release = threading.Event()
        sent: list[str] = []

        def slow_send_once(self: IngestionTransport, payload: dict) -> None:
            sent.append(payload["id"])
            started.set()
            release.wait(timeout=1.0)

        monkeypatch.setattr(IngestionTransport, "_send_once", slow_send_once)
        transport = IngestionTransport(api_key="sbat_ingest_test", queue_size=1)

        transport.send({"id": "evt_1"})
        assert started.wait(timeout=1.0)
        transport.send({"id": "evt_2"})
        start = time.monotonic()
        transport.send({"id": "evt_3"})

        assert time.monotonic() - start < 0.1
        release.set()
        assert transport.close(timeout=1.0)
        assert sent == ["evt_1", "evt_2"]

    def test_flush_returns_false_when_timeout_expires(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        release = threading.Event()

        def slow_send_once(self: IngestionTransport, payload: dict) -> None:
            release.wait(timeout=1.0)

        monkeypatch.setattr(IngestionTransport, "_send_once", slow_send_once)
        transport = IngestionTransport(api_key="sbat_ingest_test")

        transport.send({"id": "evt_123"})

        assert transport.flush(timeout=0.01) is False
        release.set()
        assert transport.close(timeout=1.0)
