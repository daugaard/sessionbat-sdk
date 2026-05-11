from __future__ import annotations

import json
from datetime import datetime

from sessionbat import SessionBat
from sessionbat.transports import MemoryTransport


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
