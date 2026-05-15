from __future__ import annotations

import json
from uuid import uuid4

from sessionbat import LangChainCallbackHandler, SessionBat, SessionBatCallbackHandler
from sessionbat.transports import MemoryTransport


class _Generation:
    def __init__(self, text: str) -> None:
        self.text = text


class _LLMResult:
    def __init__(self) -> None:
        self.generations = [[_Generation("Use the reset link.")]]
        self.llm_output = {
            "model_name": "gpt-test",
            "token_usage": {
                "prompt_tokens": 42,
                "completion_tokens": 7,
                "total_tokens": 49,
            },
        }


class _Document:
    def __init__(self) -> None:
        self.id = "doc_reset_password"
        self.page_content = "Reset your password from the sign-in page."
        self.metadata = {"score": 0.93, "title": "Reset your password"}


class TestLangChainCallbackHandler:
    def setup_method(self) -> None:
        self.transport = MemoryTransport()
        self.client = SessionBat(
            transport=self.transport,
            default_tags=["development"],
            default_context={"environment": "test"},
        )
        self.session = self.client.session(
            session_id="thread_123",
            tags=["support-bot"],
            context={"user_id": "user_123"},
        )

    def test_convenience_constructor_and_export_alias(self) -> None:
        handler = self.session.langchain_callback(tags=["langchain"])

        assert isinstance(handler, LangChainCallbackHandler)
        assert SessionBatCallbackHandler is LangChainCallbackHandler

    def test_client_convenience_constructor_does_not_require_session_id(self) -> None:
        handler = self.client.langchain_callback(tags=["langchain"])

        assert isinstance(handler, LangChainCallbackHandler)

    def test_records_llm_completion_as_assistant_response(self) -> None:
        handler = self.session.langchain_callback(
            tags=["langchain"],
            metadata={"source": "test"},
        )
        run_id = uuid4()

        handler.on_llm_start(
            {"name": "ChatOpenAI", "kwargs": {"model_name": "gpt-test"}},
            ["I am locked out"],
            run_id=run_id,
            metadata={"tenant": "acme"},
            invocation_params={"temperature": 0, "model": "gpt-test"},
        )
        handler.on_llm_end(_LLMResult(), run_id=run_id)

        event = self.transport.events[0]
        observation = event["observation"]

        assert event["type"] == "llm"
        assert event["session_id"] == "thread_123"
        assert event["interaction_id"] == str(run_id)
        assert event["tags"] == ["development", "support-bot", "langchain"]
        assert event["context"] == {"environment": "test", "user_id": "user_123"}
        assert observation["kind"] == "llm"
        assert observation["name"] == "assistant_response"
        assert observation["input"]["prompts"] == ["I am locked out"]
        assert observation["output"]["text"] == "Use the reset link."
        assert observation["metadata"]["framework"] == "langchain"
        assert observation["metadata"]["model"] == "gpt-test"
        assert observation["metadata"]["tenant"] == "acme"
        assert observation["metadata"]["langchain_run_id"] == str(run_id)
        assert observation["metrics"]["input_tokens"] == 42
        assert observation["metrics"]["output_tokens"] == 7
        assert observation["metrics"]["total_tokens"] == 49
        assert "latency_ms" in observation["metrics"]
        json.dumps(event)

    def test_records_llm_error_as_failed_assistant_response(self) -> None:
        handler = self.session.langchain_callback()
        run_id = uuid4()

        handler.on_llm_start(
            {"name": "ChatOpenAI", "kwargs": {"model_name": "gpt-test"}},
            ["I am locked out"],
            run_id=run_id,
        )
        handler.on_llm_error(RuntimeError("upstream failed"), run_id=run_id)

        observation = self.transport.events[0]["observation"]

        assert observation["kind"] == "llm"
        assert observation["metadata"]["model_name"] == "gpt-test"
        assert observation["error"] == {"type": "RuntimeError", "message": "upstream failed"}

    def test_records_tool_completion_and_error(self) -> None:
        handler = self.session.langchain_callback()
        success_run_id = uuid4()
        error_run_id = uuid4()

        handler.on_tool_start(
            {"name": "lookup_account"},
            "acct_123",
            run_id=success_run_id,
            inputs={"account_id": "acct_123"},
        )
        handler.on_tool_end({"status": "locked"}, run_id=success_run_id)
        handler.on_tool_start({"name": "send_email"}, "user_123", run_id=error_run_id)
        handler.on_tool_error(ValueError("missing template"), run_id=error_run_id)

        success = self.transport.events[0]["observation"]
        failure = self.transport.events[1]["observation"]

        assert self.transport.events[0]["type"] == "tool"
        assert self.transport.events[1]["type"] == "tool"
        assert self.transport.events[0]["interaction_id"] == str(success_run_id)
        assert self.transport.events[1]["interaction_id"] == str(error_run_id)
        assert success["kind"] == "tool"
        assert success["name"] == "lookup_account"
        assert success["input"]["inputs"] == {"account_id": "acct_123"}
        assert success["output"] == {"status": "locked"}
        assert failure["kind"] == "tool"
        assert failure["name"] == "send_email"
        assert failure["error"] == {"type": "ValueError", "message": "missing template"}

    def test_records_retrieval_completion_and_error(self) -> None:
        handler = self.session.langchain_callback()
        success_run_id = uuid4()
        error_run_id = uuid4()

        handler.on_retriever_start(
            {"name": "support_articles"},
            "reset password",
            run_id=success_run_id,
        )
        handler.on_retriever_end([_Document()], run_id=success_run_id)
        handler.on_retriever_start({"name": "support_articles"}, "billing", run_id=error_run_id)
        handler.on_retriever_error(RuntimeError("index unavailable"), run_id=error_run_id)

        success = self.transport.events[0]["observation"]
        failure = self.transport.events[1]["observation"]

        assert self.transport.events[0]["type"] == "retrieval"
        assert self.transport.events[1]["type"] == "retrieval"
        assert self.transport.events[0]["interaction_id"] == str(success_run_id)
        assert self.transport.events[1]["interaction_id"] == str(error_run_id)
        assert success["kind"] == "retrieval"
        assert success["input"] == {"query": "reset password"}
        assert success["output"]["documents"][0]["id"] == "doc_reset_password"
        assert success["output"]["documents"][0]["metadata"]["score"] == 0.93
        assert success["metrics"]["documents_found"] == 1
        assert failure["kind"] == "retrieval"
        assert failure["input"] == {"query": "billing"}
        assert failure["error"] == {"type": "RuntimeError", "message": "index unavailable"}

    def test_ignores_chain_errors(self) -> None:
        handler = self.session.langchain_callback()
        run_id = uuid4()

        handler.on_chain_error(RuntimeError("chain failed"), run_id=run_id)

        assert self.transport.events == []
