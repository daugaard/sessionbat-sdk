from __future__ import annotations

import json
import unittest
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


class LangChainCallbackHandlerTest(unittest.TestCase):
    def setUp(self) -> None:
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

        self.assertIsInstance(handler, LangChainCallbackHandler)
        self.assertIs(SessionBatCallbackHandler, LangChainCallbackHandler)

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

        self.assertEqual(event["session_id"], "thread_123")
        self.assertEqual(event["tags"], ["development", "support-bot", "langchain"])
        self.assertEqual(event["context"], {"environment": "test", "user_id": "user_123"})
        self.assertEqual(observation["kind"], "llm")
        self.assertEqual(observation["name"], "assistant_response")
        self.assertEqual(observation["input"]["prompts"], ["I am locked out"])
        self.assertEqual(observation["output"]["text"], "Use the reset link.")
        self.assertEqual(observation["metadata"]["framework"], "langchain")
        self.assertEqual(observation["metadata"]["model"], "gpt-test")
        self.assertEqual(observation["metadata"]["tenant"], "acme")
        self.assertEqual(observation["metadata"]["langchain_run_id"], str(run_id))
        self.assertEqual(observation["metrics"]["input_tokens"], 42)
        self.assertEqual(observation["metrics"]["output_tokens"], 7)
        self.assertEqual(observation["metrics"]["total_tokens"], 49)
        self.assertIn("latency_ms", observation["metrics"])
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

        self.assertEqual(observation["kind"], "llm")
        self.assertEqual(observation["metadata"]["model_name"], "gpt-test")
        self.assertEqual(observation["error"], {"type": "RuntimeError", "message": "upstream failed"})

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

        self.assertEqual(success["kind"], "tool")
        self.assertEqual(success["name"], "lookup_account")
        self.assertEqual(success["input"]["inputs"], {"account_id": "acct_123"})
        self.assertEqual(success["output"], {"status": "locked"})
        self.assertEqual(failure["kind"], "tool")
        self.assertEqual(failure["name"], "send_email")
        self.assertEqual(failure["error"], {"type": "ValueError", "message": "missing template"})

    def test_records_retrieval_completion_and_error(self) -> None:
        handler = self.session.langchain_callback()
        success_run_id = uuid4()
        error_run_id = uuid4()

        handler.on_retriever_start({"name": "support_articles"}, "reset password", run_id=success_run_id)
        handler.on_retriever_end([_Document()], run_id=success_run_id)
        handler.on_retriever_start({"name": "support_articles"}, "billing", run_id=error_run_id)
        handler.on_retriever_error(RuntimeError("index unavailable"), run_id=error_run_id)

        success = self.transport.events[0]["observation"]
        failure = self.transport.events[1]["observation"]

        self.assertEqual(success["kind"], "retrieval")
        self.assertEqual(success["input"], {"query": "reset password"})
        self.assertEqual(success["output"]["documents"][0]["id"], "doc_reset_password")
        self.assertEqual(success["output"]["documents"][0]["metadata"]["score"], 0.93)
        self.assertEqual(success["metrics"]["documents_found"], 1)
        self.assertEqual(failure["kind"], "retrieval")
        self.assertEqual(failure["input"], {"query": "billing"})
        self.assertEqual(failure["error"], {"type": "RuntimeError", "message": "index unavailable"})

    def test_ignores_chain_errors(self) -> None:
        handler = self.session.langchain_callback()
        run_id = uuid4()

        handler.on_chain_error(RuntimeError("chain failed"), run_id=run_id)

        self.assertEqual(self.transport.events, [])


if __name__ == "__main__":
    unittest.main()
