from __future__ import annotations

from langchain_core.callbacks import CallbackManager
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.tools import tool
from pydantic import Field

from sessionbat import SessionBat
from sessionbat.transports import MemoryTransport


class StaticSupportRetriever(BaseRetriever):
    documents: list[Document] = Field(default_factory=list)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: object,
    ) -> list[Document]:
        return self.documents


@tool
def lookup_account(account_id: str) -> dict[str, object]:
    """Look up account status."""
    return {"status": "locked", "password_reset_available": True}


class TestLangChainIntegration:
    def setup_method(self) -> None:
        self.transport = MemoryTransport()
        self.client = SessionBat(
            transport=self.transport,
            default_tags=["development"],
            default_context={"environment": "test"},
        )

    def test_integrates_with_langchain_callback_manager(self) -> None:
        handler = self.client.langchain_callback(tags=["langchain"])
        manager = CallbackManager([handler], metadata={"tenant": "acme"})

        llm_runs = manager.on_llm_start(
            {"name": "ChatOpenAI", "kwargs": {"model_name": "gpt-test"}},
            ["I am locked out"],
            invocation_params={"temperature": 0, "model": "gpt-test"},
        )
        llm_runs[0].on_llm_end(
            LLMResult(
                generations=[[Generation(text="Use the reset link.")]],
                llm_output={
                    "model_name": "gpt-test",
                    "token_usage": {
                        "prompt_tokens": 42,
                        "completion_tokens": 7,
                        "total_tokens": 49,
                    },
                },
            )
        )

        tool_run = manager.on_tool_start(
            {"name": "lookup_account"},
            "acct_123",
            inputs={"account_id": "acct_123"},
        )
        tool_run.on_tool_end({"status": "locked"})

        retriever_run = manager.on_retriever_start(
            {"name": "support_articles"},
            "reset password",
        )
        retriever_run.on_retriever_end(
            [
                Document(
                    id="doc_reset_password",
                    page_content="Reset your password from the sign-in page.",
                    metadata={"score": 0.93},
                )
            ]
        )

        observations = [event["observation"] for event in self.transport.events]

        assert [observation["kind"] for observation in observations] == [
            "llm",
            "tool",
            "retrieval",
        ]
        assert observations[0]["output"]["text"] == "Use the reset link."
        assert observations[0]["metrics"]["input_tokens"] == 42
        assert observations[1]["name"] == "lookup_account"
        assert observations[1]["output"] == {"status": "locked"}
        assert observations[2]["input"] == {"query": "reset password"}
        assert observations[2]["output"]["documents"][0]["id"] == "doc_reset_password"
        assert observations[2]["metrics"]["documents_found"] == 1

    def test_integrates_with_langchain_tool_runnable(self) -> None:
        handler = self.client.langchain_callback(tags=["langchain"])

        result = lookup_account.invoke(
            {"account_id": "acct_123"},
            config={
                "callbacks": [handler],
                "metadata": {"session_id": "thread_tool", "tenant": "acme"},
                "tags": ["tool"],
            },
        )

        assert result == {"status": "locked", "password_reset_available": True}
        assert len(self.transport.events) == 1

        event = self.transport.events[0]
        observation = event["observation"]

        assert event["type"] == "tool"
        assert event["session_id"] == "thread_tool"
        assert event["run_id"]
        assert event["tags"] == ["development", "langchain", "tool"]
        assert observation["kind"] == "tool"
        assert observation["name"] == "lookup_account"
        assert observation["input"]["inputs"] == {"account_id": "acct_123"}
        assert observation["output"] == result
        assert observation["metadata"]["framework"] == "langchain"
        assert observation["metadata"]["langchain_serialized_name"] == "lookup_account"
        assert observation["metadata"]["tenant"] == "acme"
        assert "latency_ms" in observation["metrics"]

    def test_integrates_with_langchain_runnable_and_fake_chat_model(self) -> None:
        handler = self.client.langchain_callback(tags=["langchain"])
        chain = ChatPromptTemplate.from_messages(
            [
                ("system", "You help users recover account access."),
                ("human", "{question}"),
            ]
        ) | FakeListChatModel(
            responses=["Use the password reset link."],
            name="FakeSupportChat",
        )

        result = chain.invoke(
            {"question": "I am locked out"},
            config={
                "callbacks": [handler],
                "metadata": {"tenant": "acme"},
                "tags": ["invoke"],
            },
        )

        assert result.content == "Use the password reset link."
        assert len(self.transport.events) == 3
        assert len({event["session_id"] for event in self.transport.events}) == 1
        assert len({event["run_id"] for event in self.transport.events}) == 1

        system_event = self.transport.events[0]
        system_observation = system_event["observation"]
        user_event = self.transport.events[1]
        user_observation = user_event["observation"]
        llm_event = self.transport.events[2]
        llm_observation = llm_event["observation"]

        assert system_event["type"] == "message"
        assert system_observation["kind"] == "message"
        assert system_observation["name"] == "system_message"
        assert system_observation["input"] == {"content": "You help users recover account access."}
        assert system_observation["metadata"]["role"] == "system"
        assert system_observation["metadata"]["framework"] == "langchain"
        assert system_observation["metadata"]["tenant"] == "acme"

        assert user_event["type"] == "message"
        assert user_observation["kind"] == "message"
        assert user_observation["name"] == "user_message"
        assert user_observation["input"] == {"content": "I am locked out"}
        assert user_observation["metadata"]["role"] == "user"
        assert user_observation["metadata"]["framework"] == "langchain"
        assert user_observation["metadata"]["tenant"] == "acme"

        assert llm_event["type"] == "llm"
        assert llm_observation["kind"] == "llm"
        assert llm_observation["name"] == "assistant_response"
        assert llm_observation["output"]["text"] == "Use the password reset link."
        assert llm_observation["metadata"]["framework"] == "langchain"
        assert llm_observation["metadata"]["chat_model"] is True
        assert llm_observation["metadata"]["langchain_serialized_name"] == "FakeSupportChat"
        assert llm_observation["metadata"]["tenant"] == "acme"
        assert "messages" in llm_observation["input"]
        assert "langchain" in llm_event["tags"]
        assert "invoke" in llm_event["tags"]

    def test_integrates_with_langchain_retrieval_chain(self) -> None:
        handler = self.client.langchain_callback(tags=["langchain"])
        retriever = StaticSupportRetriever(
            name="StaticSupportRetriever",
            documents=[
                Document(
                    id="doc_reset_password",
                    page_content="Reset your password from the sign-in page.",
                    metadata={"score": 0.93, "title": "Reset your password"},
                )
            ],
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Answer using this retrieved context: {context}"),
                ("human", "{question}"),
            ]
        )
        chain = (
            {
                "question": RunnablePassthrough(),
                "context": retriever | RunnableLambda(_format_documents),
            }
            | prompt
            | FakeListChatModel(
                responses=["Use the password reset link from the sign-in page."],
                name="FakeRagChat",
            )
        )

        result = chain.invoke(
            "How do I reset my password?",
            config={
                "callbacks": [handler],
                "metadata": {"tenant": "acme"},
                "tags": ["rag"],
            },
        )

        assert result.content == "Use the password reset link from the sign-in page."
        assert len(self.transport.events) == 4
        assert len({event["session_id"] for event in self.transport.events}) == 1
        assert len({event["run_id"] for event in self.transport.events}) == 1

        retrieval_observation = self.transport.events[0]["observation"]
        system_observation = self.transport.events[1]["observation"]
        user_observation = self.transport.events[2]["observation"]
        llm_observation = self.transport.events[3]["observation"]

        assert self.transport.events[0]["type"] == "retrieval"
        assert retrieval_observation["kind"] == "retrieval"
        assert retrieval_observation["input"] == {"query": "How do I reset my password?"}
        assert retrieval_observation["metrics"]["documents_found"] == 1
        assert retrieval_observation["output"]["documents"][0]["id"] == "doc_reset_password"
        assert (
            retrieval_observation["output"]["documents"][0]["content"]
            == "Reset your password from the sign-in page."
        )

        assert system_observation["kind"] == "message"
        assert system_observation["metadata"]["role"] == "system"
        assert (
            "Reset your password from the sign-in page." in system_observation["input"]["content"]
        )

        assert user_observation["kind"] == "message"
        assert user_observation["input"] == {"content": "How do I reset my password?"}
        assert user_observation["metadata"]["role"] == "user"

        assert llm_observation["kind"] == "llm"
        assert llm_observation["output"]["text"] == result.content
        assert llm_observation["metadata"]["langchain_serialized_name"] == "FakeRagChat"
        assert "rag" in self.transport.events[3]["tags"]

    def test_uses_langchain_metadata_session_id_when_present(self) -> None:
        handler = self.client.langchain_callback(tags=["langchain"])
        chain = ChatPromptTemplate.from_messages(
            [
                ("system", "You help users recover account access."),
                ("human", "{question}"),
            ]
        ) | FakeListChatModel(
            responses=["Use the password reset link."],
            name="FakeSupportChat",
        )

        chain.invoke(
            {"question": "I am locked out"},
            config={
                "callbacks": [handler],
                "metadata": {"session_id": "thread_123", "tenant": "acme"},
            },
        )

        assert {event["session_id"] for event in self.transport.events} == {"thread_123"}
        assert len({event["run_id"] for event in self.transport.events}) == 1

    def test_separate_langchain_invocations_get_separate_run_ids(self) -> None:
        handler = self.client.langchain_callback(tags=["langchain"])
        chain = ChatPromptTemplate.from_messages([("human", "{question}")]) | FakeListChatModel(
            responses=["First answer.", "Second answer."],
            name="FakeSupportChat",
        )

        chain.invoke({"question": "First"}, config={"callbacks": [handler]})
        first_run_ids = {event["run_id"] for event in self.transport.events}

        self.transport.events.clear()

        chain.invoke({"question": "Second"}, config={"callbacks": [handler]})
        second_run_ids = {event["run_id"] for event in self.transport.events}

        assert len(first_run_ids) == 1
        assert len(second_run_ids) == 1
        assert first_run_ids != second_run_ids


def _format_documents(documents: list[Document]) -> str:
    return "\n\n".join(document.page_content for document in documents)
