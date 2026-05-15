from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from time import monotonic
from typing import Any
from uuid import UUID

from .client import Interaction as SessionBatInteraction
from .client import Session, SessionBat

try:  # LangChain is an optional integration dependency.
    from langchain_core.callbacks import BaseCallbackHandler as _BaseCallbackHandler
except Exception:  # pragma: no cover - exercised when LangChain is not installed.
    try:
        from langchain.callbacks.base import BaseCallbackHandler as _BaseCallbackHandler
    except Exception:
        _BaseCallbackHandler = object


@dataclass(slots=True)
class _RunState:
    kind: str
    name: str
    interaction: SessionBatInteraction
    input: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    parent_run_id: str | None = None
    started_at: float = field(default_factory=monotonic)


class LangChainCallbackHandler(_BaseCallbackHandler):
    """LangChain callback handler that emits SessionBat observations.

    LangChain remains optional. If it is installed, this class subclasses its
    BaseCallbackHandler. Otherwise it still exposes the same callback methods,
    which keeps imports working in dependency-light environments.
    """

    ignore_agent = False
    ignore_chain = False
    ignore_chat_model = False
    ignore_custom_event = True
    ignore_llm = False
    ignore_retriever = False
    ignore_retry = True
    raise_error = False
    run_inline = False

    def __init__(
        self,
        target: SessionBat | Session,
        *,
        tags: list[str] | None = None,
        context: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if isinstance(target, Session):
            self.client = target.client
            self.session: Session | None = target
        else:
            self.client = target
            self.session = None
        self.tags = tags or []
        self.context = context or {}
        self.metadata = metadata or {}
        self._runs: dict[str, _RunState] = {}
        self._run_session_ids: dict[str, str] = {}
        self._root_interaction_ids: dict[str, str] = {}
        self._sessions: dict[str, Session] = {}

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any] | Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._remember_run(run_id, parent_run_id=parent_run_id, metadata=metadata)

    def on_llm_start(
        self,
        serialized: dict[str, Any] | None,
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        interaction = self._interaction_for(run_id, parent_run_id=parent_run_id, metadata=metadata)
        self._runs[_run_id(run_id)] = _RunState(
            kind="llm",
            name=_serialized_name(serialized, "llm"),
            interaction=interaction,
            input={
                "prompts": _jsonable(prompts),
                "serialized": _jsonable(serialized),
                "invocation_params": _jsonable(kwargs.get("invocation_params")),
            },
            metadata=self._metadata(
                run_id=run_id,
                parent_run_id=parent_run_id,
                serialized=serialized,
                metadata=metadata,
                extra={"langchain_callback": "on_llm_start"},
                kwargs=kwargs,
            ),
            tags=self._tags(tags),
            parent_run_id=_optional_run_id(parent_run_id),
        )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        interaction = self._interaction_for(run_id, parent_run_id=parent_run_id, metadata=metadata)
        callback_metadata = self._metadata(
            run_id=run_id,
            parent_run_id=parent_run_id,
            serialized=serialized,
            metadata=metadata,
            extra={"langchain_callback": "on_chat_model_start", "chat_model": True},
            kwargs=kwargs,
        )
        self._record_messages_from_chat_input(
            messages,
            interaction=interaction,
            metadata=callback_metadata,
            tags=tags,
        )
        self._runs[_run_id(run_id)] = _RunState(
            kind="llm",
            name=_serialized_name(serialized, "chat_model"),
            interaction=interaction,
            input={
                "messages": _jsonable(messages),
                "serialized": _jsonable(serialized),
                "invocation_params": _jsonable(kwargs.get("invocation_params")),
            },
            metadata=callback_metadata,
            tags=self._tags(tags),
            parent_run_id=_optional_run_id(parent_run_id),
        )

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        state = self._pop_run(run_id, kind="llm", name="llm", parent_run_id=parent_run_id)
        response_payload = _llm_response_payload(response)
        model_metadata = _merge_dicts(state.metadata, metadata)
        state.interaction.assistant_response(
            model=_extract_model(response=response, metadata=model_metadata),
            request=state.input,
            response=response_payload,
            metadata=_merge_dicts(
                state.metadata,
                self._callback_metadata(
                    run_id=run_id,
                    parent_run_id=parent_run_id,
                    metadata=metadata,
                    extra={"langchain_callback": "on_llm_end"},
                    kwargs=kwargs,
                ),
            ),
            metrics=_merge_dicts(_duration_metrics(state), _llm_metrics(response)),
            tags=self._tags(state.tags, tags),
            context=self.context,
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        state = self._pop_run(run_id, kind="llm", name="llm", parent_run_id=parent_run_id)
        state.interaction.assistant_response(
            model=_extract_model(metadata=_merge_dicts(state.metadata, metadata)),
            request=state.input,
            response=_jsonable(kwargs.get("response")),
            error=_error_payload(error),
            metadata=_merge_dicts(
                state.metadata,
                self._callback_metadata(
                    run_id=run_id,
                    parent_run_id=parent_run_id,
                    metadata=metadata,
                    extra={"langchain_callback": "on_llm_error"},
                    kwargs=kwargs,
                ),
            ),
            metrics=_duration_metrics(state),
            tags=self._tags(state.tags, tags),
            context=self.context,
        )

    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        interaction = self._interaction_for(run_id, parent_run_id=parent_run_id, metadata=metadata)
        self._runs[_run_id(run_id)] = _RunState(
            kind="tool",
            name=_serialized_name(serialized, "tool"),
            interaction=interaction,
            input=_tool_input(input_str, inputs),
            metadata=self._metadata(
                run_id=run_id,
                parent_run_id=parent_run_id,
                serialized=serialized,
                metadata=metadata,
                extra={"langchain_callback": "on_tool_start"},
                kwargs=kwargs,
            ),
            tags=self._tags(tags),
            parent_run_id=_optional_run_id(parent_run_id),
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        state = self._pop_run(run_id, kind="tool", name="tool", parent_run_id=parent_run_id)
        state.interaction.tool_call(
            tool_name=state.name,
            input=state.input,
            output=_output_payload(output),
            metadata=_merge_dicts(
                state.metadata,
                self._callback_metadata(
                    run_id=run_id,
                    parent_run_id=parent_run_id,
                    metadata=metadata,
                    extra={"langchain_callback": "on_tool_end"},
                    kwargs=kwargs,
                ),
            ),
            metrics=_duration_metrics(state),
            tags=self._tags(state.tags, tags),
            context=self.context,
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        state = self._pop_run(run_id, kind="tool", name="tool", parent_run_id=parent_run_id)
        state.interaction.tool_call(
            tool_name=state.name,
            input=state.input,
            error=_error_payload(error),
            metadata=_merge_dicts(
                state.metadata,
                self._callback_metadata(
                    run_id=run_id,
                    parent_run_id=parent_run_id,
                    metadata=metadata,
                    extra={"langchain_callback": "on_tool_error"},
                    kwargs=kwargs,
                ),
            ),
            metrics=_duration_metrics(state),
            tags=self._tags(state.tags, tags),
            context=self.context,
        )

    def on_retriever_start(
        self,
        serialized: dict[str, Any] | None,
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        interaction = self._interaction_for(run_id, parent_run_id=parent_run_id, metadata=metadata)
        self._runs[_run_id(run_id)] = _RunState(
            kind="retrieval",
            name=_serialized_name(serialized, "retriever"),
            interaction=interaction,
            input={"query": query, "serialized": _jsonable(serialized)},
            metadata=self._metadata(
                run_id=run_id,
                parent_run_id=parent_run_id,
                serialized=serialized,
                metadata=metadata,
                extra={"langchain_callback": "on_retriever_start"},
                kwargs=kwargs,
            ),
            tags=self._tags(tags),
            parent_run_id=_optional_run_id(parent_run_id),
        )

    def on_retriever_end(
        self,
        documents: Sequence[Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        state = self._pop_run(
            run_id,
            kind="retrieval",
            name="retriever",
            parent_run_id=parent_run_id,
        )
        docs = [_document_payload(document) for document in documents]
        state.interaction.retrieval(
            query=str(state.input.get("query", "")),
            documents=docs,
            metadata=_merge_dicts(
                state.metadata,
                self._callback_metadata(
                    run_id=run_id,
                    parent_run_id=parent_run_id,
                    metadata=metadata,
                    extra={"langchain_callback": "on_retriever_end", "retriever_name": state.name},
                    kwargs=kwargs,
                ),
            ),
            metrics=_merge_dicts(_duration_metrics(state), {"documents_found": len(docs)}),
            tags=self._tags(state.tags, tags),
            context=self.context,
        )

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        state = self._pop_run(
            run_id,
            kind="retrieval",
            name="retriever",
            parent_run_id=parent_run_id,
        )
        state.interaction.retrieval(
            query=str(state.input.get("query", "")),
            error=_error_payload(error),
            metadata=_merge_dicts(
                state.metadata,
                self._callback_metadata(
                    run_id=run_id,
                    parent_run_id=parent_run_id,
                    metadata=metadata,
                    extra={
                        "langchain_callback": "on_retriever_error",
                        "retriever_name": state.name,
                    },
                    kwargs=kwargs,
                ),
            ),
            metrics=_duration_metrics(state),
            tags=self._tags(state.tags, tags),
            context=self.context,
        )

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        return None

    def _pop_run(
        self,
        run_id: UUID,
        *,
        kind: str,
        name: str,
        parent_run_id: UUID | None,
    ) -> _RunState:
        return self._runs.pop(
            _run_id(run_id),
            _RunState(
                kind=kind,
                name=name,
                interaction=self._interaction_for(run_id, parent_run_id=parent_run_id),
                input={},
                metadata=self._callback_metadata(run_id=run_id, parent_run_id=parent_run_id),
                parent_run_id=_optional_run_id(parent_run_id),
            ),
        )

    def _metadata(
        self,
        *,
        run_id: UUID,
        parent_run_id: UUID | None,
        serialized: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
        extra: dict[str, Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return _merge_dicts(
            {"framework": "langchain"},
            self.metadata,
            _serialized_metadata(serialized),
            _invocation_metadata(kwargs),
            self._callback_metadata(
                run_id=run_id,
                parent_run_id=parent_run_id,
                metadata=metadata,
                extra=extra,
                kwargs=kwargs,
            ),
        )

    def _callback_metadata(
        self,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        callback_metadata = _merge_dicts(
            {
                "framework": "langchain",
                "langchain_run_id": _run_id(run_id),
                "langchain_parent_run_id": _optional_run_id(parent_run_id),
            },
            self.metadata,
            metadata,
            extra,
        )
        if kwargs:
            callback_metadata["langchain_kwargs"] = _jsonable(kwargs)
        return callback_metadata

    def _tags(self, *tag_sets: list[str] | None) -> list[str]:
        return _merge_tags(self.tags, *tag_sets)

    def _record_messages_from_chat_input(
        self,
        message_batches: list[list[Any]],
        *,
        interaction: SessionBatInteraction,
        metadata: dict[str, Any],
        tags: list[str] | None,
    ) -> None:
        for message in _system_and_user_messages(message_batches):
            role = _message_role(message)
            if role is None:
                continue
            interaction.message(
                role=role,
                content=_message_content(message),
                metadata=_merge_dicts(
                    metadata,
                    {
                        "langchain_message_type": _message_type(message),
                        "langchain_message_id": getattr(message, "id", None),
                    },
                ),
                tags=self._tags(tags),
                context=self.context,
            )

    def _interaction_for(
        self,
        run_id: UUID,
        *,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SessionBatInteraction:
        session = self._session_for(
            run_id,
            parent_run_id=parent_run_id,
            metadata=metadata,
        )
        sessionbat_interaction_id = self._sessionbat_interaction_id_for(
            run_id,
            parent_run_id=parent_run_id,
        )
        return session.interaction(interaction_id=sessionbat_interaction_id)

    def _session_for(
        self,
        run_id: UUID,
        *,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        if self.session is not None:
            session = self.session
            self._remember_run(run_id, parent_run_id=parent_run_id, metadata=metadata)
            return session

        session_id = self._session_id_for(
            run_id,
            parent_run_id=parent_run_id,
            metadata=metadata,
        )
        session = self._sessions.get(session_id)
        if session is None:
            session = self.client.session(session_id=session_id)
            self._sessions[session_id] = session
        self._remember_run(run_id, parent_run_id=parent_run_id, metadata=metadata)
        return session

    def _remember_run(
        self,
        run_id: UUID,
        *,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        session_id = self._session_id_for(
            run_id,
            parent_run_id=parent_run_id,
            metadata=metadata,
        )
        self._run_session_ids[_run_id(run_id)] = session_id
        self._root_interaction_ids[_run_id(run_id)] = self._sessionbat_interaction_id_for(
            run_id,
            parent_run_id=parent_run_id,
        )
        return session_id

    def _sessionbat_interaction_id_for(
        self,
        run_id: UUID,
        *,
        parent_run_id: UUID | None = None,
    ) -> str:
        if parent_run_id is None:
            return _run_id(run_id)

        parent_root_interaction_id = self._root_interaction_ids.get(_run_id(parent_run_id))
        if parent_root_interaction_id:
            return parent_root_interaction_id
        return _run_id(parent_run_id)

    def _session_id_for(
        self,
        run_id: UUID,
        *,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        metadata_session_id = _metadata_session_id(metadata)
        if metadata_session_id:
            return metadata_session_id
        if parent_run_id is not None:
            parent_session_id = self._run_session_ids.get(_run_id(parent_run_id))
            if parent_session_id:
                return parent_session_id
            return _run_id(parent_run_id)
        return _run_id(run_id)


SessionBatCallbackHandler = LangChainCallbackHandler


def _run_id(run_id: UUID | str) -> str:
    return str(run_id)


def _optional_run_id(run_id: UUID | str | None) -> str | None:
    if run_id is None:
        return None
    return _run_id(run_id)


def _merge_tags(*tag_sets: list[str] | None) -> list[str]:
    merged: list[str] = []
    for tag_set in tag_sets:
        if not tag_set:
            continue
        for tag in tag_set:
            if tag not in merged:
                merged.append(tag)
    return merged


def _merge_dicts(*values: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for value in values:
        if value:
            merged.update({key: _jsonable(item) for key, item in value.items()})
    return merged


def _serialized_name(serialized: dict[str, Any] | None, fallback: str) -> str:
    if not serialized:
        return fallback
    name = serialized.get("name")
    if isinstance(name, str) and name:
        return name
    serialized_id = serialized.get("id")
    if isinstance(serialized_id, list) and serialized_id:
        return str(serialized_id[-1])
    if isinstance(serialized_id, str) and serialized_id:
        return serialized_id
    return fallback


def _serialized_metadata(serialized: dict[str, Any] | None) -> dict[str, Any]:
    if not serialized:
        return {}
    metadata: dict[str, Any] = {}
    if "id" in serialized:
        metadata["langchain_serialized_id"] = serialized["id"]
    if "name" in serialized:
        metadata["langchain_serialized_name"] = serialized["name"]
    kwargs = serialized.get("kwargs")
    if isinstance(kwargs, Mapping):
        for key in ("model", "model_name"):
            value = kwargs.get(key)
            if isinstance(value, str) and value:
                metadata[key] = value
    return metadata


def _metadata_session_id(metadata: dict[str, Any] | None) -> str | None:
    if not metadata:
        return None
    for key in ("session_id", "thread_id", "conversation_id"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _invocation_metadata(kwargs: dict[str, Any] | None) -> dict[str, Any]:
    if not kwargs:
        return {}
    invocation_params = kwargs.get("invocation_params")
    if not isinstance(invocation_params, Mapping):
        return {}
    metadata: dict[str, Any] = {}
    for key in ("model", "model_name", "ls_model_name"):
        value = invocation_params.get(key)
        if isinstance(value, str) and value:
            metadata[key] = value
    return metadata


def _duration_metrics(state: _RunState) -> dict[str, Any]:
    return {"latency_ms": round((monotonic() - state.started_at) * 1000, 3)}


def _tool_input(input_str: str, inputs: dict[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"input": _jsonable(input_str)}
    if inputs is not None:
        payload["inputs"] = _jsonable(inputs)
    return payload


def _output_payload(output: Any) -> dict[str, Any]:
    if isinstance(output, Mapping):
        return {str(key): _jsonable(value) for key, value in output.items()}
    return {"result": _jsonable(output)}


def _error_payload(error: BaseException) -> dict[str, Any]:
    return {"type": error.__class__.__name__, "message": str(error)}


def _llm_response_payload(response: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"raw": _jsonable(response)}
    texts = _extract_generation_texts(response)
    if len(texts) == 1:
        payload["text"] = texts[0]
    elif texts:
        payload["texts"] = texts
    return payload


def _extract_generation_texts(response: Any) -> list[str]:
    texts: list[str] = []
    for generation_group in getattr(response, "generations", []) or []:
        for generation in generation_group or []:
            text = getattr(generation, "text", None)
            if isinstance(text, str) and text:
                texts.append(text)
                continue
            message = getattr(generation, "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, str) and content:
                texts.append(content)
    return texts


def _llm_metrics(response: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    llm_output = getattr(response, "llm_output", None) or {}
    if isinstance(llm_output, Mapping):
        token_usage = llm_output.get("token_usage") or llm_output.get("usage")
        if isinstance(token_usage, Mapping):
            _copy_token_metrics(metrics, token_usage)
    for generation_group in getattr(response, "generations", []) or []:
        for generation in generation_group or []:
            message = getattr(generation, "message", None)
            usage = getattr(message, "usage_metadata", None)
            if isinstance(usage, Mapping):
                _copy_token_metrics(metrics, usage)
            response_metadata = getattr(message, "response_metadata", None)
            if isinstance(response_metadata, Mapping):
                token_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
                if isinstance(token_usage, Mapping):
                    _copy_token_metrics(metrics, token_usage)
    return metrics


def _copy_token_metrics(metrics: dict[str, Any], token_usage: Mapping[str, Any]) -> None:
    token_map = {
        "prompt_tokens": "input_tokens",
        "input_tokens": "input_tokens",
        "completion_tokens": "output_tokens",
        "output_tokens": "output_tokens",
        "total_tokens": "total_tokens",
    }
    for source_key, target_key in token_map.items():
        value = token_usage.get(source_key)
        if isinstance(value, int | float):
            metrics[target_key] = value


def _extract_model(
    *,
    response: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    metadata = metadata or {}
    for key in ("model", "model_name", "ls_model_name"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    llm_output = getattr(response, "llm_output", None) if response is not None else None
    if isinstance(llm_output, Mapping):
        for key in ("model_name", "model"):
            value = llm_output.get(key)
            if isinstance(value, str) and value:
                return value
    return "unknown"


def _document_payload(document: Any) -> dict[str, Any]:
    page_content = getattr(document, "page_content", None)
    metadata = getattr(document, "metadata", None)
    doc_id = getattr(document, "id", None)
    if page_content is not None or metadata is not None or doc_id is not None:
        payload: dict[str, Any] = {
            "content": _jsonable(page_content),
            "metadata": _jsonable(metadata or {}),
        }
        if doc_id is not None:
            payload["id"] = _jsonable(doc_id)
        return payload
    if isinstance(document, Mapping):
        return {str(key): _jsonable(value) for key, value in document.items()}
    return {"content": _jsonable(document)}


def _system_and_user_messages(message_batches: list[list[Any]]) -> list[Any]:
    if not message_batches:
        return []
    messages = message_batches[-1]
    return [message for message in messages if _message_role(message) in {"system", "user"}]


def _message_role(message: Any) -> str | None:
    message_type = _message_type(message)
    if message_type in {"human", "user"}:
        return "user"
    if message_type == "system":
        return "system"
    return None


def _message_type(message: Any) -> str | None:
    message_type = getattr(message, "type", None)
    if isinstance(message_type, str):
        return message_type
    role = getattr(message, "role", None)
    if isinstance(role, str):
        return role
    if isinstance(message, Mapping):
        value = message.get("type") or message.get("role")
        if isinstance(value, str):
            return value
    return None


def _message_content(message: Any) -> Any:
    content = getattr(message, "content", None)
    if content is not None:
        return _jsonable(content)
    if isinstance(message, Mapping) and "content" in message:
        return _jsonable(message["content"])
    return _jsonable(message)


def _jsonable(value: Any, *, _depth: int = 0) -> Any:
    if _depth > 8:
        return repr(value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, BaseException):
        return _error_payload(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item, _depth=_depth + 1) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable(item, _depth=_depth + 1) for item in value]

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _jsonable(model_dump(mode="json"), _depth=_depth + 1)
        except TypeError:
            try:
                return _jsonable(model_dump(), _depth=_depth + 1)
            except Exception:
                pass
        except Exception:
            pass

    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        try:
            return _jsonable(dict_method(), _depth=_depth + 1)
        except Exception:
            pass

    to_json = getattr(value, "to_json", None)
    if callable(to_json):
        try:
            return _jsonable(to_json(), _depth=_depth + 1)
        except Exception:
            pass

    return repr(value)
