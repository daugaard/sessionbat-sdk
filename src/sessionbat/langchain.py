from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from time import monotonic
from typing import Any
from uuid import UUID

from .client import Session

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
    ignore_chain = True
    ignore_chat_model = False
    ignore_custom_event = True
    ignore_llm = False
    ignore_retriever = False
    ignore_retry = True
    raise_error = False
    run_inline = False

    def __init__(
        self,
        session: Session,
        *,
        tags: list[str] | None = None,
        context: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.session = session
        self.tags = tags or []
        self.context = context or {}
        self.metadata = metadata or {}
        self._runs: dict[str, _RunState] = {}

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
        self._runs[_run_id(run_id)] = _RunState(
            kind="llm",
            name=_serialized_name(serialized, "llm"),
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
        self._runs[_run_id(run_id)] = _RunState(
            kind="llm",
            name=_serialized_name(serialized, "chat_model"),
            input={
                "messages": _jsonable(messages),
                "serialized": _jsonable(serialized),
                "invocation_params": _jsonable(kwargs.get("invocation_params")),
            },
            metadata=self._metadata(
                run_id=run_id,
                parent_run_id=parent_run_id,
                serialized=serialized,
                metadata=metadata,
                extra={"langchain_callback": "on_chat_model_start", "chat_model": True},
                kwargs=kwargs,
            ),
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
        self.session.assistant_response(
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
        self.session.assistant_response(
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
        self._runs[_run_id(run_id)] = _RunState(
            kind="tool",
            name=_serialized_name(serialized, "tool"),
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
        self.session.tool_call(
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
        self.session.tool_call(
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
        self._runs[_run_id(run_id)] = _RunState(
            kind="retrieval",
            name=_serialized_name(serialized, "retriever"),
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
        self.session.retrieval(
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
        self.session.retrieval(
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
