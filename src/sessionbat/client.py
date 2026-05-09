from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .models import Envelope, isoformat, new_id, utc_now
from .transports import StdoutTransport, Transport


ObservationKind = Literal["message", "llm", "tool", "retrieval", "error"]
MessageRole = Literal["user", "assistant", "system", "tool"]


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
            merged.update(value)
    return merged


@dataclass(slots=True)
class SessionBat:
    transport: Transport = field(default_factory=StdoutTransport)
    api_key: str | None = None
    endpoint: str | None = None
    app: str | None = None
    default_tags: list[str] = field(default_factory=list)
    default_context: dict[str, Any] = field(default_factory=dict)

    def session(
        self,
        *,
        session_id: str,
        tags: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> "Session":
        return Session(
            client=self,
            session_id=session_id,
            tags=_merge_tags(self.default_tags, tags),
            context=_merge_dicts(self.default_context, context),
        )

    def _send(self, payload: dict[str, Any]) -> None:
        self.transport.send(payload)


@dataclass(slots=True)
class Session:
    client: SessionBat
    session_id: str
    tags: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    def message(
        self,
        *,
        role: MessageRole,
        content: Any,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        return self._record(
            kind="message",
            name=name or f"{role}_message",
            input={"content": content},
            metadata=_merge_dicts({"role": role}, metadata),
            tags=tags,
            context=context,
        )

    def user_message(
        self,
        content: Any,
        *,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        return self.message(
            role="user",
            content=content,
            metadata=metadata,
            tags=tags,
            context=context,
        )

    def assistant_response(
        self,
        *,
        model: str,
        request: dict[str, Any] | None = None,
        response: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        return self._record(
            kind="llm",
            name="assistant_response",
            input=request,
            output=response,
            error=error,
            metadata=_merge_dicts({"model": model}, metadata),
            metrics=metrics,
            tags=tags,
            context=context,
        )

    def tool_call(
        self,
        *,
        tool_name: str,
        input: dict[str, Any] | None = None,
        output: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        return self._record(
            kind="tool",
            name=tool_name,
            input=input,
            output=output,
            error=error,
            metadata=metadata,
            metrics=metrics,
            tags=tags,
            context=context,
        )

    def retrieval(
        self,
        *,
        query: str,
        documents: list[dict[str, Any]] | None = None,
        error: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        return self._record(
            kind="retrieval",
            name="retrieval",
            input={"query": query},
            output={"documents": documents or []},
            error=error,
            metadata=metadata,
            metrics=metrics,
            tags=tags,
            context=context,
        )

    def error(
        self,
        *,
        name: str,
        message: str,
        error_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        return self._record(
            kind="error",
            name=name,
            error={"type": error_type, "message": message},
            metadata=metadata,
            metrics=metrics,
            tags=tags,
            context=context,
        )

    def _record(
        self,
        *,
        kind: ObservationKind,
        name: str,
        input: dict[str, Any] | None = None,
        output: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        observation_id = new_id()
        envelope = Envelope(
            type="observation.recorded",
            tags=_merge_tags(self.tags, tags),
            context=_merge_dicts(self.context, context),
        )
        payload = envelope.as_dict()
        payload.update(
            {
                "session_id": self.session_id,
                "observation": {
                    "id": observation_id,
                    "kind": kind,
                    "name": name,
                    "recorded_at": isoformat(utc_now()),
                    "input": input,
                    "output": output,
                    "error": error,
                    "metadata": metadata or {},
                    "metrics": metrics or {},
                },
            }
        )
        self.client._send(payload)
        return observation_id
