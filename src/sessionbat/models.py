from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

SpanKind = Literal["llm", "tool", "retrieval", "chain", "custom"]
EventLevel = Literal["debug", "info", "warning", "error"]


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return uuid4().hex


def isoformat(dt: datetime) -> str:
    return dt.isoformat()


@dataclass(slots=True)
class Envelope:
    type: str
    id: str = field(default_factory=new_id)
    created_at: datetime = field(default_factory=utc_now)
    tags: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "id": self.id,
            "created_at": isoformat(self.created_at),
            "tags": self.tags,
            "context": self.context,
        }
