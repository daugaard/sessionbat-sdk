# SessionBat SDK

SessionBat is an early SDK for capturing AI session activity before deciding how much framework-specific instrumentation we need.

This repository currently focuses on a manual API built around completed observations:

- `message`
- `assistant_response`
- `tool_call`
- `retrieval`
- `error`

The current shape is intentionally simple so we can pressure-test the data model before building LangChain or OpenAI-specific adapters.

## Install

```bash
uv sync
```

## Example

```python
from sessionbat import SessionBat

client = SessionBat(
    app="support-bot",
    default_tags=["demo"],
    default_context={"environment": "development"},
)

session = client.session(
    session_id="thread_123",
    tags=["support"],
    context={
        "user_id": "user_123",
        "user_email": "person@example.com",
        "workspace_id": "ws_456",
    },
)

session.user_message("I am locked out of my account")

session.retrieval(
    query="reset password locked out",
    documents=[],
    metadata={"index": "support_articles"},
    metrics={"latency_ms": 81},
    tags=["knowledge-base"],
)

session.tool_call(
    tool_name="lookup_account",
    input={"account_id": "acct_987"},
    output={"status": "locked", "password_reset_available": True},
    metadata={"service": "account-service"},
    metrics={"latency_ms": 117, "http_status": 200},
)

session.assistant_response(
    model="gpt-5.4-mini",
    request={"messages": [{"role": "user", "content": "I am locked out of my account"}]},
    response={"text": "I found your account. Use the reset link and follow the email prompt."},
    metrics={"latency_ms": 820, "input_tokens": 142, "output_tokens": 36},
    tags=["openai"],
)

```

## Data shape

- `tags` are lightweight labels for filtering and grouping.
- `context` is user/account/application metadata attached to the whole session or a single observation.
- `metadata` is descriptive operation data such as model, provider, index, or service name.
- `metrics` is numeric operational data such as tokens, latency, cost, or HTTP status.
- `input` and `output` hold raw observation payloads.

The SDK emits raw structured payloads via a transport. The default transport prints JSON lines to stdout so traces are easy to inspect locally.

## Design notes

- Sessions are persistent identities, not bounded operations.
- Observations are recorded as completed events in one call.
- A future LangChain callback handler should map callbacks into these same observation primitives instead of inventing a second data model.
