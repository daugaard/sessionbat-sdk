# SessionBat SDK

![Tests](https://img.shields.io/github/actions/workflow/status/daugaard/sessionbat-sdk/tests.yml?branch=main)
![Ruff](https://img.shields.io/github/actions/workflow/status/daugaard/sessionbat-sdk/ruff.yml?branch=main)
![License](https://img.shields.io/github/license/daugaard/sessionbat-sdk)

SessionBat is a small Python SDK for recording AI session activity as structured
JSON events.

It is designed for teams that want to inspect what an AI app actually did in a
conversation, including:

- user messages
- assistant responses
- tool calls
- retrievals
- failures attached to the operation that failed

The default transport writes newline-delimited JSON to `stdout`, which makes it
easy to inspect events locally or pipe them into another system.

## Install

```bash
uv add sessionbat
```

Or with `pip`:

```bash
pip install sessionbat
```

For local development in this repository:

```bash
uv sync --dev
```

## Quickstart

```python
from sessionbat import SessionBat

client = SessionBat(
    app="support-bot",
    default_tags=["production"],
    default_context={"environment": "prod"},
)

session = client.session(
    session_id="thread_123",
    context={
        "user_id": "user_123",
        "workspace_id": "ws_456",
    },
)

session.user_message("I am locked out of my account")

session.retrieval(
    query="reset password locked out",
    documents=[
        {
            "id": "doc_reset_password",
            "title": "Reset your password",
            "score": 0.93,
        }
    ],
    metadata={"index": "support_articles"},
    metrics={"latency_ms": 81, "documents_found": 1},
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
)
```

That emits structured events like:

```json
{
  "type": "llm",
  "session_id": "thread_123",
  "tags": ["production"],
  "context": {"environment": "prod", "user_id": "user_123", "workspace_id": "ws_456"},
  "observation": {
    "kind": "llm",
    "name": "assistant_response"
  }
}
```

## Core API

Import the main types from `sessionbat`:

```python
from sessionbat import SessionBat, Session, LangChainCallbackHandler
```

### `SessionBat`

`SessionBat` is the client entrypoint.

```python
client = SessionBat(
    app="support-bot",
    api_key="optional",
    endpoint="optional",
)
```

Use `client.session(...)` to create a session and record observations against a
stable `session_id`.

### `Session`

A `Session` records completed observations:

- `session.user_message(content)`
- `session.message(role=..., content=...)`
- `session.assistant_response(...)`
- `session.tool_call(...)`
- `session.retrieval(...)`

Each call returns the generated observation id.

### `LangChainCallbackHandler`

The LangChain adapter maps callback events onto the same observation model.

```python
from sessionbat import SessionBat

client = SessionBat(app="support-bot")
handler = client.langchain_callback(tags=["langchain"])

result = chain.invoke(
    {"input": "I am locked out of my account"},
    config={"callbacks": [handler]},
)
```

If you already have a session object, you can attach the callback to it:

```python
handler = session.langchain_callback(tags=["langchain"])
```

## Event model

SessionBat keeps the shape intentionally small:

- `tags` are lightweight labels for filtering and grouping.
- `context` holds session-level or observation-level metadata.
- `metadata` stores descriptive operation data such as model, provider, index,
  or service name.
- `metrics` stores numeric data such as latency, tokens, cost, or HTTP status.
- `input` and `output` hold raw payloads.
- `error` is attached to the failed operation instead of being emitted as a
  separate event type.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Repository layout

- `src/sessionbat/client.py` contains the core recording API.
- `src/sessionbat/langchain.py` contains the LangChain adapter.
- `src/sessionbat/transports.py` defines the default stdout transport and the
  in-memory test transport.

## License

MIT.
