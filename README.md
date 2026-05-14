# SessionBat SDK

![Tests](https://img.shields.io/github/actions/workflow/status/daugaard/sessionbat-sdk/tests.yml?branch=main)
![Ruff](https://img.shields.io/github/actions/workflow/status/daugaard/sessionbat-sdk/ruff.yml?branch=main)
![License](https://img.shields.io/github/license/daugaard/sessionbat-sdk)

SessionBat is a Python SDK for recording AI session activity and sending it to SessionBat.

It is designed for teams that want to debug and understand what your AI app actually did, including:
- user messages
- assistant responses
- tool calls
- document retrievals

## Install

```bash
uv add sessionbat
```

Or with `pip`:

```bash
pip install sessionbat
```

## Quickstart 

### LangChain integration
The easist way to start is to use the built-in LangChain callback handler. Create a `SessionBat` client and pass the handler to your chain or agent:
```python
from sessionbat import SessionBat

client = SessionBat(app="support-bot", api_key="sbat_ingest_...")
handler = client.langchain_callback(tags=["langchain"])

# Configure your chain or agent to use the handler, for example:
result = chain.invoke(
  {"input": "I am locked out of my account"},
  config={"callbacks": [handler]},
)

# or using agents:
agent.invoke(
  {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
  config={"callbacks": [handler]},
)
```

### Custom integration
Or you can integrate directly with the core API for more control. Create a `SessionBat` client and use it to create a `Session` with a stable `session_id` and shared `context`. Then record observations against that session as they happen.

```python
from sessionbat import SessionBat

client = SessionBat(
    api_key="sbat_ingest_...",
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
    api_key="sbat_ingest_...",
    endpoint="https://ingest.sessionbat.com/api/v1/ingestion/events",
)
```

Use `client.session(...)` to create a session and record observations against a
stable `session_id`.

The SDK sends events to SessionBat ingestion by default. Pass `api_key` directly
or set `SESSIONBAT_API_KEY`. For tests or local debugging, pass an explicit
transport such as `MemoryTransport` or `StdoutTransport`.

HTTP ingestion runs in a background thread so recording observations does not
block your application on network I/O. Transient failures are retried with
bounded backoff, and queued events are flushed automatically during interpreter
shutdown. Call `client.flush()` or `client.close()` when you need to wait for
delivery before exiting a short-lived process.

### `Session`

A `Session` records completed observations:

- `session.user_message(content)`
- `session.message(role=..., content=...)`
- `session.assistant_response(...)`
- `session.tool_call(...)`
- `session.retrieval(...)`

Each call returns the generated observation id.

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
- `src/sessionbat/transports.py` defines the ingestion transport plus stdout
  and in-memory transports for debugging and tests.

## License

MIT.
