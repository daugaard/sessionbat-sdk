# SessionBat SDK

SessionBat is an early SDK for capturing AI session activity through manual
observations and framework adapters.

This repository currently focuses on a manual API built around completed observations that map cleanly onto common LangChain activity:

- `message`
- `assistant_response`
- `tool_call`
- `retrieval`

Failures are attached to the operation that failed using its `error` payload,
rather than emitted as a separate standalone event type.

The current shape is intentionally simple so we can pressure-test the data model
while adding framework-specific adapters incrementally.

## Install

```bash
uv sync
```

For development, install dev dependencies too:

```bash
uv sync --dev
```

## Example

```python
from sessionbat import SessionBat

client = SessionBat(
    app="support-bot",
    default_tags=["development", "support-bot"],
    default_context={"environment": "development"},
)

session = client.session(
    session_id="thread_123",
    context={
        "user_id": "user_123",
        "user_email": "person@example.com",
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
            "snippet": "Use the password reset link from the sign-in page to receive an email reset link.",
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

## Failure Mode Examples

The repository also includes a few pressure-test traces:

- [examples/failure_retrieval_miss.py](/home/soren/Projects/sessionbat/sessionbat-sdk/examples/failure_retrieval_miss.py): retrieval returns no documents, then the user rephrases the question
- [examples/failure_tool_loop.py](/home/soren/Projects/sessionbat/sessionbat-sdk/examples/failure_tool_loop.py): the same tool is called repeatedly with no new information
- [examples/failure_model_error.py](/home/soren/Projects/sessionbat/sessionbat-sdk/examples/failure_model_error.py): the upstream model request fails after retries

## Data shape

- `tags` are lightweight labels for stable filtering and grouping such as environment, app, or deployment.
- `context` is user/account/application metadata attached to the whole session or a single observation.
- `metadata` is descriptive operation data such as model, provider, index, or service name.
- `metrics` is numeric operational data such as tokens, latency, cost, or HTTP status.
- `input` and `output` hold raw observation payloads.

The SDK emits raw structured payloads via a transport. The default transport prints JSON lines to stdout so traces are easy to inspect locally.

## LangChain Callback

The LangChain adapter maps LangChain callbacks onto the same completed
observations. For chat models, it records rendered system and user messages
before the assistant response.

```python
from sessionbat import SessionBat

client = SessionBat(app="support-bot")
session = client.session(session_id="thread_123")
handler = session.langchain_callback(tags=["langchain"])

result = chain.invoke(
    {"input": "I am locked out of my account"},
    config={"callbacks": [handler]},
)
```

It records:

- rendered system and user chat messages as `message`
- LLM/chat model calls as `assistant_response`
- tools as `tool_call`
- retrievers as `retrieval`

Callback errors are attached to the operation that failed where possible;
chain-level errors are not emitted as standalone events.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Design notes

- Sessions are persistent identities, not bounded operations.
- Observations are recorded as completed events in one call.
- The public primitives are intentionally limited to concepts that should map directly from LangChain callbacks.
