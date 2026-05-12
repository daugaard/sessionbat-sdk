from sessionbat import SessionBat
from sessionbat.transports import StdoutTransport


def main() -> None:
    client = SessionBat(
        transport=StdoutTransport(),
        app="support-bot",
        default_tags=["development", "support-bot"],
        default_context={"environment": "development"},
    )

    session = client.session(
        session_id="thread_model_error",
        context={
            "user_id": "user_404",
            "user_email": "user404@example.com",
            "workspace_id": "ws_456",
            "plan": "starter",
        },
    )

    session.user_message("What invoices are overdue for Acme Corp?")

    session.assistant_response(
        model="gpt-5.4-mini",
        request={
            "messages": [
                {"role": "user", "content": "What invoices are overdue for Acme Corp?"},
            ]
        },
        error={
            "type": "rate_limit_error",
            "message": "OpenAI rate limit exceeded while generating response.",
        },
        metadata={"provider": "openai", "request_id": "req_123"},
        metrics={"latency_ms": 2150, "retry_count": 2},
    )


if __name__ == "__main__":
    main()
