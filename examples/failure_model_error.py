from sessionbat import SessionBat


def main() -> None:
    client = SessionBat(
        app="support-bot",
        default_tags=["demo", "failure-mode"],
        default_context={"environment": "development"},
    )

    session = client.session(
        session_id="thread_model_error",
        tags=["support"],
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
        tags=["openai"],
    )

    session.error(
        name="assistant_response_failed",
        message="The assistant could not answer because the upstream model request failed.",
        error_type="llm_request_failed",
        metadata={"provider": "openai", "customer_visible": True},
        metrics={"retry_count": 2},
        tags=["incident"],
    )


if __name__ == "__main__":
    main()
