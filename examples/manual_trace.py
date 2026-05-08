from sessionbat import SessionBat


def main() -> None:
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
            "plan": "pro",
        },
    )

    session.user_message("I am locked out of my account")

    session.retrieval(
        query="reset password locked out",
        documents=[],
        metadata={"index": "support_articles", "strategy": "semantic"},
        metrics={"latency_ms": 81, "documents_found": 0},
        tags=["knowledge-base"],
    )

    session.tool_call(
        tool_name="lookup_account",
        input={"account_id": "acct_987"},
        output={"status": "locked", "password_reset_available": True},
        metadata={"service": "account-service"},
        metrics={"latency_ms": 117, "http_status": 200},
        tags=["internal-tool"],
    )

    session.assistant_response(
        model="gpt-5.4-mini",
        request={"messages": [{"role": "user", "content": "I am locked out of my account"}]},
        response={"text": "I found your account. Use the reset link and follow the email prompt."},
        metadata={"provider": "openai"},
        metrics={"latency_ms": 820, "input_tokens": 142, "output_tokens": 36},
        tags=["openai"],
    )


if __name__ == "__main__":
    main()
