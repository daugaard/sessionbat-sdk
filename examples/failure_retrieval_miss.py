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
        session_id="thread_retrieval_miss",
        context={
            "user_id": "user_222",
            "user_email": "user222@example.com",
            "workspace_id": "ws_456",
            "plan": "pro",
        },
    )

    session.user_message("How do I export audit logs for SSO users?")

    session.retrieval(
        query="export audit logs for sso users",
        documents=[],
        metadata={
            "index": "support_articles",
            "strategy": "semantic",
            "top_k": 5,
        },
        metrics={"latency_ms": 63, "documents_found": 0},
    )

    session.assistant_response(
        model="gpt-5.4-mini",
        request={
            "messages": [
                {"role": "user", "content": "How do I export audit logs for SSO users?"},
            ]
        },
        response={
            "text": "I could not find documentation for exporting audit logs for SSO users.",
        },
        metadata={"provider": "openai"},
        metrics={"latency_ms": 704, "input_tokens": 118, "output_tokens": 24},
    )
    session.user_message(
        "Can an admin download SAML access logs anywhere?",
        metadata={"sequence": "follow_up"},
    )


if __name__ == "__main__":
    main()
