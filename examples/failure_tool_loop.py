from sessionbat import SessionBat


def main() -> None:
    client = SessionBat(
        app="support-bot",
        default_tags=["development", "support-bot"],
        default_context={"environment": "development"},
    )

    session = client.session(
        session_id="thread_tool_loop",
        context={
            "user_id": "user_777",
            "user_email": "user777@example.com",
            "workspace_id": "ws_999",
            "plan": "enterprise",
        },
    )

    session.user_message("Why is my workspace usage total wrong?")

    session.tool_call(
        tool_name="get_workspace_usage",
        input={"workspace_id": "ws_999"},
        output={"total_tokens": 192044, "cached": False},
        metadata={"service": "billing-service"},
        metrics={"latency_ms": 141, "http_status": 200, "attempt": 1},
    )

    session.tool_call(
        tool_name="get_workspace_usage",
        input={"workspace_id": "ws_999"},
        output={"total_tokens": 192044, "cached": False},
        metadata={"service": "billing-service"},
        metrics={"latency_ms": 136, "http_status": 200, "attempt": 2},
    )

    session.tool_call(
        tool_name="get_workspace_usage",
        input={"workspace_id": "ws_999"},
        output={"total_tokens": 192044, "cached": False},
        metadata={"service": "billing-service"},
        metrics={"latency_ms": 139, "http_status": 200, "attempt": 3},
    )

    session.assistant_response(
        model="gpt-5.4-mini",
        request={
            "messages": [
                {"role": "user", "content": "Why is my workspace usage total wrong?"},
            ]
        },
        response={
            "text": "I checked the usage totals again, but I still cannot determine the discrepancy.",
        },
        metadata={"provider": "openai"},
        metrics={"latency_ms": 1298, "input_tokens": 203, "output_tokens": 31},
    )


if __name__ == "__main__":
    main()
