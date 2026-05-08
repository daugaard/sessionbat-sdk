i# SessionBat

SessionBat is a B2B SaaS product that helps teams understand whether their AI applications are actually working.

Core value proposition:
> Know when your AI fails users.

SessionBat analyzes AI sessions to detect:
- unsuccessful conversations
- user frustration
- duplicate or excessive tool calls
- runaway MCP loops
- retrieval failures
- cost spikes

The core question SessionBat answers is:
> Did the AI actually help the user?

## Architecture Principles

- Keep ingestion simple and durable.
- Store raw events before processing.
- Process asynchronously.
- Preserve raw events so derived data can be rebuilt.
- Optimize for speed and simplicity.

## Tech Stack

- Ruby on Rails
- PostgreSQL
- Active Storage
- Active Job + Solid Queue
- Python for SDK (ingestion)

## Product Positioning

SessionBat is:
- Sentry for AI products.
- Honeybadger for AI applications.
- Echolocation for AI sessions.
