# homebrew-browser-use

A browser-automation agent built on the Claude Agent SDK: Claude drives a real
Playwright/Chromium session (visible live over VNC) to navigate sites, read
the DOM, extract data at scale, and produce deliverables — with a
deterministic safety layer between the model and destructive actions.

Forked from Anthropic's `browser-use-demo` reference implementation and
rebuilt around DOM-first navigation, scripted bulk extraction, and
verification of its own output.

## Quick start

```bash
cp .env.example .env        # add your ANTHROPIC_API_KEY
docker compose up --build
```

- **`localhost:8080`** — the Streamlit chat interface
- **`localhost:6080`** — live VNC view of the browser session

## Tool reference

See [DOCS.md](DOCS.md) for what each tool does and when the agent reaches
for it.

## Repo layout

- `browser_use_demo/loop.py` — the agent loop: system prompt, hooks, tool wiring
- `browser_use_demo/tools/` — the tool implementations (see DOCS.md)
- `browser_use_demo/guardrails.py` — the deterministic safety layer
- `browser_use_demo/streamlit.py` — the chat UI
- `tests/` — unit tests (`pytest`)
