# forge · Forging ideas into action

**A config-driven, multi-model AI agent with a readable codebase — learnable, and production-ready enough to actually use.**

[![CI](https://github.com/musokean/forge/actions/workflows/ci.yml/badge.svg)](https://github.com/musokean/forge/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.11%20%7C%203.13-blue.svg)](https://github.com/musokean/forge)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[中文版 README](README.zh-CN.md)

---

## Why forge?

Most agent projects fall into two camps:

- **Production-grade giants** (OpenHands, AutoGen, MetaGPT): tens of thousands of lines, powerful ecosystems — but you can't read them to understand how agents work.
- **Minimal demos** (smolagents-style ~1k lines, tutorials): readable, but they stop at "it runs" — no engineering foundation.

**forge sits in the gap**: the ~150-line ReAct loop is implemented directly, every mechanism (retry / fallback / circuit breaker / rolling summary / approval gates) is explainable, *and* it ships with a complete engineering foundation — multi-agent orchestration, a self-contained knowledge base, long-term memory, golden-set evaluation, and a zero-dependency philosophy.

**Zero heavy dependencies**: standard library + `openai` SDK + `pyyaml`. Works with any OpenAI-compatible endpoint — DeepSeek, Qwen, vLLM, Ollama, local models. Chinese-first, domestic-model friendly.

## Quick start

```bash
# install from GitHub (or clone the repo and run: pip install -e .)
pip install "git+https://github.com/musokean/forge.git"

# set your API key (env var, picked up automatically)
export DEEPSEEK_API_KEY=sk-xxx

forge                     # interactive REPL
forge "帮我算 (3+5)*2"      # one-shot question
forge --web               # browser chat UI (zero-dependency HTTP server)
forge --serve --port 8080 # HTTP API service (pip install "handcraft-agent[server]" first)
```

First run auto-generates a default `config/models.yaml` (if missing) — no config file, no crash. Edit it (or `/config` in the REPL) to switch models / roles / endpoints. Model registry → roles → debate lineup → routing → knowledge base path, all config-driven, no code changes.

## Feature highlights

| Area | What you get |
|------|-------------|
| **Core loop** | Direct ReAct loop implementation with loop-guard, streaming output, reasoning display |
| **Engineering** | Tool read-only tiers, write-operation approval gates, exponential-backoff retry, model fallback, circuit breaker, token accounting, per-step trace |
| **Multi-agent** | Parallel task fan-out, multi-role debate (pro/con/judge), supervisor plan→execute→merge, automatic task routing |
| **Context** | Token-budget truncation, rolling summary via cheap model, tool-output clipping |
| **Knowledge** | Self-contained SQLite+FTS5 knowledge base (the index *is* the source), Chinese trigram search, one-key ingest/sync/export |
| **Memory** | Cross-session user profile auto-recalled per query |
| **Reliability** | Golden-set regression (`/eval`, keyword-hit + LLM-as-judge), model-failure resilience, endpoint self-check on startup |
| **UX** | Sky-blue theme, interrupt/redirect generation (Esc / type a steer), auto tasks, Web UI |
| **Service (#14)** | HTTP API (`forge --serve`): multi-session persistence, API-key auth (loopback-only by default), per-caller rate limiting, Swagger docs at `/docs` |

## Commands

```
/reset /usage /trace /kb /export /key /model /config /circuit
/skill /memory /remember /task /eval /web /serve /help /exit
```

`/key sk-xxx` — paste a key, auto-assigns to the main model. `/config` — guided panel, no YAML hand-editing needed.

## Project layout

```
handcraft-agent/
├── config/models.yaml    # all configuration (models/roles/debate/router/kb)
├── config/golden.yaml    # golden-set eval cases
├── src/
│   ├── agent.py          # ReAct loop + context mgmt + status bar + approval
│   ├── llm.py            # openai gateway + retry + fallback + streaming + breaker
│   ├── tools.py          # 13 tools + read-only tiers + KB tools
│   ├── orchestrator.py   # parallel / debate / supervisor
│   ├── router.py         # rule-first task routing (0ms for common intents)
│   ├── knowledge.py      # SQLite+FTS5 knowledge base
│   ├── memory.py         # cross-session user profile
│   ├── eval.py           # golden-set evaluation
│   ├── web.py            # zero-dependency web chat
│   ├── server.py         # #14 HTTP API service (sessions + auth + rate limit)
│   ├── keypress.py       # interrupt/steer during generation
│   └── ...
├── main.py               # CLI entry
└── test_*.py             # milestone + stress + module tests (all mock, no network)
```

## Server mode (HTTP API)

Turn forge into an HTTP service with **sessions, auth and rate limiting** (module #14):

```bash
pip install "handcraft-agent[server]"     # optional extra: fastapi + uvicorn
forge --serve --port 8080                 # or "/serve 8080" inside the REPL
```

- **Sessions** — every conversation is persisted in SQLite (`data/sessions.db`), each with its own Agent context: clients can disconnect and resume later, or keep several threads apart.
- **Auth** — `Authorization: Bearer <key>` or `X-API-Key: <key>`. Keys come from `server.api_keys` in `config/models.yaml`, or the `FORGE_API_KEY` env var (comma-separated, `env:VAR` indirection supported). **No keys configured → loopback-only** (convenient locally; never expose that to the internet).
- **Rate limit** — `server.rate_limit_per_min` (default 60) per caller; beyond it you get `429` + `Retry-After`.
- **Write safety** — a service has no interactive approval channel, so write tools are rejected by default; keep those in the CLI. Override with `server.approve_mode` only for controlled deployments.
- Interactive docs: `http://127.0.0.1:8080/docs`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` | liveness probe (no auth) |
| GET | `/api/status` | model, session count, auth mode, rate limit |
| POST | `/api/chat` | `{"message": "...", "session_id": "optional"}` → reply + token usage |
| POST | `/api/sessions` | create a session (`{"title": "optional"}`) |
| GET | `/api/sessions` | list sessions |
| GET | `/api/sessions/{id}` | session + full message history |
| DELETE | `/api/sessions/{id}` | delete a session |

```bash
curl -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
     -d '{"message": "hello"}' http://127.0.0.1:8080/api/chat
```

## Tests

```bash
python test_router.py     # task routing (rule-first + model fallback)
python test_interrupt.py  # interrupt / steer during generation
python test_eval.py       # golden-set evaluation
python test_web.py        # web server end-to-end
python test_server.py     # HTTP API service (auth / sessions / rate limit)
python test_knowledge.py  # knowledge base
# ... plus stress tests: test_stress*.py
```

All tests run fully offline (mocked model calls) — CI-friendly.

## Contributing

This project lives with a companion knowledge base (A01–A28 concept cards) that maps every implementation detail to the underlying agent principle. Issues and PRs welcome.

## License

MIT
