# forge

**A ReAct agent you can actually read — the step loop is ~235 lines of straight-line Python, and the other 31 modules are opt-in layers you can skip.**

**Zero heavy dependencies** (stdlib + `openai` + `pyyaml`). Every mechanism is spelled out instead of hidden: retry · fallback · circuit breaker · approval gates · command sandbox · structured logs · four-role Computer Use loop. Runs against any OpenAI-compatible endpoint — DeepSeek, Qwen, vLLM, Ollama, local models.

<!-- 录好 GIF 后取消这行注释、并删掉本注释：![forge REPL — ask a question, watch the trace, interrupt mid-generation](docs/assets/forge-cli.gif) -->

[![CI](https://github.com/musokean/forge/actions/workflows/ci.yml/badge.svg)](https://github.com/musokean/forge/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/handcraft-agent.svg)](https://pypi.org/project/handcraft-agent/)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.11%20%7C%203.13-blue.svg)](https://github.com/musokean/forge)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[中文版 README](README.zh-CN.md) · [docs/](docs/) · [Releases](https://github.com/musokean/forge/releases) · [CHANGELOG](CHANGELOG.md) · 373 tests, all offline

```bash
pip install "handcraft-agent[server,device]"              # from PyPI (recommended)
pip install "git+https://github.com/musokean/forge.git"   # or: from GitHub, or clone && pip install -e .
export DEEPSEEK_API_KEY=sk-xxx
forge                        # interactive REPL
forge "帮我算 (3+5)*2"        # one-shot question
forge --web                  # browser chat UI (zero-dependency HTTP server)
```

---

## Why forge?

Most agent projects fall into two camps:

- **Production-grade giants** (OpenHands, AutoGen, MetaGPT): tens of thousands of lines, powerful ecosystems — but you can't read them to understand how agents work.
- **Minimal demos** (smolagents-style ~1k lines, tutorials): readable, but they stop at "it runs" — no engineering foundation.

**forge sits in the gap**: the ~150-line ReAct loop is implemented directly, every mechanism (retry / fallback / circuit breaker / rolling summary / approval gates) is explainable, *and* it ships with a complete engineering foundation — multi-agent orchestration, a self-contained knowledge base, long-term memory, golden-set evaluation, and a zero-dependency philosophy.

**Zero heavy dependencies**: standard library + `openai` SDK + `pyyaml`. Works with any OpenAI-compatible endpoint — DeepSeek, Qwen, vLLM, Ollama, local models. Chinese-first, domestic-model friendly.

## Quick start

```bash
pip install "handcraft-agent[server,device]"                                  # from PyPI
# or from GitHub (or clone the repo and run: pip install -e .)
pip install "git+https://github.com/musokean/forge.git"

# set your API key (env var, picked up automatically)
export DEEPSEEK_API_KEY=sk-xxx

forge                     # interactive REPL
forge "帮我算 (3+5)*2"      # one-shot question
forge --web               # browser chat UI (zero-dependency HTTP server)
forge --serve --port 8080 # HTTP API service (pip install "handcraft-agent[server]" first)
```

First run auto-generates a default `config/models.yaml` (if missing) — no config file, no crash. Edit it (or `/config` in the REPL) to switch models / roles / endpoints. Model registry → roles → debate lineup → routing → knowledge base path, all config-driven, no code changes.
    Installed with pip? It goes to `~/.forge/config/models.yaml` (or set `FORGE_CONFIG` to
    point at your own) — a config in the current directory always wins.

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
| **Client executor (#17)** | Drive remote PCs: a light executor on each machine dials out (long poll, no inbound port) and exposes shell / files / screenshot / GUI input behind two policy layers. A four-role Computer Use loop (planner → executor → evaluator → supervisor) keeps one model from being brain, hand and judge at once |
| **Voice (#11)** | Cascade voice pipeline with streaming transcription, sentence-level synthesis and barge-in; audio source and playback are injectable, so the whole mechanism is tested in CI without a microphone or a sound |
| **Who it is talking to (#18)** | Opt-in face recognition inside the voice loop: about once a second it matches who is in front of the camera against a local store of 128-dim vectors (no images kept, store kept outside the repo) and injects one line of scene text into the system prompt, so it addresses you by name. Two enrolled people: highest cross-person similarity 0.267 against 0.482 same-person, threshold 0.36 - and it says "not sure" rather than guessing |
| **Hardware (#16)** | Serial / MQTT real link behind a control plane: asset registry, staged policy, command state machine (Created→Sent→Accepted→Applied) with timeout, retries and rollback, plus agent-side temperature/runtime guards. `device_sim.py` speaks the same protocol, so the whole link is testable with no hardware |
| **Safety (#4)** | Command sandbox: Docker isolation when available (no network, read-only mount, memory/CPU/PID caps, non-root), hardened local fallback, dangerous-command blocking. Host environment is never handed to child processes — a command can no longer read your API keys |
| **Logging (#7)** | Structured JSONL logs with rotation, retention and **secret redaction**; per-run correlation ids (role/model/steps/tokens/latency); HTTP request log; `/logs` to inspect |

## Commands

```
/reset /usage /trace /kb /export /key /model /config /circuit
/skill /memory /remember /task /eval /web /serve /logs /sandbox /device /executor /help /exit
```

`/key sk-xxx` — paste a key, auto-assigns to the main model. `/config` — guided panel, no YAML hand-editing needed.

## Project layout

```
handcraft-agent/
├── config/models.yaml    # all configuration (models/roles/debate/router/kb)
├── config/golden.yaml    # golden-set eval cases
├── forge/
│   ├── agent.py          # ReAct loop + context mgmt + status bar + approval
│   ├── llm.py            # openai gateway + retry + fallback + streaming + breaker
│   ├── tools.py          # 31 tools + read-only tiers + KB tools
│   ├── orchestrator.py   # parallel / debate / supervisor
│   ├── router.py         # rule-first task routing (0ms for common intents)
│   ├── knowledge.py      # SQLite+FTS5 knowledge base
│   ├── memory.py         # cross-session user profile
│   ├── eval.py           # golden-set evaluation
│   ├── web.py            # zero-dependency web chat
│   ├── server.py         # #14 HTTP API service (sessions + auth + rate limit)
│   ├── hwproto.py        # #16 hardware protocol v1 (line-JSON + CRC + seq/ack/state)
│   ├── hwtransport.py    # #16 transports: serial (pyserial) / MQTT (paho) / memory
│   ├── hwcontrol.py      # #16 control plane: assets + policy + command state machine
│   ├── executor_hub.py   # #17 executor hub (registry + policy + command queue)
│   ├── executor.py       # #17 client executor (capabilities + path jail + long poll)
│   ├── cua.py            # #17 four-role Computer Use loop (planner/executor/evaluator/supervisor)
│   └── executor_agent.py # #17 entry point that runs on the controlled PC
│   ├── sandbox.py        # #4 command sandbox (Docker isolation / hardened local)
│   ├── logging_setup.py  # #7 structured JSONL logs (rotation, retention, redaction)
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

## Client executor (#17)

Drive other PCs. Each controlled machine runs a light executor that **only dials out** (HTTP long
poll - no inbound port, no firewall change):

```bash
# centre
forge --serve

# controlled PC (one per machine)
pip install "handcraft-agent[executor]"          # optional: adds screenshot + GUI input
python executor_agent.py --center http://<centre>:8080 --token <KEY> --id pc-01 --root D:/work

# centre REPL
/executor                       # hub overview + what is online
/executor run pc-01 "whoami"     # run a command over there
/executor cua pc-01 "open notepad and type hello"
```

Capabilities are declared by the client and filtered twice: at the hub (allow-list, staged release
`readonly`/`low_risk`/`approval`/`closed_loop`, timeout, size caps) and again on the client (allow-list,
**path jail**, size caps, and `shell` goes through that machine's own sandbox). Without the GUI extra
the client simply does not declare screenshot/input instead of pretending.

**Four-role Computer Use.** `cua_task` splits the loop so no single model is brain, hand and judge:
planner → executor → (dispatch → raw evidence) → evaluator → supervisor on repeated failure. The
evaluator only sees raw evidence, never the executor's own explanation; a completion gate catches the
common case where the executor never says "done". See `docs/executor.md`.

## Hardware (#16)

Devices are tools. `tools.py` exposes `device_status` / `device_power` / `device_level` /
`device_reset`; what sits behind them depends on config:

| `device.enabled` / `transport` | What the tools talk to |
|---|---|
| `false` or `sim` (default) | Phase 0 in-process simulator (`fake_device.py`) |
| `true` + `serial` | real serial / UART (`COM5`, `/dev/ttyUSB0`, or `socket://host:port`) |
| `true` + `mqtt` | MQTT: down `{prefix}/cmd/{device}`, up `{prefix}/up/{device}` |

**Control plane.** Talking to hardware is easy; not mis-controlling it is the hard part. Before
a command reaches the wire it passes an asset registry (unknown device aliases are refused, not
guessed), a policy engine (staged release `readonly` / `low_risk` / `approval` / `closed_loop`,
level/temperature/runtime limits, write cooldown, remote endpoints unwritable by default) and a
command state machine:

```
Created ──sent──> Sent ──ack.ok──> Accepted ──state──> Applied
                        └─ ack rejected ─> Rejected
                        └─ timeout, retries exhausted ─> Timeout   (then rollback)
```

An `ack` only means the device took the request; **only a state snapshot counts as applied**. Every
command is audited, and the agent side enforces its own over-temperature / runtime guards instead of
trusting the device alone.

**No hardware needed to verify it.** The device-side simulator speaks the same protocol over TCP:

```bash
# terminal 1 — simulated device (real protocol, real CRC)
python device_sim.py --transport socket --port 9009 --test-hooks

# terminal 2
forge                       # then:
/device mode serial socket://127.0.0.1:9009
/device connect
/device                     # device identity, policy, recent audit
/device audit 10
```

That path exercises real pyserial, real framing, real policy, real state machine — only the physical
component is simulated. `docs/hardware.md` has the protocol spec plus a reference ESP32 firmware
(`hardware/esp32_beauty_device.ino`) for the real thing.

## Voice (#11)

Talk to forge. Cascade pipeline (STT → agent → TTS) with **streaming and barge-in**:

```bash
pip install "handcraft-agent[voice]"      # sounddevice + numpy + edge-tts + openai-whisper
forge --voice                              # talk, and interrupt it mid-answer
```

- **Streaming (Phase 2)** — while you speak the captured audio is re-transcribed every second and
  the draft appears before you finish; the answer is split into sentences as it generates, and each
  finished sentence is synthesised and played immediately, so you hear sentence one while sentence
  two is still being written
- **Barge-in (Phase 3)** — the microphone keeps listening while forge thinks and talks. Start
  speaking and it stops mid-sentence, cancels the running generation and takes the half-sentence you
  already said as the next turn — no repeating yourself
- **Testable by design** — audio source and playback are injectable (real microphone / an audio file
  standing in for one / scripted synthetic audio; real ffplay or a recorder that makes no sound), and
  VAD + sentence splitting are pure state machines. So segmentation, interruption timing and
  streaming order are covered in CI: no microphone, no model, no sound
- **No microphone needed to try it**: `forge --voice --audio-source file:question.wav --voice-sink null`
  runs the whole chain (real Whisper, real model, real edge-tts) silently
- **Acoustic echo cancellation** — `forge --voice --aec` keeps the microphone live while the answer
  plays and subtracts the agent's own voice using the audio being played as the reference, so you can
  interrupt hands-free: no muting, no push-to-talk key. Pure-numpy NLMS by default (no C extension);
  it switches to in-process playback, since `ffplay` cannot hand over the samples it is playing.
- **Speakers work too** — you do not have to wear headphones:
  `forge --voice --half-duplex` mutes the microphone while the answer plays (it still hears you
  while it is *thinking*, where there is no echo to confuse it), and `forge --voice --ptt` only
  captures while you hold space — press to stop it mid-sentence, release to send that utterance.
  Full duplex with headphones still gives the smoothest barge-in.

## Who it is talking to (#18)

The voice loop can look at the camera, recognise the people in front of it, and put that into the
conversation — so it addresses you by name instead of "the person in the room". Opt-in, off by default.

```bash
pip install "handcraft-agent[vision]"      # opencv-python<5 + numpy
forge --voice --identify                   # recognises who is present, ~1 Hz
```

What it does, end to end: the camera sees a face → the face is matched against a local identity store
→ one line of scene text is injected into the system prompt before each answer → the model refers to
whoever is there. Measured on the real machine, one frame, same instant:

| Check | Result |
|-------|--------|
| Same person, enrolment vs recognition recipe | **0.894 / 0.894** — identical vectors (cosine 1.000) |
| Two people enrolled, cross-person similarity | median 0.149, **highest 0.267** |
| Same person, lowest pair | **0.482** |
| Threshold | **0.36** — sits between them, with room on both sides |
| Wrong-person matches in testing | **0** (921 frames of one person against another's enrolment) |

- **It says when it cannot tell** — with two people in frame the mouth-motion signal cannot separate
  who is speaking (the distributions overlap), so it says so instead of guessing. "I don't guess —
  guessing wrong is worse than not knowing" is the intended behaviour, not a limitation to patch over
- **Enrolling someone — say it, don't configure it.** In `--identify` (or a plain `forge` session; the
  camera just has to be reachable) stand in front of the camera and say *"enrol me as 满仓"*. It calls
  `face_enroll`, which is a **write operation**, so it asks for approval first — answer `y`. Hold still
  while it captures **12 frames across ~5 seconds** (varied poses matter: on held-out poses 12 frames
  recognise at 94%, 3 frames at 68%; more frames never caused a wrong match, it only ever answers
  "not sure"). Then check with *"who am I"*, or *"who is in the face store"* (`face_people` lists names
  and vector counts — no images, because none are ever stored). **Get the person's agreement first**:
  it is their face, and the tool is built to say so.
- **Un-enrolling someone**: say *"remove 翠花 from the face store"* → `face_forget` → approve `[y/N]` →
  every vector for that name is gone, and it cannot be recovered. Anything written with `face_note`
  goes with it. If you would rather wipe everyone at once, deleting `~/.forge/faces.db` does exactly
  that — the file holds nothing but vectors.
- **Privacy**: only 128-dimension feature vectors are stored, never images; the store lives outside
  the repository at `~/.forge/faces.db`; the camera is opened only while `--identify` runs and the
  probe closes it between reads

**Small things worth knowing**

- `--identify` is **off unless you ask for it** — no camera, no recognition, no prompts changed
- Recognition works while the answer is being spoken too, but on **speakers** the loop is
  half-duplex: it stops listening while it talks. Finish your sentence and let it finish its answer,
  or use `--ptt` if you want to cut in
- Speech recognition defaults to Whisper `base`, which is thin for Chinese. `--stt-model small`
  is noticeably better (one-off ~460 MB download)
- It knows the difference between "nobody here", "someone here I don't know" and "this is 满仓" —
  and it will name you only in the last case

