# Changelog

All notable changes to forge are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Voice mode Phase 2/3 (#11)** — streaming transcription, sentence-level TTS, and
  barge-in (speak while the agent is generating to interrupt it).
- **Using voice without headphones** — speakers and a microphone self-excite (the TTS comes back
  in through the mic and looks like a new instruction), so there are now two modes for it:
  `--half-duplex` mutes the microphone while the answer plays and reopens it after the speaker
  tail dies (you can still interrupt while it is thinking), and `--ptt` captures only while you
  hold space — pressing stops the playback, releasing submits that utterance.

### Fixed

- **`config/models.yaml` was ignored for anyone who installed the package.** Loader paths were
  built from the package directory, which after `pip install` is `site-packages` — so a config in
  the current directory was never read, and a placeholder one was auto-generated inside
  `site-packages` instead, producing "missing API key" warnings. Resolution is now
  `$FORGE_CONFIG` > `./config/models.yaml` > packaged config > `~/.forge/config/models.yaml`, and
  `/key` writes to the same file the loader reads.
- `StreamingSpeaker.stop()` queued its sentinel even when the worker thread had not started, so an
  early stop (push-to-talk pressed before the first turn) left it in the queue and the next turn's
  worker exited immediately — the answer played silently. — streaming transcription, sentence-level TTS, and
  barge-in (speak while the agent is generating to interrupt it).
- `docs/voice.md` — voice mode documentation.

## [0.4.1] - 2026-09-29

Packaging fix, no functional change. Found while verifying v0.4.0's published artifact:
installing from the tag put only `main.py` into site-packages, so a `pip install` could not
run the two entry points the docs told you to run — you had to clone the repo.

### Fixed

- `executor_agent.py` (#17 client executor) and `device_sim.py` (#16 device-side simulator)
  are now installed with the package.
- Two console scripts join `forge`:
  - `forge-executor --center http://<centre>:8080 --token <KEY> --id pc-01 --root D:/work`
  - `forge-device-sim --transport socket --port 9130`

### Verified

- v0.4.0 artifact: `/healthz` and `/api/status` report 0.4.0, `/docs` is up, all six
  `/api/executor/*` routes present.
- v0.4.1 artifact: version 0.4.1 reported in both places, `forge-executor` and
  `forge-device-sim` on PATH, `forge executor --list-caps` reporting the machine's real
  capabilities.

## [0.4.0] - 2026-09-29

**All 18 modules are in.** This release closes the last two: the hardware control plane
(#16) and the client executor (#17) — an agent that drives other PCs. Offline suite: 332
cases; CI covers 19 test files across Python 3.9 / 3.11 / 3.13.

### Added

- **Hardware control plane (#16, Phase 1)** — one protocol, two transports:
  - one message = one line of JSON with CRC and `seq`/`ack`/`state`, protocol v1; the same
    frame goes over serial and MQTT, so swapping transport does not touch the protocol
  - `pyserial`'s URL mechanism means `COM5`, a `socket://` bridge and `loop://` all run the
    same code path (the tests exercise the real one)
  - staged policy (stage, max level, max temperature, max runtime, cooldown), a command
    state machine with timeout, retry, rollback and an audit trail
  - `device_sim.py` speaks the real protocol; `hardware/esp32_beauty_device.ino` is a
    reference firmware skeleton (**not compiled or flashed here** — see `docs/hardware.md`)
  - try it: `/device` · `/device connect` · `/device mode serial socket://127.0.0.1:9132` ·
    `/device audit 5`
- **Client executor (#17)** — the agent drives other machines:
  - each controlled PC runs a light executor that **only dials out** (HTTP long poll, no
    inbound port, no firewall change, no new dependencies)
  - capabilities are declared by the client and filtered twice: at the hub (allow-list,
    staged release `readonly`/`low_risk`/`approval`/`closed_loop`, timeout, size caps) and
    again on the client (allow-list, **path jail**, size caps; `shell` runs through that
    machine's own sandbox)
  - GUI sits behind an injectable driver. Without the optional `[executor]` extra the client
    does not declare screenshot/input at all — commands fail with `E_NO_GUI` instead of
    pretending to have worked
  - **four-role Computer Use**: planner → executor → (dispatch → raw evidence) → evaluator →
    supervisor. Four separate model calls with separate prompts, bindable to different
    models per role; the evaluator only ever sees raw evidence, never the executor's own
    explanation
- **20 tools, up from 14** — the six new ones are `executor_list`, `executor_run`,
  `executor_file`, `executor_screen`, `executor_input`, `cua_task`.

### Fixed

- **Dispatched REPL commands were also sent to the model.** `/web`, `/serve`, `/logs` and
  `/device` were missing `continue`, so the command ran *and* the same line went to the
  router as a task — a silent double execution that burned tokens. `test_cli.py` now asserts
  at the source level that every `_*_command()` call is followed by `continue`.
- **Upgraded installs could not switch modes.** A config generated before a section existed
  made `/device mode …` fail with "no device section". Config writers now append the missing
  section (device and sandbox) instead of giving up.
- **The four-role loop reported finished tasks as failures.** First real-machine run: 70s,
  six steps, ❌ — on a task that had actually been completed. The executor never emitted
  `done`. Added a completion gate (when the plan runs out, the evaluator judges the whole
  task from all the evidence); the executor is now told where it may write (the client's path
  jail travels in the device brief) and to return `done` as soon as the goal is met. Same task
  afterwards: **13.2s, 3 steps, ok**.
- `pytest .` no longer collects the release copies — `build/` and `release/` are excluded via
  `norecursedirs`.
- `build/` was accidentally committed once; removed and gitignored.

## [0.3.0] - 2026-09-28

**M5 is complete.** Closes the last two modules — the tool sandbox (#4) and full logging
(#7). Offline suite: 247 cases; CI now covers every module.

### Added

- **Command sandbox (#4)** — `run_command` no longer executes straight on your machine:
  - `sandbox.mode: auto` (default) runs inside Docker when available and otherwise falls back
    to *hardened local execution*; `docker` refuses to run at all without a daemon (use in
    production); `local` / `off` for explicit control
  - container runs are locked down: `--rm --network=none`, memory/CPU/PID caps, read-only
    rootfs with a writable `/tmp`, non-root user, working directory mounted **read-only**,
    force-removed on timeout
  - the local fallback still buys real protection: host-env allowlist, dangerous-command
    patterns (`rm -rf /`, `mkfs`, `dd` to raw devices, `shutdown`…), timeout kill, output
    clipping
  - try it: `/sandbox` · `/sandbox mode docker` (hot-reloaded) · `/sandbox test echo hi`
- **Structured logs (#7)** — one JSON line per event in `data/logs/forge-YYYYMMDD.jsonl`:
  daily files with size rotation, retention pruning, level filtering, per-run correlation ids
  (role / model / steps / tools / tokens / latency) and an HTTP request log.
  Inspect with `/logs`, `/logs tail 20`, `/logs errors`.
- **CI coverage** — `test_device.py` and `test_voice.py` existed but were never wired into
  CI; they are now, together with the new sandbox/logging suite.

### Fixed

- Sandbox: a truthy default on `Sandbox.__init__(mode="auto")` silently overrode
  `sandbox.mode` from config, so `docker` / `off` policies behaved as `auto`.

### Security

- **`run_command` leaked the whole host environment to child processes** — any command could
  read `DEEPSEEK_API_KEY`. The environment is now allowlisted.
- **Secret redaction in logs**: values under `key` / `token` / `secret` / `authorization`
  fields, and anything shaped like `sk-…` / `Bearer …` / `gho_…`, are written as `***`.

### Notes

- The Docker isolation path is asserted parameter-by-parameter and needs no container
  runtime in CI; verifying it against a real container requires Docker on the host.

## [0.2.0] - 2026-09-28

Three new capability lines since v0.1.0 — and the M5 milestone closed.

### Added

- **Server mode (#14 · completes M5)** — `forge --serve` turns forge into an HTTP API:
  - **multi-session persistence** (SQLite): one Agent context per session, resume after a
    client disconnect, restore history after a restart
  - **API-key auth**: `Authorization: Bearer …` or `X-API-Key`; keys come from config
    `server.api_keys` or `FORGE_API_KEY` (`env:VAR` indirection supported). With no key
    configured it degrades to **loopback-only**
  - **rate limiting** per caller (sliding window, `429` + `Retry-After`)
  - **write tools rejected by default** (a service has no interactive approval channel);
    read-only tools work fully
  - per-request access log, `/healthz` liveness probe, Swagger UI at `/docs`
  - fastapi/uvicorn ship as an **optional extra** — the core keeps its zero-heavy-dependency
    promise and still imports without them
- **Voice mode (Phase 1)** — pluggable STT/TTS + `forge --voice`: record → Whisper → Agent →
  Edge-TTS playback. The agent core is untouched; voice only swaps the I/O.
- **Device layer (hardware Phase 0)** — a simulated triple-function beauty device
  (`fake_device.py`) exposes power / level / temperature / current as agent tools, with writes
  behind the approval gate and built-in over-temperature protection.
- Every module now has offline tests; CI covers Python 3.9 / 3.11 / 3.13 with **zero skips**.

### Fixed

- **voice** — `EdgeTTS.synthesize` no longer nests `asyncio.run` when an event loop is already
  running (it dispatches to a worker thread instead).
- **eval** — the export test no longer writes a report file into the working tree.
- **tests** — the M1/M2 assertions now check mechanisms instead of one machine's config: a
  fallback role and a multi-model debate lineup are configuration, not code requirements.

## [0.1.0] - 2026-08-24

First public release.

### Added

- **ReAct core loop** with rule-based pre-routing (greetings/simple Q&A answer in 0ms, no
  model call)
- **Multi-agent orchestration**: parallel task decomposition, debate mode, auto-routing
- **Engineering backbone**: read/write tool tiers, context truncation + rolling summaries,
  error retry + degradation, per-step tracing, write-operation approvals
- **Structured output** (JSON-schema enforced)
- **Local knowledge base** (SQLite + FTS5) and golden-set evaluation
- **Web UI** and Markdown export
- **Config-driven**: switch models/roles via `config/models.yaml`, no code changes
- Zero hard dependencies beyond `openai` + `httpx`

[Unreleased]: https://github.com/musokean/forge/compare/v0.4.1...HEAD
[0.4.1]: https://github.com/musokean/forge/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/musokean/forge/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/musokean/forge/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/musokean/forge/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/musokean/forge/releases/tag/v0.1.0
