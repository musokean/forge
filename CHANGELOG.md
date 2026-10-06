# Changelog

All notable changes to forge are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Two face-pipeline bugs found by testing on the real camera.** The detector ran a global histogram
  equalisation over every frame; on a backlit face (the room here has a bright window behind the desk)
  that turned one detectable face into zero, where the raw frame detected it and CLAHE detected it - the
  face sat at 64 against 145 for the window behind it. Preprocessing is now CLAHE by default, selectable,
  and the measurements are in the code comment. Separately, matching ranked stored *samples* rather than
  *people*, so anyone enrolled with more than one sample always had their own second sample as the
  runner-up and could never clear the margin test. A live session scored 2/10 before the fix and 10/10
  after, with same-person similarity at 0.90-0.95 against a non-face floor of -0.07-0.29.

### Fixed

- **The mouth-motion signal does not separate speech from silence, and the default implied it did.**
  Measured on this machine, same person, 15 s of each: talking gives a median mouth-motion score of
  0.0199 and sitting quietly 0.0159, maxima 0.0224 and 0.0223. The distributions overlap, so no
  threshold separates them - raising it to 0.025 would drop the talking case as well. What dominates
  the frame difference is the detector box jittering one to three pixels per frame rather than the
  mouth moving. The default is now 0.03, above the measured noise band, so the feature claims a
  clearly moving mouth and not speech; the tool description and docs say exactly that, and a
  regression test pins the measured overlap so the stronger claim cannot creep back in. Deciding who
  is speaking needs the audio VAD to say when someone is speaking, with vision only answering who.

### Added

- **YuNet face detection (#18).** A CNN detector (OpenCV's own `FaceDetectorYN`, model ~227KB) is now
  preferred when its model is present, falling back to Haar, and the choice is a `face.detector`
  setting. Measured live here: on a backlit face Haar found the face in 1 of 12 frames and YuNet in
  12 of 12 at 0.93-0.94 confidence. It also returns five landmarks, so faces can be aligned before
  embedding. `make_detector()` reports the ladder honestly and `available()` says which one is in use.
  The default score threshold is 0.9: at 0.5 a frame containing no face at all still produced a
  0.50 shoulder detection, which is exactly what the higher threshold prevents.

  On alignment, measured rather than assumed: comparing alignCrop against a tight crop on ten live
  frames of one person gives 0.892 against 0.862 median similarity (+0.030), but a plain crop with a
  25% margin scores 0.894 - so alignment buys nothing over the default path and is not a precision
  fix. All three arms recognised 10/10, far above the 0.36 threshold. The real gain from YuNet is
  detection robustness in poor light, not recognition accuracy.

- **Presence: who is here and who is speaking (#18 Phase 3)** — with several people in front of the
  camera the agent now keeps a stable identity per person as they move, and answers who is talking.
  A single microphone carries no direction information and the Haar detector returns no landmarks, so
  this does not attempt source localisation or lip reading: it combines two signals that are actually
  available, the frame-to-frame motion of the mouth region of each face and how large that face is,
  and it says "not sure" when the leader is weak or only marginally ahead. Recognition is cached per
  track and re-checked every N frames, so names do not flicker when one frame happens to miss.

- **Face recognition (#18 Phase 2)** — the agent can now *recognise* people it has been introduced to, not
  just see that someone is there. `face_enroll` stores feature vectors under a name, `face_who` answers
  "who is this", `face_people` lists the roster and `face_forget` deletes someone completely. The identity
  store keeps **only vectors, never images** (`data/faces.db`, already gitignored), matching is nearest
  neighbour with both a cosine threshold and a best-versus-runner-up margin, and anything below either bar
  comes back as *unknown* rather than a guess. The embedding model (SFace, which ships in opencv itself —
  only the .onnx file is extra) is a swappable part: a stub embedder keeps the whole decision path testable
  in CI with no model and no camera.

- **Camera and face detection (#18 Phase 1)** — two tools, `look` and `look_image`: the agent can take
  one frame from a webcam and learn whether anyone is in front of it, and where the faces are. `look`
  returns only numbers and never writes to disk; saving an annotated image is a separate tool marked as
  a write operation. A fake frame source and a stub detector keep the pipeline testable with no camera
  at all (including in CI), and neither tool is registered when opencv/numpy are missing — the same
  "do not pretend" rule the client executor follows. Ships as the `vision` extra, which caps opencv
  below 5 because OpenCV 5 removed the Haar cascades.

## [0.5.0] - 2026-10-03

Voice mode grew up, and the echo cancellation behind it was rebuilt against a real microphone.

### Added

- **Voice mode Phase 2/3 (#11)** — streaming transcription, sentence-level TTS, and barge-in (speak
  while the agent is generating to interrupt it).
- **Voice settings live in the config file**, so plain `forge --voice` can be hands-free.
  `config/models.yaml` accepts a `voice:` section (`aec`, `aec_lead_ms`, `stt_model`, `barge_ms`,
  `sink`, `half_duplex`, `ptt`, `rounds`), the command line still wins, and unknown keys are ignored.
  Set it once and the flags stop being something you retype every session.
- **Using voice without headphones** — speakers and a microphone self-excite, so there are two modes
  for it: `--half-duplex` mutes the microphone while the answer plays and reopens it once the speaker
  tail has died (you can still interrupt while it is thinking), and `--ptt` captures only while you
  hold space — pressing it stops playback, releasing submits that utterance.
- **Acoustic echo cancellation (`forge --voice --aec`)** — hands-free barge-in: the microphone stays
  live while the answer plays and the agent's own voice is subtracted, using the audio it is playing
  as the reference through a pure-numpy block NLMS filter (no C extension; `pyaec`/`speexdsp` are used
  if installed). In-process playback is the default for this mode, because `ffplay` cannot hand over
  the samples it is playing.
- **Residual echo suppression**, the stage after the filter. A laptop's speaker-to-microphone path is
  not linear — driver enhancement, clipping, chassis vibration — and a linear filter cannot model it:
  on the test machine the recording correlates with the played audio at 0.045, so no filter length
  helps. The suppressor estimates how much of the microphone is echo from the two energies and ducks
  it unless someone is clearly louder, which keeps the agent's voice out of the transcription while a
  real interruption still gets through.
- **Online delay estimation.** The reference has to line up with the microphone, and the total device
  delay is a property of the hardware: 460-520ms here (output buffer, input buffer and acoustics)
  against the 26ms the driver reports, so a constant cannot work. It is measured from the energy
  envelopes, which survive a non-linear path even though the waveforms do not, using a long window
  and a correlation curve accumulated across blocks — a single window's peak jumps between 250 and
  786ms on a weak echo.
- `docs/voice.md` — voice mode documentation, including which modes let you interrupt and what an
  interruption does.

### Fixed

- **`config/models.yaml` was ignored for anyone who installed the package.** Loader paths were built
  from the package directory, which after `pip install` is `site-packages` — so a config in the
  current directory was never read, and a placeholder one was auto-generated inside `site-packages`
  instead, producing "missing API key" warnings. Resolution is now `$FORGE_CONFIG` >
  `./config/models.yaml` > packaged config > `~/.forge/config/models.yaml`, and `/key` writes to the
  same file the loader reads.
- **Only the first half of an interrupted sentence was transcribed.** Barge-in transcribed the audio
  the moment it crossed the threshold, so everything the user said afterwards — a few hundred
  milliseconds of speech in practice — was lost. It now stops playback, waits for the user to finish,
  and transcribes the stitched utterance.
- **The sentence being synthesized was played after an interruption.** `StreamingSpeaker.stop()` only
  set a flag that the next round cleared, so a sentence already in synthesis could still reach the
  speaker. Rounds now carry an epoch and a worker whose epoch has passed retires.
- **The echo canceller's reference was anchored to the wrong moment.** It was anchored when the
  listener started, which is 100-300ms before playback begins, and then advanced by sample count —
  so any late or dropped block shifted it permanently, and the filter could not converge at all on
  the real machine. The reference is now looked up from the capture time of each microphone block.
- **An echo canceller safety net that switched the feature off.** The bypass meant to stop a
  misbehaving filter from making the link louder fired on any frame whose residual was louder than
  its input, with no margin, so 54% of frames were bypassed during convergence. It now requires a
  smoothed violation, and the real bound is a cap on the filter's norm.
- `StreamingSpeaker.stop()` queued its sentinel even when the worker thread had not started, so an
  early stop (push-to-talk pressed before the first turn) left it in the queue and the next turn's
  worker exited immediately — the answer played silently.

### Verified

- 413 tests pass, 25 of them covering echo cancellation (synthetic path, weak echo, double-talk,
  bypass, delay estimation).
- On the test machine, with echo cancellation enabled, a playback leaves a residual of 0.0084
  against the voice-activity threshold of 0.012 and the listener emits no `speech_end`: it no longer
  transcribes its own speech. Delay estimates were 480, 486 and 493ms across a playback, a 13ms
  spread, with correlations of 0.45 to 0.51.
- The clean synthetic echo path converges to an ERLE of 55.7dB with nothing bypassed, which is what
  says the reference timing is right.

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

[Unreleased]: https://github.com/musokean/forge/compare/v0.5.0...HEAD
[Unreleased]: https://github.com/musokean/forge/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/musokean/forge/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/musokean/forge/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/musokean/forge/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/musokean/forge/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/musokean/forge/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/musokean/forge/releases/tag/v0.1.0
