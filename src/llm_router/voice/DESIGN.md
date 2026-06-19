# Voice Assistant Pipeline — design

Status: **prototyping (started 2026-06-18)**. Forge task: "Voice assistant pipeline — Moshi + TTS + LLM handoff" (P3, xl, novel).

The idea: a local, low-latency voice assistant where **Moshi** handles natural full-duplex
conversation and quick replies, and **hands off** to the LLM fleet (via the router's
auto-router) when a query needs real reasoning. Long-form answers are spoken with a
dedicated **TTS** layer.

## Verified building blocks (measured 2026-06-18)

| Layer | Component | Where | Status / perf |
|---|---|---|---|
| Conversation | **Moshi q8** (full-duplex S2S) | hypatia `moshi-backend.service` :8998 | live, ~10 GB, sub-200 ms, systemd unit |
| Reasoning | **Router auto-router** (`auto`/`auto-full`) | euclid+delphi :4010 | live — classifies complexity, routes to Qwen3.6 / Nemotron / Zen |
| TTS (long-form) | **Orpheus 3B** | euclid Arc/OpenVINO :5397 | works but **~0.5× realtime** on Arc → move to CUDA |
| STT (non-Moshi input) | **Whisper large-v3-turbo** | euclid :5404 (`stt`) | live |

### TTS decision
- **Fish S2-Pro**: best quality but **~22 GB + ~0.6–1 tok/s** (no compile) — too heavy/slow for co-residency. Reserve for a dedicated 24 GB box / offline.
- **Orpheus 3B**: ~4.5 GB, good quality. The euclid Arc deployment is ~0.5× realtime; on **CUDA it hits ~realtime** and **fits hypatia's ~25 GB headroom** alongside Qwen3.6 + Moshi. → **Plan: run Orpheus on CUDA (hypatia, llama.cpp) for the pipeline.** Prototype against the existing euclid Orpheus first (functional, just slow) to build the plumbing.
- **Kokoro**: lightweight CPU fallback (currently down on delphi).
- Short conversational turns use **Moshi's own voice**; the TTS layer is only for long-form handoff answers.

## Architecture

```
        ┌─────────── Moshi (hypatia :8998) ───────────┐
user ⇄  │  full-duplex conversation, quick replies     │
voice   │  detects "needs deeper reasoning" → handoff  │
        └───────────────────┬─────────────────────────┘
                            │ query text
                            ▼
              Handoff orchestrator  (this package)
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
   Router auto-router (:4010)     filler to Moshi
   classify → Qwen3.6 / Nemotron  ("let me look that up…")
   / Zen → stream tokens
                │ tokens
                ▼
   TTS (Orpheus on CUDA) — sentence-chunked streaming
                │ audio
                ▼
        played back to user; Moshi resumes control
```

## Decisions
- **Handoff routing reuses the router's auto-router** (it already does embedding-based
  complexity classification) — no new classifier.
- **Orchestrator is Python** (Moshi client, audio handling, websockets are Python-native).
  Lives in `src/llm_router/voice/`.
- **Single endpoint where possible**: the router fronts both chat *and* TTS models
  (`auto-full`, `orpheus`/`tts`), so the orchestrator can use one base URL + key.
- Prototype transport = PCM/WAV at boundaries; optimize to streaming / Moshi's Mimi codec later.

## Open questions (revisit as we build)
1. **Handoff trigger**: explicit wake-phrase ("look that up") vs. Moshi confidence vs.
   auto-router difficulty score on every turn? v1 = explicit / standalone.
2. **Frontend**: voice-only (CLI/daemon) vs. a small web UI? (Moshi already ships a web UI.)
3. **Barge-in**: let the user interrupt long-form TTS playback (Moshi supports interruption natively).

## Roadmap
- [x] **Slice 1 — handoff + TTS half** (`handoff_tts.py`): text → router LLM → Orpheus → audio, with per-stage timing. *Proves the reasoning+speech path independent of Moshi.* Measured: LLM ttft ~248 ms; Orpheus-on-Arc ~1.94× RTF.
- [x] **Slice 2 — sentence-chunked streaming TTS** (`streaming_tts.py`): producer streams the LLM + segments sentences; consumer synthesizes + concatenates concurrently. Aggressive first chunk (clause/word-cap) for fast first audio. **Measured TTFA: ~57 s (slice 1) → 20.9 s (sentence-chunked) → 8.6 s (aggressive first chunk).** Total wall-clock still gated by Orpheus-on-Arc (~1.9× RTF) → Slice 3.
- [x] **Slice 3 — Orpheus on CUDA** (hypatia). llama.cpp CUDA (GB10) serving Orpheus-3b **Q4_K_M** on :5406 (`orpheus-llama.service`) + orpheus-fastapi SNAC decode on :5397 (`orpheus-fastapi.service`); orchestrator default repointed to hypatia. **Measured: Q8 → 1.79× RTF (58 tok/s); Q4_K_M → 1.17× RTF (83 tok/s).** The GB10's LPDDR5X (~273 GB/s) makes a *dense* 3B memory-bandwidth-bound, so Q4 is the sweet spot (sub-1.0 would need Q3/IQ3 at a quality cost). End-to-end (streaming + CUDA Q4): **TTFA 8.6→6.1 s, total 61→35 s, RTF 1.92→1.17×.** TTFA floor (~6 s) = LLM-first-clause + orpheus per-request overhead; masked in the full pipeline by Moshi conversational filler.
  - Deploy gotchas (hypatia/aarch64): copied archimedes `/opt/llama-cpp-cuda` (same GB10); needs `libportaudio2`; old requirements pins (numpy 1.24/psutil 5.9.0) have no py3.12 aarch64 wheels — install current versions; only gcc-13 present (`CC=gcc-13`); app.py hardcodes uvicorn `reload=True` → must set `reload=False` for systemd (reload worker orphans + holds the port).
- [~] **Slice 4 — Moshi integration** (in progress). `moshi_client.py`: headless async websocket client for moshi-backend `/api/chat`. Protocol (see `~/code/moshi/rust/protocol.md`): each msg = 1 type byte + payload — **MT=1** Opus audio (24kHz mono, via `sphn`), **MT=2** UTF-8 text (Moshi's transcript), **MT=3** control (Start/EndTurn/Pause/Restart). The reference client (`moshi/client.py`) uses live mic/speaker; ours is file/stream-driven so it can run as a service and mediate the handoff.
  - **Handoff design:** orchestrator connects to Moshi as a client, monitors the **MT=2 text stream** for a trigger (explicit phrase v1, e.g. "look that up" / "ask the expert"; confidence/auto-router classification later). On trigger → **MT=3 Pause** Moshi, emit filler, run the `streaming_tts` orchestrator (auto-router LLM → CUDA Orpheus), play the answer audio, then **MT=3 Restart/EndTurn** to resume Moshi. The ~6 s TTS TTFA floor is hidden behind Moshi's filler.
  - **Topology:** the voice orchestrator (Moshi client + handoff) runs **on hypatia**, co-located with Moshi (:8998) + Orpheus (:5397) — all localhost; only the router (:4010) is remote (LAN). Deps: aiohttp, numpy, `sphn` (kyutai Opus lib).
  - **`handoff.py` — DONE + live-tested (turn-based v1).** A turn comes in as audio → Whisper STT (euclid :5399 `/v3/audio/transcriptions`) → `detect_handoff` (trigger phrase) → route: **expert** (`stream_pipeline` → auto-router LLM → CUDA Orpheus) or **Moshi** (`moshi_client.converse`). Verified end-to-end: *"Ask the expert. Why is the sky blue?"* → EXPERT (query "Why is the sky blue?", LLM→Orpheus, RTF 1.22×); *"Hey, how are you doing today?"* → MOSHI (conversational reply). Deferred to later: auto-router complexity classification (vs. explicit phrase), MT=3 pause/resume for *continuous* full-duplex (v1 is turn-based), and "filler-while-thinking".
- [~] **Slice 5 — live browser frontend (orchestrator-as-Moshi-proxy).** Layers 1–3 (`server.py`) **DONE + live-tested**: transparent Moshi proxy; energy-VAD turn detection + Whisper trigger with **arm-on-phrase** (a trigger phrase with no trailing question arms; the next turn becomes the query — handles split utterances); handoff action mutes Moshi → streams LLM→Orpheus answer to the browser as realtime Opus → resumes. Validated: "ask the expert" / "why is the sky blue?" → 27 s expert answer streamed through the proxy. (Impl note: turn-based VAD, not the originally-sketched mid-stream detection; `--router-url` needed since the orchestrator is off the router host.)
  **Frontend + deploy — DONE.** The proxy serves Moshi's web-client dist (`--web-dir`) on the same origin (it connects to our `/api/chat` automatically) and forwards the client's gen-param query string upstream. Deployed as **`llm-voice-proxy.service`** on hypatia (:8999), behind the hub Caddy at **`https://voice.bcc.sh`** (both hubs, HA; `voice` added to the CoreDNS split-horizon regex; valid Let's Encrypt cert → browser mic works). **Open https://voice.bcc.sh in a browser to use it.**
  **Polish — DONE:** (a) **filler-while-thinking** — a pre-rendered "Let me look that up." clip (synth'd at startup, cached) plays during the LLM-TTFT gap; (b) **text fix** — during a handoff the proxy now mutes Moshi's *text* too (MT=2), not just audio, and **injects the expert's text** (filler + answer) as MT=2 so the browser shows what's actually being spoken; (c) **barge-in** — sustained user voice during the answer (energy VAD, `bargein_thresh`/`bargein_s`) sets a cancel event → expert stream stops, Moshi resumes (browser echo-cancellation assumed; needs real-browser validation). Original step sketch:
  1. `server.py` — ws endpoint; per connection, open a `MoshiClient` to localhost:8998 and relay MT frames both ways.
  2. **Continuous trigger** — chunked/streaming Whisper (or VAD-segmented) on inbound user audio in parallel with Moshi; detect trigger/complexity mid-stream (not turn-end).
  3. **Handoff** — MT=3 Pause Moshi → filler clip → `stream_pipeline` (LLM→Orpheus) streamed to the browser as MT=1 frames → MT=3 Restart.
  4. **Deploy** — orchestrator systemd unit on hypatia; serve the web client.
  - **Constraint:** browser mic (`getUserMedia`) needs a **secure context** → must serve over real HTTPS (hub Caddy front door, e.g. `voice.bcc.sh`), not the self-signed :8998 cert. This basically mandates the Caddy front door.
  - Open: streaming-STT choice (euclid Whisper /v3 is non-streaming → chunk it, or add a streaming STT); keep explicit-phrase trigger first, auto-router classification later; barge-in.
