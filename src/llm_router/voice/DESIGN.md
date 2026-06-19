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
- [ ] Slice 5 — STT path for non-Moshi inputs; barge-in; frontend.
