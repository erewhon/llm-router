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
- [ ] **Slice 6 — bespoke front end.** Make the handoff a first-class UX instead of a reskinned Moshi client. *Why:* the stock client shows one undifferentiated text stream and can't tell you whether Moshi or the expert is speaking (we inject expert text as if it were Moshi's), has no handoff status, no controls, no source/citation display.
  - **Approach — fork, don't rebuild.** Fork `~/code/moshi/client` (Vite/React, `src/{decoder,protocol,pages,components}`): **reuse `src/decoder` + `src/protocol`** (mic capture, Opus encode/decode workers, ws framing — the hard, solved parts); rewrite `pages`/`components` for the handoff UI. Lives in the llm-router repo (pnpm/Vite project, e.g. `clients/voice-web/`).
  - **Protocol extension — additive, backward-compatible.** Server (`server.py`) emits **MT=4 MetaData (JSON)** status/source events; the stock client *ignores* unknown MT=4 so it keeps working, the bespoke FE renders them. Event shape e.g. `{"phase":"listening|transcribing|routing|thinking|speaking|moshi","source":"moshi|expert","model":"<auto-router pick>","query":"…"}`. This also lets us stop the "expert text masquerading as Moshi MT=2" hack — send expert text on a labeled channel.
  - **UI features:** source attribution (You / Moshi / 🧠 Expert, distinct styling); handoff status/phase indicator; controls (push-to-talk, a manual "ask the expert" button vs. only the voice phrase, explicit stop/interrupt = deliberate barge-in, mute); routing transparency (which model, latency, sources); attributed transcript.
  - **Steps:** (1) server emits MT=4 phase/source events at each transition (turn → trigger → routing → speaking → resume); (2) fork client → `clients/voice-web/`, reuse decoder/protocol, build the UI; (3) wire MT=4 → UI state; (4) `pnpm build` → point `llm-voice-proxy --web-dir` at the new dist; (5) deploy (same `voice.bcc.sh`, just a different dist).
  - **Also fold in here:** smoother audio buffering (the pump currently inserts brief silences between expert sentences while Orpheus synthesizes the next — pre-synthesize/buffer ahead); a visible "thinking" affordance during the TTFT gap.
  - Effort: medium frontend project (days). Backend orchestration is done; this is UI + a small additive protocol change.

  ### Slice 6 progress
  - [x] **Step 1 — server MT=4 protocol (additive, backward-compatible).** `server.py`:
    - **Server → browser** voice status events on MT=4 (JSON): `{"kind":"voice","phase":"trigger|thinking|speaking|moshi|cancelled","source":"expert|moshi","model":<resolved>,"query":<str>}`. Lifecycle: `trigger` (query captured) → `thinking` (filler playing, LLM TTFT, `source:expert`) → `speaking` (first audio out, carries the auto-router's **resolved** model, e.g. `Qwen/Qwen3.6-35B-A3B-FP8`) → `moshi` (resumed) / `cancelled` (barge-in). The stock Moshi client decodes MT=4 as metadata and `safeParse`-ignores it (no crash).
    - **Browser → server** control on MT=4: `{"cmd":"ask","query"?:<str>}` (direct query, or no query = *arm* the next spoken turn) and `{"cmd":"stop"}` (explicit interrupt). The bridge intercepts these and does **not** forward them to Moshi.
    - **`GET /api/voice/config`** → `{triggers, llm, filler, handoff_enabled}` for the FE to render the wake-phrase hint + manual button without a ws round-trip.
    - **Resolved-model transparency:** `handoff_tts.stream_deltas(..., on_model=cb)` fires once with the response `model` field; the server forwards it in the `speaking` event.
    - **Text attribution by phase (not a separate channel):** expert text still goes over MT=2 (Moshi muted during handoff), and the FE attributes MT=2 to the current `source` from the phase events — avoids a risky double text channel while still letting the FE label who's speaking.
    - **Verified headless** (`ask` command → expert path): event order `trigger→thinking→speaking(model=Qwen3.6-35B)→moshi`, 625 expert audio frames, full Rayleigh-scattering answer. Deployed live to hypatia `llm-voice-proxy`.
  - [x] **Step 2 — bespoke FE scaffolded** at `clients/voice-web/` (pnpm/Vite/React/TS + Tailwind). Clean-room reassembly of the upstream Kyutai client: **reused** the playback worklet (`audio-processor.ts`, verbatim), the opus encode/decode workers (`opus-recorder`, copied into `public/assets` by `scripts/copy-workers.mjs` — same pattern the upstream client ships the decoder), and the binary framing (`protocol.ts`). **Ours:** `audio-engine.ts` (framework-free mic+playback wiring), `use-voice.ts` (handoff state machine over MT=4), React UI. See `clients/voice-web/README.md`.
  - [x] **Step 3 — MT=4 → UI wired.** `use-voice.ts` maps phase events → phase/speaker/model; attributes MT=2 text to the current source (You/Moshi/🧠 Expert); renders a `source:"user"` turn as a "You" bubble. UI: `PhaseBar` (listening/thinking/speaking + model), attributed `Transcript`, `Controls` (Start/End, 🧠 Ask the expert [type a query or empty=arm next turn], Stop=interrupt, mic mute + level meter).
  - [x] **Step 4 — `pnpm build`** green (tsc clean; ~163 kB JS gzip 53 kB). Worklet builds to a plain `registerProcessor` script (verified, not worker-wrapped); all `/assets/*` worker refs resolve.
  - [x] **Step 5 — deployed.** dist rsynced to `hypatia:/home/erewhon/voice/webclient/dist/` (replacing the stock client); `--web-dir` already points there; live at **https://voice.bcc.sh**. Index + all assets serve 200 (wasm as `application/wasm`). Backend (`server.py` with MT=4 + `on_turn`) redeployed; headless lifecycle re-verified PASS.
  - [x] **Browser-validated 2026-06-19** (headless Chromium via Playwright, synthetic-mic + the real deployed site). Full flow works: Connect → Moshi full-duplex conversation (attributed "Moshi" bubbles) → manual "Ask the expert" → filler + streamed expert answer ("🧠 Expert" bubble) → resume; source attribution, phase indicator, mute, and **barge-in** all confirmed.
  - **Bugs found + fixed during validation:**
    1. **Connect → immediate disconnect** (the reported bug). Root cause: `connect()` started the opus recorder *before* opening the ws, so the opening Ogg pages (OpusHead **BOS** + OpusTags) were produced while `ws` was null and dropped — Moshi's Ogg reader then saw a mid-stream start and killed the connection with `OggReadError::InvalidData` ("Constraint violated" in the moshi-backend log). Fix: defer `recorder.start()` to `ws.onopen` (`AudioEngine.startCapture()`) so the BOS page is the first frame Moshi sees. Verified: first sent frame is now `header_type=0x02` (BOS) and Moshi streams audio+text back.
    2. **Stale bundle after deploy.** `index.html` was browser-cached, so redeploys silently served the old JS. Fix: the proxy serves `index.html` with `Cache-Control: no-cache` (hashed `/assets` stay immutable). *(One-time: an already-cached browser needs a single hard refresh to pick up the header.)*
    3. **Noise "You …" bubbles.** Whisper transcribes silence/noise as "." / "…"; the FE now skips user turns with no word chars.
  - **Text/audio alignment fixed (2026-06-19):** per-sentence the server now synthesizes *before* emitting the caption, so a sentence's text + audio arrive paired (~10–60 ms apart) instead of the text racing ~a full sentence ahead. Because `pump.push()` is non-blocking, the next sentence synthesizes while the current one plays → inter-sentence gaps drop to ~1 frame. Verified in-browser via the ws timeline.
  - **Moshi-listening-during-handoff fixed (2026-06-19):** the proxy was still forwarding the user's mic to Moshi during the expert answer, so Moshi accumulated the question (+ TTS echo) while paused and blurted a reply the instant it resumed. `Bridge._browser_to_moshi` now drops upstream mic frames while `suppress` is on (the tap still gets them for turn-detection/barge-in). **KEY:** moshi-backend *ignores* Control frames (`MsgType::Control => {}` in stream_both.rs) — so PAUSE/RESTART are no-ops; muting/withholding audio is the *only* lever over Moshi.
  - **Spoken-question-leaks-to-Moshi fixed (2026-06-19):** with the "Ask the expert" button (arm mode), the question is spoken *before* the handoff starts (handoff begins only after the turn is transcribed), so Moshi heard it and answered after the expert. Fix: clicking arm now mutes Moshi immediately (`suppress["on"]=True` on `cmd:ask` with no query) so it never hears the spoken question, with a `ARM_TIMEOUT_S=12s` auto-disarm (and manual Stop disarms) if no question follows. Verified in-browser: arming produces a clean 12 s gap in Moshi audio, then resume. *(The voice-phrase path — "ask the expert, …" in one breath — still lets Moshi hear it; not cleanly fixable via the proxy since we can't un-hear a completed utterance. The button is the clean path.)*
  - **Per-sentence playback stall (remote) — adaptive jitter buffer (2026-06-19):** ws-timeline showed the server delivers smoothly (realtime, ~82 ms inter-sentence gaps), so stalls were client-side underruns: the worklet's pre-roll buffer is tuned tiny (~80 ms) for Moshi's tight local loop and starves on a jittery/remote link. The worklet now accepts a `setBuffer` message; the FE sets a larger pre-roll (`EXPERT_BUFFER_MS=500`) on entering the expert phase (one-way TTS can afford the latency) and reverts to `MOSHI_BUFFER_MS=80` on resume. *(User confirmed "a lot smoother" at 300 ms; bumped to 500.)*
  - **Whisper noise-hallucination filter (2026-06-19):** Whisper emits stock phrases on silence/noise ("Thank you", "you", subtitle credits like "Субтитры…/DimaTorzok", "♪", "."), which were showing as junk "You" turns. `_is_noise_transcript()` drops them server-side before both `on_turn` (transcript) and trigger detection (exact-phrase set + substring markers + a "<2 alnum chars" guard); real utterances incl. short ones like "yes" pass.
  - **Latency breakdown (measured 2026-06-19, "what causes rainbows"):** the bottleneck is **Orpheus (TTS), not the expert LLM**. LLM (Qwen3.6-35B via auto-router): TTFT 0.2 s, first sentence 0.5 s, full answer ~1–2.5 s. Orpheus: ~1.0–1.17× realtime, with ~0.7 s fixed per-request overhead → first-chunk synth dominates the start. Live pipeline: filler at 0 s, **expert answer voice at ~4.2 s** (LLM ~0.5 s + Orpheus first-chunk + 500 ms buffer; filler covers most of it), then realtime playback.
  - **Conciseness system prompt (2026-06-19):** strengthened `PipelineConfig.system_prompt` to "at most two short sentences … only longer if asked." Cut "what causes rainbows" from **5 sentences / 34 s → 2 sentences / 11 s** of speech (and LLM gen 2.6 s → 0.9 s). Biggest single UX lever since total time is Orpheus-bound.
  - **Polish still to fold in:** the ~2 s gap between the (~2 s) filler ending and the answer starting (~4.2 s) — a longer/looping filler or pre-synthesized first clip would cover it; a faster TTS (e.g. Kokoro) would cut TTFA + total since Orpheus is the floor; finer Moshi turn segmentation in the transcript.
