# voice-web — bespoke handoff-aware voice front end

The browser UI for the LLM Router voice assistant (Slice 6). It connects to the
voice proxy (`llm_router.voice.server`, deployed as `llm-voice-proxy` on hypatia,
fronted at `https://voice.bcc.sh`) and renders the **handoff** as first-class UX
that the stock Moshi client can't: it shows **who is speaking** (You / Moshi /
🧠 Expert), the handoff **phase**, and the auto-router's **resolved model**.

## What's reused vs. ours

This is a clean-room reassembly of the upstream Kyutai Moshi web client
(`~/code/moshi/client`, Apache-2.0), keeping the hard, solved audio plumbing and
rewriting the UI:

- **Reused** (essentially verbatim): `src/audio-processor.ts` (the playback
  AudioWorklet jitter buffer), the opus encode/decode workers (`opus-recorder`,
  copied into `public/assets` by `scripts/copy-workers.mjs`), and the binary
  framing (`src/protocol.ts`).
- **Ours**: `src/audio-engine.ts` (framework-free mic+playback wiring),
  `src/use-voice.ts` (the handoff state machine over the MT=4 protocol), and the
  React UI in `src/App.tsx` + `src/components/`.

## Protocol (additive, see ../../src/llm_router/voice/server.py)

- **Server → browser** MT=4 JSON: `{"kind":"voice","phase":…,"source":…,"model":…,"query":…,"text":…}`.
  Drives phase/source/model and a `source:"user"` event carries the user's turn.
- **Browser → server** MT=4 JSON: `{"cmd":"ask","query"?}` (typed question, or
  empty = arm the next spoken turn) and `{"cmd":"stop"}` (interrupt). The proxy
  intercepts these; they're never forwarded to Moshi.
- `GET /api/voice/config` → `{triggers, llm, filler, handoff_enabled}`.

## Develop

```sh
pnpm install
# point /api (incl. ws) at a running proxy; localhost is a secure context so the
# mic works in dev. e.g. on hypatia, or set VITE_BACKEND to a reachable proxy.
VITE_BACKEND=http://127.0.0.1:8999 pnpm dev
```

## Build + deploy

```sh
pnpm install && pnpm build        # -> dist/  (prebuild copies the opus workers)
# ship dist to hypatia and point the proxy's --web-dir at it:
rsync -a --delete dist/ hypatia:/home/erewhon/voice/webclient/dist/
ssh hypatia 'sudo systemctl restart llm-voice-proxy.service'
```

Then open <https://voice.bcc.sh>. The proxy serves `dist/index.html` at `/` and
`dist/assets` at `/assets`, same-origin with `/api/chat`.
