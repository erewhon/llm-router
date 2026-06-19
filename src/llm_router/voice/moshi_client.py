"""Slice 4: headless Moshi websocket client (foundation for the handoff).

Moshi (kyutai full-duplex speech-to-speech) speaks a binary websocket protocol
at `/api/chat` — each message is one type byte + payload (see
`~/code/moshi/rust/protocol.md`):

    MT=0 Handshake    MT=1 Audio (Opus, 24kHz mono ogg)    MT=2 Text (UTF-8)
    MT=3 Control (0=Start 1=EndTurn 2=Pause 3=Restart)
    MT=4 MetaData (JSON)    MT=5 Error    MT=6 Ping

The official client (`moshi/client.py`) drives a live mic/speaker via
`sounddevice`. This one is file/stream-driven so the orchestrator can run as a
headless service: stream a PCM utterance in as the "user", collect Moshi's
response audio + its MT=2 text transcript. Monitoring that text stream is how
the handoff (Slice 4) detects when to route a query to the LLM fleet.

Deps: aiohttp, numpy, sphn (kyutai's Opus lib). Run on hypatia (Moshi local).

    python -m llm_router.voice.moshi_client --in question.wav --out reply.wav
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import wave
from collections.abc import Callable

import aiohttp
import numpy as np
import sphn

# message type bytes
MT_HANDSHAKE = 0
MT_AUDIO = 1
MT_TEXT = 2
MT_CONTROL = 3
MT_METADATA = 4
MT_ERROR = 5
MT_PING = 6
# control bytes
CTL_START = 0
CTL_END_TURN = 1
CTL_PAUSE = 2
CTL_RESTART = 3

SAMPLE_RATE = 24000
FRAME_SIZE = 1920  # 80 ms at 24 kHz — Moshi's frame
FRAME_SECONDS = FRAME_SIZE / SAMPLE_RATE


def load_wav_mono24k(path: str) -> np.ndarray:
    """Load a WAV as float32 mono [-1, 1] at 24 kHz (assumes already 24 kHz)."""
    with contextlib.closing(wave.open(path, "rb")) as w:
        n, ch, sr = w.getnframes(), w.getnchannels(), w.getframerate()
        raw = w.readframes(n)
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    if sr != SAMPLE_RATE:  # cheap linear resample; fine for speech input
        idx = np.linspace(0, len(pcm) - 1, int(len(pcm) * SAMPLE_RATE / sr))
        pcm = np.interp(idx, np.arange(len(pcm)), pcm).astype(np.float32)
    return pcm


def save_wav_mono24k(path: str, pcm: np.ndarray) -> float:
    """Write float32 mono PCM to a 24 kHz WAV. Returns duration in seconds."""
    clipped = np.clip(pcm, -1.0, 1.0)
    data = (clipped * 32767.0).astype(np.int16)
    with contextlib.closing(wave.open(path, "wb")) as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(data.tobytes())
    return len(pcm) / SAMPLE_RATE


class MoshiClient:
    """Minimal headless client for moshi-backend's /api/chat websocket."""

    def __init__(self, url: str = "ws://127.0.0.1:8998/api/chat") -> None:
        self.url = url
        self._opus_writer = sphn.OpusStreamWriter(SAMPLE_RATE)
        self._opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)

    async def converse(
        self,
        pcm_in: np.ndarray,
        max_seconds: float = 30.0,
        tail_silence_s: float = 2.0,
        on_text: Callable[[str], None] | None = None,
    ) -> tuple[np.ndarray, str]:
        """Stream `pcm_in` to Moshi at realtime pace, collect reply audio + text.

        Returns (reply_pcm, transcript). Stops after `max_seconds`, or once
        Moshi has spoken and then gone quiet for `tail_silence_s`.
        """
        out_chunks: list[np.ndarray] = []
        text_parts: list[str] = []
        # SSL: ws:// for plain; the server uses self-signed TLS, so callers pass
        # wss:// + we disable verification for the homelab cert.
        connector = aiohttp.TCPConnector(ssl=False) if self.url.startswith("wss") else None
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.ws_connect(self.url, max_msg_size=0) as ws:
                sender = asyncio.create_task(self._send_pcm(ws, pcm_in))
                loop = asyncio.get_event_loop()
                start = loop.time()
                last_audio = None
                spoke = False
                # Poll with a short timeout so the time/silence bounds are checked
                # even when Moshi is idle (an `async for` would block on recv).
                while True:
                    now = loop.time()
                    if now - start > max_seconds:
                        break
                    if spoke and last_audio and (now - last_audio) > tail_silence_s:
                        break
                    try:
                        message = await asyncio.wait_for(ws.receive(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    if message.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSING,
                        aiohttp.WSMsgType.ERROR,
                    ):
                        break
                    if message.type != aiohttp.WSMsgType.BINARY:
                        continue
                    data = message.data
                    if not data:
                        continue
                    kind = data[0]
                    if kind == MT_AUDIO:
                        pcm = self._opus_reader.append_bytes(data[1:])
                        if pcm is not None and len(pcm):
                            out_chunks.append(pcm)
                            spoke = True
                            last_audio = loop.time()
                    elif kind == MT_TEXT:
                        tok = data[1:].decode("utf-8", "replace")
                        text_parts.append(tok)
                        if on_text:
                            on_text(tok)
                    elif kind == MT_ERROR:
                        text_parts.append(f"[error] {data[1:].decode('utf-8','replace')}")
                        break
                sender.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await sender
        reply = np.concatenate(out_chunks) if out_chunks else np.zeros(0, np.float32)
        return reply, "".join(text_parts)

    async def _send_pcm(self, ws: aiohttp.ClientWebSocketResponse, pcm: np.ndarray) -> None:
        """Encode PCM to Opus and send MT=1 frames at ~realtime pace."""
        for i in range(0, len(pcm), FRAME_SIZE):
            frame = pcm[i : i + FRAME_SIZE]
            if len(frame) < FRAME_SIZE:
                frame = np.pad(frame, (0, FRAME_SIZE - len(frame)))
            opus = self._opus_writer.append_pcm(frame)
            if opus:
                await ws.send_bytes(bytes([MT_AUDIO]) + opus)
            await asyncio.sleep(FRAME_SECONDS)


async def _amain(args: argparse.Namespace) -> int:
    client = MoshiClient(url=args.url)
    pcm_in = load_wav_mono24k(args.infile)
    print(f"sending {len(pcm_in) / SAMPLE_RATE:.1f}s of audio to {args.url}")
    reply, text = await client.converse(
        pcm_in,
        max_seconds=args.max_seconds,
        tail_silence_s=args.tail_silence,
        on_text=lambda t: print(t, end="", flush=True),
    )
    dur = save_wav_mono24k(args.out, reply)
    print(f"\n--- transcript ---\n{text}\n--- reply: {dur:.1f}s audio -> {args.out} ---")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Slice 4: headless Moshi client")
    ap.add_argument("--in", dest="infile", required=True, help="input wav (the 'user' utterance)")
    ap.add_argument("--out", default="/tmp/moshi_reply.wav", help="output wav for Moshi's reply")
    ap.add_argument("--url", default="ws://127.0.0.1:8998/api/chat", help="moshi-backend ws url")
    ap.add_argument("--max-seconds", type=float, default=30.0, help="max conversation window")
    ap.add_argument("--tail-silence", type=float, default=2.0, help="stop after this much trailing silence")
    args = ap.parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
