"""Slice 5: browser-facing voice proxy (orchestrator-as-Moshi-proxy).

Speaks Moshi's binary websocket protocol to the browser and relays to real
Moshi (localhost:8998), so Moshi's own web client works unmodified — point it at
this server's `/api/chat`. A tap decodes the inbound user audio, detects
end-of-turn by energy (VAD), transcribes the turn (Whisper), and on a handoff
trigger phrase: mutes Moshi (audio AND text), plays a short filler, streams the
LLM→Orpheus answer to the browser as realtime Opus + its text, then resumes.
Barge-in: if the user talks during the answer, it's cancelled and Moshi resumes.

Layers: (1) bridge [`Bridge`], (2) turn tap + STT trigger [`TurnTap`],
(3) handoff action w/ filler + text + barge-in [`VoiceSession`]. See DESIGN.md.

    ROUTER_API_KEY=... python -m llm_router.voice.server --port 8999 \
        --moshi wss://127.0.0.1:8998/api/chat --stt-url http://192.168.42.240:5399/v3 \
        --web-dir /path/to/dist
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import logging
import pathlib
import time
import wave
from collections.abc import Callable

import httpx
import numpy as np
import sphn
from aiohttp import ClientSession, TCPConnector, WSMsgType, web

from llm_router.voice.handoff import DEFAULT_TRIGGERS
from llm_router.voice.handoff_tts import PipelineConfig, stream_deltas, synth_bytes
from llm_router.voice.streaming_tts import early_first_chunk, extract_sentences

log = logging.getLogger("voice.server")

MT_AUDIO = 1
MT_TEXT = 2
MT_CONTROL = 3
CTL_PAUSE = 2
CTL_RESTART = 3
SAMPLE_RATE = 24000
FRAME_SIZE = 1920
FRAME_SECONDS = FRAME_SIZE / SAMPLE_RATE


def _pcm_to_wav_bytes(pcm: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with contextlib.closing(wave.open(buf, "wb")) as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def _wav_bytes_to_pcm24k(wav_bytes: bytes) -> np.ndarray:
    with contextlib.closing(wave.open(io.BytesIO(wav_bytes), "rb")) as w:
        n, ch, sr = w.getnframes(), w.getnchannels(), w.getframerate()
        raw = w.readframes(n)
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    if sr != SAMPLE_RATE:
        idx = np.linspace(0, len(pcm) - 1, int(len(pcm) * SAMPLE_RATE / sr))
        pcm = np.interp(idx, np.arange(len(pcm)), pcm).astype(np.float32)
    return pcm


class HandoffState:
    """Shared between the tap (sets cancel on barge-in) and the session."""

    def __init__(self) -> None:
        self.active = False
        self.cancel = asyncio.Event()


class Bridge:
    """One browser <-> Moshi session; relays MT frames, with a tap + mute hook."""

    def __init__(
        self,
        browser: web.WebSocketResponse,
        moshi_url: str,
        on_user_audio: Callable[[bytes], None] | None = None,
        suppress_downstream: Callable[[], bool] | None = None,
    ) -> None:
        self.browser = browser
        self.moshi_url = moshi_url
        self.on_user_audio = on_user_audio
        self.suppress_downstream = suppress_downstream
        self.up = None

    async def run(self) -> None:
        connector = TCPConnector(ssl=False) if self.moshi_url.startswith("wss") else None
        async with ClientSession(connector=connector) as sess:
            async with sess.ws_connect(self.moshi_url, max_msg_size=0) as up:
                self.up = up
                tasks = {
                    asyncio.create_task(self._browser_to_moshi(up)),
                    asyncio.create_task(self._moshi_to_browser(up)),
                }
                _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for t in pending:
                    t.cancel()

    async def _browser_to_moshi(self, up) -> None:
        async for msg in self.browser:
            if msg.type == WSMsgType.BINARY:
                data = msg.data
                await up.send_bytes(data)
                if self.on_user_audio and data and data[0] == MT_AUDIO:
                    self.on_user_audio(data[1:])
            elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.ERROR):
                break

    async def _moshi_to_browser(self, up) -> None:
        async for msg in up:
            if msg.type == WSMsgType.BINARY:
                # During a handoff, mute BOTH Moshi's audio and its text — the
                # session injects the expert's audio+text instead.
                if (
                    self.suppress_downstream
                    and self.suppress_downstream()
                    and msg.data[:1] in (bytes([MT_AUDIO]), bytes([MT_TEXT]))
                ):
                    continue
                await self.browser.send_bytes(msg.data)
            elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.ERROR):
                break

    async def send_audio_to_browser(self, opus_payload: bytes) -> None:
        await self.browser.send_bytes(bytes([MT_AUDIO]) + opus_payload)

    async def send_text_to_browser(self, text: str) -> None:
        await self.browser.send_bytes(bytes([MT_TEXT]) + text.encode("utf-8"))

    async def control_moshi(self, ctl: int) -> None:
        if self.up is not None:
            with contextlib.suppress(Exception):
                await self.up.send_bytes(bytes([MT_CONTROL, ctl]))


class TurnTap:
    """Energy-VAD end-of-turn → Whisper → handoff trigger; barge-in during handoff."""

    def __init__(
        self,
        stt_url: str,
        stt_model: str,
        triggers: list[str],
        on_trigger: Callable[[str], object],
        state: HandoffState,
        energy_thresh: float = 0.01,
        bargein_thresh: float = 0.02,
        bargein_s: float = 0.4,
        silence_gap_s: float = 0.8,
        min_turn_s: float = 0.6,
        max_turn_s: float = 16.0,
    ) -> None:
        self.stt_url = stt_url
        self.stt_model = stt_model
        self.triggers = triggers
        self.on_trigger = on_trigger
        self.state = state
        self.energy_thresh = energy_thresh
        self.bargein_thresh = bargein_thresh
        self.bargein_n = int(bargein_s * SAMPLE_RATE)
        self.silence_gap_s = silence_gap_s
        self.min_turn_n = int(min_turn_s * SAMPLE_RATE)
        self.max_turn_n = int(max_turn_s * SAMPLE_RATE)
        self._reader = sphn.OpusStreamReader(SAMPLE_RATE)
        self._pcm = np.zeros(0, np.float32)
        self._speech_seen = False
        self._last_voice: float | None = None
        self._armed = False
        self._bargein_voiced = 0
        self._busy = False

    def feed(self, opus_payload: bytes) -> None:
        pcm = self._reader.append_bytes(opus_payload)
        if pcm is None or not len(pcm):
            return
        self._pcm = np.concatenate((self._pcm, pcm))[-self.max_turn_n :]
        rms = float(np.sqrt(np.mean(pcm**2)))
        if rms > self.energy_thresh:
            self._speech_seen = True
            self._last_voice = time.monotonic()
        # Barge-in: sustained voice while the expert answer is playing -> cancel.
        if self.state.active:
            if rms > self.bargein_thresh:
                self._bargein_voiced += len(pcm)
                if self._bargein_voiced > self.bargein_n:
                    self.state.cancel.set()
            else:
                self._bargein_voiced = 0
        else:
            self._bargein_voiced = 0

    def _reset(self) -> None:
        self._pcm = np.zeros(0, np.float32)
        self._speech_seen = False
        self._last_voice = None

    async def loop(self) -> None:
        async with httpx.AsyncClient(timeout=30.0) as client:
            while True:
                await asyncio.sleep(0.25)
                if self._busy or self.state.active:
                    continue
                if not (
                    self._speech_seen
                    and self._last_voice is not None
                    and (time.monotonic() - self._last_voice) > self.silence_gap_s
                    and len(self._pcm) > self.min_turn_n
                ):
                    continue
                turn = self._pcm.copy()
                self._reset()
                self._busy = True
                try:
                    text = await self._stt(client, turn)
                    if not text:
                        continue
                    log.info("turn: %r", text)
                    low = text.lower()
                    matched = next((t for t in self.triggers if t in low), None)
                    if matched:
                        after = text[low.index(matched) + len(matched) :].lstrip(" ,.:;-—?!").strip()
                        if after:
                            log.info("TRIGGER -> expert query: %r", after)
                            await self.on_trigger(after)
                        else:
                            log.info("armed: heard trigger, awaiting the question")
                            self._armed = True
                    elif self._armed:
                        self._armed = False
                        log.info("TRIGGER (armed) -> expert query: %r", text)
                        await self.on_trigger(text)
                except Exception:  # noqa: BLE001
                    log.exception("tap error")
                finally:
                    self._busy = False

    async def _stt(self, client: httpx.AsyncClient, pcm: np.ndarray) -> str:
        resp = await client.post(
            f"{self.stt_url}/audio/transcriptions",
            files={"file": ("turn.wav", _pcm_to_wav_bytes(pcm), "audio/wav")},
            data={"model": self.stt_model},
        )
        resp.raise_for_status()
        return resp.json().get("text", "").strip()


class VoiceSession:
    """Handoff action: mute Moshi, filler, stream LLM→Orpheus (audio+text), resume.

    Honors barge-in via `state.cancel` (set by the tap when the user talks over
    the answer).
    """

    def __init__(self, bridge: Bridge, pcfg: PipelineConfig, suppress: dict,
                 state: HandoffState, app: web.Application) -> None:
        self.bridge = bridge
        self.pcfg = pcfg
        self.suppress = suppress
        self.state = state
        self.app = app

    async def on_trigger(self, query: str) -> None:
        self.state.cancel.clear()
        self.state.active = True
        self.suppress["on"] = True
        await self.bridge.control_moshi(CTL_PAUSE)
        writer = sphn.OpusStreamWriter(SAMPLE_RATE)
        try:
            filler_pcm = self.app.get("filler_pcm")
            if filler_pcm is not None:
                await self.bridge.send_text_to_browser(self.app["filler_text"] + " ")
                await self._send_pcm(writer, filler_pcm)
            if not self.state.cancel.is_set():
                await self._stream_expert(query, writer)
        except Exception:
            log.exception("expert path failed")
        finally:
            await self.bridge.control_moshi(CTL_RESTART)
            self.suppress["on"] = False
            self.state.active = False
            if self.state.cancel.is_set():
                log.info("handoff cancelled (barge-in)")

    async def _stream_expert(self, query: str, writer: sphn.OpusStreamWriter) -> None:
        async with httpx.AsyncClient(timeout=self.pcfg.request_timeout) as client:

            async def speak(text: str) -> None:
                await self.bridge.send_text_to_browser(text + " ")
                wav = await synth_bytes(self.pcfg, client, text)
                await self._send_pcm(writer, _wav_bytes_to_pcm24k(wav))

            buf = ""
            first_done = False
            async for delta in stream_deltas(self.pcfg, client, query):
                if self.state.cancel.is_set():
                    return
                buf += delta
                to_say: list[str] = []
                if not first_done:
                    chunk, buf = early_first_chunk(buf, 12, 36)
                    if not chunk:
                        continue
                    to_say.append(chunk)
                    first_done = True
                sents, buf = extract_sentences(buf, 16)
                to_say += sents
                for s in to_say:
                    if self.state.cancel.is_set():
                        return
                    await speak(s)
            if buf.strip() and not self.state.cancel.is_set():
                await speak(buf.strip())

    async def _send_pcm(self, writer: sphn.OpusStreamWriter, pcm: np.ndarray) -> None:
        for i in range(0, len(pcm), FRAME_SIZE):
            if self.state.cancel.is_set():
                return
            frame = pcm[i : i + FRAME_SIZE]
            if len(frame) < FRAME_SIZE:
                frame = np.pad(frame, (0, FRAME_SIZE - len(frame)))
            opus = writer.append_pcm(frame)
            if opus:
                await self.bridge.send_audio_to_browser(opus)
            await asyncio.sleep(FRAME_SECONDS)  # realtime pace


async def _ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    app = request.app
    peer = request.remote
    moshi_url = app["moshi_url"]
    if request.query_string:  # forward the client's generation params to Moshi
        moshi_url += ("&" if "?" in moshi_url else "?") + request.query_string
    log.info("browser %s connected; bridging to %s", peer, moshi_url)

    suppress = {"on": False}
    bridge = Bridge(ws, moshi_url, suppress_downstream=lambda: suppress["on"])
    coros = [bridge.run()]

    pcfg: PipelineConfig = app["pcfg"]
    if app.get("stt_url") and pcfg.router_key:
        state = HandoffState()
        session = VoiceSession(bridge, pcfg, suppress, state, app)
        tap = TurnTap(app["stt_url"], app["stt_model"], app["triggers"], session.on_trigger, state)
        bridge.on_user_audio = tap.feed
        coros.append(tap.loop())
    elif app.get("stt_url"):
        log.warning("stt-url set but no ROUTER_API_KEY — handoff disabled (passthrough only)")

    tasks = {asyncio.create_task(c) for c in coros}
    try:
        _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
    except Exception:
        log.exception("session error")
    finally:
        log.info("browser %s disconnected", peer)
    return ws


async def _health(_: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def _prefetch_filler(app: web.Application) -> None:
    """Synth the filler clip once Orpheus answers (so the first handoff isn't slow)."""
    pcfg: PipelineConfig = app["pcfg"]
    text = app["filler_text"]
    if not (text and pcfg.router_key and app.get("stt_url")):
        return
    for _ in range(60):
        try:
            async with httpx.AsyncClient(timeout=30.0) as c:
                wav = await synth_bytes(pcfg, c, text)
            app["filler_pcm"] = _wav_bytes_to_pcm24k(wav)
            log.info("filler clip ready (%.1fs): %r", len(app["filler_pcm"]) / SAMPLE_RATE, text)
            return
        except Exception as e:  # noqa: BLE001
            log.debug("filler synth retry: %s", e)
            await asyncio.sleep(5)
    log.warning("filler synth gave up; handoffs will have no filler")


def make_app(
    moshi_url: str,
    pcfg: PipelineConfig,
    stt_url: str | None = None,
    stt_model: str = "whisper-large-v3-turbo",
    triggers: list[str] | None = None,
    web_dir: str | None = None,
    filler_text: str = "Let me look that up.",
) -> web.Application:
    app = web.Application()
    app["moshi_url"] = moshi_url
    app["pcfg"] = pcfg
    app["stt_url"] = stt_url
    app["stt_model"] = stt_model
    app["triggers"] = triggers or list(DEFAULT_TRIGGERS)
    app["filler_text"] = filler_text
    app["filler_pcm"] = None
    app.router.add_get("/api/chat", _ws_handler)
    app.router.add_get("/health", _health)
    if web_dir:  # serve Moshi's web client; same-origin -> it connects to our /api/chat
        wd = pathlib.Path(web_dir)

        async def _index(_: web.Request) -> web.FileResponse:
            return web.FileResponse(wd / "index.html")

        app.router.add_get("/", _index)
        app.router.add_static("/assets", wd / "assets")
    async def _on_startup(a: web.Application) -> None:  # fire-and-forget filler prefetch
        a["_filler_task"] = asyncio.create_task(_prefetch_filler(a))

    app.on_startup.append(_on_startup)
    return app


def main() -> int:
    ap = argparse.ArgumentParser(description="Slice 5: Moshi-proxy voice server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8999)
    ap.add_argument("--moshi", default="wss://127.0.0.1:8998/api/chat", help="upstream Moshi ws")
    ap.add_argument("--stt-url", default=None, help="Whisper /v3 base (enables the handoff)")
    ap.add_argument("--stt-model", default="whisper-large-v3-turbo")
    ap.add_argument("--llm", default="auto-full", help="expert-path router model")
    ap.add_argument("--router-url", default=None, help="router base url (remote when off-host)")
    ap.add_argument("--tts-url", default=None, help="Orpheus base for the expert path")
    ap.add_argument("--web-dir", default=None, help="serve a web-client dist from / (frontend)")
    ap.add_argument("--filler", default="Let me look that up.", help="filler phrase during TTFT")
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")

    p_over: dict = {"llm_model": args.llm}
    if args.router_url:
        p_over["router_url"] = args.router_url
    if args.tts_url:
        p_over["tts_url"] = args.tts_url
    pcfg = PipelineConfig(**p_over)

    web.run_app(
        make_app(
            args.moshi, pcfg, stt_url=args.stt_url, stt_model=args.stt_model,
            web_dir=args.web_dir, filler_text=args.filler,
        ),
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
