"""Slice 5: browser-facing voice proxy (orchestrator-as-Moshi-proxy).

Speaks Moshi's binary websocket protocol to the browser and relays to real
Moshi (localhost:8998), so Moshi's own web client works unmodified — just point
it at this server's `/api/chat`. A tap decodes the inbound user audio and runs
rolling-window Whisper on it to detect a handoff trigger mid-stream; when it
fires, the handoff layer pauses Moshi, plays the LLM→Orpheus answer to the
browser as MT=1 frames, then resumes.

Layers: (1) transparent bridge [`Bridge`], (2) STT trigger tap [`TriggerTap`],
(3) handoff action — pause/Orpheus-stream/resume [`VoiceSession.on_trigger`].
See DESIGN.md (Slice 5). Run on hypatia (localhost to Moshi + Orpheus):

    python -m llm_router.voice.server --port 8999 --moshi wss://127.0.0.1:8998/api/chat \
        --stt-url http://192.168.42.240:5399/v3
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import logging
import wave
from collections.abc import Callable

import httpx
import numpy as np
import sphn
from aiohttp import ClientSession, TCPConnector, WSMsgType, web

from llm_router.voice.handoff import DEFAULT_TRIGGERS, detect_handoff

log = logging.getLogger("voice.server")

MT_AUDIO = 1
MT_TEXT = 2
MT_CONTROL = 3
CTL_PAUSE = 2
CTL_RESTART = 3
SAMPLE_RATE = 24000


class Bridge:
    """A single browser <-> Moshi session: relays MT frames both ways.

    `on_user_audio(payload)` receives the Opus payload of each inbound MT=1
    frame from the browser (the tap). `suppress_downstream()` lets the handoff
    layer mute Moshi→browser audio while it plays the expert answer.
    """

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
        self.up = None  # upstream ws (set in run)

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
                # During a handoff we mute Moshi's audio but keep its text.
                if self.suppress_downstream and self.suppress_downstream() and msg.data[:1] == bytes(
                    [MT_AUDIO]
                ):
                    continue
                await self.browser.send_bytes(msg.data)
            elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.ERROR):
                break

    async def send_audio_to_browser(self, opus_payload: bytes) -> None:
        await self.browser.send_bytes(bytes([MT_AUDIO]) + opus_payload)

    async def control_moshi(self, ctl: int) -> None:
        if self.up is not None:
            with contextlib.suppress(Exception):
                await self.up.send_bytes(bytes([MT_CONTROL, ctl]))


def _pcm_to_wav_bytes(pcm: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with contextlib.closing(wave.open(buf, "wb")) as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


class TriggerTap:
    """Decodes inbound user Opus, runs rolling-window Whisper, fires on trigger."""

    def __init__(
        self,
        stt_url: str,
        stt_model: str,
        triggers: list[str],
        on_trigger: Callable[[str], "asyncio.Future"],
        window_s: float = 6.0,
        check_interval_s: float = 2.5,
        min_new_s: float = 1.5,
    ) -> None:
        self.stt_url = stt_url
        self.stt_model = stt_model
        self.triggers = triggers
        self.on_trigger = on_trigger
        self.window_n = int(window_s * SAMPLE_RATE)
        self.check_interval_s = check_interval_s
        self.min_new_n = int(min_new_s * SAMPLE_RATE)
        self._reader = sphn.OpusStreamReader(SAMPLE_RATE)
        self._pcm = np.zeros(0, np.float32)
        self._seen = 0  # samples at last transcription
        self._busy = False

    def feed(self, opus_payload: bytes) -> None:
        pcm = self._reader.append_bytes(opus_payload)
        if pcm is not None and len(pcm):
            self._pcm = np.concatenate((self._pcm, pcm))[-self.window_n * 2 :]

    async def loop(self) -> None:
        async with httpx.AsyncClient(timeout=30.0) as client:
            while True:
                await asyncio.sleep(self.check_interval_s)
                if self._busy or len(self._pcm) - self._seen < self.min_new_n:
                    continue
                window = self._pcm[-self.window_n :]
                self._seen = len(self._pcm)
                try:
                    text = await self._stt(client, window)
                except Exception as e:  # noqa: BLE001
                    log.warning("stt error: %s", e)
                    continue
                if not text:
                    continue
                log.info("heard: %r", text)
                is_h, query = detect_handoff(text, self.triggers)
                if is_h:
                    log.info("TRIGGER -> expert query: %r", query)
                    self._busy = True
                    try:
                        await self.on_trigger(query)
                    finally:
                        self._busy = False
                        self._pcm = np.zeros(0, np.float32)  # avoid re-firing
                        self._seen = 0

    async def _stt(self, client: httpx.AsyncClient, pcm: np.ndarray) -> str:
        resp = await client.post(
            f"{self.stt_url}/audio/transcriptions",
            files={"file": ("turn.wav", _pcm_to_wav_bytes(pcm), "audio/wav")},
            data={"model": self.stt_model},
        )
        resp.raise_for_status()
        return resp.json().get("text", "").strip()


async def _ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    app = request.app
    peer = request.remote
    log.info("browser %s connected; bridging to %s", peer, app["moshi_url"])

    suppress = {"on": False}
    tap = None
    if app.get("stt_url"):
        # Layer 3 (handoff action) lands here; for now the trigger just logs and
        # briefly mutes Moshi as a placeholder so the wiring is exercised.
        async def on_trigger(query: str) -> None:
            log.info("handoff requested for %r (action: layer 3)", query)

        tap = TriggerTap(
            app["stt_url"], app["stt_model"], app["triggers"], on_trigger
        )

    bridge = Bridge(
        ws,
        app["moshi_url"],
        on_user_audio=tap.feed if tap else None,
        suppress_downstream=(lambda: suppress["on"]) if tap else None,
    )
    coros = [bridge.run()]
    if tap:
        coros.append(tap.loop())
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


def make_app(
    moshi_url: str,
    stt_url: str | None = None,
    stt_model: str = "whisper-large-v3-turbo",
    triggers: list[str] | None = None,
) -> web.Application:
    app = web.Application()
    app["moshi_url"] = moshi_url
    app["stt_url"] = stt_url
    app["stt_model"] = stt_model
    app["triggers"] = triggers or list(DEFAULT_TRIGGERS)
    app.router.add_get("/api/chat", _ws_handler)
    app.router.add_get("/health", _health)
    return app


def main() -> int:
    ap = argparse.ArgumentParser(description="Slice 5: Moshi-proxy voice server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8999)
    ap.add_argument("--moshi", default="wss://127.0.0.1:8998/api/chat", help="upstream Moshi ws")
    ap.add_argument("--stt-url", default=None, help="Whisper /v3 base (enables the trigger tap)")
    ap.add_argument("--stt-model", default="whisper-large-v3-turbo")
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")
    web.run_app(
        make_app(args.moshi, stt_url=args.stt_url, stt_model=args.stt_model),
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
