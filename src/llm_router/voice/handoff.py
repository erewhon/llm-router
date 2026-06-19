"""Slice 4: the handoff orchestrator — route a spoken turn to Moshi or the LLM.

A turn arrives as user audio. We transcribe it (Whisper STT), then decide:

- **conversational** → **Moshi** (full-duplex, fast, natural)        [moshi_client]
- **expert / complex** → **LLM handoff**: auto-router LLM → CUDA Orpheus TTS
                                                                     [streaming_tts]

v1 trigger = an explicit phrase ("ask the expert", "look that up", …); the text
after the phrase becomes the expert query. Everything else goes to Moshi. The
planned upgrade is auto-router complexity classification instead of a phrase.

Runs co-located with Moshi (:8998) + Orpheus on hypatia in production; for dev it
can run anywhere the four services are reachable.

    ROUTER_API_KEY=... uv run python -m llm_router.voice.handoff \
        --in turn.wav --out answer.wav --moshi-url wss://192.168.42.52:8998/api/chat
"""

from __future__ import annotations

import argparse
import asyncio
import os

import httpx
from pydantic import BaseModel, Field

from llm_router.voice.handoff_tts import PipelineConfig
from llm_router.voice.moshi_client import MoshiClient, load_wav_mono24k, save_wav_mono24k
from llm_router.voice.streaming_tts import stream_pipeline

DEFAULT_TRIGGERS = [
    "ask the expert",
    "look that up",
    "look this up",
    "look it up",
    "search for",
    "search the web",
]


class HandoffConfig(BaseModel):
    moshi_url: str = "ws://127.0.0.1:8998/api/chat"
    stt_url: str = "http://192.168.42.240:5399/v3"
    stt_model: str = "whisper-large-v3-turbo"
    triggers: list[str] = Field(default_factory=lambda: list(DEFAULT_TRIGGERS))
    moshi_max_seconds: float = 30.0
    stt_timeout: float = 60.0


class TurnResult(BaseModel):
    path: str  # "expert" | "moshi"
    user_text: str
    query: str | None = None  # the expert query when path == "expert"
    out_path: str
    detail: str = ""


def detect_handoff(text: str, triggers: list[str]) -> tuple[bool, str]:
    """If `text` contains a trigger phrase, return (True, query-after-phrase)."""
    low = text.lower()
    for t in triggers:
        i = low.find(t)
        if i != -1:
            after = text[i + len(t) :].lstrip(" ,.:;-—?!").strip()
            return True, (after or text.strip())
    return False, ""


async def transcribe(cfg: HandoffConfig, client: httpx.AsyncClient, audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        resp = await client.post(
            f"{cfg.stt_url}/audio/transcriptions",
            files={"file": (os.path.basename(audio_path), f, "audio/wav")},
            data={"model": cfg.stt_model},
            timeout=cfg.stt_timeout,
        )
    resp.raise_for_status()
    return resp.json().get("text", "").strip()


async def handle_turn(
    hcfg: HandoffConfig, pcfg: PipelineConfig, audio_path: str, out_path: str
) -> TurnResult:
    """Transcribe a turn, decide Moshi vs expert, produce the answer audio."""
    async with httpx.AsyncClient() as client:
        user_text = await transcribe(hcfg, client, audio_path)

    is_handoff, query = detect_handoff(user_text, hcfg.triggers)
    if is_handoff:
        res = await stream_pipeline(pcfg, query, out_path)
        return TurnResult(
            path="expert",
            user_text=user_text,
            query=query,
            out_path=out_path,
            detail=f"ttfa={res.ttfa_s:.2f}s audio={res.audio_s:.2f}s rtf={res.realtime_factor:.2f}x",
        )

    pcm = load_wav_mono24k(audio_path)
    reply, transcript = await MoshiClient(hcfg.moshi_url).converse(
        pcm, max_seconds=hcfg.moshi_max_seconds
    )
    dur = save_wav_mono24k(out_path, reply)
    return TurnResult(
        path="moshi",
        user_text=user_text,
        out_path=out_path,
        detail=f"moshi_said={transcript.strip()!r} audio={dur:.2f}s",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Slice 4: handoff orchestrator (Moshi vs LLM)")
    ap.add_argument("--in", dest="infile", required=True, help="user turn audio (wav)")
    ap.add_argument("--out", default="/tmp/handoff_answer.wav", help="answer audio out")
    ap.add_argument("--moshi-url", default=None, help="moshi-backend ws url")
    ap.add_argument("--stt-url", default=None, help="Whisper /v3 base url")
    ap.add_argument("--llm", default="auto-full", help="expert-path router model")
    ap.add_argument("--tts-url", default=None, help="Orpheus base url for the expert path")
    args = ap.parse_args()

    h_over: dict = {}
    if args.moshi_url:
        h_over["moshi_url"] = args.moshi_url
    if args.stt_url:
        h_over["stt_url"] = args.stt_url
    hcfg = HandoffConfig(**h_over)

    p_over: dict = {"llm_model": args.llm}
    if args.tts_url:
        p_over["tts_url"] = args.tts_url
    pcfg = PipelineConfig(**p_over)
    if not pcfg.router_key:
        print("error: ROUTER_API_KEY not set", flush=True)
        return 2

    res = asyncio.run(handle_turn(hcfg, pcfg, args.infile, args.out))
    print(f"\n[{res.path.upper()}] heard: {res.user_text!r}")
    if res.query is not None:
        print(f"  expert query: {res.query!r}")
    print(f"  {res.detail}\n  -> {res.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
