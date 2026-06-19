"""Slice 1 of the voice pipeline: text -> LLM (router handoff) -> TTS -> audio.

This proves the "deep reasoning + high-quality speech" half of the assistant
(see DESIGN.md), independent of the Moshi conversational front-end. It streams
the LLM answer from the router (measuring time-to-first-token), then synthesizes
it with Orpheus and reports the realtime factor.

The streaming helpers here (`stream_deltas`, `synth_bytes`) are reused by
`streaming_tts.py` (Slice 2).

Run from the repo root:

    ROUTER_API_KEY=... uv run python -m llm_router.voice.handoff_tts \
        "In one sentence, why is the sky blue?" --llm coder --out /tmp/answer.wav

`--llm auto-full` exercises the auto-router (the production handoff path).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import time
import wave
from collections.abc import AsyncIterator, Callable

import httpx
from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    """Endpoints + knobs for the handoff->TTS slices."""

    router_url: str = Field(default="http://127.0.0.1:4010")
    router_key: str = Field(default_factory=lambda: os.environ.get("ROUTER_API_KEY", ""))
    llm_model: str = "auto-full"
    system_prompt: str = (
        "You are a concise voice assistant. Answer in plain spoken prose, "
        "no markdown, no lists, suitable for text-to-speech."
    )
    # TTS: hypatia CUDA Orpheus (Slice 3) — Q4_K_M on llama.cpp CUDA, ~1.18x RTF
    # (near realtime) vs euclid Arc's ~1.9x. euclid (192.168.42.240:5397) remains
    # a fallback.
    tts_url: str = "http://192.168.42.52:5397"
    tts_model: str = "orpheus"
    tts_voice: str = "tara"
    request_timeout: float = 600.0


class LLMResult(BaseModel):
    text: str
    ttft_s: float | None  # time to first streamed token
    total_s: float
    n_chunks: int


class TTSResult(BaseModel):
    out_path: str
    synth_s: float
    audio_s: float

    @property
    def realtime_factor(self) -> float:
        """synth_time / audio_duration. <1 is faster than realtime."""
        return self.synth_s / self.audio_s if self.audio_s else float("inf")


async def stream_deltas(
    cfg: PipelineConfig,
    client: httpx.AsyncClient,
    prompt: str,
    on_model: Callable[[str], None] | None = None,
) -> AsyncIterator[str]:
    """Yield content deltas from a streaming router chat completion.

    If `on_model` is given, it's called once with the response's `model` field
    (for `auto-full` this is the auto-router's resolved pick — useful for
    routing transparency in the voice UI).
    """
    payload = {
        "model": cfg.llm_model,
        "messages": [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {cfg.router_key}"}
    model_reported = False
    async with client.stream(
        "POST", f"{cfg.router_url}/v1/chat/completions", json=payload, headers=headers
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if on_model and not model_reported and obj.get("model"):
                model_reported = True
                with contextlib.suppress(Exception):
                    on_model(obj["model"])
            delta = (obj.get("choices") or [{}])[0].get("delta", {}).get("content")
            if delta:
                yield delta


async def run_llm(cfg: PipelineConfig, client: httpx.AsyncClient, prompt: str) -> LLMResult:
    """Collect a full streamed answer plus timings (Slice 1, non-pipelined)."""
    started = time.perf_counter()
    ttft: float | None = None
    parts: list[str] = []
    async for delta in stream_deltas(cfg, client, prompt):
        if ttft is None:
            ttft = time.perf_counter() - started
        parts.append(delta)
    return LLMResult(
        text="".join(parts).strip(),
        ttft_s=ttft,
        total_s=time.perf_counter() - started,
        n_chunks=len(parts),
    )


async def synth_bytes(cfg: PipelineConfig, client: httpx.AsyncClient, text: str) -> bytes:
    """Synthesize `text` via the OpenAI-compatible /v1/audio/speech endpoint."""
    payload = {"model": cfg.tts_model, "input": text, "voice": cfg.tts_voice}
    resp = await client.post(f"{cfg.tts_url}/v1/audio/speech", json=payload)
    resp.raise_for_status()
    return resp.content


def _wav_duration(path: str) -> float:
    with contextlib.closing(wave.open(path, "rb")) as w:
        return w.getnframes() / float(w.getframerate())


async def run_tts(
    cfg: PipelineConfig, client: httpx.AsyncClient, text: str, out_path: str
) -> TTSResult:
    started = time.perf_counter()
    content = await synth_bytes(cfg, client, text)
    with open(out_path, "wb") as f:
        f.write(content)
    synth_s = time.perf_counter() - started
    return TTSResult(out_path=out_path, synth_s=synth_s, audio_s=_wav_duration(out_path))


async def pipeline(cfg: PipelineConfig, prompt: str, out_path: str) -> tuple[LLMResult, TTSResult]:
    async with httpx.AsyncClient(timeout=cfg.request_timeout) as client:
        llm = await run_llm(cfg, client, prompt)
        if not llm.text:
            raise RuntimeError("LLM returned no text")
        tts = await run_tts(cfg, client, llm.text, out_path)
    return llm, tts


def main() -> int:
    ap = argparse.ArgumentParser(description="Voice pipeline slice 1: handoff -> TTS")
    ap.add_argument("prompt", help="the question to answer and speak")
    ap.add_argument("--llm", default=None, help="router model (default: auto-full)")
    ap.add_argument("--voice", default=None, help="TTS voice (default: tara)")
    ap.add_argument("--tts-url", default=None, help="override TTS base URL")
    ap.add_argument("--out", default="/tmp/voice_answer.wav", help="output wav path")
    args = ap.parse_args()

    overrides = {}
    if args.llm:
        overrides["llm_model"] = args.llm
    if args.voice:
        overrides["tts_voice"] = args.voice
    if args.tts_url:
        overrides["tts_url"] = args.tts_url
    cfg = PipelineConfig(**overrides)
    if not cfg.router_key:
        print("error: ROUTER_API_KEY not set", flush=True)
        return 2

    llm, tts = asyncio.run(pipeline(cfg, args.prompt, args.out))

    ttft = f"{llm.ttft_s * 1000:.0f} ms" if llm.ttft_s is not None else "n/a"
    print(f"\n--- answer ({cfg.llm_model}) ---\n{llm.text}\n")
    print(f"LLM:  ttft={ttft}  total={llm.total_s:.2f}s  chunks={llm.n_chunks}")
    print(
        f"TTS:  synth={tts.synth_s:.2f}s  audio={tts.audio_s:.2f}s  "
        f"realtime_factor={tts.realtime_factor:.2f}x  -> {tts.out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
