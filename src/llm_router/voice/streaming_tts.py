"""Slice 2 of the voice pipeline: sentence-chunked streaming TTS.

Instead of waiting for the whole LLM answer and then synthesizing it in one
shot (Slice 1), this streams the answer, flushes complete sentences to TTS as
soon as they form, and synthesizes them while the LLM keeps generating. The
payoff is **time-to-first-audio (TTFA)**: the user hears the start of the
answer seconds sooner, even on a slow TTS backend.

A producer task streams the LLM and pushes sentences onto a queue; a consumer
task synthesizes each sentence and appends the audio. They overlap, so TTFA is
roughly (first-sentence latency + one sentence of synth) rather than
(whole answer + whole synth).

Run from the repo root:

    ROUTER_API_KEY=... uv run python -m llm_router.voice.streaming_tts \
        "Explain in 3 sentences why the sky is blue." --llm coder --out /tmp/answer.wav
"""

from __future__ import annotations

import argparse
import asyncio
import io
import re
import time
import wave

import httpx
from pydantic import BaseModel

from llm_router.voice.handoff_tts import (
    PipelineConfig,
    _wav_duration,
    stream_deltas,
    synth_bytes,
)

# A sentence ends at .!? followed by whitespace. Short fragments (abbreviations
# like "e.g." or "Mr.") are coalesced into the next chunk via _MIN_CHUNK_CHARS.
_SENT_END = re.compile(r"[.!?](?=\s)")


def extract_sentences(buf: str, min_chars: int) -> tuple[list[str], str]:
    """Pull complete sentences (>= min_chars) off the front of `buf`.

    Returns (sentences, remainder). A terminator that would yield a too-short
    chunk is skipped so the fragment merges into the following sentence.
    """
    out: list[str] = []
    search_from = 0
    while True:
        m = _SENT_END.search(buf, search_from)
        if not m:
            break
        idx = m.end()
        cand = buf[:idx].strip()
        if len(cand) < min_chars:
            search_from = idx  # too short — keep scanning for a later terminator
            continue
        out.append(cand)
        buf = buf[idx:].lstrip()
        search_from = 0
    return out, buf


# For the FIRST chunk we break early — at any clause boundary or a word cap — so
# first audio plays fast. Later chunks use full sentences (better prosody).
_CLAUSE_END = re.compile(r"[,;:.!?](?=\s)")


def early_first_chunk(buf: str, min_chars: int, max_chars: int) -> tuple[str, str]:
    """Return (first_chunk, remainder), or ("", buf) if not ready to flush yet."""
    if len(buf.strip()) < min_chars:
        return "", buf
    m = _CLAUSE_END.search(buf)
    if m and len(buf[: m.end()].strip()) >= min_chars:
        idx = m.end()
        return buf[:idx].strip(), buf[idx:].lstrip()
    if len(buf) >= max_chars:  # no boundary yet — hard-cut at the last word
        cut = buf.rfind(" ", 0, max_chars)
        cut = cut if cut > 0 else max_chars
        return buf[:cut].strip(), buf[cut:].lstrip()
    return "", buf


class StreamResult(BaseModel):
    out_path: str
    ttfa_s: float | None  # time to first audio chunk ready
    total_s: float  # wall-clock to last audio chunk
    audio_s: float  # duration of concatenated audio
    n_sentences: int

    @property
    def realtime_factor(self) -> float:
        return self.total_s / self.audio_s if self.audio_s else float("inf")


async def stream_pipeline(
    cfg: PipelineConfig,
    prompt: str,
    out_path: str,
    min_chunk_chars: int = 16,
    first_chunk_min_chars: int = 12,
    first_chunk_max_chars: int = 36,
) -> StreamResult:
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    started = time.perf_counter()
    state: dict = {"params": None, "frames": [], "ttfa": None, "n": 0}

    async def produce(client: httpx.AsyncClient) -> None:
        buf = ""
        first_done = False
        async for delta in stream_deltas(cfg, client, prompt):
            buf += delta
            if not first_done:  # break early for a fast first audio chunk
                chunk, buf = early_first_chunk(buf, first_chunk_min_chars, first_chunk_max_chars)
                if not chunk:
                    continue
                await queue.put(chunk)
                first_done = True
            sents, buf = extract_sentences(buf, min_chunk_chars)
            for s in sents:
                await queue.put(s)
        if buf.strip():
            await queue.put(buf.strip())
        await queue.put(None)  # sentinel

    async def consume(client: httpx.AsyncClient) -> None:
        while True:
            sent = await queue.get()
            if sent is None:
                break
            wav_bytes = await synth_bytes(cfg, client, sent)
            if state["ttfa"] is None:
                state["ttfa"] = time.perf_counter() - started
            with wave.open(io.BytesIO(wav_bytes), "rb") as w:
                if state["params"] is None:
                    state["params"] = w.getparams()
                state["frames"].append(w.readframes(w.getnframes()))
            state["n"] += 1

    async with httpx.AsyncClient(timeout=cfg.request_timeout) as client:
        await asyncio.gather(produce(client), consume(client))

    if not state["frames"]:
        raise RuntimeError("no audio produced")
    with wave.open(out_path, "wb") as w:
        w.setparams(state["params"])
        w.writeframes(b"".join(state["frames"]))

    return StreamResult(
        out_path=out_path,
        ttfa_s=state["ttfa"],
        total_s=time.perf_counter() - started,
        audio_s=_wav_duration(out_path),
        n_sentences=state["n"],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Voice pipeline slice 2: streaming chunked TTS")
    ap.add_argument("prompt", help="the question to answer and speak")
    ap.add_argument("--llm", default=None, help="router model (default: auto-full)")
    ap.add_argument("--voice", default=None, help="TTS voice (default: tara)")
    ap.add_argument("--tts-url", default=None, help="override TTS base URL")
    ap.add_argument("--out", default="/tmp/voice_answer_streamed.wav", help="output wav path")
    ap.add_argument("--min-chunk-chars", type=int, default=16, help="min sentence length to flush")
    ap.add_argument("--first-chunk-max-chars", type=int, default=36, help="cap on the fast first chunk")
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

    res = asyncio.run(
        stream_pipeline(
            cfg,
            args.prompt,
            args.out,
            args.min_chunk_chars,
            first_chunk_max_chars=args.first_chunk_max_chars,
        )
    )
    ttfa = f"{res.ttfa_s:.2f}s" if res.ttfa_s is not None else "n/a"
    print(
        f"streamed: ttfa={ttfa}  total={res.total_s:.2f}s  audio={res.audio_s:.2f}s  "
        f"sentences={res.n_sentences}  rtf={res.realtime_factor:.2f}x  -> {res.out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
