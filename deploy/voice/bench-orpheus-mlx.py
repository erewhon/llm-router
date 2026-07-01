#!/usr/bin/env python3
"""bench-orpheus-mlx.py — HONEST latency bench for Orpheus TTS on an mlx-audio server.

Orpheus emits substantial leading/trailing SILENCE (~0.6s lead + ~0.6s trail was
measured on 4-bit), which inflates any RTF computed against the raw clip length.
This bench measures the *speech* portion and reports both, plus a real
time-to-first-speech (streams the response and timestamps when the first
non-silent sample actually arrives — not just the HTTP header).

Per prompt it reports:
  sRTF  = compute / SPEECH duration      <- the honest throughput number
  rRTF  = compute / raw (incl. silence)  <- what a naive bench shows (optimistic)
  TTFS  = wall-clock to first speech sample transmitted (perceived onset)
  lead/trail = silence trimmed off each end

Deps: soundfile + numpy (already in the mlx-audio venv). HTTP via stdlib urllib.

Prereqs (on the Mac), same as the shell bench:
  uv venv --python 3.12 && source .venv/bin/activate
  uv pip install -U 'mlx-audio[all]' 'setuptools<81'
  # server crashes on mlx >= 0.31.2 (thread-local streams); pin the matched pair:
  uv pip install 'mlx==0.31.1' 'mlx-lm==0.31.1'
  mlx_audio.server --host 127.0.0.1 --port 8000   # in another terminal

Usage:
  python bench-orpheus-mlx.py
  MODEL=mlx-community/orpheus-3b-0.1-ft-6bit python bench-orpheus-mlx.py
  SERVER=http://127.0.0.1:9000 VOICE=tara python bench-orpheus-mlx.py

NOTE: requests are serial on purpose — concurrent requests crash the server
      (Blaizzy/mlx-audio#638). In production you'd also TRIM the leading silence
      before playback (it's a known Orpheus artifact and pure onset latency).
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import urllib.request

try:
    import numpy as np
    import soundfile as sf
except ImportError:
    sys.exit("Missing deps. In the venv: uv pip install soundfile numpy")

SERVER = os.environ.get("SERVER", "http://127.0.0.1:8000")
MODEL = os.environ.get("MODEL", "mlx-community/orpheus-3b-0.1-ft-4bit")
VOICE = os.environ.get("VOICE", "tara")
SILENCE_FRAC = 0.02  # amplitude threshold as a fraction of the clip's peak

PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In a hole in the ground there lived a hobbit, and it was cozy.",
    "She sells seashells by the seashore on a sunny summer afternoon.",
    "To be, or not to be, that is the question worth asking today.",
]

# subtype -> bytes per sample, to map a sample index back to a byte offset
_BPS = {"PCM_U8": 1, "PCM_S8": 1, "PCM_16": 2, "PCM_24": 3, "PCM_32": 4,
        "FLOAT": 4, "DOUBLE": 8}


def synth(text: str) -> tuple[bytes, float, list[tuple[int, float]]]:
    """POST to /v1/audio/speech, streaming the reply.

    Returns (wav_bytes, compute_seconds, arrivals) where arrivals is a list of
    (cumulative_bytes_received, wall_clock_timestamp) so we can later find when
    a given byte offset landed.
    """
    body = json.dumps({"model": MODEL, "input": text, "voice": VOICE}).encode()
    req = urllib.request.Request(
        f"{SERVER}/v1/audio/speech",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    arrivals: list[tuple[int, float]] = []
    chunks: list[bytes] = []
    cum = 0
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            cum += len(chunk)
            chunks.append(chunk)
            arrivals.append((cum, time.perf_counter() - t0))  # seconds since request start
    compute = time.perf_counter() - t0
    return b"".join(chunks), compute, arrivals


def analyze(data: bytes):
    """Return (sr, total_s, lead_s, trail_s, speech_s, first_speech_byte)."""
    audio, sr = sf.read(io.BytesIO(data), always_2d=True)
    frames, ch = audio.shape
    sub = sf.info(io.BytesIO(data)).subtype
    bps = _BPS.get(sub, 2)
    data_offset = max(0, len(data) - frames * ch * bps)  # WAV header (+pre-chunks)

    mono = audio.mean(axis=1)
    amp = np.abs(mono)
    peak = float(amp.max()) if frames else 0.0
    voiced = np.where(amp > peak * SILENCE_FRAC)[0] if peak > 0 else np.array([], int)

    total_s = frames / sr if sr else 0.0
    if len(voiced) == 0:  # all silence — degrade gracefully
        return sr, total_s, total_s, 0.0, 0.0, len(data)
    first, last = int(voiced[0]), int(voiced[-1])
    lead_s = first / sr
    trail_s = (frames - 1 - last) / sr
    speech_s = (last - first + 1) / sr
    first_speech_byte = data_offset + first * ch * bps
    return sr, total_s, lead_s, trail_s, speech_s, first_speech_byte


def main() -> None:
    print(f"Model:  {MODEL}")
    print(f"Server: {SERVER}   Voice: {VOICE}")
    print("Warming up (loads the model; excluded from results)...")
    synth("warmup one two three")
    print()

    hdr = ("sRTF", "rRTF", "TTFS", "compute", "speech", "total", "lead", "trail", "prompt")
    print(f"{hdr[0]:<6}{hdr[1]:<6}{hdr[2]:<8}{hdr[3]:<9}{hdr[4]:<8}"
          f"{hdr[5]:<8}{hdr[6]:<7}{hdr[7]:<7}{hdr[8]}")

    tot_c = tot_speech = tot_total = tot_ttfs = 0.0
    for p in PROMPTS:
        data, compute, arrivals = synth(p)
        _sr, total_s, lead_s, trail_s, speech_s, first_byte = analyze(data)
        # wall-clock from request start until the first speech sample was transmitted
        onset = next((rel for cum, rel in arrivals if cum >= first_byte), compute)

        srtf = compute / speech_s if speech_s else float("nan")
        rrtf = compute / total_s if total_s else float("nan")
        print(f"{srtf:<6.3f}{rrtf:<6.3f}{onset:<8.3f}{compute:<9.3f}"
              f"{speech_s:<8.3f}{total_s:<8.3f}{lead_s:<7.3f}{trail_s:<7.3f}{p[:40]}")
        tot_c += compute
        tot_speech += speech_s
        tot_total += total_s
        tot_ttfs += onset

    n = len(PROMPTS)
    print()
    print(f"Speech-only : {tot_c:.3f}s compute for {tot_speech:.3f}s speech  "
          f"->  RTF {tot_c / tot_speech:.3f}  ({tot_speech / tot_c:.2f}x realtime)   <- honest")
    print(f"Raw (silence): {tot_c:.3f}s compute for {tot_total:.3f}s audio   "
          f"->  RTF {tot_c / tot_total:.3f}  ({tot_total / tot_c:.2f}x realtime)   <- optimistic")
    print(f"Speech is {100 * tot_speech / tot_total:.0f}% of output; "
          f"mean time-to-first-speech {tot_ttfs / n:.3f}s")


if __name__ == "__main__":
    main()
