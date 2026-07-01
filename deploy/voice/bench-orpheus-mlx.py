#!/usr/bin/env python3
"""bench-orpheus-mlx.py — HONEST RTF bench for Orpheus TTS on an mlx-audio server.

Orpheus emits substantial leading/trailing SILENCE (~25-35% of the clip), which
inflates any RTF computed against the raw clip length. This bench trims the
silence and reports speech-only RTF alongside the raw (optimistic) one.

Per prompt it reports:
  sRTF  = compute / SPEECH duration      <- the honest throughput number
  rRTF  = compute / raw (incl. silence)  <- what a naive bench shows (optimistic)
  compute = wall-clock to synthesize the whole clip
  speech/total = trimmed vs raw duration
  lead/trail   = silence trimmed off each end (verify against the wav files)

Note on latency: the server returns WAV, whose length-prefixed header can't be
written until synthesis finishes, so it buffers the whole clip — there is no
progressive audio. First-audio latency therefore ~= `compute` (you wait for the
full synth), and on playback the clip still opens with `lead` seconds of silence.
To feel responsive you must chunk the input text into short utterances; a
streaming-onset metric would be meaningless against this non-streaming path.

Deps: soundfile + numpy (already in the mlx-audio venv). HTTP via stdlib urllib.

Prereqs (on the Mac), same as before:
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
      before playback (a known Orpheus artifact and pure onset latency).
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


def synth(text: str) -> tuple[bytes, float]:
    """POST to /v1/audio/speech; return (wav_bytes, compute_seconds)."""
    body = json.dumps({"model": MODEL, "input": text, "voice": VOICE}).encode()
    req = urllib.request.Request(
        f"{SERVER}/v1/audio/speech",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
    return data, time.perf_counter() - t0


def analyze(data: bytes) -> tuple[float, float, float, float]:
    """Return (total_s, lead_s, trail_s, speech_s) by trimming near-silence."""
    audio, sr = sf.read(io.BytesIO(data), always_2d=True)
    frames, _ch = audio.shape
    mono = audio.mean(axis=1)
    amp = np.abs(mono)
    peak = float(amp.max()) if frames else 0.0
    voiced = np.where(amp > peak * SILENCE_FRAC)[0] if peak > 0 else np.array([], int)

    total_s = frames / sr if sr else 0.0
    if len(voiced) == 0:  # all silence — degrade gracefully
        return total_s, total_s, 0.0, 0.0
    first, last = int(voiced[0]), int(voiced[-1])
    return total_s, first / sr, (frames - 1 - last) / sr, (last - first + 1) / sr


def main() -> None:
    print(f"Model:  {MODEL}")
    print(f"Server: {SERVER}   Voice: {VOICE}")
    print("Warming up (loads the model; excluded from results)...")
    synth("warmup one two three")
    print()

    hdr = ("sRTF", "rRTF", "compute", "speech", "total", "lead", "trail", "prompt")
    print(f"{hdr[0]:<6}{hdr[1]:<6}{hdr[2]:<9}{hdr[3]:<8}"
          f"{hdr[4]:<8}{hdr[5]:<7}{hdr[6]:<7}{hdr[7]}")

    tot_c = tot_speech = tot_total = 0.0
    for p in PROMPTS:
        data, compute = synth(p)
        total_s, lead_s, trail_s, speech_s = analyze(data)
        srtf = compute / speech_s if speech_s else float("nan")
        rrtf = compute / total_s if total_s else float("nan")
        print(f"{srtf:<6.3f}{rrtf:<6.3f}{compute:<9.3f}{speech_s:<8.3f}"
              f"{total_s:<8.3f}{lead_s:<7.3f}{trail_s:<7.3f}{p[:40]}")
        tot_c += compute
        tot_speech += speech_s
        tot_total += total_s

    print()
    print(f"Speech-only : {tot_c:.3f}s compute for {tot_speech:.3f}s speech  "
          f"->  RTF {tot_c / tot_speech:.3f}  ({tot_speech / tot_c:.2f}x realtime)   <- honest")
    print(f"Raw (silence): {tot_c:.3f}s compute for {tot_total:.3f}s audio   "
          f"->  RTF {tot_c / tot_total:.3f}  ({tot_total / tot_c:.2f}x realtime)   <- optimistic")
    print(f"Speech is {100 * tot_speech / tot_total:.0f}% of output "
          f"(the rest is lead/trail silence). First-audio ~= compute (server buffers WAV).")


if __name__ == "__main__":
    main()
