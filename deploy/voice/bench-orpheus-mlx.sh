#!/usr/bin/env bash
# bench-orpheus-mlx.sh — measure Orpheus (mlx-audio server) latency on an Apple
# Silicon Mac. Reports two things per prompt:
#   RTF  = compute time / audio duration   (<1.0 = faster than realtime; throughput)
#   TTFA = time to first audio byte         (perceived latency; what makes it feel instant)
# Depends only on curl + macOS built-ins (afinfo, awk).
#
# Prereqs (on the Mac):
#   uv venv --python 3.12 && source .venv/bin/activate
#   uv pip install -U 'mlx-audio[all]' 'setuptools<81'   # [all] pulls webrtcvad;
#                                                         # pin avoids pkg_resources break
#   # IMPORTANT: the mlx-audio server crashes with "There is no Stream(gpu, 0) in
#   # current thread" on mlx >= 0.31.2 (thread-local streams). Pin the matched
#   # pre-0.31.2 pair (mlx-audio 0.4.4 floors both at >=0.31.1):
#   uv pip install 'mlx==0.31.1' 'mlx-lm==0.31.1'
#   mlx_audio.server --host 127.0.0.1 --port 8000        # start in another terminal
#
# Usage:
#   ./bench-orpheus-mlx.sh
#   MODEL=mlx-community/orpheus-3b-0.1-ft-6bit ./bench-orpheus-mlx.sh
#   SERVER=http://127.0.0.1:9000 VOICE=tara ./bench-orpheus-mlx.sh
#
# NOTE: keep PROMPTS free of " and \ — kept dependency-free (no JSON escaping).
# NOTE: requests are serial on purpose — concurrent requests crash the server
#       (Blaizzy/mlx-audio#638), a separate bug still present on 0.31.1.
set -euo pipefail

SERVER="${SERVER:-http://127.0.0.1:8000}"
MODEL="${MODEL:-mlx-community/orpheus-3b-0.1-ft-4bit}"
VOICE="${VOICE:-tara}"

PROMPTS=(
  "The quick brown fox jumps over the lazy dog."
  "In a hole in the ground there lived a hobbit, and it was cozy."
  "She sells seashells by the seashore on a sunny summer afternoon."
  "To be, or not to be, that is the question worth asking today."
)

synth() {  # $1=text $2=outfile ; echoes "TTFA TOTAL" seconds (curl timings)
  local body
  body=$(printf '{"model":"%s","input":"%s","voice":"%s"}' "$MODEL" "$1" "$VOICE")
  curl -s -X POST "$SERVER/v1/audio/speech" \
    -H "Content-Type: application/json" \
    -d "$body" -o "$2" -w '%{time_starttransfer} %{time_total}'
}

echo "Model:  $MODEL"
echo "Server: $SERVER   Voice: $VOICE"
echo "Warming up (loads the 3B; excluded from results)..."
synth "warmup one two three" /tmp/orpheus_warm.wav >/dev/null
echo

tot_c=0; tot_d=0; tot_t=0; n=0
printf "%-6s %-7s %-9s %-9s %s\n" "RTF" "TTFA" "compute" "audio" "prompt"
for p in "${PROMPTS[@]}"; do
  out="/tmp/orpheus_bench_$n.wav"
  res=$(synth "$p" "$out")           # "TTFA TOTAL"
  ttfa=${res%% *}
  c=${res##* }
  d=$(afinfo "$out" | awk '/estimated duration/ {print $3}')
  rtf=$(awk -v c="$c" -v d="$d" 'BEGIN{ if (d>0) printf "%.3f", c/d; else print "n/a" }')
  printf "%-6s %-7s %-9s %-9s %.46s\n" "$rtf" "${ttfa}s" "${c}s" "${d}s" "$p"
  tot_c=$(awk -v a="$tot_c" -v b="$c"    'BEGIN{printf "%.6f", a+b}')
  tot_d=$(awk -v a="$tot_d" -v b="$d"    'BEGIN{printf "%.6f", a+b}')
  tot_t=$(awk -v a="$tot_t" -v b="$ttfa" 'BEGIN{printf "%.6f", a+b}')
  n=$((n+1))
done

echo
awk -v c="$tot_c" -v d="$tot_d" 'BEGIN{
  if (c>0) printf "Overall: %.3fs compute for %.3fs audio  ->  RTF %.3f  (%.2fx realtime)\n", c, d, c/d, d/c
}'
awk -v t="$tot_t" -v n="$n" 'BEGIN{
  if (n>0) printf "Mean TTFA (first audio byte): %.3fs\n", t/n
}'
