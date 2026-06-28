#!/usr/bin/env bash
# Block until the NetBird mesh IP is assigned to a local interface.
# netbird.service reaching "active" does NOT guarantee wt0 has its IP yet, so
# the llm-router-postgres systemd unit calls this before publishing the
# DB's mesh-bound port (otherwise the bind fails and Docker drops the mapping).
set -euo pipefail

ip="${1:-100.72.235.6}"
for _ in $(seq 1 90); do
  if ip -4 addr show | grep -qF "${ip}/"; then
    exit 0
  fi
  sleep 1
done

echo "wait-netbird-ip: ${ip} not assigned after 90s" >&2
exit 1
