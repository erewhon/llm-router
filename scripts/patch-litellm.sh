#!/bin/sh
# Workaround for litellm 1.82.0 packaging bug:
# litellm/types/ and litellm/types/proxy/ are missing __init__.py,
# causing "No module named 'litellm.types.proxy.litellm_pre_call_utils'"
# on tool-calling requests through the proxy.
#
# Run after: uv sync --extra litellm
# Remove when litellm fixes the bug upstream.

set -e

SITE=$(python -c "import site; print(site.getsitepackages()[0])")
TYPES_DIR="$SITE/litellm/types"

if [ -d "$TYPES_DIR" ] && [ ! -f "$TYPES_DIR/__init__.py" ]; then
    touch "$TYPES_DIR/__init__.py"
    echo "Created $TYPES_DIR/__init__.py"
fi

if [ -d "$TYPES_DIR/proxy" ] && [ ! -f "$TYPES_DIR/proxy/__init__.py" ]; then
    touch "$TYPES_DIR/proxy/__init__.py"
    echo "Created $TYPES_DIR/proxy/__init__.py"
fi

echo "litellm patched OK"
