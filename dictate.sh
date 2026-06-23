#!/usr/bin/env bash
# Launch the native push-to-talk dictation client from Git Bash (Windows side).
#
# Run this from Git Bash on Windows — NOT from WSL — because the client needs the
# OS that owns your keyboard and microphone. Make sure the server (python app.py)
# is already running in WSL.
set -euo pipefail
cd "$(dirname "$0")"

# Prefer the Windows "py" launcher; fall back to python on PATH.
if command -v py >/dev/null 2>&1; then
    PY=py
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "Python not found on PATH. Install Windows Python from https://python.org" >&2
    exit 1
fi

if ! "$PY" dictate_client.py "$@"; then
    echo
    echo "Client exited with an error. If modules are missing, run:" >&2
    echo "    $PY -m pip install -r requirements-client.txt" >&2
    exit 1
fi
