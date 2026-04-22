#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
WHISPER_MODEL="${WHISPER_MODEL:-turbo}"

check_dependencies() {
    local failed=0

    echo "--- Checking dependencies ---"

    # ffmpeg
    if ! command -v ffmpeg &>/dev/null; then
        echo "[MISSING] ffmpeg — install with: sudo apt install ffmpeg"
        failed=1
    else
        echo "[OK] ffmpeg $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')"
    fi

    # Python venv
    if [[ ! -f "$VENV/bin/python" ]]; then
        echo "[MISSING] Python virtual environment at $VENV"
        echo "          Create it with: python3 -m venv .venv && pip install -r requirements.txt"
        failed=1
    else
        echo "[OK] venv $VENV"
    fi

    # Python packages (quick import check)
    if [[ -f "$VENV/bin/python" ]]; then
        local packages=("flask" "whisper" "torch" "deep_translator")
        for pkg in "${packages[@]}"; do
            if ! "$VENV/bin/python" -c "import $pkg" &>/dev/null; then
                echo "[MISSING] Python package '$pkg' — run: pip install -r requirements.txt"
                failed=1
            else
                echo "[OK] python package: $pkg"
            fi
        done
    fi

    return $failed
}

# Try to start the app; on failure, run dependency checks
start_app() {
    echo "Starting speech-to-text app (model=$WHISPER_MODEL)..."
    cd "$SCRIPT_DIR"
    source "$VENV/bin/activate"
    WHISPER_MODEL="$WHISPER_MODEL" python app.py
}

if ! start_app; then
    echo ""
    echo "App failed to start. Running dependency checks..."
    echo ""
    if ! check_dependencies; then
        echo ""
        echo "Fix the issues above and re-run ./start.sh"
        exit 1
    else
        echo ""
        echo "All dependencies look OK — the error may be in the app itself. Check the logs above."
        exit 1
    fi
fi
