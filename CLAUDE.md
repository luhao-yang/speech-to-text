# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

```bash
source .venv/bin/activate
python app.py
```

The server starts on `http://0.0.0.0:5000`. First run downloads the Whisper model weights (~1.5 GB) to `~/.cache/whisper/`.

To use a different Whisper model:
```bash
WHISPER_MODEL=large-v3 python app.py
```

## Architecture

Single-file Flask backend (`app.py`) + single-page frontend (`templates/index.html`).

**Startup flow:** Whisper model is loaded into memory at process start (not per-request). Device is auto-detected — CUDA if available, otherwise CPU.

**Request flow:**
1. Browser captures audio via `MediaRecorder` API → WebM/Opus blob
2. `POST /transcribe` — saves blob to `tempfile`, runs `model.transcribe()`, deletes temp file, returns `{text, language, elapsed}`
3. `POST /translate` (optional) — calls `GoogleTranslator` (requires internet), returns `{translated}`

**Key globals in `app.py`:**
- `DEVICE` — `"cuda"` or `"cpu"`, set at import time
- `MODEL_NAME` — from `WHISPER_MODEL` env var, default `"turbo"`
- `model` — the loaded Whisper model instance (module-level singleton)
- `TRANSLATION_TARGETS` — dict of language codes shown in the translation dropdown
- `DICTATION_LANGUAGES` — dict of language codes for the input language selector (passed to Whisper's `language` param; `"auto"` means let Whisper detect)

## Dependencies

Install into the `.venv` virtual environment:
```bash
pip install -r requirements.txt
```

For GPU acceleration, replace the CPU PyTorch build:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121  # adjust for your CUDA version
```

System requirement: `ffmpeg` must be installed (`sudo apt install ffmpeg`) — Whisper uses it to decode audio.
