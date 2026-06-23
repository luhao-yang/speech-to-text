# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

```bash
source .venv/bin/activate
python app.py
```

The server starts on `http://0.0.0.0:5000`. First run downloads the model weights
(faster-whisper format, ~1.5 GB for `turbo`/`large-v3`) to the Hugging Face cache
(`~/.cache/huggingface/`).

To use a different Whisper model or precision:
```bash
WHISPER_MODEL=large-v3 python app.py
WHISPER_COMPUTE_TYPE=int8_float16 python app.py   # trade accuracy for speed/VRAM
```

INFO-level and per-request logs are on by default. Set `STT_VERBOSE=0` to quiet
logging down to WARNING (and suppress werkzeug access logs).

## Architecture

Flask backend (`app.py`) + streaming helper (`streaming.py`) + single-page frontend
(`templates/index.html`). Inference runs on **faster-whisper** (CTranslate2), not the
original `openai-whisper` package.

**Startup flow:** the model is loaded into memory once at process start (not
per-request). Device is auto-detected — CUDA if available, otherwise CPU. Compute
type defaults to `float16` on GPU and `int8` on CPU.

**Two transcription modes:**

1. **Batch** (`POST /transcribe`) — browser records a full clip via `MediaRecorder`,
   uploads the blob, server saves it to a `tempfile`, runs `model.transcribe(...)`,
   deletes the temp file, returns `{text, language, elapsed}`. Best for files.

2. **Live** (`WebSocket /ws/stream`) — browser captures mic audio in a 16 kHz
   `AudioContext`, an `AudioWorklet` emits int16 PCM frames over the socket. The
   server feeds them to an `OnlineTranscriber` (`streaming.py`) that re-transcribes a
   growing buffer and commits words via **LocalAgreement-2**: only the leading words
   two consecutive hypotheses agree on are emitted as `final`; the rest stream as
   `partial` (live preview). This is the low-latency dictation path.

`POST /translate` (optional) — calls `GoogleTranslator` (requires internet), returns
`{translated}`. `GET /health` — liveness probe returning model/device.

**WebSocket protocol (`/ws/stream`):** client sends an optional first text frame
`{"language": "en"}`, then binary int16-PCM/16 kHz/mono frames, then `{"action":
"stop"}` to flush. Server replies with JSON text frames `{"type": "partial"|"final"|
"error", ...}`; the final frame carries `"done": true` and the full `committed` text.

**Key globals in `app.py`:**
- `DEVICE` — `"cuda"` or `"cpu"`, set at import time
- `COMPUTE_TYPE` — from `WHISPER_COMPUTE_TYPE`, defaults per device
- `MODEL_NAME` — from `WHISPER_MODEL` env var, default `"turbo"`
- `model` — the loaded `faster_whisper.WhisperModel` (module-level singleton)
- `MODEL_LOCK` — serialises model access; faster-whisper isn't safe to call
  concurrently from multiple WebSocket connections
- `STREAM_MIN_CHUNK_SECONDS` — how much new audio to buffer before each streaming pass
- `TRANSLATION_TARGETS` / `DICTATION_LANGUAGES` — language dropdown contents

**Key pieces in `streaming.py`:**
- `HypothesisBuffer` — the LocalAgreement-2 commit logic over word timestamps
- `OnlineTranscriber` — one per WebSocket connection; owns the audio buffer, runs
  `model.transcribe(..., word_timestamps=True, vad_filter=True)`, trims committed audio
- `pcm16_to_float32` — converts incoming int16 PCM bytes to the float32 array Whisper wants

## Native dictation client

`dictate_client.py` is a standalone push-to-talk client (not part of the Flask
process). It reuses the `/ws/stream` WebSocket: hold a hotkey (F9), it captures
the mic via `sounddevice`, streams int16/16 kHz PCM to the server, waits for the
`done: true` frame, and types the committed text at the cursor via `pynput`.
Final-text-only — nothing is injected until you release the key.

Must run with **Windows** Python (global hotkeys + keystroke injection need the
OS that owns the keyboard); the server can stay in WSL and is reached over
`localhost`. Launchers: `dictate.bat` (cmd) and `dictate.sh` (Git Bash). Client
deps live in `requirements-client.txt` (`sounddevice`, `websocket-client`,
`pynput`), separate from the server's `requirements.txt`.

## Dependencies

Install into the `.venv` virtual environment:
```bash
pip install -r requirements.txt
```

For GPU acceleration, replace the CPU PyTorch build:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121  # adjust for your CUDA version
```

System requirement: `ffmpeg` must be installed (`sudo apt install ffmpeg`) — faster-whisper
uses it to decode uploaded audio files (the live path sends raw PCM and needs no ffmpeg).
