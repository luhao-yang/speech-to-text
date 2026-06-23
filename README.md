# Speech-to-Text

A local, self-hosted speech-to-text web application powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (a fast CTranslate2 reimplementation of OpenAI Whisper) and [Flask](https://flask.palletsprojects.com/).  
Dictate in **real time** and watch the transcript appear as you speak, or record/upload a clip for batch transcription. See the detected language, copy the result, and optionally translate it into 20+ languages — all running entirely on your own machine.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Running the App](#running-the-app)
7. [Using the App](#using-the-app)
8. [Configuration](#configuration)
9. [API Reference](#api-reference)
10. [Whisper Model Comparison](#whisper-model-comparison)
11. [Troubleshooting](#troubleshooting)

---

## How It Works

There are two transcription modes:

**⚡ Live dictation (default)** — low-latency streaming over a WebSocket:

```
Browser (microphone)                          Flask server (app.py)
      │                                              │
      │  16 kHz AudioContext + AudioWorklet          │
      │  emit raw int16 PCM frames                   │
      │ ───────────  ws://…/ws/stream  ───────────►  │
      │                                     OnlineTranscriber (streaming.py)
      │                                     re-transcribes a growing buffer,
      │                                     commits agreed words (LocalAgreement-2)
      │  ◄──── {"type":"partial", …} live preview ──┤
      │  ◄──── {"type":"final",   …} committed text ┤
      ▼
   Committed text + greyed live tail, updating as you speak
```

**📁 Record & upload (batch)** — record a whole clip, then transcribe:

```
Browser  ──MediaRecorder (WebM/Opus)──►  POST /transcribe  ──►  {text, language, elapsed}
                                          (temp file → faster-whisper → delete)
```

Either way, the transcript can optionally be sent to `POST /translate` (Google
Translate via deep-translator, needs internet) and translated into 20+ languages.

### Component breakdown

| Component | Role |
|-----------|------|
| **Flask + flask-sock** (`app.py`) | Web framework; serves the UI, REST endpoints, and the `/ws/stream` WebSocket |
| **faster-whisper** | Local, offline speech recognition (CTranslate2); ~4× faster than `openai-whisper`, supports 99+ languages |
| **streaming.py** | `OnlineTranscriber` + LocalAgreement-2 logic that turns Whisper into a low-latency streaming transcriber |
| **PyTorch** | Provides CUDA device detection; faster-whisper runs on GPU when available |
| **deep-translator** | Thin wrapper around Google Translate; used for the optional translation step |
| **MediaRecorder / AudioWorklet** | Browser APIs that capture microphone audio; no plugins needed |

---

## Features

- ⚡ **Live dictation** – streaming transcription that appears as you speak (~1 s latency), with stable committed text and a live preview tail
- 🎙️ **Batch mode** – record (or upload) a whole clip and transcribe it in one shot
- 🔤 **faster-whisper** – `turbo` model by default; ~4× faster and lower VRAM than `openai-whisper`; all processing is local and offline
- 🌐 **Automatic language detection** – Whisper identifies the spoken language automatically
- 📋 **Copy button** – one-click copy of the transcript to the clipboard
- 🔄 **Translation** – translate the transcript to 20+ languages (requires internet for Google Translate)
- 🩺 **Health check** – `GET /health` for liveness probes
- ⚡ **CUDA support** – automatically uses an NVIDIA GPU when available; falls back to CPU seamlessly

---

## Project Structure

```
speech-to-text/
│
├── app.py                  # Flask application entry point
│   ├── DEVICE              # Auto-detects CUDA or CPU
│   ├── MODEL_NAME          # Whisper model to load (env-configurable)
│   ├── COMPUTE_TYPE        # float16 (GPU) / int8 (CPU), env-configurable
│   ├── TRANSLATION_TARGETS # Dict of supported translation languages
│   ├── GET  /              # Serves index.html
│   ├── GET  /health        # Liveness probe (model + device)
│   ├── POST /transcribe    # Accepts audio, returns transcript JSON (batch)
│   ├── POST /translate     # Accepts text + target lang, returns translation JSON
│   └── WS   /ws/stream     # Live streaming transcription (partial/final frames)
│
├── streaming.py            # Streaming transcription engine
│   ├── HypothesisBuffer    # LocalAgreement-2 word-commit logic
│   └── OnlineTranscriber   # Per-connection buffer + faster-whisper driver
│
├── templates/
│   └── index.html          # Single-page UI (HTML + CSS + vanilla JS)
│       ├── Mode toggle          (Live dictation / Record & upload)
│       ├── Transcript area      (committed text + live preview + copy button)
│       ├── Meta badges          (detected language, elapsed time)
│       └── Translation section  (language dropdown + output area)
│
├── dictate_client.py       # Native push-to-talk dictation client (runs Windows-side)
├── dictate.bat             # Launcher for the client (Windows / cmd)
├── dictate.sh              # Launcher for the client (Git Bash)
│
├── requirements.txt        # Server dependencies (pip)
├── requirements-client.txt # Native dictation client dependencies (Windows Python)
└── README.md               # This file
```

---

## Prerequisites

### System packages

| Package | Why it's needed |
|---------|----------------|
| **Python 3.9+** | Runtime for Flask and Whisper |
| **ffmpeg** | Whisper uses ffmpeg internally to decode audio files |
| **Git** | Cloning the repository |
| **CUDA toolkit** *(optional)* | Enables GPU acceleration (NVIDIA GPUs only) |

### Hardware

- **CPU-only**: any modern x86-64 CPU works; transcription will be slower
- **GPU (recommended)**: NVIDIA GPU with CUDA support dramatically speeds up transcription  
  (e.g. `turbo` model: ~10× real-time on CPU vs ~80× real-time on a modern GPU)

---

## Installation

These steps target **WSL 2 (Ubuntu 22.04 / 24.04)** but work on any Ubuntu/Debian system.

### Step 1 — Install system dependencies

```bash
sudo apt update && sudo apt install -y python3 python3-pip python3-venv ffmpeg git
```

Verify ffmpeg is installed:

```bash
ffmpeg -version
```

### Step 2 — Clone the repository

```bash
git clone https://github.com/luhao-yang/speech-to-text.git
cd speech-to-text
```

### Step 3 — Create a Python virtual environment

Using a virtual environment keeps project dependencies isolated from your system Python.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt will change to show `(.venv)` when the environment is active.  
To deactivate later: `deactivate`

### Step 4 — Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:

| Package | Version | Purpose |
|---------|---------|---------|
| `flask` | 3.1.3 | Web framework / HTTP server |
| `flask-sock` | 0.7.0 | WebSocket support for the live streaming endpoint |
| `faster-whisper` | 1.2.1 | Speech-to-text model (CTranslate2) |
| `deep-translator` | 1.11.4 | Translation via Google Translate |
| `torch` | ≥ 2.0 | PyTorch (CUDA device detection) |
| `numpy` | 2.4.3 | Numerical arrays |
| `werkzeug` | 3.1.6 | WSGI utilities (Flask dependency) |

### Step 5 — (GPU only) Install CUDA-enabled PyTorch

If you have an NVIDIA GPU and the CUDA toolkit installed, replace the CPU-only PyTorch build with a CUDA-enabled one for much faster transcription.

First, check your CUDA version:

```bash
nvcc --version
# or
nvidia-smi
```

Then install the matching PyTorch build (see https://pytorch.org/get-started/locally/ for all options):

```bash
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU is detected:

```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

## Running the App

```bash
# Make sure the virtual environment is active
source .venv/bin/activate

# Start the Flask server
python app.py
```

Expected output:

```
INFO:app:Loading Whisper model 'turbo' on device 'cuda' (compute_type=float16) …
INFO:app:Whisper model loaded.
 * Running on http://0.0.0.0:5000
```

> **First run only:** faster-whisper will automatically download the `turbo` model weights (~1.5 GB) from Hugging Face and cache them in `~/.cache/huggingface/`. Subsequent starts are instant.

Open your browser and navigate to:

```
http://localhost:5000
```

> **WSL note:** If you're running WSL and want to access the app from your Windows browser, use `http://localhost:5000` — WSL 2 automatically forwards ports from the Linux VM to Windows.

To stop the server: press `Ctrl+C`

---

## Using the App

### Live dictation (default)

1. Open **http://localhost:5000** in your browser.
2. Make sure **⚡ Live dictation** is selected in the mode toggle.
3. Click **Start Recording** and speak.
4. The transcript streams in as you talk: committed words appear in white, the
   live (still-changing) preview appears greyed at the end.
5. Click **Stop** to finish — the final words are flushed and committed.

### Record & upload (batch)

1. Select **📁 Record & upload** in the mode toggle.
2. Click **Start Recording**, speak, then **Stop Recording**.
3. The whole clip is sent to the server; a spinner shows while it transcribes. Once complete:
   - The **transcript** appears in the text area.
   - The **detected language** badge updates (e.g. `en`, `zh`, `fr`).
   - The **elapsed time** badge shows how many seconds the transcription took.

### Native push-to-talk dictation (no browser)

`dictate_client.py` turns the app into a system-wide dictation tool: hold a
hotkey, speak, release, and the transcript is **typed at your cursor** in
whatever application is focused — no browser, no copy/paste.

It's a thin client over the same `/ws/stream` WebSocket the web UI uses, so the
server (`python app.py`) runs unchanged.

> **WSL users:** run the client with **Windows** Python (e.g. from Git Bash or
> `cmd`), *not* inside WSL. Global hotkeys and "type at the cursor" only work
> from the OS that owns your keyboard and windows. The server can stay in WSL —
> the client reaches it over `localhost`, just like the browser.

1. Install the client dependencies with your Windows Python:
   ```bash
   pip install -r requirements-client.txt
   ```
2. Make sure the server is running (`python app.py`, in WSL is fine).
3. Launch the client:
   - **Git Bash:** `./dictate.sh`
   - **Windows:** double-click `dictate.bat` (or run it from `cmd`)
4. **Hold F9**, speak, then **release** — the text appears at your cursor.
   Press **Ctrl+C** in the client window to quit.

Configurable constants at the top of `dictate_client.py`:

| Constant | Default | Purpose |
|----------|---------|---------|
| `HOTKEY` | `F9` | Push-to-talk key (held while speaking). |
| `LANGUAGE` | `"en"` | ISO code (e.g. `"en"`) or `"auto"` to detect. Auto-detect misfires on short clips, so pin it if you can. |
| `TRAILING_SPACE` | `True` | Append a space after each dictation. |
| `AUTO_GAIN` | `True` | Adaptively normalize mic level toward a healthy volume (the browser does this for free; raw capture doesn't). |
| `MIC_GAIN` | `1.0` | Extra manual gain multiplier on top of `AUTO_GAIN`. Raise it (e.g. `2.0`) if dictation is still too quiet/inaccurate. |
| `SERVER_URL` | `ws://localhost:5000/ws/stream` | Where the server is. |

> **Accuracy note:** the browser applies noise suppression and auto-gain to the
> mic by default, which the raw native capture can't fully match. `AUTO_GAIN`
> closes most of the gap; if accuracy is still poor, try raising `MIC_GAIN` or
> moving closer to the mic.

> Text is injected after you release the key (the server runs a final pass), so
> it feels like "speak → release → text lands" rather than word-by-word.
> Keystroke simulation occasionally needs the terminal run as administrator to
> type into elevated apps.

### Copy to clipboard

- Click the **Copy** button (bottom-right of the transcript box) to copy the full transcript.  
  The button briefly turns green to confirm the copy.

### Translation

1. After transcription, choose a target language from the dropdown (e.g. *Chinese (Simplified)*).
2. Click **Translate**.
3. The translated text appears in the box below the dropdown.

> Translation requires an internet connection (uses Google Translate via `deep-translator`).

---

## Configuration

The app is configured via environment variables set before running `python app.py`.

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `turbo` | Whisper model to load. See [model comparison](#whisper-model-comparison) below. |
| `WHISPER_COMPUTE_TYPE` | `float16` (GPU) / `int8` (CPU) | CTranslate2 precision. Options include `float16`, `int8_float16`, `int8`. Lower precision = faster / less VRAM, slightly less accurate. |
| `STT_VERBOSE` | `1` (on) | INFO-level logging, including the dev server's per-request access logs. Set to `0` to quiet logs down to WARNING. |

### Examples

```bash
# Use the large-v3 model for maximum accuracy
WHISPER_MODEL=large-v3 python app.py

# Use the tiny model for minimum memory usage / fastest speed
WHISPER_MODEL=tiny python app.py
```

---

## API Reference

The Flask server exposes the following endpoints consumed by the frontend.

### `POST /transcribe`

Accepts an audio file, runs Whisper, and returns the transcript (batch mode).

**Request** — `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `audio` | file | Audio recording (WebM, OGG, WAV, MP3, etc.) |

**Response** — `application/json`

```json
{
  "text": "Hello, this is a test.",
  "language": "en",
  "elapsed": 1.83
}
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Transcribed text |
| `language` | string | ISO 639-1 language code detected by Whisper |
| `elapsed` | float | Time in seconds taken by the transcription |

**Error response**

```json
{ "error": "No audio file provided." }
```

---

### `POST /translate`

Translates a text string into the requested language.

**Request** — `application/json`

```json
{
  "text": "Hello, this is a test.",
  "target": "zh-CN"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Text to translate |
| `target` | string | Target language code (see supported list below) |

**Supported target language codes**

| Code | Language | Code | Language |
|------|----------|------|----------|
| `en` | English | `ko` | Korean |
| `zh-CN` | Chinese (Simplified) | `pt` | Portuguese |
| `zh-TW` | Chinese (Traditional) | `ru` | Russian |
| `es` | Spanish | `ar` | Arabic |
| `fr` | French | `hi` | Hindi |
| `de` | German | `it` | Italian |
| `ja` | Japanese | `nl` | Dutch |
| `pl` | Polish | `sv` | Swedish |
| `tr` | Turkish | `vi` | Vietnamese |
| `th` | Thai | `id` | Indonesian |

**Response** — `application/json`

```json
{ "translated": "你好，这是一个测试。" }
```

---

### `WS /ws/stream`

Live streaming transcription over a WebSocket. Used by the **Live dictation** mode.

**Client → server**

1. (Optional) one text frame selecting the language: `{"language": "en"}` — omit or
   use `"auto"` to let Whisper detect it.
2. Binary frames of raw **little-endian int16 PCM, mono, 16 kHz**.
3. A text frame `{"action": "stop"}` to flush the remaining audio as final.

**Server → client** — JSON text frames:

```json
{ "type": "partial", "text": "live preview that may still change" }
{ "type": "final",   "text": "newly committed words", "committed": "full text so far" }
{ "type": "final",   "text": "…", "committed": "…", "done": true }
{ "type": "error",   "message": "…" }
```

`partial` frames are a live preview that may be rewritten; `final` frames are
committed (stable) via the LocalAgreement-2 policy. The frame with `"done": true`
marks the end of the stream.

---

### `GET /health`

Liveness probe.

**Response** — `application/json`

```json
{ "status": "ok", "model": "turbo", "device": "cuda" }
```

---

## Whisper Model Comparison

Choose a model based on your hardware and accuracy requirements.

| Model | Size | VRAM | Relative speed | Accuracy |
|-------|------|------|----------------|----------|
| `tiny` | 39 M params | ~1 GB | ~32× real-time | ★★☆☆☆ |
| `base` | 74 M params | ~1 GB | ~16× real-time | ★★★☆☆ |
| `small` | 244 M params | ~2 GB | ~6× real-time | ★★★★☆ |
| `medium` | 769 M params | ~5 GB | ~2× real-time | ★★★★☆ |
| `large-v3` | 1550 M params | ~10 GB | ~1× real-time | ★★★★★ |
| `turbo` *(default)* | 809 M params | ~6 GB | ~8× real-time | ★★★★★ |

> **`turbo`** is the recommended default — it offers near-`large` accuracy at much faster speed.  
> Use `tiny` or `base` if you have limited RAM/VRAM and need a quick response.

---

## Troubleshooting

### `ffmpeg not found`
```bash
sudo apt install -y ffmpeg
```

### `CUDA available: False` despite having a GPU
- Ensure the NVIDIA driver is installed: `nvidia-smi`
- Install a CUDA-enabled PyTorch build (see [Step 5](#step-5--gpu-only-install-cuda-enabled-pytorch))
- In WSL, ensure you are using WSL 2 and have the [CUDA WSL-Ubuntu driver](https://developer.nvidia.com/cuda/wsl) installed on Windows

### Browser cannot access the microphone
- Ensure you are using `http://localhost:5000` (not an IP address) — browsers only grant microphone access on `localhost` or HTTPS origins
- Check browser permissions: click the lock/info icon in the address bar → allow microphone

### First transcription is very slow
- The Whisper model is loaded into memory at server start, but the first inference warms up CUDA kernels. Subsequent requests will be faster.

### `OSError: [Errno 28] No space left on device`
- The model cache lives in `~/.cache/huggingface/`. Free up disk space or point the cache elsewhere:
  ```bash
  export HF_HOME=/path/to/large/disk
  python app.py
  ```

### Live dictation shows no text
- Live mode needs microphone access on a `localhost` or HTTPS origin (see the microphone note above).
- Check the browser console for WebSocket errors and the server log for `Streaming error`.
- The first streaming pass warms up CUDA kernels and can lag a second or two; it settles quickly.

### Port 5000 already in use
```bash
# Find the process using port 5000
sudo lsof -i :5000
# Kill it, then restart
kill <PID>
python app.py
```
