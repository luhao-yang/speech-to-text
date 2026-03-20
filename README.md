# Speech-to-Text

A local, self-hosted speech-to-text web application powered by [OpenAI Whisper](https://github.com/openai/whisper) and [Flask](https://flask.palletsprojects.com/).  
Record audio directly in your browser, transcribe it with Whisper, see the detected language and elapsed time, and optionally translate the result into 20+ languages — all running entirely on your own machine.

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

```
Browser (microphone)
      │
      │  1. User clicks "Start Recording"
      │     MediaRecorder captures audio as WebM/Opus blob
      │
      ▼
Flask server  (app.py)
      │
      │  2. POST /transcribe
      │     Audio blob saved to a temporary file
      │     OpenAI Whisper loads the file and runs the STT model
      │     Detected language + transcript + elapsed time returned as JSON
      │
      │  3. (Optional) POST /translate
      │     Transcript text sent to Google Translate via deep-translator
      │     Translated text returned as JSON
      │
      ▼
Browser UI
      │
      └─ Displays transcript, detected language, elapsed time
         Copy-to-clipboard button
         Translated text (if requested)
```

### Component breakdown

| Component | Role |
|-----------|------|
| **Flask** (`app.py`) | Python web framework; serves the UI and exposes REST endpoints |
| **OpenAI Whisper** | Local, offline speech recognition model; supports 99+ languages |
| **PyTorch** | Deep-learning runtime used by Whisper; uses CUDA GPU when available |
| **deep-translator** | Thin wrapper around Google Translate; used for the optional translation step |
| **MediaRecorder API** | Browser API that captures microphone audio; no plugins needed |

---

## Features

- 🎙️ **Browser recording** – click once to record, click again to stop; no extra software needed
- 🔤 **Whisper transcription** – `turbo` model by default; all processing is local and offline
- 🌐 **Automatic language detection** – Whisper identifies the spoken language automatically
- ⏱️ **Elapsed time** – shows how long the transcription job took in seconds
- 📋 **Copy button** – one-click copy of the transcript to the clipboard
- 🔄 **Translation** – translate the transcript to 20+ languages (requires internet for Google Translate)
- ⚡ **CUDA support** – automatically uses an NVIDIA GPU when available; falls back to CPU seamlessly

---

## Project Structure

```
speech-to-text/
│
├── app.py                  # Flask application entry point
│   ├── DEVICE              # Auto-detects CUDA or CPU
│   ├── MODEL_NAME          # Whisper model to load (env-configurable)
│   ├── TRANSLATION_TARGETS # Dict of supported translation languages
│   ├── GET  /              # Serves index.html
│   ├── POST /transcribe    # Accepts audio, returns transcript JSON
│   └── POST /translate     # Accepts text + target lang, returns translation JSON
│
├── templates/
│   └── index.html          # Single-page UI (HTML + CSS + vanilla JS)
│       ├── Recording controls   (Start / Stop button)
│       ├── Transcript area      (result + copy button)
│       ├── Meta badges          (detected language, elapsed time)
│       └── Translation section  (language dropdown + output area)
│
├── requirements.txt        # Python dependencies (pip)
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
| `flask` | ≥ 3.0 | Web framework / HTTP server |
| `openai-whisper` | ≥ 20231117 | Speech-to-text model |
| `deep-translator` | ≥ 1.11.4 | Translation via Google Translate |
| `torch` | ≥ 2.0 | PyTorch (Whisper's ML runtime) |
| `numpy` | ≥ 1.24 | Numerical arrays (Whisper dependency) |
| `werkzeug` | ≥ 3.0 | WSGI utilities (Flask dependency) |

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
INFO:__main__:Loading Whisper model 'turbo' on device 'cuda' …
INFO:__main__:Whisper model loaded.
 * Running on http://0.0.0.0:5000
```

> **First run only:** Whisper will automatically download the `turbo` model weights (~1.5 GB) from the internet and cache them in `~/.cache/whisper/`. Subsequent starts are instant.

Open your browser and navigate to:

```
http://localhost:5000
```

> **WSL note:** If you're running WSL and want to access the app from your Windows browser, use `http://localhost:5000` — WSL 2 automatically forwards ports from the Linux VM to Windows.

To stop the server: press `Ctrl+C`

---

## Using the App

### Transcription

1. Open **http://localhost:5000** in your browser.
2. Click **Start Recording** — the button turns red and pulses to show it is listening.
3. Speak clearly into your microphone.
4. Click **Stop Recording** — the audio is automatically sent to the server for transcription.
5. While processing, a spinner appears. Once complete:
   - The **transcript** appears in the text area.
   - The **detected language** badge updates (e.g. `en`, `zh`, `fr`).
   - The **elapsed time** badge shows how many seconds the transcription took.

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

### Examples

```bash
# Use the large-v3 model for maximum accuracy
WHISPER_MODEL=large-v3 python app.py

# Use the tiny model for minimum memory usage / fastest speed
WHISPER_MODEL=tiny python app.py
```

---

## API Reference

The Flask server exposes two JSON endpoints consumed by the frontend.

### `POST /transcribe`

Accepts an audio file, runs Whisper, and returns the transcript.

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
- The Whisper model cache lives in `~/.cache/whisper/`. Free up disk space or point the cache elsewhere:
  ```bash
  export XDG_CACHE_HOME=/path/to/large/disk
  python app.py
  ```

### Port 5000 already in use
```bash
# Find the process using port 5000
sudo lsof -i :5000
# Kill it, then restart
kill <PID>
python app.py
```
