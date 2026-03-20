# Speech-to-Text

A local speech-to-text web application powered by [OpenAI Whisper](https://github.com/openai/whisper) and [Flask](https://flask.palletsprojects.com/).  
Record audio directly in your browser, transcribe it with Whisper, see the detected language and elapsed time, and optionally translate the result into 20+ languages.

## Features

- 🎙️ **Browser recording** – click once to record, click again to stop
- 🔤 **Transcription** – Whisper `turbo` model by default (configurable)
- 🌐 **Language detection** – displays the language Whisper detected
- ⏱️ **Elapsed time** – shows how long the STT job took
- 📋 **Copy button** – one-click copy of the transcript
- 🔄 **Translation** – translate to 20+ languages via Google Translate (deep-translator)
- ⚡ **CUDA support** – automatically uses GPU when available; falls back to CPU

## Requirements

- Python 3.9+
- [ffmpeg](https://ffmpeg.org/) (required by Whisper for audio decoding)
- CUDA-compatible GPU + CUDA toolkit *(optional, for GPU acceleration)*

## Setup (WSL / Ubuntu)

```bash
# 1. Install ffmpeg
sudo apt update && sudo apt install -y ffmpeg

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. (GPU only) Install a CUDA-enabled PyTorch build — see https://pytorch.org/get-started/locally/
#    Example for CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Running

```bash
python app.py
```

Open your browser at **http://localhost:5000**.

> **First run:** Whisper will download the `turbo` model (~1.5 GB) on first start-up.

## Configuration

| Environment variable | Default  | Description                              |
|----------------------|----------|------------------------------------------|
| `WHISPER_MODEL`      | `turbo`  | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`, `turbo`) |

```bash
WHISPER_MODEL=large python app.py
```

## Usage

1. Click **Start Recording** and speak.
2. Click **Stop Recording** — transcription begins automatically.
3. The transcript, detected language, and elapsed time appear on screen.
4. Click **Copy** to copy the transcript to your clipboard.
5. Choose a target language from the dropdown and click **Translate**.

## Project Structure

```
speech-to-text/
├── app.py              # Flask application & Whisper integration
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Single-page UI
└── README.md
```
