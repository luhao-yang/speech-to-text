import os
import json
import time
import tempfile
import logging
import threading

import torch
from faster_whisper import WhisperModel
from flask import Flask, jsonify, render_template, request
from flask_sock import Sock
from deep_translator import GoogleTranslator

from streaming import SAMPLE_RATE, OnlineTranscriber, pcm16_to_float32

# INFO-level logs (including the dev server's per-request access logs) are on by
# default. Set STT_VERBOSE=0 to quiet them down to WARNING.
VERBOSE = os.environ.get("STT_VERBOSE", "1").lower() in ("1", "true", "yes", "on")

logging.basicConfig(level=logging.INFO if VERBOSE else logging.WARNING)
logger = logging.getLogger(__name__)
# Keep our own intentional startup/error messages regardless of verbosity.
logger.setLevel(logging.INFO)
if not VERBOSE:
    # Silence the dev server's per-request access log spam.
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload limit
sock = Sock(app)

# flask-sock / simple-websocket hard-code the permessage-deflate extension, but
# compressed frames get mangled by the Werkzeug dev server and browsers reject
# them ("Invalid frame header"). Our JSON frames are tiny, so we decline
# compression by making the server-side extension refuse to negotiate.
import simple_websocket.ws as _sws  # noqa: E402
from wsproto.extensions import PerMessageDeflate as _PerMessageDeflate  # noqa: E402


class _NoCompression(_PerMessageDeflate):
    def accept(self, offer):  # always decline -> no compression negotiated
        return None


_sws.PerMessageDeflate = _NoCompression

# faster-whisper's transcribe is not safe to call concurrently from multiple
# WebSocket connections, so serialise access to the shared model.
MODEL_LOCK = threading.Lock()

# Process a streaming window once this much new audio has arrived (seconds).
STREAM_MIN_CHUNK_SECONDS = 1.0

# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# float16 on GPU, int8 on CPU — good speed/quality defaults; overridable via env.
COMPUTE_TYPE = os.environ.get(
    "WHISPER_COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8"
)
MODEL_NAME = os.environ.get("WHISPER_MODEL", "turbo")

logger.info(
    "Loading Whisper model '%s' on device '%s' (compute_type=%s) …",
    MODEL_NAME,
    DEVICE,
    COMPUTE_TYPE,
)
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
logger.info("Whisper model loaded.")

# ---------------------------------------------------------------------------
# Translation helpers
# ---------------------------------------------------------------------------

# Languages supported by deep-translator / Google Translate that are also
# commonly detected by Whisper.  The key is the ISO 639-1 code used by
# deep-translator; the value is the human-readable label shown in the UI.
TRANSLATION_TARGETS = {
    "en": "English",
    "zh-CN": "Chinese (Simplified)",
    "zh-TW": "Chinese (Traditional)",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "sv": "Swedish",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
}

DICTATION_LANGUAGES = {
    "auto": "Auto-detect",
    "en": "English",
    "zh": "Chinese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "sv": "Swedish",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
}

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html", translation_targets=TRANSLATION_TARGETS, dictation_languages=DICTATION_LANGUAGES)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """Receive an audio file, run Whisper, and return JSON results."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided."}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    # Save to a temp file so Whisper can read it
    suffix = os.path.splitext(audio_file.filename)[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        audio_file.save(tmp_path)

    language = request.form.get("language", "auto")

    try:
        start = time.perf_counter()
        transcribe_kwargs = {"task": "transcribe", "vad_filter": True}
        if language and language != "auto":
            transcribe_kwargs["language"] = language
        with MODEL_LOCK:
            segments, info = model.transcribe(tmp_path, **transcribe_kwargs)
            # segments is a generator — iterate to run the transcription.
            text = "".join(segment.text for segment in segments).strip()
        elapsed = time.perf_counter() - start

        detected_language = info.language or "unknown"

        return jsonify(
            {
                "text": text,
                "language": detected_language,
                "elapsed": round(elapsed, 2),
            }
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Transcription error")
        return jsonify({"error": str(exc)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.route("/translate", methods=["POST"])
def translate():
    """Translate text to the requested target language."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    target = data.get("target", "en")

    if not text:
        return jsonify({"error": "No text provided."}), 400

    if target not in TRANSLATION_TARGETS:
        return jsonify({"error": f"Unsupported target language: {target}"}), 400

    try:
        translated = GoogleTranslator(source="auto", target=target).translate(text)
        return jsonify({"translated": translated})
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Translation error")
        return jsonify({"error": str(exc)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME, "device": DEVICE})


# ---------------------------------------------------------------------------
# Real-time streaming transcription (WebSocket)
# ---------------------------------------------------------------------------
#
# Protocol (all server->client messages are JSON text frames):
#   client -> server:
#     - first text frame (optional): {"language": "en"}  (omit / "auto" to detect)
#     - binary frames: raw little-endian int16 PCM, mono, 16 kHz
#     - text frame {"action": "stop"}: flush remaining audio as final
#   server -> client:
#     - {"type": "partial", "text": "..."}   live preview, may change
#     - {"type": "final",   "text": "...", "committed": "<full text so far>"}
#     - {"type": "error",   "message": "..."}


@sock.route("/ws/stream")
def stream(ws):
    transcriber = OnlineTranscriber(model, MODEL_LOCK)
    pending_samples = 0
    min_samples = int(STREAM_MIN_CHUNK_SECONDS * SAMPLE_RATE)

    def emit(committed_words, partial):
        if committed_words:
            ws.send(json.dumps({
                "type": "final",
                "text": "".join(w[2] for w in committed_words).strip(),
                "committed": transcriber.committed_text,
            }))
        ws.send(json.dumps({"type": "partial", "text": partial}))

    try:
        while True:
            data = ws.receive()
            if data is None:
                break

            # Control / config frames arrive as text (JSON).
            if isinstance(data, str):
                try:
                    msg = json.loads(data)
                except ValueError:
                    continue
                if "language" in msg:
                    lang = msg["language"]
                    transcriber.language = None if lang in (None, "", "auto") else lang
                if msg.get("action") == "stop":
                    break
                continue

            # Binary frame = PCM audio.
            transcriber.add_audio(pcm16_to_float32(data))
            pending_samples += len(data) // 2
            if pending_samples >= min_samples:
                pending_samples = 0
                committed_words, partial = transcriber.process()
                emit(committed_words, partial)

        # Connection ending or stop requested: do a final pass + flush.
        committed_words, partial = transcriber.process()
        emit(committed_words, partial)
        remaining = transcriber.finish()
        ws.send(json.dumps({
            "type": "final",
            "text": "".join(w[2] for w in remaining).strip(),
            "committed": transcriber.committed_text,
            "done": True,
        }))
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Streaming error")
        try:
            ws.send(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:  # pylint: disable=broad-except
            pass


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
