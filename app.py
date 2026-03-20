import os
import time
import tempfile
import logging

import torch
import whisper
from flask import Flask, jsonify, render_template, request
from deep_translator import GoogleTranslator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload limit

# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = os.environ.get("WHISPER_MODEL", "turbo")

logger.info("Loading Whisper model '%s' on device '%s' …", MODEL_NAME, DEVICE)
model = whisper.load_model(MODEL_NAME, device=DEVICE)
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
        transcribe_kwargs = {"task": "transcribe"}
        if language and language != "auto":
            transcribe_kwargs["language"] = language
        result = model.transcribe(tmp_path, **transcribe_kwargs)
        elapsed = time.perf_counter() - start

        detected_language = result.get("language", "unknown")
        text = result.get("text", "").strip()

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
