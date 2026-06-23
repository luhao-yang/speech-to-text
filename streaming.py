"""Low-latency streaming transcription on top of faster-whisper.

Whisper is not a streaming model — it transcribes 30-second windows.  To get
stable, low-latency output we use the *LocalAgreement-2* policy popularised by
``whisper_streaming``: we repeatedly re-transcribe a growing audio buffer and
only *commit* the leading words that two consecutive hypotheses agree on.  The
agreed prefix is emitted as "final" (it won't change); the rest is emitted as
"partial" (a live preview that may still be rewritten).

The buffer is trimmed once committed audio gets long, so cost stays bounded.
"""

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class HypothesisBuffer:
    """Holds committed words and applies LocalAgreement-2 to new hypotheses."""

    def __init__(self):
        self.committed_in_buffer = []  # list of (start, end, word)
        self.buffer = []               # previous hypothesis tail (uncommitted)
        self.new = []                  # latest hypothesis
        self.last_committed_time = 0.0

    def insert(self, new_words, offset):
        """Add a fresh hypothesis (word timestamps shifted by ``offset``)."""
        new_words = [(a + offset, b + offset, t) for a, b, t in new_words]
        # Only consider words at or after what we've already committed.
        self.new = [(a, b, t) for a, b, t in new_words
                    if a > self.last_committed_time - 0.1]

        if not self.new:
            return
        a, _, _ = self.new[0]
        if abs(a - self.last_committed_time) < 1 and self.committed_in_buffer:
            # Drop a leading n-gram that repeats words already committed,
            # which happens because windows overlap.
            cn = len(self.committed_in_buffer)
            nn = len(self.new)
            for i in range(1, min(cn, nn, 5) + 1):
                c = " ".join(self.committed_in_buffer[-j][2]
                             for j in range(i, 0, -1))
                tail = " ".join(self.new[j - 1][2] for j in range(1, i + 1))
                if c == tail:
                    for _ in range(i):
                        self.new.pop(0)
                    break

    def flush(self):
        """Commit the longest prefix shared by the last two hypotheses."""
        committed = []
        while self.new and self.buffer:
            if self.new[0][2] == self.buffer[0][2]:
                committed.append(self.new[0])
                self.last_committed_time = self.new[0][1]
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.committed_in_buffer.extend(committed)
        return committed

    def pop_committed(self, time):
        """Forget committed words that end before ``time`` (buffer trimming)."""
        while self.committed_in_buffer and self.committed_in_buffer[0][1] <= time:
            self.committed_in_buffer.pop(0)


class OnlineTranscriber:
    """Drives a faster-whisper model over a streamed audio buffer.

    One instance per WebSocket connection.  Not shared between connections.
    """

    # Re-transcribe once the buffer grows this long (seconds).
    BUFFER_TRIM_SECONDS = 18.0

    def __init__(self, model, model_lock, language=None):
        self.model = model
        self.model_lock = model_lock
        self.language = language  # None -> auto-detect
        self.audio = np.zeros(0, dtype=np.float32)
        self.buffer_time_offset = 0.0
        self.hypothesis = HypothesisBuffer()
        self.committed = []  # list of (start, end, word)

    def add_audio(self, pcm_float32):
        self.audio = np.append(self.audio, pcm_float32)

    @property
    def committed_text(self):
        return "".join(w[2] for w in self.committed).strip()

    def _transcribe(self, init_prompt):
        with self.model_lock:
            segments, _ = self.model.transcribe(
                self.audio,
                language=self.language,
                task="transcribe",
                beam_size=5,
                word_timestamps=True,
                condition_on_previous_text=True,
                initial_prompt=init_prompt or None,
                vad_filter=True,
            )
            words = []
            for seg in segments:
                if seg.words:
                    words.extend((w.start, w.end, w.word) for w in seg.words)
            return words

    def process(self):
        """Run one transcription pass. Returns (committed_words, partial_text)."""
        # Prompt the model with recently committed text for continuity.
        init_prompt = "".join(w[2] for w in self.committed[-40:])
        words = self._transcribe(init_prompt)

        self.hypothesis.insert(words, self.buffer_time_offset)
        newly_committed = self.hypothesis.flush()
        self.committed.extend(newly_committed)

        self._maybe_trim()

        partial = "".join(w[2] for w in self.hypothesis.buffer).strip()
        return newly_committed, partial

    def _maybe_trim(self):
        """Drop already-committed audio once the buffer gets long."""
        buffered_seconds = len(self.audio) / SAMPLE_RATE
        if buffered_seconds <= self.BUFFER_TRIM_SECONDS or not self.committed:
            return
        cut_time = self.committed[-1][1]
        cut_samples = int((cut_time - self.buffer_time_offset) * SAMPLE_RATE)
        if cut_samples <= 0:
            return
        self.audio = self.audio[cut_samples:]
        self.buffer_time_offset = cut_time
        self.hypothesis.pop_committed(cut_time)

    def finish(self):
        """Flush any remaining buffered hypothesis as final text."""
        remaining = self.hypothesis.buffer
        self.committed.extend(remaining)
        self.hypothesis.buffer = []
        return remaining


def pcm16_to_float32(data: bytes) -> np.ndarray:
    """Convert little-endian int16 PCM bytes to a float32 array in [-1, 1]."""
    if len(data) % 2:
        data = data[:-1]  # guard against a torn frame
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
