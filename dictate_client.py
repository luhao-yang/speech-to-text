"""Native push-to-talk dictation client.

Hold a hotkey, speak, release — the transcribed text is typed at the cursor in
whatever application is focused. No browser, no copy/paste.

This is a *thin client* for the streaming server in ``app.py``: it captures the
mic, streams int16/16 kHz/mono PCM to the ``/ws/stream`` WebSocket, waits for the
final committed transcript, and types it via simulated keystrokes.

IMPORTANT (WSL users): run this with **Windows** Python, not inside WSL. Global
hotkeys and "type at the cursor" only work from the OS that owns the keyboard and
the focused windows. The server can stay in WSL — this client reaches it over
``localhost`` just like the browser does.

Setup (in a Windows terminal):
    pip install sounddevice websocket-client pynput
    python dictate_client.py

Then hold F9, speak, and release.
"""

import json
import queue
import threading

import sounddevice as sd
import websocket  # websocket-client
from pynput import keyboard

# --- Config -----------------------------------------------------------------
SERVER_URL = "ws://localhost:5000/ws/stream"
SAMPLE_RATE = 16_000          # server expects 16 kHz mono int16
BLOCKSIZE = 1_600             # 0.1 s of audio per mic callback
LANGUAGE = "auto"            # ISO code (e.g. "en") or "auto" to detect
HOTKEY = keyboard.Key.f9      # hold to dictate
TRAILING_SPACE = True         # append a space after each dictation
# ---------------------------------------------------------------------------

_kb = keyboard.Controller()


class Dictation:
    """One reusable push-to-talk session manager."""

    def __init__(self):
        self._lock = threading.Lock()
        self._active = False
        self._audio_q: "queue.Queue[bytes]" = queue.Queue()
        self._ws = None
        self._stream = None
        self._committed = ""
        self._send_thread = None
        self._recv_thread = None

    # -- mic capture --------------------------------------------------------
    def _mic_cb(self, indata, frames, time_info, status):  # noqa: ARG002
        if status:
            print("audio:", status)
        # RawInputStream hands us a bytes-like buffer of raw int16 PCM.
        self._audio_q.put(bytes(indata))

    def _send_loop(self):
        while self._active:
            try:
                chunk = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._ws.send_binary(chunk)
            except Exception:  # socket closed underneath us
                break

    def _recv_loop(self):
        while True:
            try:
                raw = self._ws.recv()
            except Exception:
                break
            if not raw:
                break
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            # Every final/done frame carries the full transcript so far.
            if msg.get("committed") is not None:
                self._committed = msg["committed"]
            if msg.get("done"):
                break

    # -- lifecycle ----------------------------------------------------------
    def start(self):
        with self._lock:
            if self._active:
                return
            self._active = True
            self._committed = ""
            self._audio_q = queue.Queue()
        try:
            self._ws = websocket.create_connection(SERVER_URL, max_size=None)
            self._ws.send(json.dumps({"language": LANGUAGE}))
        except Exception as exc:
            print(f"could not connect to {SERVER_URL}: {exc}")
            self._active = False
            return

        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self._recv_thread.start()
        self._send_thread.start()

        self._stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE,
            dtype="int16", channels=1, callback=self._mic_cb,
        )
        self._stream.start()
        print("● recording — release to transcribe")

    def stop(self):
        with self._lock:
            if not self._active:
                return
            self._active = False

        # Stop the mic, then flush any audio still queued before signalling stop.
        self._stream.stop()
        self._stream.close()
        self._send_thread.join(timeout=5)
        while True:
            try:
                self._ws.send_binary(self._audio_q.get_nowait())
            except queue.Empty:
                break
            except Exception:
                break

        try:
            self._ws.send(json.dumps({"action": "stop"}))
        except Exception:
            pass
        self._recv_thread.join(timeout=15)
        try:
            self._ws.close()
        except Exception:
            pass

        text = self._committed.strip()
        if text:
            print(f"✎ {text}")
            _kb.type(text + (" " if TRAILING_SPACE else ""))
        else:
            print("… nothing transcribed")


def main():
    dictation = Dictation()
    held = False  # guard against key-repeat firing on_press while held

    def on_press(key):
        nonlocal held
        if key == HOTKEY and not held:
            held = True
            threading.Thread(target=dictation.start, daemon=True).start()

    def on_release(key):
        nonlocal held
        if key == HOTKEY and held:
            held = False
            threading.Thread(target=dictation.stop, daemon=True).start()

    print(f"Push-to-talk dictation ready. Hold {HOTKEY} to dictate. Ctrl+C to quit.")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == "__main__":
    main()
