"""
talktype — push-to-talk transcription.

Hold Ctrl + Mouse-Back to record, release to transcribe. Text appears
at your cursor. Win+Q to quit.

Why this combo: two-handed (Ctrl=left hand, mouse-back=right thumb),
ergonomic for high-frequency use, no conflict with any default Windows
shortcut. Trade-off: in browser, mouse-back may briefly fire navigate-
back before talktype intercepts (selective suppress would need a
WH_MOUSE_LL hook; not implemented).

Requirements: pip install sounddevice openai keyboard pynput numpy
"""

import io
import os
import sys
import wave
import queue
import threading
import logging
import numpy as np
import sounddevice as sd
import keyboard
from pynput import mouse as pmouse
from pynput import keyboard as pkb
from openai import OpenAI

# --- Config ---

# Mouse button to hold for push-to-talk. x1 = back button (most mice
# with a thumb-side back/forward pair). Change to pmouse.Button.x2 for
# the forward button.
RECORD_BUTTON = pmouse.Button.x1
SAMPLE_RATE = 16000
CHANNELS = 1
LANGUAGE = None  # None = auto-detect. Set to "sv", "en" etc. to force a language
SILENCE_THRESHOLD = 300
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_type.log")

# --- Logging ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("talktype")


# --- Audio feedback (piano-like WAV tones via winsound) ---

SOUNDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")
SOUND_ENABLED = True  # set to False to disable all sounds

def _play_wav(name):
    """Play a WAV file from sounds/ dir. Non-blocking, no conflict with recording."""
    if not SOUND_ENABLED:
        return
    def _do():
        try:
            path = os.path.join(SOUNDS_DIR, name)
            if sys.platform == "win32":
                import winsound
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_NODEFAULT)
            else:
                import wave as _wave
                with _wave.open(path) as wf:
                    data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                    sd.play(data.astype(np.float32) / 32768, samplerate=wf.getframerate(), blocking=True)
        except Exception:
            pass
    threading.Thread(target=_do, daemon=True).start()

def beep_start(): _play_wav("start.wav")
def beep_done():  _play_wav("done.wav")
def beep_error(): _play_wav("error.wav")


# --- UI ---

def set_title(text):
    """Set console window title via ctypes (no subprocess, no focus steal)."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleTitleW(text)
        except Exception:
            pass


# --- Audio helpers ---

def audio_to_wav(audio):
    """Convert int16 numpy audio to an in-memory WAV file."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    buf.seek(0)
    buf.name = "recording.wav"
    return buf


def is_silence(audio):
    """True if audio is too quiet to contain speech."""
    return np.max(np.abs(audio)) < SILENCE_THRESHOLD


# --- Transcription ---

def transcribe(client, audio):
    """Send audio to Whisper API, return text."""
    kwargs = dict(model="whisper-1", file=audio_to_wav(audio), response_format="text")
    if LANGUAGE:
        kwargs["language"] = LANGUAGE
    result = client.audio.transcriptions.create(**kwargs)
    return result.strip()


# --- Recording session ---

class RecordingSession:
    """Hold-to-record, release-to-transcribe."""

    def __init__(self, client):
        self._client = client
        self._active_session = None
        self._transcribe_queue = queue.Queue()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._transcribe_worker, daemon=True)
        self._worker.start()

    def start(self):
        with self._lock:
            if self._active_session is not None:
                return

            frames = []

            def on_audio(indata, frame_count, time_info, status):
                if status:
                    log.warning("audio status: %s", status)
                frames.append(indata.copy())

            session = {"frames": frames, "stream": None}
            try:
                session["stream"] = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                    callback=on_audio,
                )
                session["stream"].start()
            except Exception as e:
                if session["stream"] is not None:
                    try:
                        session["stream"].close()
                    except Exception:
                        pass
                log.error("recording start failed: %s", e)
                return

            self._active_session = session

        set_title("talktype [RECORDING]")
        beep_start()
        log.info("recording...")

    def stop(self):
        with self._lock:
            if self._active_session is None:
                return
            session = self._active_session
            self._active_session = None

        stream = session["stream"]
        if stream is not None:
            try:
                stream.stop()
            except Exception as e:
                log.warning("stream stop failed: %s", e)
            try:
                stream.close()
            except Exception as e:
                log.warning("stream close failed: %s", e)

        frames = session["frames"]
        if not frames:
            log.warning("no audio captured")
            set_title("talktype")
            return

        self._transcribe_queue.put(np.concatenate(frames, axis=0))

    def _transcribe_worker(self):
        while True:
            audio = self._transcribe_queue.get()
            try:
                self._transcribe_and_type(audio)
            finally:
                self._transcribe_queue.task_done()

    def _transcribe_and_type(self, audio):
        duration = len(audio) / SAMPLE_RATE
        if duration < 0.3:
            log.info("too short (%.1fs), skipping", duration)
            set_title("talktype")
            return

        if is_silence(audio):
            log.info("silent audio (peak=%d), skipping", np.max(np.abs(audio)))
            set_title("talktype")
            return

        set_title("talktype [transcribing...]")
        log.info("transcribing %.1fs...", duration)

        try:
            text = transcribe(self._client, audio)
            if not text:
                log.warning("empty transcription")
                set_title("talktype")
                return

            log.info("got: %s", text)
            keyboard.write(text + " ")
            log.info("typed %d chars", len(text))
            beep_done()

        except Exception as e:
            log.error("transcription failed: %s", e)
            beep_error()

        set_title("talktype")


# --- Main ---

def main():
    api_key = os.environ.get("OPENAI_API_KEY", "").strip().strip('"').strip("'")
    if not api_key:
        log.error("OPENAI_API_KEY is not set.")
        raise SystemExit(1)

    client = OpenAI(api_key=api_key)
    session = RecordingSession(client)

    set_title("talktype")
    log.info("talktype ready. Hold Ctrl + Mouse-Back to record, release to transcribe.")
    log.info("Win+Q to quit.")
    log.info("Language: %s | Log: %s", LANGUAGE, LOG_FILE)

    # Track Ctrl-state without suppressing keyboard input. We only need
    # to know "is Ctrl held right now?" when the mouse button event fires.
    ctrl_held = {"value": False}

    def on_key_press(key):
        if key in (pkb.Key.ctrl, pkb.Key.ctrl_l, pkb.Key.ctrl_r):
            ctrl_held["value"] = True

    def on_key_release(key):
        if key in (pkb.Key.ctrl, pkb.Key.ctrl_l, pkb.Key.ctrl_r):
            ctrl_held["value"] = False

    def on_click(_x, _y, button, pressed):
        if button != RECORD_BUTTON:
            return
        if pressed:
            # Only arm recording when Ctrl was already held at click-down.
            # Releasing the button mid-record always stops, regardless of
            # whether Ctrl is still held — natural push-to-talk semantics.
            if ctrl_held["value"]:
                session.start()
        else:
            session.stop()

    key_listener = pkb.Listener(on_press=on_key_press, on_release=on_key_release)
    mouse_listener = pmouse.Listener(on_click=on_click)
    key_listener.start()
    mouse_listener.start()

    try:
        keyboard.wait("windows+q")
    finally:
        key_listener.stop()
        mouse_listener.stop()
    log.info("Bye.")


if __name__ == "__main__":
    main()
