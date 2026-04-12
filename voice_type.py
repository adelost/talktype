"""
talktype — push-to-talk transcription.

Hold F9 to record, release to transcribe. Text appears at your cursor.
Key is suppressed so apps never see it. Ctrl+C to quit.

Requirements: pip install sounddevice openai keyboard numpy
"""

import io
import os
import sys
import wave
import threading
import logging
import numpy as np
import sounddevice as sd
import keyboard
from openai import OpenAI

# --- Config ---

HOTKEY = "f9"
SAMPLE_RATE = 16000
CHANNELS = 1
LANGUAGE = "sv"
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


# --- Audio feedback ---

def _play_tone(freq, duration_ms=80, volume=0.15, fade_ms=15):
    """Play a soft sine tone via sounddevice. Non-blocking."""
    def _do():
        try:
            sr = 22050
            samples = int(sr * duration_ms / 1000)
            t = np.linspace(0, duration_ms / 1000, samples, endpoint=False)
            tone = (np.sin(2 * np.pi * freq * t) * volume).astype(np.float32)
            fade = int(sr * fade_ms / 1000)
            if fade > 0 and len(tone) > fade * 2:
                tone[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
                tone[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
            sd.play(tone, samplerate=sr, blocking=True)
        except Exception:
            pass
    threading.Thread(target=_do, daemon=True).start()


def beep_start(): _play_tone(660, 60, 0.12)
def beep_done():  _play_tone(520, 50, 0.10); _play_tone(660, 70, 0.12)
def beep_error(): _play_tone(280, 150, 0.15)


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
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_to_wav(audio),
        language=LANGUAGE,
        response_format="text",
    )
    return result.strip()


# --- Recording session ---

class RecordingSession:
    """Hold-to-record, release-to-transcribe."""

    def __init__(self, client):
        self._client = client
        self._frames = []
        self._stream = None
        self._active = False
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            if self._active:
                return
            self._active = True
            self._frames.clear()

        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                callback=self._on_audio,
            )
            self._stream.start()
        except Exception as e:
            log.error("recording start failed: %s", e)
            with self._lock:
                self._active = False
            return

        set_title("talktype [RECORDING]")
        beep_start()
        log.info("recording...")

    def stop(self):
        with self._lock:
            if not self._active:
                return
            self._active = False
            audio = np.concatenate(self._frames, axis=0) if self._frames else None
            self._frames.clear()

        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                log.warning("stream close failed: %s", e)
            self._stream = None

        if audio is not None:
            threading.Thread(
                target=self._transcribe_and_type, args=(audio,), daemon=True
            ).start()

    def _on_audio(self, indata, frame_count, time_info, status):
        if status:
            log.warning("audio status: %s", status)
        with self._lock:
            if self._active:
                self._frames.append(indata.copy())

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
    log.info("talktype ready. Hold %s to record, release to transcribe.", HOTKEY.upper())
    log.info("Ctrl+C to quit.")
    log.info("Language: %s | Log: %s", LANGUAGE, LOG_FILE)

    keyboard.on_press_key(HOTKEY, lambda _: session.start(), suppress=True)
    keyboard.on_release_key(HOTKEY, lambda _: session.stop(), suppress=True)

    try:
        keyboard.wait("ctrl+c")
    except KeyboardInterrupt:
        pass
    log.info("Bye.")


if __name__ == "__main__":
    main()
