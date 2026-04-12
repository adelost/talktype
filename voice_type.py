"""
talktype — push-to-talk transcription with streaming.

Hold F9 to record. While held, audio is transcribed in chunks every
few seconds — text appears at your cursor as you speak. Release F9
to flush the final chunk.

Key is suppressed so apps never see it. Ctrl+C to quit.

Requirements: pip install sounddevice openai keyboard numpy
"""

import io
import os
import sys
import time
import wave
import threading
import logging
import numpy as np
import sounddevice as sd
import keyboard
from openai import OpenAI
from chunking import find_last_sentence_boundary

# --- Config ---

HOTKEY = "f9"
SAMPLE_RATE = 16000
CHANNELS = 1
LANGUAGE = "sv"
CHUNK_INTERVAL_S = 5
OVERLAP_S = 0.3
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


# --- Audio feedback (soft sine tones, not system beeps) ---

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


# --- UI helpers ---

def set_title(text):
    """Set console window title (visible in taskbar). Uses ctypes to avoid
    os.system('title ...') which spawns cmd.exe and can steal focus."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleTitleW(text)
        except Exception:
            pass


# --- Audio buffer ---

class AudioBuffer:
    """Thread-safe audio frame buffer with snapshot + trim operations."""

    def __init__(self):
        self._frames = []
        self._lock = threading.Lock()

    def append(self, frame):
        with self._lock:
            self._frames.append(frame.copy())

    def snapshot(self):
        """Return all buffered audio as a single numpy array, or None."""
        with self._lock:
            if not self._frames:
                return None
            return np.concatenate(self._frames, axis=0)

    def trim_to(self, keep_samples):
        """Keep only the last N samples, discard the rest."""
        with self._lock:
            if not self._frames:
                return
            audio = np.concatenate(self._frames, axis=0)
            if keep_samples <= 0 or keep_samples >= len(audio):
                return
            self._frames.clear()
            self._frames.append(audio[-keep_samples:])

    def clear(self):
        with self._lock:
            self._frames.clear()


# --- Transcription ---

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


def transcribe(client, audio):
    """Send audio to Whisper API, return result with word timestamps."""
    return client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_to_wav(audio),
        language=LANGUAGE,
        response_format="verbose_json",
        timestamp_granularities=["word"],
    )


def type_text(text):
    """Type text at cursor position via keyboard.write (SendInput)."""
    keyboard.write(text + " ")


# --- Chunk processing (composed method) ---

def process_chunk(client, buf, is_final=False):
    """Transcribe buffered audio and type completed sentences.

    For mid-stream chunks: finds the last sentence boundary, types up to
    that point, and trims the buffer. For the final chunk: types everything.
    """
    audio = buf.snapshot()
    if audio is None or not has_enough_audio(audio):
        return

    duration = len(audio) / SAMPLE_RATE
    set_title(f"talktype [transcribing {duration:.0f}s...]")
    log.info("chunk: transcribing %.1fs (final=%s)...", duration, is_final)

    try:
        result = transcribe(client, audio)
        text = (result.text or "").strip()
        words = getattr(result, "words", None) or []

        if not text:
            log.warning("chunk: empty transcription")
            return

        if is_final:
            type_final(text, buf)
        else:
            type_until_boundary(text, words, duration, buf)

    except Exception as e:
        log.error("chunk transcription failed: %s", e)
        beep_error()
        set_title("talktype [RECORDING]")


SILENCE_THRESHOLD = 300  # int16 peak amplitude below this = silence

def has_enough_audio(audio):
    """At least 0.3s of audio with actual speech (not silence).
    Whisper hallucinates on silent audio ('Tack till elever...' etc.)."""
    if len(audio) < SAMPLE_RATE * 0.3:
        return False
    peak = np.max(np.abs(audio))
    if peak < SILENCE_THRESHOLD:
        log.info("chunk: skipping silent audio (peak=%d)", peak)
        return False
    return True


def type_final(text, buf):
    """Final chunk: type everything, clear buffer."""
    log.info("final: %s", text)
    type_text(text)
    log.info("typed %d chars", len(text))
    buf.clear()
    beep_done()
    set_title("talktype")


def type_until_boundary(text, words, duration, buf):
    """Mid-stream: type text up to the last sentence boundary, trim buffer."""
    boundary = find_last_sentence_boundary(words, text)
    if boundary is None:
        log.info("chunk: no sentence boundary yet, waiting... (%s)", text[:60])
        set_title("talktype [RECORDING]")
        return

    sentence, cut_time = boundary
    log.info("chunk: sentence boundary at %.1fs: %s", cut_time, sentence)
    type_text(sentence)
    log.info("typed %d chars", len(sentence))

    keep_from = max(0, cut_time - OVERLAP_S)
    keep_samples = int((duration - keep_from) * SAMPLE_RATE)
    buf.trim_to(keep_samples)
    set_title("talktype [RECORDING]")


# --- Recording session ---

class RecordingSession:
    """Manages mic capture + periodic chunk streaming."""

    def __init__(self, client):
        self._client = client
        self._buf = AudioBuffer()
        self._stream = None
        self._active = False
        self._lock = threading.Lock()

    @property
    def active(self):
        return self._active

    def start(self):
        with self._lock:
            if self._active:
                return
            self._active = True
            self._buf.clear()

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

        threading.Thread(target=self._chunk_loop, daemon=True).start()
        set_title("talktype [RECORDING]")
        beep_start()
        log.info("recording (streaming every %ds)...", CHUNK_INTERVAL_S)

    def stop(self):
        with self._lock:
            if not self._active:
                return
            self._active = False

        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                log.warning("stream close failed: %s", e)
            self._stream = None

        threading.Thread(
            target=lambda: process_chunk(self._client, self._buf, is_final=True),
            daemon=True,
        ).start()

    def _on_audio(self, indata, frame_count, time_info, status):
        if status:
            log.warning("audio status: %s", status)
        self._buf.append(indata)

    def _chunk_loop(self):
        while True:
            time.sleep(CHUNK_INTERVAL_S)
            if not self._active:
                return
            process_chunk(self._client, self._buf, is_final=False)


# --- Main ---

def main():
    api_key = os.environ.get("OPENAI_API_KEY", "").strip().strip('"').strip("'")
    if not api_key:
        log.error("OPENAI_API_KEY is not set.")
        raise SystemExit(1)

    client = OpenAI(api_key=api_key)
    session = RecordingSession(client)

    set_title("talktype")
    log.info("talktype ready. Hold %s to record, text streams as you speak.", HOTKEY.upper())
    log.info("Ctrl+C to quit.")
    log.info("Language: %s | Chunk: %ds | Log: %s", LANGUAGE, CHUNK_INTERVAL_S, LOG_FILE)

    keyboard.on_press_key(HOTKEY, lambda _: session.start(), suppress=True)
    keyboard.on_release_key(HOTKEY, lambda _: session.stop(), suppress=True)

    try:
        keyboard.wait("ctrl+c")
    except KeyboardInterrupt:
        pass
    log.info("Bye.")


if __name__ == "__main__":
    main()
