"""
Voice Type — push-to-talk transcription with global hotkey.

Hold F9 to record, release to transcribe and type at cursor.
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

# --- Config ---

HOTKEY = "f9"  # suppressed — apps never see it, no focus steal
SAMPLE_RATE = 16000
CHANNELS = 1
LANGUAGE = "sv"
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_type.log")

# --- Logging (file + console) ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("voice-type")

# --- State ---

_api_key = os.environ.get("OPENAI_API_KEY", "").strip().strip('"').strip("'")
client = OpenAI(api_key=_api_key)
recording = False
audio_frames = []
stream = None
lock = threading.Lock()


def start_recording():
    global recording, audio_frames, stream
    with lock:
        if recording:
            return
        recording = True
        audio_frames = []

    def callback(indata, frames, time_info, status):
        if status:
            log.warning("audio status: %s", status)
        audio_frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        callback=callback,
    )
    stream.start()
    log.info("recording...")


def stop_and_transcribe():
    global recording, stream
    with lock:
        if not recording:
            return
        recording = False

    if stream:
        stream.stop()
        stream.close()
        stream = None

    if not audio_frames:
        log.warning("no audio captured")
        return

    audio = np.concatenate(audio_frames, axis=0)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    buf.seek(0)
    buf.name = "recording.wav"

    duration = len(audio) / SAMPLE_RATE
    log.info("transcribing %.1fs...", duration)

    try:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=buf,
            language=LANGUAGE,
            response_format="text",
        )
        text = result.strip()
        if not text:
            log.warning("empty transcription")
            return

        log.info("got: %s", text)
        # Type at cursor — keyboard.write uses SendInput, works in any app
        keyboard.write(text)
        log.info("typed %d chars", len(text))

    except Exception as e:
        log.error("transcription failed: %s", e)


def on_press(event):
    if not recording:
        start_recording()


def on_release(event):
    if recording:
        threading.Thread(target=stop_and_transcribe, daemon=True).start()


def main():
    log.info("Voice Type ready. Hold %s to record, release to transcribe.", HOTKEY.upper())
    log.info("Ctrl+C to quit.")
    log.info("Language: %s | Model: whisper-1 | Log: %s", LANGUAGE, LOG_FILE)

    # suppress=True: the key never reaches the active app — no focus steal,
    # no side effects. This is how Ventrilo/Discord push-to-talk works.
    keyboard.on_press_key(HOTKEY, on_press, suppress=True)
    keyboard.on_release_key(HOTKEY, on_release, suppress=True)

    try:
        keyboard.wait("ctrl+c")
    except KeyboardInterrupt:
        pass
    log.info("Bye.")


if __name__ == "__main__":
    main()
