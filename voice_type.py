"""
Voice Type — push-to-talk transcription with global hotkey.

Hold F9 to record, release to transcribe and type at cursor.
Key is suppressed so apps never see it. Ctrl+C to quit.

Requirements: pip install sounddevice openai keyboard numpy
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
active_recording = None
transcribe_queue = queue.Queue()
lock = threading.Lock()


def create_recording_session():
    frames = []

    def callback(indata, frame_count, time_info, status):
        if status:
            log.warning("audio status: %s", status)
        frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        callback=callback,
    )
    return {"frames": frames, "stream": stream}


def start_recording():
    global active_recording
    with lock:
        if active_recording is not None:
            return

        session = None
        try:
            session = create_recording_session()
            session["stream"].start()
        except Exception as e:
            if session is not None:
                try:
                    session["stream"].close()
                except Exception:
                    pass
            log.error("recording start failed: %s", e)
            return

        active_recording = session

    log.info("recording...")


def stop_recording():
    global active_recording
    with lock:
        if active_recording is None:
            return None
        session = active_recording
        active_recording = None

    stream = session["stream"]

    if stream:
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
        return None

    return np.concatenate(frames, axis=0)


def transcribe_and_type(audio):
    duration = len(audio) / SAMPLE_RATE
    log.info("transcribing %.1fs...", duration)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    buf.seek(0)
    buf.name = "recording.wav"

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
        # Type at cursor — keyboard.write uses SendInput, works in any app.
        # Trailing space so consecutive recordings flow naturally:
        # "first sentence." + " " + "second sentence."
        keyboard.write(text + " ")
        log.info("typed %d chars", len(text))

    except Exception as e:
        log.error("transcription failed: %s", e)


def transcribe_worker():
    while True:
        audio = transcribe_queue.get()
        try:
            transcribe_and_type(audio)
        finally:
            transcribe_queue.task_done()


def on_press(event):
    start_recording()


def on_release(event):
    audio = stop_recording()
    if audio is not None:
        transcribe_queue.put(audio)


def main():
    if not _api_key:
        log.error("OPENAI_API_KEY is not set.")
        raise SystemExit(1)

    log.info("Voice Type ready. Hold %s to record, release to transcribe.", HOTKEY.upper())
    log.info("Ctrl+C to quit.")
    log.info("Language: %s | Model: whisper-1 | Log: %s", LANGUAGE, LOG_FILE)

    threading.Thread(target=transcribe_worker, daemon=True).start()

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
