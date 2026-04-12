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

# --- Audio feedback (soft tones via sounddevice) ---

def _play_tone(freq, duration_ms=80, volume=0.15, fade_ms=15):
    def _do():
        try:
            sr = 22050
            t = np.linspace(0, duration_ms / 1000, int(sr * duration_ms / 1000), endpoint=False)
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
def beep_chunk(): _play_tone(520, 40, 0.08)
def beep_done():  _play_tone(520, 50, 0.10); _play_tone(660, 70, 0.12)
def beep_error(): _play_tone(280, 150, 0.15)

def set_title(text):
    if sys.platform == "win32":
        os.system(f"title {text}")

# --- Config ---

HOTKEY = "f9"
SAMPLE_RATE = 16000
CHANNELS = 1
LANGUAGE = "sv"
CHUNK_INTERVAL_S = 5       # send to Whisper every N seconds while held
OVERLAP_S = 0.3            # audio overlap between chunks to avoid cut words
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

# --- State ---

_api_key = os.environ.get("OPENAI_API_KEY", "").strip().strip('"').strip("'")
client = OpenAI(api_key=_api_key)
lock = threading.Lock()
recording = False
frames = []           # raw int16 audio frames (appended by callback)
stream = None
chunk_timer = None     # periodic timer thread


# --- Audio helpers ---

def frames_to_audio():
    """Snapshot current frames as a numpy array."""
    with lock:
        if not frames:
            return None
        return np.concatenate(frames, axis=0)


def trim_frames_to(keep_samples):
    """Remove all frames except the last `keep_samples` worth of audio."""
    with lock:
        if not frames:
            return
        all_audio = np.concatenate(frames, axis=0)
        if keep_samples <= 0 or keep_samples >= len(all_audio):
            return
        frames.clear()
        frames.append(all_audio[-keep_samples:])


def audio_to_wav(audio):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    buf.seek(0)
    buf.name = "recording.wav"
    return buf


# --- Transcription with word timestamps ---

SENTENCE_ENDS = set(".!?")

def transcribe_with_timestamps(audio):
    """Send audio to Whisper, get back text + word-level timestamps."""
    buf = audio_to_wav(audio)
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=buf,
        language=LANGUAGE,
        response_format="verbose_json",
        timestamp_granularities=["word"],
    )
    return result


def find_last_sentence_boundary(words, full_text):
    """
    Find the last sentence boundary in the transcription.

    Whisper's word objects DON'T include punctuation (word="videon" even
    when full text says "videon!"), so we search the full text for sentence
    endings, count words up to that point, and look up the timestamp from
    the words array.

    Returns (text_up_to_boundary, cut_timestamp) or None if no boundary.
    """
    if not words or not full_text:
        return None

    # Find the last sentence-ending punctuation followed by a space or end
    last_boundary_pos = -1
    for i, ch in enumerate(full_text):
        if ch in SENTENCE_ENDS:
            last_boundary_pos = i

    if last_boundary_pos == -1:
        return None

    text_up_to = full_text[:last_boundary_pos + 1].strip()
    if not text_up_to:
        return None

    # Count words in the sliced text to find the corresponding word index
    word_count = len(text_up_to.split())
    word_idx = min(word_count - 1, len(words) - 1)

    if word_idx < 0:
        return None

    cut_time = words[word_idx].end if hasattr(words[word_idx], "end") else words[word_idx]["end"]
    return text_up_to, cut_time


# --- Chunk streaming logic ---

def process_chunk(is_final=False):
    """Transcribe current audio, type completed sentences, trim buffer."""
    audio = frames_to_audio()
    if audio is None or len(audio) < SAMPLE_RATE * 0.3:
        # Less than 0.3s — not worth transcribing
        return

    duration = len(audio) / SAMPLE_RATE
    set_title(f"talktype [transcribing {duration:.0f}s...]")
    log.info("chunk: transcribing %.1fs (final=%s)...", duration, is_final)

    try:
        result = transcribe_with_timestamps(audio)
        words = getattr(result, "words", None) or []
        full_text = (result.text or "").strip()

        if not full_text:
            log.warning("chunk: empty transcription")
            return

        if is_final:
            # Final chunk: type everything, clear buffer
            log.info("final: %s", full_text)
            keyboard.write(full_text + " ")
            log.info("typed %d chars", len(full_text))
            with lock:
                frames.clear()
            beep_done()
            set_title("talktype")
            return

        # Mid-stream: find last sentence boundary
        boundary = find_last_sentence_boundary(words, full_text)
        if boundary is None:
            # No complete sentence yet — wait for more audio
            log.info("chunk: no sentence boundary yet, waiting... (%s)", full_text[:60])
            set_title("talktype [RECORDING]")
            return

        text, cut_time = boundary
        log.info("chunk: sentence boundary at %.1fs: %s", cut_time, text)
        keyboard.write(text + " ")
        log.info("typed %d chars", len(text))
        beep_chunk()

        # Trim buffer: keep audio from cut_time onwards (with overlap)
        keep_from = max(0, cut_time - OVERLAP_S)
        keep_samples = int((duration - keep_from) * SAMPLE_RATE)
        trim_frames_to(keep_samples)

        set_title("talktype [RECORDING]")

    except Exception as e:
        log.error("chunk transcription failed: %s", e)
        beep_error()
        set_title("talktype [RECORDING]" if recording else "talktype")


def chunk_loop():
    """Periodic timer that fires process_chunk while recording."""
    while True:
        time.sleep(CHUNK_INTERVAL_S)
        with lock:
            if not recording:
                return
        process_chunk(is_final=False)


# --- Recording lifecycle ---

def start_recording():
    global recording, stream, chunk_timer

    with lock:
        if recording:
            return
        recording = True
        frames.clear()

    def callback(indata, frame_count, time_info, status):
        if status:
            log.warning("audio status: %s", status)
        with lock:
            frames.append(indata.copy())

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            callback=callback,
        )
        stream.start()
    except Exception as e:
        log.error("recording start failed: %s", e)
        with lock:
            recording = False
        return

    # Start periodic chunk timer
    chunk_timer = threading.Thread(target=chunk_loop, daemon=True)
    chunk_timer.start()

    set_title("talktype [RECORDING]")
    beep_start()
    log.info("recording (streaming every %ds)...", CHUNK_INTERVAL_S)


def stop_recording():
    global recording, stream

    with lock:
        if not recording:
            return
        recording = False

    if stream:
        try:
            stream.stop()
            stream.close()
        except Exception as e:
            log.warning("stream close failed: %s", e)
        stream = None

    # Process whatever is left in the buffer as final chunk
    threading.Thread(target=lambda: process_chunk(is_final=True), daemon=True).start()


# --- Hotkey ---

def on_press(event):
    start_recording()

def on_release(event):
    stop_recording()


# --- Main ---

def main():
    if not _api_key:
        log.error("OPENAI_API_KEY is not set.")
        raise SystemExit(1)

    set_title("talktype")
    log.info("talktype ready. Hold %s to record, text streams as you speak.", HOTKEY.upper())
    log.info("Ctrl+C to quit.")
    log.info("Language: %s | Chunk: %ds | Log: %s", LANGUAGE, CHUNK_INTERVAL_S, LOG_FILE)

    keyboard.on_press_key(HOTKEY, on_press, suppress=True)
    keyboard.on_release_key(HOTKEY, on_release, suppress=True)

    try:
        keyboard.wait("ctrl+c")
    except KeyboardInterrupt:
        pass
    log.info("Bye.")


if __name__ == "__main__":
    main()
