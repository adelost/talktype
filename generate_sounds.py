"""Generate soft notification sounds as WAV files. Run once."""
import numpy as np
import wave
import os

DIR = os.path.dirname(os.path.abspath(__file__))
SR = 22050


def piano_tone(freq, duration_s=0.15, volume=0.3):
    """Sine wave with fast attack + exponential decay (piano-like)."""
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    envelope = np.exp(-t * 12)  # fast decay
    tone = np.sin(2 * np.pi * freq * t) * envelope * volume
    return tone.astype(np.float32)


def save_wav(path, audio):
    int16 = (audio * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(int16.tobytes())
    print(f"saved {path} ({len(int16)/SR:.2f}s)")


# Start: single note (C5 = 523 Hz)
save_wav(os.path.join(DIR, "sounds", "start.wav"), piano_tone(523, 0.12, 0.25))

# Done: two notes (G4 + C5, quick arpeggio)
g4 = piano_tone(392, 0.10, 0.20)
c5 = piano_tone(523, 0.15, 0.25)
gap = np.zeros(int(SR * 0.05), dtype=np.float32)
save_wav(os.path.join(DIR, "sounds", "done.wav"), np.concatenate([g4, gap, c5]))

# Error: low note (C3 = 131 Hz, longer decay)
save_wav(os.path.join(DIR, "sounds", "error.wav"), piano_tone(131, 0.25, 0.3))
