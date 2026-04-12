# talktype

Hold a key, speak, release. Text appears at your cursor.

Works in any app — editor, browser, terminal, chat. The key is suppressed at the OS level so it never triggers anything in the active window.

Uses OpenAI's Whisper API for transcription. Fast enough for real conversations (~1-2s turnaround).

## Setup

```
pip install sounddevice openai keyboard numpy
```

Set your OpenAI API key:
```
# Windows
setx OPENAI_API_KEY sk-proj-...

# Linux/Mac
export OPENAI_API_KEY=sk-proj-...
```

## Usage

```
python voice_type.py
```

- **F9** (hold) — start recording
- **F9** (release) — transcribe and type at cursor
- **Ctrl+C** — quit

## Configuration

Edit the top of `voice_type.py`:

```python
HOTKEY = "f9"       # any key name: f9, scroll_lock, pause, etc.
LANGUAGE = "sv"     # ISO language code for Whisper
```

## How it works

1. `keyboard` library hooks F9 globally with `suppress=True` (like Ventrilo push-to-talk)
2. `sounddevice` records from your default mic while the key is held
3. Audio is sent to OpenAI Whisper API as WAV
4. `keyboard.write()` types the result via SendInput — works in the focused window

## Requirements

- Windows (uses Win32 keyboard hooks)
- Python 3.10+
- A microphone
- OpenAI API key
