# talktype

Hold a key, speak, release. Text appears at your cursor.

Works in any app — editor, browser, terminal, chat. The key is suppressed at the OS level so it never triggers anything in the active window.

Uses OpenAI's Whisper API for transcription. Auto-detects language.

## Install

```
pip install talktype
```

Set your OpenAI API key:
```
setx OPENAI_API_KEY sk-proj-...
```

## Usage

```
talktype
```

- **F9** (hold) — start recording
- **F9** (release) — transcribe and type at cursor
- **Ctrl+C** — quit

## Configuration

Edit `talktype/main.py` or set before running:

```python
HOTKEY = "f9"           # any key name: f9, scroll_lock, pause, etc.
LANGUAGE = None         # None = auto-detect, or "sv", "en", "de" etc.
SOUND_ENABLED = True    # audio feedback on record start/done
```

## How it works

1. `keyboard` library hooks F9 globally with `suppress=True` (like push-to-talk in Discord/Ventrilo)
2. `sounddevice` records from your default mic while the key is held
3. On release, audio is sent to OpenAI Whisper API
4. `keyboard.write()` types the result via SendInput at your cursor

## Platform support

- **Windows** — fully supported
- **Linux** — requires root (`keyboard` lib needs `/dev/input` access)
- **Mac** — not yet supported (`keyboard` lib doesn't work without accessibility hacks)

## Requirements

- Python 3.10+
- A microphone
- OpenAI API key

## License

MIT
