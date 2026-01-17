# Media Transcriber

Transcribe audio and video files using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with optional LLM-powered summaries, chapter markers, entity extraction, and topic tagging.

## Features

- **Fast transcription** using faster-whisper (CTranslate2-based, 4x faster than OpenAI Whisper)
- **Speaker diarization** - identify who's speaking (requires HuggingFace token)
- **Subtitle generation** - SRT and WebVTT formats
- **LLM-powered analysis** (via Ollama or OpenAI):
  - Summary generation
  - Chapter markers (YouTube-compatible)
  - Named entity extraction (people, companies, games, events, places, products)
  - Topic and keyword tagging
- **Parallel processing** - transcribe multiple files simultaneously
- **Combined export** - merge all transcripts into a single document
- **Search** - search across existing transcripts
- **Domain vocabulary** - built-in pinball-specific terms for improved accuracy

## Installation

```bash
# Clone the repository
git clone https://github.com/colinize/media-transcriber.git
cd media-transcriber

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install faster-whisper and other requirements
pip install faster-whisper rich torch
```

### Optional: Speaker Diarization

Requires a HuggingFace token and pyannote.audio:

```bash
pip install pyannote.audio

# Get a free token at https://huggingface.co/settings/tokens
# Accept model terms at https://huggingface.co/pyannote/speaker-diarization-3.1
export HF_TOKEN='your_token_here'
```

### Optional: LLM Features (Summaries, Chapters, Entities, Topics)

Choose one:

**Ollama (free, local):**
```bash
brew install ollama
ollama pull llama3.2
```

**OpenAI:**
```bash
export OPENAI_API_KEY='your_api_key_here'
```

## Usage

### Basic Transcription

```bash
# Transcribe all media files in a folder
python transcriber.py /path/to/folder

# Use a different model (tiny, base, small, medium, large-v2, large-v3)
python transcriber.py /path/to/folder --model medium

# Transcribe multiple folders
python transcriber.py /path/to/folder1 /path/to/folder2
```

### Parallel Processing

```bash
# Transcribe 3 files simultaneously
python transcriber.py /path/to/folder --workers 3
python transcriber.py /path/to/folder -w 3
```

### Speaker Diarization

```bash
# Identify speakers (requires HF_TOKEN)
python transcriber.py /path/to/folder --diarize
```

### Subtitle Generation

```bash
# Generate SRT subtitles
python transcriber.py /path/to/folder --srt

# Generate WebVTT subtitles
python transcriber.py /path/to/folder --vtt

# Both formats
python transcriber.py /path/to/folder --srt --vtt
```

### LLM-Powered Features

```bash
# Generate summaries
python transcriber.py /path/to/folder --summarize

# Generate chapter markers
python transcriber.py /path/to/folder --chapters

# Extract named entities (people, companies, games, etc.)
python transcriber.py /path/to/folder --entities

# Extract topics and keywords
python transcriber.py /path/to/folder --topics

# All analysis features at once
python transcriber.py /path/to/folder --summarize --chapters --entities --topics
```

### Process Existing Transcripts

```bash
# Generate summaries for existing transcripts only
python transcriber.py /path/to/folder --summarize-only

# Generate chapters for existing transcripts
python transcriber.py /path/to/folder --chapters-only

# Extract entities from existing transcripts
python transcriber.py /path/to/folder --entities-only

# Extract topics from existing transcripts
python transcriber.py /path/to/folder --topics-only
```

### Combined Document Export

```bash
# Export all transcripts to a single markdown file
python transcriber.py /path/to/folder --export

# Export existing transcripts only (no transcription)
python transcriber.py /path/to/folder --export-only

# Export as plain text
python transcriber.py /path/to/folder --export-only --export-format txt
```

### Search Transcripts

```bash
# Search for a term across all transcripts
python transcriber.py /path/to/folder --search "search term"

# Case-sensitive search
python transcriber.py /path/to/folder --search "Steve Ritchie" --case-sensitive
```

### Other Options

```bash
# Specify language (default: auto-detect)
python transcriber.py /path/to/folder --language en

# Disable built-in vocabulary
python transcriber.py /path/to/folder --no-vocab

# Use custom vocabulary file
python transcriber.py /path/to/folder --vocab /path/to/vocab.txt

# Force overwrite existing transcripts
python transcriber.py /path/to/folder --force

# Specify LLM provider
python transcriber.py /path/to/folder --summarize --summarize-provider openai

# Specify LLM model
python transcriber.py /path/to/folder --summarize --summarize-model gpt-4o
```

## Output Files

| File | Description |
|------|-------------|
| `*.transcript.md` | Timestamped transcript in Markdown |
| `*.summary.md` | LLM-generated summary |
| `*.chapters.txt` | YouTube-compatible chapter markers |
| `*.entities.json` | Extracted named entities |
| `*.topics.json` | Extracted topics and keywords |
| `*.srt` | SRT subtitle file |
| `*.vtt` | WebVTT subtitle file |
| `*_all_transcripts.md` | Combined document (markdown) |
| `*_all_transcripts.txt` | Combined document (plain text) |

## Supported Formats

**Video:** `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.m4v`

**Audio:** `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.aac`

## Models

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `tiny` | 39M | Fastest | Lower |
| `base` | 74M | Fast | Basic |
| `small` | 244M | Medium | Good |
| `medium` | 769M | Slower | Better |
| `large-v2` | 1.5B | Slow | High |
| `large-v3` | 1.5B | Slow | Highest (default) |
| `distil-large-v3` | 756M | Medium | High |

## License

MIT
