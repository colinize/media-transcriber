#!/usr/bin/env python3
"""
Media Transcriber - Transcribe audio/video files using faster-whisper
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

console = Console()

# Summary generation settings
SUMMARY_SYSTEM_PROMPT = """You are a podcast/media summarizer. Create a concise, well-structured summary of the transcript provided.

Your summary should include:
1. **Overview** (2-3 sentences): What is this content about? Who are the speakers/hosts?
2. **Key Topics** (bullet points): Main subjects discussed
3. **Notable Quotes or Insights** (if any): Interesting statements worth highlighting
4. **Key Takeaways** (2-4 bullets): Most important points for the listener

Keep the summary informative but concise (aim for 200-400 words). Use markdown formatting."""

SUMMARY_USER_PROMPT = """Please summarize the following transcript:

---
{transcript}
---

Provide a structured summary with Overview, Key Topics, Notable Quotes (if any), and Key Takeaways."""

# Chapter detection prompts
CHAPTERS_SYSTEM_PROMPT = """You are a podcast/video chapter creator. Analyze the transcript and identify distinct topic segments.

For each chapter:
1. Identify when the topic changes significantly
2. Create a clear, concise title (2-6 words)
3. Use the EXACT timestamp from the transcript for when the topic begins

Output format (one chapter per line):
00:00 Introduction
02:15 Topic One Title
08:42 Another Topic
etc.

Guidelines:
- First chapter should start at 00:00
- Aim for 4-12 chapters for typical podcast episodes
- Chapters should be at least 2 minutes apart
- Use clear, descriptive titles
- Only include major topic changes, not minor tangents"""

CHAPTERS_USER_PROMPT = """Analyze this transcript and create chapter markers. Use ONLY timestamps that appear in the transcript.

---
{transcript}
---

Output ONLY the chapter list in this format:
00:00 Title
MM:SS Title
...

No other text, just the chapter list."""

# Named entity extraction prompts
ENTITIES_SYSTEM_PROMPT = """You are a named entity extractor. Extract all significant entities from the transcript.

Categories to extract:
- **people**: Names of people mentioned (hosts, guests, designers, players, etc.)
- **companies**: Companies and organizations mentioned
- **games**: Pinball machines, video games, or other games mentioned
- **events**: Tournaments, expos, conventions mentioned
- **places**: Cities, venues, locations mentioned
- **products**: Products, parts, or equipment mentioned

Rules:
- Only include entities that are clearly named (not generic references)
- Use the most complete/formal version of names
- Deduplicate - each entity only once per category
- For games, include manufacturer if mentioned (e.g., "Stern's Godzilla")

Output as JSON with category keys and arrays of strings."""

ENTITIES_USER_PROMPT = """Extract all named entities from this transcript:

---
{transcript}
---

Output valid JSON with these keys: people, companies, games, events, places, products
Each key should have an array of strings (entity names).
If a category has no entities, use an empty array.

Example format:
{{"people": ["Steve Ritchie", "Keith Elwin"], "companies": ["Stern Pinball"], "games": ["Godzilla", "Foo Fighters"], "events": ["Pinball Expo 2024"], "places": ["Chicago"], "products": []}}"""

# Keyword/topic extraction prompts
TOPICS_SYSTEM_PROMPT = """You are a topic and keyword extractor. Analyze the transcript and identify the main themes, topics, and keywords.

Extract:
- **main_topics**: The 3-5 primary subjects discussed (high-level themes)
- **keywords**: 10-20 specific terms that capture key content (names, technical terms, specific items)
- **themes**: 2-4 overarching themes or categories the content falls into
- **tone**: The overall tone (e.g., "educational", "casual conversation", "interview", "news/updates")

Rules:
- main_topics should be descriptive phrases (3-8 words each)
- keywords should be single words or short phrases (1-3 words)
- themes should be broad categories
- Be specific to the actual content, not generic

Output as JSON."""

TOPICS_USER_PROMPT = """Extract topics and keywords from this transcript:

---
{transcript}
---

Output valid JSON with these keys: main_topics, keywords, themes, tone
Example format:
{{"main_topics": ["New pinball machine releases", "Tournament strategy discussion"], "keywords": ["multiball", "Stern", "combo shots", "high scores"], "themes": ["Pinball gaming", "Competition"], "tone": "casual conversation"}}"""

# Global diarization pipeline (loaded once if needed)
_diarization_pipeline = None

# Supported file extensions
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.avi', '.webm', '.m4v'}
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac'}
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

# Pinball-specific vocabulary to improve transcription accuracy
PINBALL_VOCABULARY = """
Pinball, pinball machine, playfield, backglass, translite, apron, lockdown bar.
Stern Pinball, Jersey Jack Pinball, Chicago Gaming, American Pinball, Spooky Pinball,
Dutch Pinball, Multimorphic, deeproot Pinball, Haggis Pinball, Flip N Out Pinball.
Steve Ritchie, Keith Elwin, Eric Meunier, Pat Lawlor, Dennis Nordman, John Popadiuk,
George Gomez, Dwight Sullivan, John Borg, Lonnie Ropp, Brian Eddy, Joe Balcer.
Gary Stern, Jack Guarnieri, Zach Sharpe, Josh Sharpe, Roger Sharpe, Bowen Kerins.
IFPA, PAPA, Pinburgh, Expo, TPF, Texas Pinball Festival, Northwest Pinball Show.
multiball, jackpot, super jackpot, wizard mode, combo, ramp, orbit, spinner,
pop bumper, slingshot, flipper, drop target, stand-up target, scoop, VUK,
ball lock, kickback, outlane, inlane, drain, tilt, slam tilt, extra ball.
EM, electromechanical, solid state, DMD, dot matrix display, LCD, LED.
Bally, Williams, Gottlieb, Data East, Sega, Capcom, Premier, Midway.
Attack from Mars, Medieval Madness, The Addams Family, Twilight Zone,
Theatre of Magic, Monster Bash, Creature from the Black Lagoon, Indiana Jones,
Jurassic Park, Godzilla, Lord of the Rings, Spider-Man, Batman, Iron Man,
Deadpool, Avengers, Venom, James Bond, Mandalorian, Foo Fighters, Led Zeppelin,
Rush, AC/DC, Metallica, KISS, Guns N Roses, Aerosmith, Black Knight,
Funhouse, Whirlwind, Taxi, High Speed, Road Show, No Good Gofers,
Scared Stiff, Tales from the Crypt, Elvira, Ghostbusters, Stranger Things,
John Wick, Jaws, The Big Lebowski, Total Nuclear Annihilation, Heist, Alien,
Willy Wonka, Toy Story, Houdini, Dialed In, Alice in Wonderland, Dune,
Legends of Valhalla, Galactic Tank Force, Celts, Cactus Canyon, Big Bang Bar.
"""


def load_diarization_pipeline():
    """Load the pyannote diarization pipeline (requires HuggingFace token)."""
    global _diarization_pipeline

    if _diarization_pipeline is not None:
        return _diarization_pipeline

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        console.print("[red]Error:[/red] HF_TOKEN environment variable required for diarization")
        console.print("Get a free token at: https://huggingface.co/settings/tokens")
        console.print("Then: export HF_TOKEN='your_token_here'")
        sys.exit(1)

    from pyannote.audio import Pipeline

    _diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )

    # Use GPU if available
    import torch
    if torch.cuda.is_available():
        _diarization_pipeline.to(torch.device("cuda"))

    return _diarization_pipeline


def run_diarization(file_path: Path) -> list[dict]:
    """Run speaker diarization on an audio file."""
    pipeline = load_diarization_pipeline()

    diarization = pipeline(str(file_path))

    # Convert to list of segments with speaker labels
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    return segments


def assign_speakers_to_segments(whisper_segments: list, diarization_segments: list) -> list:
    """Assign speaker labels to whisper segments based on diarization."""

    def get_speaker_at_time(time: float) -> str:
        """Find which speaker is talking at a given time."""
        for seg in diarization_segments:
            if seg["start"] <= time <= seg["end"]:
                return seg["speaker"]
        return "Unknown"

    # Assign speaker to each whisper segment based on its midpoint
    labeled_segments = []
    for seg in whisper_segments:
        midpoint = (seg.start + seg.end) / 2
        speaker = get_speaker_at_time(midpoint)
        labeled_segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "speaker": speaker
        })

    return labeled_segments


def format_timestamp(seconds: float) -> str:
    """Convert seconds to [MM:SS] or [HH:MM:SS] format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"
    return f"[{minutes:02d}:{secs:02d}]"


def format_duration(seconds: float) -> str:
    """Format duration as human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_vtt_timestamp(seconds: float) -> str:
    """Convert seconds to VTT timestamp format (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def generate_srt(segments: list, has_speakers: bool = False) -> str:
    """Generate SRT subtitle format from segments."""
    lines = []
    for i, segment in enumerate(segments, 1):
        start = segment.start if hasattr(segment, 'start') else segment["start"]
        end = segment.end if hasattr(segment, 'end') else segment["end"]
        text = segment.text if hasattr(segment, 'text') else segment["text"]

        if has_speakers:
            speaker = segment.get("speaker", "")
            text = f"[{speaker}] {text.strip()}"
        else:
            text = text.strip()

        lines.append(str(i))
        lines.append(f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)


def generate_vtt(segments: list, has_speakers: bool = False) -> str:
    """Generate WebVTT subtitle format from segments."""
    lines = ["WEBVTT", ""]
    for segment in segments:
        start = segment.start if hasattr(segment, 'start') else segment["start"]
        end = segment.end if hasattr(segment, 'end') else segment["end"]
        text = segment.text if hasattr(segment, 'text') else segment["text"]

        if has_speakers:
            speaker = segment.get("speaker", "")
            text = f"<v {speaker}>{text.strip()}"
        else:
            text = text.strip()

        lines.append(f"{format_vtt_timestamp(start)} --> {format_vtt_timestamp(end)}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)


def find_media_files(folder: Path) -> list[Path]:
    """Find all supported media files in folder."""
    files = []
    for ext in MEDIA_EXTENSIONS:
        files.extend(folder.glob(f"*{ext}"))
        files.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(set(files))


def load_vocabulary_file(vocab_path: Path) -> str:
    """Load vocabulary terms from a file."""
    if not vocab_path.exists():
        console.print(f"[red]Error:[/red] Vocabulary file not found: {vocab_path}")
        sys.exit(1)

    content = vocab_path.read_text().strip()

    # Handle both one-per-line and comma-separated formats
    if '\n' in content:
        terms = [line.strip() for line in content.split('\n') if line.strip()]
    else:
        terms = [term.strip() for term in content.split(',') if term.strip()]

    return ', '.join(terms)


def check_ollama_available() -> bool:
    """Check if Ollama is running and available."""
    import subprocess
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def summarize_with_ollama(transcript: str, model: str = "llama3.2") -> str:
    """Generate summary using local Ollama."""
    import subprocess

    prompt = SUMMARY_USER_PROMPT.format(transcript=transcript)

    # Build the request
    request_data = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    })

    result = subprocess.run(
        ["curl", "-s", "http://localhost:11434/api/chat", "-d", request_data],
        capture_output=True,
        text=True,
        timeout=300  # 5 minute timeout for long transcripts
    )

    if result.returncode != 0:
        raise RuntimeError(f"Ollama request failed: {result.stderr}")

    response = json.loads(result.stdout)
    return response.get("message", {}).get("content", "")


def summarize_with_openai(transcript: str, model: str = "gpt-4o-mini") -> str:
    """Generate summary using OpenAI API."""
    import subprocess

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    prompt = SUMMARY_USER_PROMPT.format(transcript=transcript)

    # Build the request
    request_data = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    })

    result = subprocess.run(
        [
            "curl", "-s", "https://api.openai.com/v1/chat/completions",
            "-H", "Content-Type: application/json",
            "-H", f"Authorization: Bearer {api_key}",
            "-d", request_data
        ],
        capture_output=True,
        text=True,
        timeout=120
    )

    if result.returncode != 0:
        raise RuntimeError(f"OpenAI request failed: {result.stderr}")

    response = json.loads(result.stdout)

    if "error" in response:
        raise RuntimeError(f"OpenAI API error: {response['error'].get('message', 'Unknown error')}")

    return response.get("choices", [{}])[0].get("message", {}).get("content", "")


def generate_summary(transcript: str, provider: str = "auto", model: str | None = None) -> str:
    """Generate a summary of the transcript using LLM.

    Args:
        transcript: The transcript text to summarize
        provider: "ollama", "openai", or "auto" (tries ollama first, then openai)
        model: Model name (defaults based on provider)

    Returns:
        Generated summary text
    """
    # Truncate very long transcripts to fit context window
    max_chars = 100000  # ~25k tokens, safe for most models
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars] + "\n\n[Transcript truncated due to length...]"

    if provider == "auto":
        # Try Ollama first (free, local)
        if check_ollama_available():
            console.print("  [dim]Using Ollama (local)[/dim]")
            return summarize_with_ollama(transcript, model or "llama3.2")
        # Fall back to OpenAI if API key is set
        elif os.environ.get("OPENAI_API_KEY"):
            console.print("  [dim]Using OpenAI API[/dim]")
            return summarize_with_openai(transcript, model or "gpt-4o-mini")
        else:
            raise RuntimeError(
                "No LLM available for summarization.\n"
                "Options:\n"
                "  1. Install Ollama: brew install ollama && ollama pull llama3.2\n"
                "  2. Set OPENAI_API_KEY environment variable"
            )
    elif provider == "ollama":
        return summarize_with_ollama(transcript, model or "llama3.2")
    elif provider == "openai":
        return summarize_with_openai(transcript, model or "gpt-4o-mini")
    else:
        raise ValueError(f"Unknown provider: {provider}")


def chapters_with_ollama(transcript: str, model: str = "llama3.2") -> str:
    """Generate chapters using local Ollama."""
    import subprocess

    prompt = CHAPTERS_USER_PROMPT.format(transcript=transcript)

    request_data = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": CHAPTERS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    })

    result = subprocess.run(
        ["curl", "-s", "http://localhost:11434/api/chat", "-d", request_data],
        capture_output=True,
        text=True,
        timeout=300
    )

    if result.returncode != 0:
        raise RuntimeError(f"Ollama request failed: {result.stderr}")

    response = json.loads(result.stdout)
    return response.get("message", {}).get("content", "")


def chapters_with_openai(transcript: str, model: str = "gpt-4o-mini") -> str:
    """Generate chapters using OpenAI API."""
    import subprocess

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    prompt = CHAPTERS_USER_PROMPT.format(transcript=transcript)

    request_data = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": CHAPTERS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    })

    result = subprocess.run(
        [
            "curl", "-s", "https://api.openai.com/v1/chat/completions",
            "-H", "Content-Type: application/json",
            "-H", f"Authorization: Bearer {api_key}",
            "-d", request_data
        ],
        capture_output=True,
        text=True,
        timeout=120
    )

    if result.returncode != 0:
        raise RuntimeError(f"OpenAI request failed: {result.stderr}")

    response = json.loads(result.stdout)

    if "error" in response:
        raise RuntimeError(f"OpenAI API error: {response['error'].get('message', 'Unknown error')}")

    return response.get("choices", [{}])[0].get("message", {}).get("content", "")


def generate_chapters(transcript: str, provider: str = "auto", model: str | None = None) -> str:
    """Generate chapter markers from a transcript using LLM.

    Args:
        transcript: The transcript text with timestamps
        provider: "ollama", "openai", or "auto"
        model: Model name (defaults based on provider)

    Returns:
        Chapter list in YouTube-compatible format
    """
    # Truncate very long transcripts
    max_chars = 100000
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars] + "\n\n[Transcript truncated...]"

    if provider == "auto":
        if check_ollama_available():
            console.print("  [dim]Using Ollama (local)[/dim]")
            return chapters_with_ollama(transcript, model or "llama3.2")
        elif os.environ.get("OPENAI_API_KEY"):
            console.print("  [dim]Using OpenAI API[/dim]")
            return chapters_with_openai(transcript, model or "gpt-4o-mini")
        else:
            raise RuntimeError(
                "No LLM available for chapter generation.\n"
                "Options:\n"
                "  1. Install Ollama: brew install ollama && ollama pull llama3.2\n"
                "  2. Set OPENAI_API_KEY environment variable"
            )
    elif provider == "ollama":
        return chapters_with_ollama(transcript, model or "llama3.2")
    elif provider == "openai":
        return chapters_with_openai(transcript, model or "gpt-4o-mini")
    else:
        raise ValueError(f"Unknown provider: {provider}")


def entities_with_ollama(transcript: str, model: str = "llama3.2") -> dict:
    """Extract entities using local Ollama."""
    import subprocess

    prompt = ENTITIES_USER_PROMPT.format(transcript=transcript)

    request_data = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": ENTITIES_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "format": "json"  # Request JSON output
    })

    result = subprocess.run(
        ["curl", "-s", "http://localhost:11434/api/chat", "-d", request_data],
        capture_output=True,
        text=True,
        timeout=300
    )

    if result.returncode != 0:
        raise RuntimeError(f"Ollama request failed: {result.stderr}")

    response = json.loads(result.stdout)
    content = response.get("message", {}).get("content", "{}")

    # Parse the JSON response
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        import re
        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"people": [], "companies": [], "games": [], "events": [], "places": [], "products": []}


def entities_with_openai(transcript: str, model: str = "gpt-4o-mini") -> dict:
    """Extract entities using OpenAI API."""
    import subprocess

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    prompt = ENTITIES_USER_PROMPT.format(transcript=transcript)

    request_data = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": ENTITIES_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"}
    })

    result = subprocess.run(
        [
            "curl", "-s", "https://api.openai.com/v1/chat/completions",
            "-H", "Content-Type: application/json",
            "-H", f"Authorization: Bearer {api_key}",
            "-d", request_data
        ],
        capture_output=True,
        text=True,
        timeout=120
    )

    if result.returncode != 0:
        raise RuntimeError(f"OpenAI request failed: {result.stderr}")

    response = json.loads(result.stdout)

    if "error" in response:
        raise RuntimeError(f"OpenAI API error: {response['error'].get('message', 'Unknown error')}")

    content = response.get("choices", [{}])[0].get("message", {}).get("content", "{}")
    return json.loads(content)


def extract_entities(transcript: str, provider: str = "auto", model: str | None = None) -> dict:
    """Extract named entities from a transcript using LLM.

    Args:
        transcript: The transcript text
        provider: "ollama", "openai", or "auto"
        model: Model name (defaults based on provider)

    Returns:
        Dictionary with entity categories as keys
    """
    # Truncate very long transcripts
    max_chars = 100000
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars] + "\n\n[Transcript truncated...]"

    if provider == "auto":
        if check_ollama_available():
            console.print("  [dim]Using Ollama (local)[/dim]")
            return entities_with_ollama(transcript, model or "llama3.2")
        elif os.environ.get("OPENAI_API_KEY"):
            console.print("  [dim]Using OpenAI API[/dim]")
            return entities_with_openai(transcript, model or "gpt-4o-mini")
        else:
            raise RuntimeError(
                "No LLM available for entity extraction.\n"
                "Options:\n"
                "  1. Install Ollama: brew install ollama && ollama pull llama3.2\n"
                "  2. Set OPENAI_API_KEY environment variable"
            )
    elif provider == "ollama":
        return entities_with_ollama(transcript, model or "llama3.2")
    elif provider == "openai":
        return entities_with_openai(transcript, model or "gpt-4o-mini")
    else:
        raise ValueError(f"Unknown provider: {provider}")


def topics_with_ollama(transcript: str, model: str = "llama3.2") -> dict:
    """Extract topics/keywords using local Ollama."""
    import subprocess

    prompt = TOPICS_USER_PROMPT.format(transcript=transcript)

    request_data = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": TOPICS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "format": "json"
    })

    result = subprocess.run(
        ["curl", "-s", "http://localhost:11434/api/chat", "-d", request_data],
        capture_output=True,
        text=True,
        timeout=300
    )

    if result.returncode != 0:
        raise RuntimeError(f"Ollama request failed: {result.stderr}")

    response = json.loads(result.stdout)
    content = response.get("message", {}).get("content", "{}")

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        import re
        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"main_topics": [], "keywords": [], "themes": [], "tone": "unknown"}


def topics_with_openai(transcript: str, model: str = "gpt-4o-mini") -> dict:
    """Extract topics/keywords using OpenAI API."""
    import subprocess

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    prompt = TOPICS_USER_PROMPT.format(transcript=transcript)

    request_data = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": TOPICS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"}
    })

    result = subprocess.run(
        [
            "curl", "-s", "https://api.openai.com/v1/chat/completions",
            "-H", "Content-Type: application/json",
            "-H", f"Authorization: Bearer {api_key}",
            "-d", request_data
        ],
        capture_output=True,
        text=True,
        timeout=120
    )

    if result.returncode != 0:
        raise RuntimeError(f"OpenAI request failed: {result.stderr}")

    response = json.loads(result.stdout)

    if "error" in response:
        raise RuntimeError(f"OpenAI API error: {response['error'].get('message', 'Unknown error')}")

    content = response.get("choices", [{}])[0].get("message", {}).get("content", "{}")
    return json.loads(content)


def extract_topics(transcript: str, provider: str = "auto", model: str | None = None) -> dict:
    """Extract topics and keywords from a transcript using LLM.

    Args:
        transcript: The transcript text
        provider: "ollama", "openai", or "auto"
        model: Model name (defaults based on provider)

    Returns:
        Dictionary with main_topics, keywords, themes, tone
    """
    max_chars = 100000
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars] + "\n\n[Transcript truncated...]"

    if provider == "auto":
        if check_ollama_available():
            console.print("  [dim]Using Ollama (local)[/dim]")
            return topics_with_ollama(transcript, model or "llama3.2")
        elif os.environ.get("OPENAI_API_KEY"):
            console.print("  [dim]Using OpenAI API[/dim]")
            return topics_with_openai(transcript, model or "gpt-4o-mini")
        else:
            raise RuntimeError(
                "No LLM available for topic extraction.\n"
                "Options:\n"
                "  1. Install Ollama: brew install ollama && ollama pull llama3.2\n"
                "  2. Set OPENAI_API_KEY environment variable"
            )
    elif provider == "ollama":
        return topics_with_ollama(transcript, model or "llama3.2")
    elif provider == "openai":
        return topics_with_openai(transcript, model or "gpt-4o-mini")
    else:
        raise ValueError(f"Unknown provider: {provider}")


# Thread-safe console lock for parallel processing
_console_lock = Lock()


def export_combined_document(
    folders: list[Path],
    output_path: Path | None = None,
    format_type: str = "markdown"
) -> Path:
    """Combine all transcripts into a single document.

    Args:
        folders: List of folders containing transcripts
        output_path: Optional output path (defaults to first folder)
        format_type: 'markdown' or 'txt'

    Returns:
        Path to the created combined document
    """
    # Collect all transcripts
    all_transcripts = []
    for folder in folders:
        transcripts = find_transcripts(folder)
        all_transcripts.extend(transcripts)

    if not all_transcripts:
        raise ValueError("No transcripts found to export")

    # Sort by filename for consistent ordering
    all_transcripts.sort(key=lambda p: p.name.lower())

    # Determine output path
    if output_path is None:
        folder_name = folders[0].name if len(folders) == 1 else "combined"
        ext = ".md" if format_type == "markdown" else ".txt"
        output_path = folders[0] / f"{folder_name}_all_transcripts{ext}"

    # Build combined document
    lines = []

    if format_type == "markdown":
        # Markdown format with table of contents
        lines.append("# Combined Transcripts")
        lines.append("")
        lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
        lines.append(f"**Total Files**: {len(all_transcripts)}  ")
        if len(folders) > 1:
            lines.append(f"**Folders**: {len(folders)}")
        lines.append("")

        # Table of contents
        lines.append("## Table of Contents")
        lines.append("")
        for i, transcript in enumerate(all_transcripts, 1):
            source_name = transcript.stem.replace('.transcript', '')
            # Create anchor-friendly ID
            anchor = source_name.lower().replace(' ', '-').replace('.', '')
            lines.append(f"{i}. [{source_name}](#{anchor})")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Add each transcript
        for transcript in all_transcripts:
            content = transcript.read_text()
            source_name = transcript.stem.replace('.transcript', '')

            # Add section header
            lines.append(f"## {source_name}")
            lines.append("")

            # Skip the original title line and metadata, just get content after ---
            content_lines = content.split('\n')
            in_header = True
            for line in content_lines:
                if line.strip() == '---' and in_header:
                    in_header = False
                    continue
                if not in_header:
                    lines.append(line)

            lines.append("")
            lines.append("---")
            lines.append("")

    else:
        # Plain text format
        lines.append("COMBINED TRANSCRIPTS")
        lines.append("=" * 50)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"Total Files: {len(all_transcripts)}")
        lines.append("")
        lines.append("=" * 50)
        lines.append("")

        for transcript in all_transcripts:
            content = transcript.read_text()
            source_name = transcript.stem.replace('.transcript', '')

            lines.append(f"FILE: {source_name}")
            lines.append("-" * 50)
            lines.append("")

            # Extract just the transcript content (after ---)
            content_lines = content.split('\n')
            in_header = True
            for line in content_lines:
                if line.strip() == '---' and in_header:
                    in_header = False
                    continue
                if not in_header:
                    # Remove markdown formatting for plain text
                    clean_line = line.replace('**', '').replace('*', '')
                    lines.append(clean_line)

            lines.append("")
            lines.append("=" * 50)
            lines.append("")

    # Write the combined document
    output_path.write_text('\n'.join(lines))
    return output_path


def process_single_file(
    media_file: Path,
    model,
    model_name: str,
    language: str | None,
    vocabulary: str | None,
    diarize: bool,
    generate_srt_flag: bool,
    generate_vtt_flag: bool,
    summarize_flag: bool,
    chapters_flag: bool,
    entities_flag: bool,
    topics_flag: bool,
    summarize_provider: str,
    summarize_model: str | None,
    force: bool,
    file_index: int,
    total_files: int,
    show_folder: bool,
) -> dict:
    """Process a single media file. Thread-safe worker function.

    Returns dict with keys: status ('processed', 'skipped', 'error'), file, error_msg
    """
    transcript_path = media_file.with_suffix(media_file.suffix + ".transcript.md")
    result = {"file": media_file.name, "status": "processed", "error_msg": None}

    # Thread-safe output
    def log(msg: str):
        with _console_lock:
            console.print(msg)

    # Show file being processed
    if show_folder:
        log(f"[bold][{file_index}/{total_files}][/bold] [dim]{media_file.parent.name}/[/dim]{media_file.name}")
    else:
        log(f"[bold][{file_index}/{total_files}][/bold] {media_file.name}")

    # Check if transcript exists
    if transcript_path.exists() and not force:
        log("  [yellow]⊘[/yellow] Skipped: transcript already exists")
        log("  [dim](use --force to overwrite)[/dim]")
        log("")
        result["status"] = "skipped"
        return result

    try:
        # Transcribe
        markdown, segments, duration, has_speakers = transcribe_file(
            media_file, model, model_name, language, vocabulary, diarize
        )

        # Save markdown transcript
        transcript_path.write_text(markdown)
        log(f"  [green]✓[/green] Saved: {transcript_path.name}")

        # Generate SRT if requested
        if generate_srt_flag:
            srt_path = media_file.with_suffix(media_file.suffix + ".srt")
            srt_content = generate_srt(segments, has_speakers)
            srt_path.write_text(srt_content)
            log(f"  [green]✓[/green] Saved: {srt_path.name}")

        # Generate VTT if requested
        if generate_vtt_flag:
            vtt_path = media_file.with_suffix(media_file.suffix + ".vtt")
            vtt_content = generate_vtt(segments, has_speakers)
            vtt_path.write_text(vtt_content)
            log(f"  [green]✓[/green] Saved: {vtt_path.name}")

        # Generate summary if requested
        if summarize_flag:
            summary_path = transcript_path.with_suffix(".md").with_suffix(".summary.md")
            log("  [cyan]Generating summary...[/cyan]")
            try:
                summary = generate_summary(
                    markdown,
                    provider=summarize_provider,
                    model=summarize_model
                )
                summary_content = f"# Summary: {media_file.name}\n\n"
                summary_content += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n"
                summary_content += f"**Source**: {transcript_path.name}\n\n"
                summary_content += "---\n\n"
                summary_content += summary
                summary_path.write_text(summary_content)
                log(f"  [green]✓[/green] Saved: {summary_path.name}")
            except Exception as sum_err:
                log(f"  [yellow]![/yellow] Summary failed: {sum_err}")

        # Generate chapters if requested
        if chapters_flag:
            chapters_path = transcript_path.with_suffix(".md").with_suffix(".chapters.txt")
            log("  [cyan]Generating chapters...[/cyan]")
            try:
                chapters = generate_chapters(
                    markdown,
                    provider=summarize_provider,
                    model=summarize_model
                )
                chapters_content = f"# Chapters: {media_file.name}\n"
                chapters_content += f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                chapters_content += f"# Copy the timestamps below to YouTube description\n\n"
                chapters_content += chapters.strip()
                chapters_path.write_text(chapters_content)
                log(f"  [green]✓[/green] Saved: {chapters_path.name}")
            except Exception as ch_err:
                log(f"  [yellow]![/yellow] Chapters failed: {ch_err}")

        # Extract entities if requested
        if entities_flag:
            entities_path = transcript_path.with_suffix(".md").with_suffix(".entities.json")
            log("  [cyan]Extracting entities...[/cyan]")
            try:
                entities = extract_entities(
                    markdown,
                    provider=summarize_provider,
                    model=summarize_model
                )
                entities_path.write_text(json.dumps(entities, indent=2))
                log(f"  [green]✓[/green] Saved: {entities_path.name}")
                # Show entity counts
                total_entities = sum(len(v) for v in entities.values() if isinstance(v, list))
                non_empty = [k for k, v in entities.items() if isinstance(v, list) and v]
                if non_empty:
                    log(f"  [dim]Found {total_entities} entities: {', '.join(non_empty)}[/dim]")
            except Exception as ent_err:
                log(f"  [yellow]![/yellow] Entities failed: {ent_err}")

        # Extract topics if requested
        if topics_flag:
            topics_path = transcript_path.with_suffix(".md").with_suffix(".topics.json")
            log("  [cyan]Extracting topics...[/cyan]")
            try:
                topics = extract_topics(
                    markdown,
                    provider=summarize_provider,
                    model=summarize_model
                )
                topics_path.write_text(json.dumps(topics, indent=2))
                log(f"  [green]✓[/green] Saved: {topics_path.name}")
                # Show topic summary
                main_topics = topics.get("main_topics", [])
                keywords = topics.get("keywords", [])
                if main_topics:
                    log(f"  [dim]Topics: {', '.join(main_topics[:3])}{'...' if len(main_topics) > 3 else ''}[/dim]")
                if keywords:
                    log(f"  [dim]Keywords: {len(keywords)} extracted[/dim]")
            except Exception as top_err:
                log(f"  [yellow]![/yellow] Topics failed: {top_err}")

        log("")
        return result

    except Exception as e:
        log(f"  [red]✗[/red] Error: {e}")
        log("")
        result["status"] = "error"
        result["error_msg"] = str(e)
        return result


def generate_markdown(filename: str, segments: list, duration: float, model_name: str, has_speakers: bool = False) -> str:
    """Generate formatted markdown transcript from segments."""
    lines = [
        f"# Transcript: {filename}",
        "",
        f"**Transcribed**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Model**: whisper-{model_name}  ",
        f"**Duration**: {format_duration(duration)}",
    ]

    if has_speakers:
        # Count unique speakers
        speakers = set(seg.get("speaker", "Unknown") for seg in segments)
        lines.append(f"**Speakers**: {len(speakers)}")

    lines.extend(["", "---", ""])

    if has_speakers:
        # Group consecutive segments by speaker for cleaner output
        current_speaker = None
        for segment in segments:
            speaker = segment.get("speaker", "Unknown")
            timestamp = format_timestamp(segment["start"])
            text = segment["text"].strip()

            if speaker != current_speaker:
                if current_speaker is not None:
                    lines.append("")  # Add blank line between speakers
                lines.append(f"**{speaker}**")
                current_speaker = speaker

            lines.append(f"{timestamp} {text}")

        lines.append("")
    else:
        for segment in segments:
            # Handle both faster-whisper segment objects and dicts
            start = segment.start if hasattr(segment, 'start') else segment["start"]
            text = segment.text if hasattr(segment, 'text') else segment["text"]
            timestamp = format_timestamp(start)
            lines.append(f"{timestamp} {text.strip()}")
            lines.append("")

    return "\n".join(lines)


def transcribe_file(file_path: Path, model, model_name: str, language: str | None, vocabulary: str | None, diarize: bool = False) -> tuple[str, list, float, bool]:
    """Transcribe a single file and return (markdown, segments, duration, has_speakers)."""
    options = {
        "beam_size": 5,
        "vad_filter": True,  # Skip silent sections
        "vad_parameters": {"min_silence_duration_ms": 500},
    }

    if language and language != "auto":
        options['language'] = language

    if vocabulary:
        options['initial_prompt'] = vocabulary

    segments, info = model.transcribe(str(file_path), **options)
    duration = info.duration if info.duration else 0

    # Collect segments with progress bar
    segment_list = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Transcribing {format_duration(duration)}",
            total=duration if duration > 0 else 100
        )

        for segment in segments:
            segment_list.append(segment)
            if duration > 0:
                progress.update(task, completed=segment.end)

    # Run speaker diarization if requested
    if diarize:
        console.print("  [cyan]Running speaker diarization...[/cyan]")
        diarization_segments = run_diarization(file_path)
        labeled_segments = assign_speakers_to_segments(segment_list, diarization_segments)
        markdown = generate_markdown(file_path.name, labeled_segments, duration, model_name, has_speakers=True)
        return markdown, labeled_segments, duration, True

    markdown = generate_markdown(file_path.name, segment_list, duration, model_name)
    return markdown, segment_list, duration, False


def find_transcripts(folder: Path) -> list[Path]:
    """Find all transcript files in folder."""
    return sorted(folder.glob("*.transcript.md"))


def search_transcripts(folder: Path, query: str, case_sensitive: bool = False) -> None:
    """Search all transcripts for a query string."""
    transcripts = find_transcripts(folder)

    if not transcripts:
        print(f"No transcripts found in: {folder}")
        print("Run transcription first: python transcriber.py <folder>")
        sys.exit(0)

    print(f"Searching {len(transcripts)} transcript(s) for: \"{query}\"")
    print()

    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(re.escape(query), flags)

    total_matches = 0
    files_with_matches = 0

    for transcript in transcripts:
        content = transcript.read_text()
        lines = content.split('\n')

        # Extract source filename from first line
        source_file = transcript.stem.replace('.transcript', '')

        matches = []
        for i, line in enumerate(lines):
            if pattern.search(line):
                # Get timestamp if present
                matches.append(line.strip())

        if matches:
            files_with_matches += 1
            total_matches += len(matches)

            print(f"📄 {source_file}")
            print("-" * 40)
            for match in matches:
                # Highlight the query in the match
                highlighted = pattern.sub(f"\033[1;33m{query}\033[0m", match)
                print(f"  {highlighted}")
            print()

    # Summary
    print("=" * 50)
    if total_matches:
        print(f"Found {total_matches} match(es) in {files_with_matches} file(s)")
    else:
        print("No matches found.")


def get_device_and_compute():
    """Detect the best available device and compute type."""
    import torch

    if torch.cuda.is_available():
        return "cuda", "float16"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple Silicon - use CPU with int8 for better compatibility
        return "cpu", "int8"
    else:
        return "cpu", "int8"


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video files using faster-whisper (large-v3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Transcribe single folder:
    python transcriber.py /path/to/folder
    python transcriber.py /path/to/folder --model medium

  Transcribe multiple folders:
    python transcriber.py /path/to/folder1 /path/to/folder2 /path/to/folder3
    python transcriber.py ~/podcasts/*/   # shell glob for all subfolders

  Parallel processing:
    python transcriber.py /path/to/folder --workers 2    # 2 files at once
    python transcriber.py /path/to/folder -w 3           # 3 parallel workers

  Other options:
    python transcriber.py /path/to/folder --language en
    python transcriber.py /path/to/folder --no-vocab  # disable pinball vocabulary
    python transcriber.py /path/to/folder --diarize   # identify speakers (requires HF_TOKEN)

  Search transcripts:
    python transcriber.py /path/to/folder --search "pinball"
    python transcriber.py /path/to/folder --search "Steve Ritchie" --case-sensitive

  Generate summaries:
    python transcriber.py /path/to/folder --summarize           # transcribe + summarize
    python transcriber.py /path/to/folder --summarize-only      # summarize existing transcripts
    python transcriber.py /path/to/folder --summarize-only --summarize-provider openai

  Generate chapters:
    python transcriber.py /path/to/folder --chapters            # transcribe + chapters
    python transcriber.py /path/to/folder --chapters-only       # chapters for existing transcripts
    python transcriber.py /path/to/folder --summarize --chapters  # both summary and chapters

  Export combined document:
    python transcriber.py /path/to/folder --export              # transcribe + export combined
    python transcriber.py /path/to/folder --export-only         # export existing transcripts
    python transcriber.py /path/to/folder --export-only --export-format txt  # plain text

  Extract named entities:
    python transcriber.py /path/to/folder --entities            # transcribe + extract entities
    python transcriber.py /path/to/folder --entities-only       # entities for existing transcripts
    python transcriber.py /path/to/folder --summarize --entities  # summary + entities

  Extract topics and keywords:
    python transcriber.py /path/to/folder --topics              # transcribe + extract topics
    python transcriber.py /path/to/folder --topics-only         # topics for existing transcripts
    python transcriber.py /path/to/folder --entities --topics   # extract both entities and topics

  Speaker diarization setup:
    1. Get free token: https://huggingface.co/settings/tokens
    2. Accept model terms: https://huggingface.co/pyannote/speaker-diarization-3.1
    3. export HF_TOKEN='your_token_here'

  Summary LLM setup (one of these):
    - Ollama (free, local): brew install ollama && ollama pull llama3.2
    - OpenAI: export OPENAI_API_KEY='your_api_key_here'
        """
    )
    parser.add_argument(
        "folders",
        type=Path,
        nargs="+",
        help="Path(s) to folder(s) containing media files"
    )
    parser.add_argument(
        "--search",
        metavar="QUERY",
        help="Search transcripts for a term instead of transcribing"
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Make search case-sensitive (default: case-insensitive)"
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3", "distil-large-v3"],
        help="Whisper model size (default: large-v3)"
    )
    parser.add_argument(
        "--language",
        default="auto",
        help="Language code (e.g., 'en', 'es') or 'auto' for detection (default: auto)"
    )
    parser.add_argument(
        "--no-vocab",
        action="store_true",
        help="Disable vocabulary prompt entirely"
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        metavar="FILE",
        help="Custom vocabulary file (one term per line, or comma-separated)"
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization (requires HF_TOKEN env var)"
    )
    parser.add_argument(
        "--srt",
        action="store_true",
        help="Also generate SRT subtitle file"
    )
    parser.add_argument(
        "--vtt",
        action="store_true",
        help="Also generate WebVTT subtitle file"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing transcripts"
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Generate summary after transcription (requires Ollama or OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Only generate summaries for existing transcripts (no transcription)"
    )
    parser.add_argument(
        "--summarize-provider",
        choices=["auto", "ollama", "openai"],
        default="auto",
        help="LLM provider for summaries (default: auto - tries ollama, then openai)"
    )
    parser.add_argument(
        "--summarize-model",
        metavar="MODEL",
        help="Model for summarization (default: llama3.2 for ollama, gpt-4o-mini for openai)"
    )
    parser.add_argument(
        "--chapters",
        action="store_true",
        help="Generate chapter markers after transcription (requires Ollama or OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--chapters-only",
        action="store_true",
        help="Only generate chapters for existing transcripts (no transcription)"
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel transcription workers (default: 1, max recommended: 3)"
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export all transcripts to a single combined document after transcription"
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export existing transcripts to combined document (no transcription)"
    )
    parser.add_argument(
        "--export-format",
        choices=["markdown", "txt"],
        default="markdown",
        help="Format for combined export (default: markdown)"
    )
    parser.add_argument(
        "--entities",
        action="store_true",
        help="Extract named entities (people, companies, games, etc.) after transcription"
    )
    parser.add_argument(
        "--entities-only",
        action="store_true",
        help="Only extract entities from existing transcripts (no transcription)"
    )
    parser.add_argument(
        "--topics",
        action="store_true",
        help="Extract topics and keywords after transcription"
    )
    parser.add_argument(
        "--topics-only",
        action="store_true",
        help="Only extract topics from existing transcripts (no transcription)"
    )

    args = parser.parse_args()

    # Validate all folders
    folders = []
    for folder_arg in args.folders:
        folder = folder_arg.expanduser().resolve()
        if not folder.exists():
            print(f"Error: Folder not found: {folder}")
            sys.exit(1)
        if not folder.is_dir():
            print(f"Error: Not a directory: {folder}")
            sys.exit(1)
        folders.append(folder)

    # Search mode - search across all provided folders
    if args.search:
        for folder in folders:
            if len(folders) > 1:
                console.rule(f"[bold]{folder.name}")
            search_transcripts(folder, args.search, args.case_sensitive)
        return

    # Export-only mode - combine existing transcripts into single document
    if args.export_only:
        console.print("Exporting transcripts to combined document...")
        try:
            output_path = export_combined_document(folders, format_type=args.export_format)
            console.print(f"[green]✓[/green] Exported to: {output_path}")
            console.print(f"  Format: {args.export_format}")
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)
        return

    # Entities-only mode - extract entities from existing transcripts
    if args.entities_only:
        all_transcripts = []
        for folder in folders:
            all_transcripts.extend(find_transcripts(folder))

        if not all_transcripts:
            console.print("[yellow]No transcripts found for entity extraction.[/yellow]")
            console.print("Run transcription first, or check the folder path.")
            sys.exit(0)

        console.print(f"Found [bold]{len(all_transcripts)}[/bold] transcript(s) for entity extraction")
        console.print()

        extracted = 0
        skipped = 0
        errors = []

        for i, transcript_path in enumerate(all_transcripts, 1):
            entities_path = transcript_path.with_suffix(".md").with_suffix(".entities.json")

            # Show folder name if processing multiple folders
            if len(folders) > 1:
                console.print(f"[bold][{i}/{len(all_transcripts)}][/bold] [dim]{transcript_path.parent.name}/[/dim]{transcript_path.name}")
            else:
                console.print(f"[bold][{i}/{len(all_transcripts)}][/bold] {transcript_path.name}")

            # Check if entities exist
            if entities_path.exists() and not args.force:
                console.print("  [yellow]⊘[/yellow] Skipped: entities already exist")
                skipped += 1
                console.print()
                continue

            try:
                transcript_text = transcript_path.read_text()
                console.print("  [cyan]Extracting entities...[/cyan]")

                entities = extract_entities(
                    transcript_text,
                    provider=args.summarize_provider,
                    model=args.summarize_model
                )

                # Save as JSON
                entities_path.write_text(json.dumps(entities, indent=2))
                console.print(f"  [green]✓[/green] Saved: {entities_path.name}")

                # Show entity counts
                total_entities = sum(len(v) for v in entities.values() if isinstance(v, list))
                non_empty = [k for k, v in entities.items() if isinstance(v, list) and v]
                if non_empty:
                    console.print(f"  [dim]Found {total_entities} entities: {', '.join(non_empty)}[/dim]")
                extracted += 1

            except Exception as e:
                console.print(f"  [red]✗[/red] Error: {e}")
                errors.append((transcript_path.name, str(e)))

            console.print()

        # Summary
        console.rule("[bold]Summary")
        console.print(f"[green]Extracted:[/green] {extracted} file(s)")
        if skipped:
            console.print(f"[yellow]Skipped:[/yellow] {skipped} file(s) (already had entities)")
        if errors:
            console.print(f"[red]Errors:[/red] {len(errors)} file(s)")
            for filename, error in errors:
                console.print(f"  [red]•[/red] {filename}: {error}")
        return

    # Topics-only mode - extract topics from existing transcripts
    if args.topics_only:
        all_transcripts = []
        for folder in folders:
            all_transcripts.extend(find_transcripts(folder))

        if not all_transcripts:
            console.print("[yellow]No transcripts found for topic extraction.[/yellow]")
            console.print("Run transcription first, or check the folder path.")
            sys.exit(0)

        console.print(f"Found [bold]{len(all_transcripts)}[/bold] transcript(s) for topic extraction")
        console.print()

        extracted = 0
        skipped = 0
        errors = []

        for i, transcript_path in enumerate(all_transcripts, 1):
            topics_path = transcript_path.with_suffix(".md").with_suffix(".topics.json")

            # Show folder name if processing multiple folders
            if len(folders) > 1:
                console.print(f"[bold][{i}/{len(all_transcripts)}][/bold] [dim]{transcript_path.parent.name}/[/dim]{transcript_path.name}")
            else:
                console.print(f"[bold][{i}/{len(all_transcripts)}][/bold] {transcript_path.name}")

            # Check if topics exist
            if topics_path.exists() and not args.force:
                console.print("  [yellow]⊘[/yellow] Skipped: topics already exist")
                skipped += 1
                console.print()
                continue

            try:
                transcript_text = transcript_path.read_text()
                console.print("  [cyan]Extracting topics...[/cyan]")

                topics = extract_topics(
                    transcript_text,
                    provider=args.summarize_provider,
                    model=args.summarize_model
                )

                # Save as JSON
                topics_path.write_text(json.dumps(topics, indent=2))
                console.print(f"  [green]✓[/green] Saved: {topics_path.name}")

                # Show topic summary
                main_topics = topics.get("main_topics", [])
                keywords = topics.get("keywords", [])
                if main_topics:
                    console.print(f"  [dim]Topics: {', '.join(main_topics[:3])}{'...' if len(main_topics) > 3 else ''}[/dim]")
                if keywords:
                    console.print(f"  [dim]Keywords: {len(keywords)} extracted[/dim]")
                extracted += 1

            except Exception as e:
                console.print(f"  [red]✗[/red] Error: {e}")
                errors.append((transcript_path.name, str(e)))

            console.print()

        # Summary
        console.rule("[bold]Summary")
        console.print(f"[green]Extracted:[/green] {extracted} file(s)")
        if skipped:
            console.print(f"[yellow]Skipped:[/yellow] {skipped} file(s) (already had topics)")
        if errors:
            console.print(f"[red]Errors:[/red] {len(errors)} file(s)")
            for filename, error in errors:
                console.print(f"  [red]•[/red] {filename}: {error}")
        return

    # Summarize-only mode - generate summaries for existing transcripts
    if args.summarize_only:
        all_transcripts = []
        for folder in folders:
            all_transcripts.extend(find_transcripts(folder))

        if not all_transcripts:
            console.print("[yellow]No transcripts found to summarize.[/yellow]")
            console.print("Run transcription first, or check the folder path.")
            sys.exit(0)

        console.print(f"Found [bold]{len(all_transcripts)}[/bold] transcript(s) to summarize")
        console.print()

        summarized = 0
        skipped = 0
        errors = []

        for i, transcript_path in enumerate(all_transcripts, 1):
            summary_path = transcript_path.with_suffix(".md").with_suffix(".summary.md")

            # Show folder name if processing multiple folders
            if len(folders) > 1:
                console.print(f"[bold][{i}/{len(all_transcripts)}][/bold] [dim]{transcript_path.parent.name}/[/dim]{transcript_path.name}")
            else:
                console.print(f"[bold][{i}/{len(all_transcripts)}][/bold] {transcript_path.name}")

            # Check if summary exists
            if summary_path.exists() and not args.force:
                console.print("  [yellow]⊘[/yellow] Skipped: summary already exists")
                skipped += 1
                console.print()
                continue

            try:
                transcript_text = transcript_path.read_text()
                console.print("  [cyan]Generating summary...[/cyan]")

                summary = generate_summary(
                    transcript_text,
                    provider=args.summarize_provider,
                    model=args.summarize_model
                )

                # Create summary markdown
                source_file = transcript_path.stem.replace('.transcript', '')
                summary_content = f"# Summary: {source_file}\n\n"
                summary_content += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n"
                summary_content += f"**Source**: {transcript_path.name}\n\n"
                summary_content += "---\n\n"
                summary_content += summary

                summary_path.write_text(summary_content)
                console.print(f"  [green]✓[/green] Saved: {summary_path.name}")
                summarized += 1

            except Exception as e:
                console.print(f"  [red]✗[/red] Error: {e}")
                errors.append((transcript_path.name, str(e)))

            console.print()

        # Summary
        console.rule("[bold]Summary")
        console.print(f"[green]Summarized:[/green] {summarized} file(s)")
        if skipped:
            console.print(f"[yellow]Skipped:[/yellow] {skipped} file(s) (already had summaries)")
        if errors:
            console.print(f"[red]Errors:[/red] {len(errors)} file(s)")
            for filename, error in errors:
                console.print(f"  [red]•[/red] {filename}: {error}")
        return

    # Chapters-only mode - generate chapters for existing transcripts
    if args.chapters_only:
        all_transcripts = []
        for folder in folders:
            all_transcripts.extend(find_transcripts(folder))

        if not all_transcripts:
            console.print("[yellow]No transcripts found for chapter generation.[/yellow]")
            console.print("Run transcription first, or check the folder path.")
            sys.exit(0)

        console.print(f"Found [bold]{len(all_transcripts)}[/bold] transcript(s) for chapter generation")
        console.print()

        generated = 0
        skipped = 0
        errors = []

        for i, transcript_path in enumerate(all_transcripts, 1):
            chapters_path = transcript_path.with_suffix(".md").with_suffix(".chapters.txt")

            # Show folder name if processing multiple folders
            if len(folders) > 1:
                console.print(f"[bold][{i}/{len(all_transcripts)}][/bold] [dim]{transcript_path.parent.name}/[/dim]{transcript_path.name}")
            else:
                console.print(f"[bold][{i}/{len(all_transcripts)}][/bold] {transcript_path.name}")

            # Check if chapters exist
            if chapters_path.exists() and not args.force:
                console.print("  [yellow]⊘[/yellow] Skipped: chapters already exist")
                skipped += 1
                console.print()
                continue

            try:
                transcript_text = transcript_path.read_text()
                console.print("  [cyan]Generating chapters...[/cyan]")

                chapters = generate_chapters(
                    transcript_text,
                    provider=args.summarize_provider,
                    model=args.summarize_model
                )

                # Create chapters file (YouTube-compatible format)
                source_file = transcript_path.stem.replace('.transcript', '')
                chapters_content = f"# Chapters: {source_file}\n"
                chapters_content += f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                chapters_content += f"# Copy the timestamps below to YouTube description\n\n"
                chapters_content += chapters.strip()

                chapters_path.write_text(chapters_content)
                console.print(f"  [green]✓[/green] Saved: {chapters_path.name}")
                generated += 1

            except Exception as e:
                console.print(f"  [red]✗[/red] Error: {e}")
                errors.append((transcript_path.name, str(e)))

            console.print()

        # Summary
        console.rule("[bold]Summary")
        console.print(f"[green]Generated:[/green] {generated} chapter file(s)")
        if skipped:
            console.print(f"[yellow]Skipped:[/yellow] {skipped} file(s) (already had chapters)")
        if errors:
            console.print(f"[red]Errors:[/red] {len(errors)} file(s)")
            for filename, error in errors:
                console.print(f"  [red]•[/red] {filename}: {error}")
        return

    # Check for diarization requirements early
    if args.diarize and not os.environ.get("HF_TOKEN"):
        console.print("[red]Error:[/red] HF_TOKEN environment variable required for diarization")
        console.print("Get a free token at: [link]https://huggingface.co/settings/tokens[/link]")
        console.print("Then: [cyan]export HF_TOKEN='your_token_here'[/cyan]")
        sys.exit(1)

    # Transcribe mode - import faster_whisper only when needed
    from faster_whisper import WhisperModel

    # Collect all media files from all folders
    all_media_files = []
    for folder in folders:
        media_files = find_media_files(folder)
        all_media_files.extend(media_files)

    if not all_media_files:
        print(f"No media files found in: {', '.join(str(f) for f in folders)}")
        print(f"Supported formats: {', '.join(sorted(MEDIA_EXTENSIONS))}")
        sys.exit(0)

    console.print(f"Found [bold]{len(all_media_files)}[/bold] media file(s) in [bold]{len(folders)}[/bold] folder(s)")
    console.print(f"Model: [cyan]whisper-{args.model}[/cyan]")

    # Detect device
    device, compute_type = get_device_and_compute()
    console.print(f"Device: [cyan]{device}[/cyan] ({compute_type})")

    # Determine vocabulary to use
    if args.no_vocab:
        vocabulary = None
        console.print("Vocabulary: [yellow]disabled[/yellow]")
    elif args.vocab:
        vocabulary = load_vocabulary_file(args.vocab)
        console.print(f"Vocabulary: [green]custom file[/green] ({args.vocab.name})")
    else:
        vocabulary = PINBALL_VOCABULARY
        console.print("Vocabulary: [green]pinball-specific terms[/green]")

    if args.diarize:
        console.print("Diarization: [green]speaker identification enabled[/green]")
    console.print()

    # Load model once
    with console.status("[bold cyan]Loading Whisper model...", spinner="dots"):
        model = WhisperModel(args.model, device=device, compute_type=compute_type)
    console.print("[green]✓[/green] Whisper model loaded")

    # Load diarization model if needed
    if args.diarize:
        with console.status("[bold cyan]Loading diarization model...", spinner="dots"):
            load_diarization_pipeline()
        console.print("[green]✓[/green] Diarization model loaded")

    if args.workers > 1:
        console.print(f"Workers: [cyan]{args.workers}[/cyan] (parallel processing)")
    console.print()

    # Prepare common arguments for worker
    language = None if args.language == "auto" else args.language
    show_folder = len(folders) > 1
    total_files = len(all_media_files)

    # Process files
    processed = 0
    skipped = 0
    errors = []

    if args.workers == 1:
        # Sequential processing (original behavior)
        for i, media_file in enumerate(all_media_files, 1):
            result = process_single_file(
                media_file=media_file,
                model=model,
                model_name=args.model,
                language=language,
                vocabulary=vocabulary,
                diarize=args.diarize,
                generate_srt_flag=args.srt,
                generate_vtt_flag=args.vtt,
                summarize_flag=args.summarize,
                chapters_flag=args.chapters,
                entities_flag=args.entities,
                topics_flag=args.topics,
                summarize_provider=args.summarize_provider,
                summarize_model=args.summarize_model,
                force=args.force,
                file_index=i,
                total_files=total_files,
                show_folder=show_folder,
            )
            if result["status"] == "processed":
                processed += 1
            elif result["status"] == "skipped":
                skipped += 1
            else:
                errors.append((result["file"], result["error_msg"]))
    else:
        # Parallel processing
        console.print(f"[dim]Starting parallel transcription with {args.workers} workers...[/dim]")
        console.print()

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for i, media_file in enumerate(all_media_files, 1):
                future = executor.submit(
                    process_single_file,
                    media_file=media_file,
                    model=model,
                    model_name=args.model,
                    language=language,
                    vocabulary=vocabulary,
                    diarize=args.diarize,
                    generate_srt_flag=args.srt,
                    generate_vtt_flag=args.vtt,
                    summarize_flag=args.summarize,
                    chapters_flag=args.chapters,
                    entities_flag=args.entities,
                    topics_flag=args.topics,
                    summarize_provider=args.summarize_provider,
                    summarize_model=args.summarize_model,
                    force=args.force,
                    file_index=i,
                    total_files=total_files,
                    show_folder=show_folder,
                )
                futures[future] = media_file

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result["status"] == "processed":
                        processed += 1
                    elif result["status"] == "skipped":
                        skipped += 1
                    else:
                        errors.append((result["file"], result["error_msg"]))
                except Exception as e:
                    media_file = futures[future]
                    errors.append((media_file.name, str(e)))

    # Summary
    console.rule("[bold]Summary")
    console.print(f"[green]Processed:[/green] {processed} file(s)")
    if skipped:
        console.print(f"[yellow]Skipped:[/yellow] {skipped} file(s) (already had transcripts)")
    if errors:
        console.print(f"[red]Errors:[/red] {len(errors)} file(s)")
        for filename, error in errors:
            console.print(f"  [red]•[/red] {filename}: {error}")

    # Export combined document if requested
    if args.export and processed > 0:
        console.print()
        console.print("Exporting transcripts to combined document...")
        try:
            output_path = export_combined_document(folders, format_type=args.export_format)
            console.print(f"[green]✓[/green] Exported to: {output_path}")
        except ValueError as e:
            console.print(f"[yellow]![/yellow] Export failed: {e}")


if __name__ == "__main__":
    main()
