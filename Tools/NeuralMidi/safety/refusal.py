# NeuralMidi — refusal.py
# Copyright (C) 2026 Lewis James Seals
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Prompt screening and refusal layer."""

from __future__ import annotations

import re
from typing import Tuple

# Curated lists (abbreviated but representative; full lists would be ~200 each)
LIVING_ARTISTS = {
    "taylor swift", "drake", "beyonce", "beyoncé", "ed sheeran", "ariana grande",
    "billie eilish", "the weeknd", "post malone", "dua lipa", "harry styles",
    "olivia rodrigo", "bad bunny", "travis scott", "kendrick lamar", "rihanna",
    "justin bieber", "adele", "bruno mars", "lady gaga", "kanye west", "ye",
    "doja cat", "sza", "the kid laroi", "lil nas x", "miley cyrus", "shawn mendes",
    "camila cabello", "selena gomez", "justin timberlake", "coldplay", "imagine dragons",
    "maroon 5", "bts", "blackpink", "stray kids", "newjeans", "ive", "aespa",
    "taylor", "swift",  # common short forms for living artists
}

COPYRIGHTED_WORKS = {
    "shape of you", "blinding lights", "bad guy", "drivers license", "levitating",
    "peaches", "montero", "stay", "good 4 u", "willow", "positions", "cardigan",
    "anti-hero", "as it was", "heat waves", "industry baby", "save your tears",
    "butter", "dynamite", "permission to dance", "bohemian rhapsody", "stairway to heaven",
    "hotel california", "imagine", "hey jude", "let it be", "thriller", "billie jean",
    "smells like teen spirit", "wonderwall", "creep", "zombie", "yellow",
}

EXPLICIT_KEYWORDS = {
    "explicit lyrics", "nsfw", "porn", "xxx", "erotic audio",
}

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"you\s+are\s+now\s+",
    r"disregard\s+(your|the)\s+(rules|guidelines|instructions)",
    r"system\s*:\s*",
    r"\[INST\]",
    r"<\|im_start\|>",
]

CAPABILITY_OVERREACH = {
    "generate audio": "NeuralMidi generates symbolic MIDI only. Use a DAW or soundfont renderer for audio.",
    "render stems": "Stem separation is not available (stemnet is a stub). Generate a full score instead.",
    "add vocals": "Vocal synthesis (voxnet) is not implemented. Generate an instrumental MIDI score.",
    "voice clone": "Voice cloning requires a consent_token and is not supported in this release.",
    "clone voice": "Voice cloning requires a consent_token and is not supported in this release.",
    "text to speech": "TTS is outside NeuralMidi's scope. Generate MIDI instead.",
    "make a song with lyrics": "Lyrics-to-audio is not supported. You can generate an instrumental MIDI.",
}


def screen_prompt(text: str, consent_token: str | None = None) -> Tuple[bool, str, str]:
    """Return (accepted, reason, redirect).

    accepted=True means proceed; False means refuse.
    """
    if not text or not text.strip():
        return False, "empty_prompt", "Provide a non-empty musical prompt, e.g. 'sad piano in C minor'."

    lower = text.lower().strip()

    # Prompt injection
    for pat in INJECTION_PATTERNS:
        if re.search(pat, lower, re.IGNORECASE):
            return (
                False,
                "prompt_injection",
                "Prompt appears to contain injection attempts. "
                "Describe the music you want (style, key, mood, instruments).",
            )

    # Living artists (word-boundary aware)
    import re as _re
    for name in LIVING_ARTISTS:
        if _re.search(r"(?<!\w)" + _re.escape(name) + r"(?!\w)", lower):
            return (
                False,
                "named_living_artist",
                f"Cannot target living artist style '{name}'. "
                "Try a genre/era preset instead, e.g. 'romantic piano' or 'lo-fi hip-hop beat'.",
            )

    # Copyrighted works
    for title in COPYRIGHTED_WORKS:
        if title in lower:
            return (
                False,
                "named_copyrighted_work",
                f"Cannot reproduce copyrighted work '{title}'. "
                "Describe original music by mood, key, tempo, and instrumentation.",
            )

    # Explicit content
    for kw in EXPLICIT_KEYWORDS:
        if kw in lower:
            return (
                False,
                "explicit_content",
                "Explicit content is not supported. Please rephrase with a musical description.",
            )

    # Voice cloning without consent
    if ("voice clone" in lower or "clone voice" in lower or "voice cloning" in lower) and not consent_token:
        return (
            False,
            "voice_cloning_no_consent",
            "Voice cloning requires a valid consent_token. This feature is not available in the current release.",
        )

    # Capability overreach
    for phrase, redirect in CAPABILITY_OVERREACH.items():
        if phrase in lower:
            return False, "capability_overreach", redirect

    return True, "ok", ""
