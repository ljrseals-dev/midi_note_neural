# NeuralMidi — next_root.py
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

"""Key / mode resolver for NeuralMidi.

Priority:
  1. Prompt control tokens (KEY*)
  2. Explicit key word in prompt text
  3. Measured from notes (Krumhansl-Schmuckler)
  4. Seeded fallback
"""

from __future__ import annotations

import re
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Sequence

from .tokenizer import Vocab, Note

# Krumhansl-Schmuckler key profiles (major / minor), normalized later
_KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_NOTE_ALIASES = {
    "c": 0, "c#": 1, "db": 1, "d": 2, "d#": 3, "eb": 3,
    "e": 4, "f": 5, "f#": 6, "gb": 6, "g": 7, "g#": 8, "ab": 8,
    "a": 9, "a#": 10, "bb": 10, "b": 11,
}
_MODE_ALIASES = {
    "major": "major", "maj": "major", "ionian": "major",
    "minor": "minor", "min": "minor", "aeolian": "minor",
    "harmonic minor": "harmonic_minor", "harmonic_minor": "harmonic_minor",
    "melodic minor": "melodic_minor", "melodic_minor": "melodic_minor",
    "dorian": "dorian", "mixolydian": "mixolydian",
    "phrygian": "phrygian", "lydian": "lydian",
}


@dataclass
class KeyContext:
    root: int          # 0-11
    mode: str          # one of vocab.modes
    confidence: float  # 0..1
    source: str        # "control" | "text" | "measured" | "fallback"


def _correlation(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    if da < 1e-9 or db < 1e-9:
        return 0.0
    return num / (da * db)


def _measure_key(notes: List[Note]) -> Tuple[int, str, float]:
    """Krumhansl-Schmuckler: return (root, mode, confidence)."""
    if not notes:
        return 0, "major", 0.0
    weights = [0.0] * 12
    for pitch, _s, dur, vel in notes:
        pc = pitch % 12
        weights[pc] += max(1, dur) * (vel / 127.0 + 0.1)
    best_r, best_m, best_c = 0, "major", -1.0
    for root in range(12):
        rotated = weights[root:] + weights[:root]
        c_maj = _correlation(rotated, _KS_MAJOR)
        c_min = _correlation(rotated, _KS_MINOR)
        if c_maj > best_c:
            best_r, best_m, best_c = root, "major", c_maj
        if c_min > best_c:
            best_r, best_m, best_c = root, "minor", c_min
    # map correlation (-1..1) roughly to 0..1 confidence
    conf = max(0.0, min(1.0, (best_c + 1) / 2))
    return best_r, best_m, conf


def _parse_key_from_text(text: str) -> Optional[Tuple[int, str]]:
    if not text:
        return None
    lower = text.lower()
    # patterns: "in C minor", "C major", "key of A dorian", "Am", "C#m"
    # Try explicit "in X mode" first
    m = re.search(
        r"\b(?:in|key(?:\s+of)?)\s+([a-g][#b]?)\s*"
        r"(major|minor|maj|min|dorian|mixolydian|phrygian|lydian|"
        r"harmonic\s*minor|melodic\s*minor)?\b",
        lower,
    )
    if m:
        note = m.group(1).replace(" ", "")
        mode_raw = (m.group(2) or "major").replace(" ", "_")
        root = _NOTE_ALIASES.get(note)
        mode = _MODE_ALIASES.get(mode_raw, "major")
        if root is not None:
            return root, mode
    # Short forms: Am, C#m, Bbmaj
    m2 = re.search(r"\b([a-g][#b]?)(m|maj|min)?\b", lower)
    if m2:
        note = m2.group(1)
        mod = m2.group(2)
        root = _NOTE_ALIASES.get(note)
        if root is not None:
            mode = "minor" if mod in ("m", "min") else "major"
            return root, mode
    return None


def next_root(
    ctrl: Optional[dict],
    notes: Optional[List[Note]],
    prompt_text: str,
    seed: int,
) -> KeyContext:
    """Resolve key/mode with the documented priority order."""
    vocab = Vocab()

    # 1. Control tokens
    if ctrl and "key" in ctrl:
        k = ctrl["key"]
        if isinstance(k, (list, tuple)) and len(k) >= 2:
            return KeyContext(int(k[0]) % 12, str(k[1]), 1.0, "control")
        if isinstance(k, int):
            return KeyContext(k % 12, "major", 1.0, "control")

    # 2. Explicit key word in prompt text
    parsed = _parse_key_from_text(prompt_text or "")
    if parsed is not None:
        root, mode = parsed
        return KeyContext(root, mode, 0.9, "text")

    # 3. Measured from notes
    if notes:
        root, mode, conf = _measure_key(notes)
        if conf > 0.5:
            return KeyContext(root, mode, conf, "measured")

    # 4. Seeded fallback
    rng = random.Random(seed)
    root = rng.randint(0, 11)
    mode = rng.choice(["major", "minor"])
    return KeyContext(root, mode, 0.0, "fallback")


def scale_pitch_classes(root: int, mode: str) -> List[int]:
    """Return the 7 (or fewer) pitch classes of the scale."""
    intervals = {
        "major": [0, 2, 4, 5, 7, 9, 11],
        "minor": [0, 2, 3, 5, 7, 8, 10],
        "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
        "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
        "dorian": [0, 2, 3, 5, 7, 9, 10],
        "mixolydian": [0, 2, 4, 5, 7, 9, 10],
        "phrygian": [0, 1, 3, 5, 7, 8, 10],
        "lydian": [0, 2, 4, 6, 7, 9, 11],
    }
    iv = intervals.get(mode, intervals["major"])
    return [(root + i) % 12 for i in iv]
