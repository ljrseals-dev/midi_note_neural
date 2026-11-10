# NeuralMidi — api.py (Juncture host adapter)
# Copyright (C) 2026 Lewis James Seals
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Call NeuralMidi from Python without the CLI, mido or torch.

Same steps as `neuralmidi generate` (screen the prompt, find the key, sample under the
grammar, watermark, write, audit) with the safety layer unchanged. Returns a dict instead of
printing, and writes the file with Juncture's pure-Python writer.
"""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any, Dict

from .next_root import next_root
from .next_seq import next_seq
from .safety.audit import append_audit
from .safety.refusal import screen_prompt
from .safety.watermark import watermark_notes
from .neuralmidi import load_environment, load_vocab

PPQ = 480
TICKS_PER_STEP = PPQ // 4  # one sixteenth


def generate(prompt: str, out_path: str | Path, *, seed: int = 0, tempo: float = 120.0,
             register: str = "mid", max_tokens: int | None = None, program: int = 0) -> Dict[str, Any]:
    """Generate one MIDI file from a prompt. Raises PermissionError if the prompt is refused.

    No checkpoint is used: the sampler runs on the grammar and scale constraints alone,
    which is the same path the CLI takes when `--ckpt` is not given.
    """
    from engine.smf import write_smf  # host module; imported here so the package stays importable alone

    if register not in ("low", "mid", "high"):
        raise ValueError("register must be low, mid or high")
    if not 30 <= float(tempo) <= 240:
        raise ValueError("tempo must be between 30 and 240 BPM")
    env = load_environment()
    vocab = load_vocab()
    request_id = str(uuid.uuid4())

    accepted, reason, redirect = screen_prompt(prompt or "")
    if not accepted:
        append_audit(request_id, "none", seed, prompt or "", "none", False)
        raise PermissionError(f"refused: {reason}. {redirect}")

    key = next_root(None, None, prompt, seed)
    ids, notes, stop = next_seq(
        model=None, ctrl={}, n_tokens=int(max_tokens or env.get("max_tokens", 256)),
        prompt_ids=[vocab.BOS], temperature=env.get("temperature", 0.85),
        top_k=env.get("top_k", 50), top_p=env.get("top_p", 0.9),
        repetition_penalty=env.get("repetition_penalty", 1.15),
        enforce_grammar=True, enforce_scale=True, enforce_register=True,
        key=key, register=register, seed=int(seed), window=env.get("window", 2048),
    )
    notes = watermark_notes(notes, request_id, "none")

    ev: list[tuple] = [
        (0, "name", "NeuralMidi", 0, 0),
        (0, "text", f"NeuralMidi GPL-3.0-or-later | req={request_id} | ckpt=none | seed={seed}", 0, 0),
        (0, "tempo", int(round(60_000_000 / float(tempo))), 0, 0),
        (0, "pc", int(program), 0, 0),
    ]
    for pitch, start, dur, vel in notes:
        on = int(start) * TICKS_PER_STEP
        ev.append((on, "on", int(pitch), max(1, min(127, int(vel))), 0))
        ev.append((on + max(1, int(dur)) * TICKS_PER_STEP, "off", int(pitch), 0, 0))
    data = write_smf([ev], ppq=PPQ)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    append_audit(request_id, "none", seed, prompt, hashlib.sha256(data).hexdigest()[:16], True)

    step_s = 60.0 / float(tempo) / 4.0
    return {
        "path": str(out), "request_id": request_id, "stop": stop, "notes": len(notes),
        "duration_s": round(max((s + d for _, s, d, _ in notes), default=0) * step_s, 2),
        "key_root": key.root, "seed": int(seed), "tempo": float(tempo), "register": register,
        "license": "GPL-3.0-or-later",
        "note_list": [[int(p), int(s), int(d), int(v)] for p, s, d, v in notes],
    }
