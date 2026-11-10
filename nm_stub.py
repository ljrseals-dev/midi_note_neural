"""
nm_stub.py — stand-in for Tools.NeuralMidi until the real package is on PYTHONPATH.

Real swap: delete this file's usage and import from Tools.NeuralMidi instead.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ─── Minimal Vocab stand-in ───────────────────────────────────────────────────

@dataclass
class _Vocab:
    PAD: int = 0
    BOS: int = 1
    EOS: int = 2
    VOCAB_SIZE: int = 247

    def hash(self) -> str:
        return "stub-hash"


def load_vocab() -> _Vocab:
    return _Vocab()


def build_model(cfg: Dict[str, Any], vocab_size: int):
    """Returns a tiny module with a logits_for callable attribute."""
    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(16, vocab_size)

        def logits_for(self, ids: List[int]):
            # unused by stub generate path
            return None

    m = _M()
    m.logits_for = m.logits_for  # type: ignore
    return m


def screen_prompt(prompt: str) -> Tuple[bool, str, str]:
    blocked = ("taylor swift", "drake", "bad_word")
    low = (prompt or "").lower()
    for term in blocked:
        if term in low:
            return False, "living_artist_or_blocked", "Please rephrase without naming living artists."
    return True, "ok", ""


def generate_track(
    prompt: str,
    seed: int,
    register: str = "mid",
    role: str = "melody",
    context_ids: Optional[List[int]] = None,
) -> Tuple[List[Tuple[int, int, int, int]], List[int]]:
    """Returns (notes, new_ids). notes = (pitch, start_step, dur, vel)."""
    rng = random.Random(seed + (len(context_ids) if context_ids else 0))
    if register == "low":
        lo, hi = 28, 52
    elif register == "high":
        lo, hi = 60, 96
    else:
        lo, hi = 40, 76

    n_notes = {
        "drums": rng.randint(16, 40),
        "bass": rng.randint(8, 20),
        "chords": rng.randint(6, 16),
        "melody": rng.randint(12, 28),
    }.get(role, rng.randint(10, 24))

    notes: List[Tuple[int, int, int, int]] = []
    step = 0
    for _ in range(n_notes):
        if role == "drums":
            pitch = rng.choice([36, 38, 42, 46, 49])
        else:
            pitch = rng.randint(lo, hi)
        dur = rng.randint(1, 4)
        vel = rng.randint(45, 110)
        notes.append((pitch, step, dur, vel))
        step += dur + rng.randint(0, 2)

    new_ids = [rng.randint(3, 210) for _ in range(len(notes) * 4)]
    return notes, new_ids


def cmd_train(args: Any) -> int:
    """Stub matching neuralmidi.cmd_train(argparse.Namespace) → exit code."""
    ckpt = getattr(args, "ckpt", "checkpoints/user_ckpt.pt")
    steps = int(getattr(args, "steps", 100))
    preset = getattr(args, "preset", "tiny")
    corpus = getattr(args, "train", "corpus/")

    model = nn.Linear(8, 8)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(max(1, steps)):
        x = torch.randn(4, 8)
        y = torch.randn(4, 8)
        optim.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        optim.step()

    os.makedirs(os.path.dirname(ckpt) or ".", exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": {"preset": preset},
            "vocab_hash": "stub-hash",
            "steps": steps,
            "corpus": corpus,
        },
        ckpt,
    )
    return 0
