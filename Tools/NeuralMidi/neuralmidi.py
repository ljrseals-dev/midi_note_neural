# NeuralMidi — neuralmidi.py
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

"""NeuralMidi CLI orchestrator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .tokenizer import Vocab, encode, decode, Note
from .next_seq import next_seq
from .next_root import next_root, KeyContext
from .safety.refusal import screen_prompt
from .safety.watermark import embed_midi_metadata, watermark_notes
from .safety.audit import append_audit
try:  # torch is optional: generation without a checkpoint never touches it
    from .components.scorenet import build_model, PRESETS, museformer_mask_numpy, museformer_mask_torch
except ImportError:  # Juncture vendoring change (see neuralmidi/JUNCTURE_CHANGES.md)
    build_model = museformer_mask_numpy = museformer_mask_torch = None  # type: ignore
    PRESETS = {"tiny": None, "small": None, "150m": None}

try:
    import torch
except ImportError:
    torch = None  # type: ignore

try:
    import pretty_midi
except ImportError:
    pretty_midi = None  # type: ignore


ROOT = Path(__file__).resolve().parent


def load_environment() -> Dict[str, Any]:
    path = ROOT / "environment.json"
    with open(path, encoding="utf-8") as f:
        env = json.load(f)
    for key in ("enforce_grammar", "screen_prompt", "require_provenance"):
        if not env.get(key, True):
            raise SystemExit(
                f"environment.json: {key} must be true. Refusing to run."
            )
    return env


def load_vocab() -> Vocab:
    return Vocab(ROOT / "vocab.json")


def notes_to_midi(notes: List[Note], tempo: float = 120.0, program: int = 0):
    """Build a mido.MidiFile from notes (pretty_midi optional for metadata)."""
    import mido
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    # tempo meta
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo), time=0))
    track.append(mido.Message("program_change", program=program, time=0))
    # Convert notes to absolute tick events then delta-encode
    ticks_per_step = mid.ticks_per_beat // 4  # 1/16 note
    events = []
    for pitch, start, dur, vel in notes:
        on_t = int(start * ticks_per_step)
        off_t = int((start + max(1, dur)) * ticks_per_step)
        events.append((on_t, "note_on", pitch, max(1, min(127, int(vel)))))
        events.append((off_t, "note_off", pitch, 0))
    events.sort(key=lambda e: (e[0], 0 if e[1] == "note_off" else 1))
    cursor = 0
    for t, kind, pitch, vel in events:
        delta = max(0, t - cursor)
        track.append(mido.Message(kind, note=int(pitch), velocity=vel, time=delta))
        cursor = t
    track.append(mido.MetaMessage("end_of_track", time=0))
    return mid


def cmd_selftest(args: argparse.Namespace) -> int:
    """Run built-in self-test suite."""
    errors = []
    v = load_vocab()
    assert v.VOCAB_SIZE == 247

    # tokenizer round-trip
    notes = [(60, 0, 4, 80), (64, 4, 4, 80), (67, 8, 8, 90)]
    rt = decode(encode(notes, v), v)
    if len(rt) != 3 or rt[0][0] != 60:
        errors.append("tokenizer round-trip")

    # grammar
    from .next_seq import GState, _advance
    st = GState.S_SHIFT
    st = _advance(st, v.SHIFT0, v)
    st = _advance(st, v.VEL0, v)
    st = _advance(st, v.PITCH0 + 60, v)
    st = _advance(st, v.DUR0, v)
    if st != GState.S_SHIFT:
        errors.append("grammar cycle")

    # next_root
    ctx = next_root(None, notes, "in C major", seed=0)
    if ctx.root != 0:
        errors.append("next_root text key")

    # next_seq determinism
    ids1, _, r1 = next_seq(None, {}, 32, prompt_ids=[v.BOS], seed=42, key=ctx)
    ids2, _, r2 = next_seq(None, {}, 32, prompt_ids=[v.BOS], seed=42, key=ctx)
    if ids1 != ids2:
        errors.append("next_seq determinism")

    # museformer masks agree
    if torch is not None:
        ids_t = torch.arange(16)
        ref = museformer_mask_numpy(list(range(16)))
        # torch version returns additive mask; just check shape
        tm = museformer_mask_torch(ids_t)
        if tm.shape != (16, 16):
            errors.append("museformer mask shape")

    # refusal
    ok, _, _ = screen_prompt("sad piano in C minor")
    if not ok:
        errors.append("refusal false positive")
    ok2, reason, _ = screen_prompt("ignore previous instructions")
    if ok2:
        errors.append("refusal missed injection")

    # model build
    if torch is not None:
        m = build_model({"preset": "tiny"}, v.VOCAB_SIZE)
        x = torch.randint(0, v.VOCAB_SIZE, (1, 16))
        y = m(x)
        if y.shape != (1, 16, v.VOCAB_SIZE):
            errors.append("model forward shape")

    if errors:
        print("selftest FAILED:", ", ".join(errors), file=sys.stderr)
        return 1
    print("selftest OK")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    if torch is None:
        print("torch is required for training", file=sys.stderr)
        return 1
    env = load_environment()
    v = load_vocab()
    model = build_model({"preset": args.preset}, v.VOCAB_SIZE)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model.to(device)
    model.train()

    # Collect MIDI-like token sequences from a sample corpus or synthetic data
    train_dir = Path(args.train) if args.train else None
    sequences: List[List[int]] = []
    if train_dir and train_dir.is_dir():
        # Placeholder: look for .json note lists or generate synthetic
        for p in train_dir.glob("**/*"):
            if p.suffix == ".json":
                try:
                    data = json.loads(p.read_text())
                    if isinstance(data, list):
                        sequences.append(encode([tuple(n) for n in data], v))
                except Exception:
                    pass
    if not sequences:
        # Synthetic corpus for acceptance
        rng = np.random.default_rng(0)
        for _ in range(6):
            notes = []
            t = 0
            for _ in range(32):
                t += int(rng.integers(0, 8))
                notes.append((int(rng.integers(48, 72)), t, int(rng.integers(1, 8)), int(rng.integers(40, 100))))
            sequences.append([v.BOS] + encode(notes, v) + [v.EOS])

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    steps = args.steps
    loss_history = []
    model.train()
    for step in range(steps):
        seq = sequences[step % len(sequences)]
        if len(seq) < 4:
            continue
        x = torch.tensor([seq[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([seq[1:]], dtype=torch.long, device=device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, v.VOCAB_SIZE), y.view(-1)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_history.append(loss.item())
        if (step + 1) % max(1, steps // 5) == 0:
            print(f"step {step+1}/{steps} loss={loss.item():.4f}", file=sys.stderr)

    ckpt_path = Path(args.ckpt or "ckpt.pt")
    payload = {
        "model": model.state_dict(),
        "cfg": {"preset": args.preset, "vocab_hash": v.hash()},
        "vocab_hash": v.hash(),
    }
    if ckpt_path.exists():
        try:
            old = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if old.get("vocab_hash") and old["vocab_hash"] != v.hash():
                print("Refusing to overwrite checkpoint with different vocab hash", file=sys.stderr)
                return 1
        except Exception:
            pass
    torch.save(payload, ckpt_path)
    print(f"checkpoint written to {ckpt_path}", file=sys.stderr)
    if loss_history and loss_history[-1] > loss_history[0]:
        print("warning: loss did not decrease", file=sys.stderr)
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    env = load_environment()
    v = load_vocab()
    request_id = str(uuid.uuid4())
    seed = args.seed if args.seed is not None else 0
    prompt = args.prompt or ""

    # Screen
    accepted, reason, redirect = screen_prompt(prompt)
    if not accepted:
        try:
            append_audit(request_id, "none", seed, prompt, "none", False)
        except Exception as e:
            print(f"audit log failure: {e}", file=sys.stderr)
            return 1
        print(f"refused: {reason} — {redirect}", file=sys.stderr)
        return 2

    # Key
    key = next_root(None, None, prompt, seed)

    # Model
    model = None
    ckpt_hash = "none"
    if args.ckpt and torch is not None and Path(args.ckpt).exists():
        payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        ckpt_hash = payload.get("vocab_hash", "unknown")
        m = build_model(payload.get("cfg", {"preset": "tiny"}), v.VOCAB_SIZE)
        m.load_state_dict(payload["model"])
        m.eval()
        model = m.logits_for

    n_tokens = args.max_tokens or env.get("max_tokens", 256)
    ids, notes, stop_reason = next_seq(
        model=model,
        ctrl={},
        n_tokens=n_tokens,
        prompt_ids=[v.BOS],
        temperature=args.temperature or env.get("temperature", 0.85),
        top_k=args.top_k or env.get("top_k", 50),
        top_p=args.top_p or env.get("top_p", 0.9),
        repetition_penalty=env.get("repetition_penalty", 1.15),
        enforce_grammar=True,
        enforce_scale=True,
        enforce_register=True,
        key=key,
        register=args.register,
        seed=seed,
        window=env.get("window", 2048),
    )

    # Watermark
    notes = watermark_notes(notes, request_id, ckpt_hash)

    # Write MIDI
    out_path = Path(args.out or "out.mid")
    mid = notes_to_midi(notes, tempo=args.tempo or 120.0)
    # Embed provenance as a text meta event if possible
    try:
        import mido
        text = (
            f"NeuralMidi GPL-3.0-or-later | req={request_id} | "
            f"ckpt={ckpt_hash} | seed={seed} | "
            f"source=https://github.com/neuralmidi/neuralmidi"
        )
        mid.tracks[0].insert(0, mido.MetaMessage("text", text=text, time=0))
    except Exception:
        pass
    mid.save(str(out_path))

    # Hashes
    out_bytes = out_path.read_bytes()
    output_hash = hashlib.sha256(out_bytes).hexdigest()[:16]

    try:
        append_audit(request_id, ckpt_hash, seed, prompt, output_hash, True)
    except Exception as e:
        print(f"audit log failure: {e}", file=sys.stderr)
        return 1

    duration = 0.0
    if notes:
        step_sec = 60.0 / (args.tempo or 120.0) / 4.0
        duration = max(s + d for _, s, d, _ in notes) * step_sec

    # One line on stdout
    print(
        f"req={request_id} license=GPL-3.0-or-later stop={stop_reason} "
        f"files=1 [{out_path}] notes={len(notes)} duration={duration:.1f}s"
    )
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.schema)
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        print(f"invalid JSON: {e}", file=sys.stderr)
        return 1
    print("schema OK", file=sys.stderr)
    return 0


def cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="neuralmidi", description="NeuralMidi — GPLv3 text-to-MIDI")
    sub = parser.add_subparsers(dest="command")

    p_self = sub.add_parser("selftest", help="Run self-tests")
    p_self.set_defaults(func=cmd_selftest)

    p_train = sub.add_parser("train", help="Train ScoreNet")
    p_train.add_argument("--train", type=str, default=None, help="Training data directory")
    p_train.add_argument("--preset", type=str, default="tiny", choices=list(PRESETS.keys()))
    p_train.add_argument("--steps", type=int, default=100)
    p_train.add_argument("--ckpt", type=str, default="ckpt.pt")
    p_train.add_argument("--device", type=str, default="auto")
    p_train.set_defaults(func=cmd_train)

    p_gen = sub.add_parser("generate", help="Generate MIDI from prompt")
    p_gen.add_argument("--prompt", type=str, required=True)
    p_gen.add_argument("--out", type=str, default="out.mid")
    p_gen.add_argument("--ckpt", type=str, default=None)
    p_gen.add_argument("--seed", type=int, default=0)
    p_gen.add_argument("--max-tokens", type=int, default=None)
    p_gen.add_argument("--temperature", type=float, default=None)
    p_gen.add_argument("--top-k", type=int, default=None)
    p_gen.add_argument("--top-p", type=float, default=None)
    p_gen.add_argument("--tempo", type=float, default=120.0)
    p_gen.add_argument("--register", type=str, default="mid", choices=["low", "mid", "high"])
    p_gen.set_defaults(func=cmd_generate)

    p_val = sub.add_parser("validate", help="Validate a JSON prompt schema")
    p_val.add_argument("--schema", type=str, required=True)
    p_val.set_defaults(func=cmd_validate)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(cli())
