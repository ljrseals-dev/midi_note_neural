# NeuralMidi Python Tool — Detailed Documentation

This document covers the internal architecture of the Python package, how to train, generate, extend presets, and extend the vocabulary.

---

## Architecture overview

```
prompt text
    │
    ▼
safety/refusal.screen_prompt()     ← refuse living artists, copyrighted works,
    │                                 injection, capability overreach
    ▼
next_root()                        ← key/mode: control → text → KS measure → fallback
    │
    ▼
ScoreNet (optional checkpoint)     ← decoder-only Transformer, or uniform prior
    │
    ▼
next_seq()                         ← grammar mask → scale mask → register mask
    │                                 → repetition penalty → temperature / top-k / top-p
    │                                 → CPU multinomial sample
    ▼
watermark_notes()                  ← ±1 velocity on first note of each bar
    │
    ▼
MIDI writer (mido) + metadata      ← GPLv3 notice, request_id, ckpt_hash, seed, source URL
    │
    ▼
audit.append_audit()               ← append-only ~/.neuralmidi/audit.log
```

---

## How it works

### Vocabulary (`vocab.json`)

Single source of truth. Both Python and (future) C++ load this file. Token ranges:

| Range | Meaning |
|-------|---------|
| 0–2 | PAD, BOS, EOS |
| 3–34 | SHIFT (inter-onset interval in 1/16 steps) |
| 35–50 | VEL (16 velocity bins) |
| 51–178 | PITCH (MIDI 0–127) |
| 179–210 | DUR (duration in 1/16 steps) |
| 211+ | Control tokens (KEY, TEMPO, DENSITY, REGISTER, SEP) |

Never hardcode IDs. Always go through `Vocab`.

### Tokenizer

```python
from Tools.NeuralMidi.tokenizer import Vocab, encode, decode

v = Vocab()
tokens = encode([(60, 0, 4, 80), (64, 4, 4, 90)], v)
notes  = decode(tokens, v)   # exact round-trip (velocity quantized)
```

Notes are `(pitch, start_step, duration_steps, velocity)`. Steps are 1/16 notes.

### Grammar

Legal transitions only:

```
SHIFT → VEL → PITCH → DUR → SHIFT
EOS allowed only from SHIFT
```

Implemented in `next_seq._grammar_logits` / `_advance`. Any illegal transition raises or is masked out during sampling.

### Sampling (`next_seq`)

1. Preserve prompt in the context window.
2. Apply grammar mask.
3. At PITCH positions: apply scale mask and register mask.
4. Apply repetition penalty to music tokens only.
5. Temperature → top-k → top-p.
6. Sample on CPU (`numpy.random.Generator`) for cross-device determinism.
7. Return `(ids, notes, stop_reason)` where `stop_reason ∈ {eos, max, invalid}`.

Same `(seed, prompt, params)` always yields identical `ids`.

### Key resolution (`next_root`)

Priority order:

1. Prompt control tokens (`ctrl["key"]`)
2. Explicit key word in prompt text (`in C minor`, `D dorian`, …)
3. Krumhansl-Schmuckler measurement from existing notes
4. Seeded random fallback

---

## Training

```bash
neuralmidi train \
  --train /path/to/corpus \   # optional; synthetic data used if absent
  --preset tiny \             # tiny | small | 150m
  --steps 2000 \
  --ckpt ckpt.pt \
  --device auto
```

- Optimizer: AdamW, cosine LR schedule (simple constant LR in the current loop).
- Checkpoint stores `model` state dict, `cfg`, and `vocab_hash`.
- Refuses to overwrite a checkpoint whose `vocab_hash` differs from the current vocabulary.
- Loss is printed to stderr every ~20 % of steps.

Presets:

| Name | d | layers | heads | ff |
|------|---|--------|-------|-----|
| tiny | 128 | 4 | 4 | 512 |
| small | 512 | 8 | 8 | 2048 |
| 150m | 768 | 12 | 12 | 3072 |

---

## Generation

```bash
neuralmidi generate \
  --prompt "melancholy nocturne in C# minor, sparse, mid register" \
  --out nocturne.mid \
  --ckpt ckpt.pt \
  --seed 7 \
  --max-tokens 400 \
  --temperature 0.85 \
  --top-k 50 \
  --top-p 0.9 \
  --tempo 72 \
  --register mid
```

Without a checkpoint the sampler uses a uniform prior under the grammar/scale/register constraints (useful for testing).

---

## Adding presets

### Composer

Edit `presets/composers.json`:

```json
{
  "my_style": {
    "display": "My Style",
    "genre": "electronic",
    "era": "contemporary",
    "mood": "dark",
    "tempo_range": [90, 120],
    "density": "medium",
    "register": "mid",
    "instruments": {"synth": 0.6, "drums": 0.4}
  }
}
```

### Instrument

Edit `presets/instruments.json`:

```json
{
  "my_patch": {
    "display": "My Patch",
    "gm_program": 81,
    "category": "synth"
  }
}
```

`gm_program` is the General MIDI program number used by a future renderer / soundfont loader.

Run the preset tests after changes:

```bash
pytest Tools/NeuralMidi/tests/test_presets.py -q
```

---

## Extending the vocabulary

1. Edit `vocab.json` (increment `version` if the layout changes incompatibly).
2. Update `Vocab` in `tokenizer.py` if new named fields are added.
3. Update grammar masks in `next_seq.py` if new token classes participate in the note cycle.
4. Re-train; old checkpoints with a different `vocab_hash` will be refused.
5. Never silently change ranges — mismatches must fail the load.

---

## Safety layer

| Module | Role |
|--------|------|
| `safety/refusal.py` | Screen prompt; return `(accepted, reason, redirect)` |
| `safety/watermark.py` | Metadata text event + ±1 velocity bar watermark |
| `safety/audit.py` | Append-only log under `~/.neuralmidi/audit.log` with `fcntl.flock` |

Refusal categories: named living artists, named copyrighted works, voice cloning without consent, explicit keywords, capability overreach (stubs), prompt injection.

If the audit log cannot be written, generation refuses.

---

## Stub components

`alignnet`, `voxnet`, `rendernet`, `stemnet`, `mixnet` raise `NotImplementedError`. The refusal layer redirects users who request audio rendering, stems, or vocals toward supported symbolic generation.

---

## Testing

```bash
pytest Tools/NeuralMidi/tests/ -q
# Expected: 30 passed
```

Coverage targets (Phase 8 acceptance): ≥ 85 % of `Tools/NeuralMidi/`.

---

## License

GPL-3.0-or-later. See the repository root `LICENSE`, `NOTICE`, and `LICENSING.md`.
