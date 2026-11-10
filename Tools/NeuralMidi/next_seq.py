# NeuralMidi — next_seq.py
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

"""Grammar-constrained, scale-aware sampler for NeuralMidi."""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .tokenizer import Vocab, decode, Note
from .next_root import KeyContext, scale_pitch_classes


class GState(IntEnum):
    S_SHIFT = 0
    S_VEL = 1
    S_PITCH = 2
    S_DUR = 3


def _grammar_logits(state: GState, vocab: Vocab) -> np.ndarray:
    """Return a boolean mask of legal tokens for the current grammar state.
    True = allowed.
    """
    mask = np.zeros(vocab.VOCAB_SIZE, dtype=bool)
    if state == GState.S_SHIFT:
        mask[vocab.SHIFT0 : vocab.SHIFT0 + vocab.SHIFT_COUNT] = True
        mask[vocab.EOS] = True
    elif state == GState.S_VEL:
        mask[vocab.VEL0 : vocab.VEL0 + vocab.VEL_COUNT] = True
    elif state == GState.S_PITCH:
        mask[vocab.PITCH0 : vocab.PITCH0 + vocab.PITCH_COUNT] = True
    elif state == GState.S_DUR:
        mask[vocab.DUR0 : vocab.DUR0 + vocab.DUR_COUNT] = True
    return mask


def _advance(state: GState, token: int, vocab: Vocab) -> GState:
    """Advance grammar state given a legal token. Raises on illegal."""
    if state == GState.S_SHIFT:
        if token == vocab.EOS:
            return GState.S_SHIFT  # terminal; caller stops
        if vocab.is_shift(token):
            return GState.S_VEL
        raise ValueError(f"illegal token {token} in S_SHIFT")
    if state == GState.S_VEL:
        if vocab.is_vel(token):
            return GState.S_PITCH
        raise ValueError(f"illegal token {token} in S_VEL")
    if state == GState.S_PITCH:
        if vocab.is_pitch(token):
            return GState.S_DUR
        raise ValueError(f"illegal token {token} in S_PITCH")
    if state == GState.S_DUR:
        if vocab.is_dur(token):
            return GState.S_SHIFT
        raise ValueError(f"illegal token {token} in S_DUR")
    raise ValueError(f"unknown state {state}")


def _scale_mask(vocab: Vocab, key: KeyContext, register: Optional[str] = None) -> np.ndarray:
    """Boolean mask over vocab: True for pitch tokens that are in-scale (and register)."""
    mask = np.zeros(vocab.VOCAB_SIZE, dtype=bool)
    pcs = set(scale_pitch_classes(key.root, key.mode))
    # register bands (MIDI pitch)
    if register == "low":
        lo, hi = 21, 48
    elif register == "high":
        lo, hi = 72, 108
    else:  # mid or None
        lo, hi = 36, 84
    for p in range(vocab.PITCH_COUNT):
        pitch = p  # 0-127
        if pitch < lo or pitch > hi:
            continue
        if (pitch % 12) in pcs:
            mask[vocab.PITCH0 + p] = True
    # Always allow at least something
    if not mask[vocab.PITCH0 : vocab.PITCH0 + vocab.PITCH_COUNT].any():
        mask[vocab.PITCH0 : vocab.PITCH0 + vocab.PITCH_COUNT] = True
    return mask


def _apply_repetition_penalty(
    logits: np.ndarray,
    history: Sequence[int],
    penalty: float,
    vocab: Vocab,
) -> np.ndarray:
    if penalty <= 1.0 or not history:
        return logits
    out = logits.copy()
    for t in history:
        if vocab.is_shift(t) or vocab.is_vel(t) or vocab.is_pitch(t) or vocab.is_dur(t):
            if out[t] > 0:
                out[t] /= penalty
            else:
                out[t] *= penalty
    return out


def _top_k_top_p_filter(logits: np.ndarray, top_k: int, top_p: float) -> np.ndarray:
    """Return filtered logits (set discarded to -inf)."""
    filtered = logits.copy()
    # top-k
    if top_k > 0 and top_k < len(filtered):
        thresh = np.partition(filtered, -top_k)[-top_k]
        filtered[filtered < thresh] = -np.inf
    # top-p (nucleus)
    if 0.0 < top_p < 1.0:
        sorted_idx = np.argsort(filtered)[::-1]
        sorted_logits = filtered[sorted_idx]
        # softmax for cumulative
        mx = np.max(sorted_logits[np.isfinite(sorted_logits)]) if np.any(np.isfinite(sorted_logits)) else 0.0
        ex = np.exp(sorted_logits - mx)
        ex[~np.isfinite(sorted_logits)] = 0.0
        probs = ex / (ex.sum() + 1e-12)
        cum = np.cumsum(probs)
        # keep tokens until cum >= top_p
        keep = cum <= top_p
        if not keep.any():
            keep[0] = True
        # also keep the first token that crosses
        first_cross = np.searchsorted(cum, top_p, side="left")
        if first_cross < len(keep):
            keep[first_cross] = True
        discard = sorted_idx[~keep]
        filtered[discard] = -np.inf
    return filtered


def _sample_cpu(logits: np.ndarray, rng: np.random.Generator) -> int:
    """Softmax + multinomial on CPU for cross-device determinism."""
    mx = np.max(logits[np.isfinite(logits)]) if np.any(np.isfinite(logits)) else 0.0
    ex = np.exp(logits - mx)
    ex[~np.isfinite(logits)] = 0.0
    s = ex.sum()
    if s <= 0 or not np.isfinite(s):
        # fallback: uniform over finite
        finite = np.isfinite(logits)
        if not finite.any():
            return 0
        idxs = np.flatnonzero(finite)
        return int(rng.choice(idxs))
    probs = ex / s
    return int(rng.choice(len(probs), p=probs))


def next_seq(
    model: Any,
    ctrl: Dict[str, Any],
    n_tokens: int,
    *,
    prompt_ids: Optional[List[int]] = None,
    temperature: float = 0.85,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    enforce_grammar: bool = True,
    enforce_scale: bool = True,
    enforce_register: bool = True,
    key: Optional[KeyContext] = None,
    register: Optional[str] = None,
    seed: int = 0,
    window: int = 2048,
) -> Tuple[List[int], List[Note], str]:
    """Generate a continuation under grammar / scale / register constraints.

    Returns (ids, notes, stop_reason) where stop_reason in {"eos", "max", "invalid"}.
    Sampling is performed on CPU for determinism.
    """
    vocab = Vocab()
    rng = np.random.default_rng(seed)

    ids: List[int] = list(prompt_ids or [vocab.BOS])
    # Preserve prompt in context window
    if len(ids) > window:
        ids = ids[-window:]

    state = GState.S_SHIFT
    # If prompt ends mid-note, recover state by replaying
    for t in ids:
        if t == vocab.BOS or t == vocab.PAD or t == vocab.SEP:
            continue
        try:
            if enforce_grammar:
                state = _advance(state, t, vocab)
        except ValueError:
            state = GState.S_SHIFT

    stop_reason = "max"
    for _ in range(n_tokens):
        # Obtain logits from model (or uniform if model is None / stub)
        if model is None:
            logits = np.zeros(vocab.VOCAB_SIZE, dtype=np.float64)
        else:
            # model is expected to accept list[int] and return 1-d logits
            raw = model(ids[-window:])
            logits = np.asarray(raw, dtype=np.float64).ravel()
            if len(logits) < vocab.VOCAB_SIZE:
                padded = np.full(vocab.VOCAB_SIZE, -1e9)
                padded[: len(logits)] = logits
                logits = padded

        # Grammar mask
        if enforce_grammar:
            gmask = _grammar_logits(state, vocab)
            logits = np.where(gmask, logits, -np.inf)

        # Scale / register mask at PITCH positions
        if state == GState.S_PITCH and (enforce_scale or enforce_register):
            if key is None:
                key = KeyContext(0, "major", 0.0, "fallback")
            smask = _scale_mask(
                vocab,
                key,
                register if enforce_register else None,
            )
            if enforce_scale or enforce_register:
                logits = np.where(smask, logits, -np.inf)

        # Repetition penalty (music tokens only)
        logits = _apply_repetition_penalty(logits, ids, repetition_penalty, vocab)

        # Temperature
        if temperature > 0:
            logits = logits / max(temperature, 1e-6)
        else:
            # greedy
            choice = int(np.nanargmax(logits))
            ids.append(choice)
            try:
                state = _advance(state, choice, vocab) if enforce_grammar else state
            except ValueError:
                stop_reason = "invalid"
                break
            if choice == vocab.EOS:
                stop_reason = "eos"
                break
            continue

        # top-k / top-p
        logits = _top_k_top_p_filter(logits, top_k, top_p)

        # Sample on CPU
        choice = _sample_cpu(logits, rng)
        ids.append(choice)

        if choice == vocab.EOS:
            stop_reason = "eos"
            break

        if enforce_grammar:
            try:
                state = _advance(state, choice, vocab)
            except ValueError:
                stop_reason = "invalid"
                break

    notes = decode(ids, vocab)
    return ids, notes, stop_reason
