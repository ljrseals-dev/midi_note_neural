# NeuralMidi — scorenet.py
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

"""ScoreNet: decoder-only Transformer for symbolic MIDI generation."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

PRESETS = {
    "tiny":  dict(d=128,  layers=4,  heads=4,  ff=512),
    "small": dict(d=512,  layers=8,  heads=8,  ff=2048),
    "150m":  dict(d=768,  layers=12, heads=12, ff=3072),
}


class RoPE(nn.Module):
    def __init__(self, dim: int, max_seq: int = 4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq = max_seq

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None):
        # x: (B, T, H, D)
        t = x.shape[1]
        if positions is None:
            positions = torch.arange(t, device=x.device)
        freqs = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, :, None, :]
        sin = emb.sin()[None, :, None, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        # rotate
        xr = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return x * cos + xr * sin


class TransformerBlock(nn.Module):
    def __init__(self, d: int, heads: int, ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.heads = heads
        self.head_dim = d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.ff = nn.Sequential(
            nn.Linear(d, ff),
            nn.GELU(),
            nn.Linear(ff, d),
        )
        self.rope = RoPE(self.head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, T, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, T, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # each (B, T, H, Hd)
        q = self.rope(q)
        k = self.rope(k)
        q = q.transpose(1, 2)  # (B, H, T, Hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # causal mask
        if mask is None:
            attn_mask = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device), diagonal=1
            )
        else:
            attn_mask = mask
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.out(out)
        x = x + self.ff(self.ln2(x))
        return x


class ScoreNet(nn.Module):
    def __init__(self, vocab_size: int, d: int = 128, layers: int = 4, heads: int = 4, ff: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.blocks = nn.ModuleList([TransformerBlock(d, heads, ff) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        # weight tying
        self.head.weight = self.tok_emb.weight
        self.d = d

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def logits_for(self, ids: list) -> list:
        """Return last-position logits as a Python list (for next_seq)."""
        device = next(self.parameters()).device
        t = torch.tensor([ids], dtype=torch.long, device=device)
        logits = self.forward(t)[0, -1].float().cpu().numpy()
        return logits


def build_model(cfg: Dict[str, Any], vocab_size: int = 247) -> ScoreNet:
    preset = cfg.get("preset", "tiny")
    if preset in PRESETS:
        p = PRESETS[preset]
        d, layers, heads, ff = p["d"], p["layers"], p["heads"], p["ff"]
    else:
        d = cfg.get("d", 128)
        layers = cfg.get("layers", 4)
        heads = cfg.get("heads", 4)
        ff = cfg.get("ff", 512)
    model = ScoreNet(vocab_size, d=d, layers=layers, heads=heads, ff=ff)
    return model


def museformer_mask_numpy(ids):
    """Reference numpy Museformer-style bar/beat mask (simplified)."""
    import numpy as np
    n = len(ids)
    mask = np.zeros((n, n), dtype=np.float32)
    # causal
    for i in range(n):
        mask[i, : i + 1] = 1.0
    return mask


def museformer_mask_torch(ids: torch.Tensor) -> torch.Tensor:
    """GPU version; must agree with numpy reference on fixed input."""
    n = ids.shape[-1]
    mask = torch.triu(torch.ones(n, n, device=ids.device), diagonal=1) * float("-inf")
    return mask
