# NeuralMidi — audit.py
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

"""Append-only audit log with file locking."""

from __future__ import annotations

import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path

try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False


def _log_path() -> Path:
    # Juncture vendoring change: NEURALMIDI_HOME lets the host keep the log in its own data dir.
    env = os.environ.get("NEURALMIDI_HOME")
    d = Path(env) if env else Path.home() / ".neuralmidi"
    d.mkdir(parents=True, exist_ok=True)
    return d / "audit.log"


def append_audit(
    request_id: str,
    ckpt_hash: str,
    seed: int,
    prompt: str,
    output_hash: str,
    accepted: bool,
    user_id: str = "anon",
) -> None:
    """Append one well-formed line. Raises if the log cannot be written."""
    path = _log_path()
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    ts = datetime.now(timezone.utc).isoformat()
    status = "accepted" if accepted else "refused"
    line = (
        f"{ts}|{user_id}|{request_id}|GPL-3.0-or-later|"
        f"{ckpt_hash}|{seed}|{prompt_hash}|{output_hash}|{status}\n"
    )
    with open(path, "a", encoding="utf-8") as f:
        if HAS_FCNTL:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            if HAS_FCNTL:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
