"""
MidiNote Neural Host — production-shaped FastAPI bridge.

Today: runs on nm_stub.
When Tools.NeuralMidi is on PYTHONPATH, set USE_REAL = True (or auto-detect).
Orchestration stays here; next_seq is called directly (not via thin api.py).
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import struct
import zipfile
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ─── Real vs stub (graceful fallback) ───────────────────────────────────────
# NEURALMIDI_REAL=1 enables Tools.NeuralMidi. If the import fails, fall back to stub.
_want_real = os.environ.get("NEURALMIDI_REAL", "0").lower() in ("1", "true", "yes")
USE_REAL = False
next_seq = None  # type: ignore
next_root = None  # type: ignore
torch = None  # type: ignore

if _want_real:
    try:
        from Tools.NeuralMidi.tokenizer import Vocab  # type: ignore  # noqa: F401
        from Tools.NeuralMidi.next_seq import next_seq  # type: ignore
        from Tools.NeuralMidi.next_root import next_root  # type: ignore
        from Tools.NeuralMidi.safety.refusal import screen_prompt  # type: ignore
        from Tools.NeuralMidi.neuralmidi import build_model, load_vocab, cmd_train  # type: ignore
        import torch

        USE_REAL = True
        print("⚡ RUNNING REAL NEURALMIDI CORE (GPL-3.0-or-later)")
    except ImportError as e:
        print(f"❌ NEURALMIDI_REAL=1 but import failed: {e}")
        print("⚠️  Falling back to STUB mode.")
        _want_real = False

if not USE_REAL:
    import nm_stub as _stub

    def load_vocab():
        return _stub.load_vocab()

    def build_model(cfg, vocab_size):
        return _stub.build_model(cfg, vocab_size)

    def screen_prompt(prompt: str):
        return _stub.screen_prompt(prompt)

    def cmd_train(args):
        return _stub.cmd_train(args)

    print("🛠️  RUNNING STUB MODE (no real model required)")

try:
    import uvicorn
except ImportError:
    uvicorn = None  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("MidiNoteHost")

CKPT_DIR = Path("checkpoints")
CORPUS_DIR = Path("corpus")
STATIC_DIR = Path(__file__).resolve().parent / "static"


# ─── Model cache ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=5)
def get_cached_model(ckpt_path: str):
    """
    Real path loads .pt once, checks vocab_hash, returns logits_for callable.
    Stub / missing file → None (uniform prior).
    """
    logger.info("cache load: %s (real=%s)", ckpt_path, USE_REAL)
    v = load_vocab()

    path = Path(ckpt_path)
    if not path.is_file():
        path = CKPT_DIR / ckpt_path
    if not path.is_file():
        logger.warning("checkpoint %s not found — uniform prior", ckpt_path)
        return None

    if not USE_REAL:
        # Stub: just mark as cached; generate path still uses generate_track
        logger.info("stub cache marker for %s", ckpt_path)
        return f"stub:{ckpt_path}"

    assert torch is not None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(str(path), map_location=device, weights_only=False)
    if payload.get("vocab_hash") != v.hash():
        raise ValueError(
            f"Vocab mismatch! ckpt={payload.get('vocab_hash')} current={v.hash()}"
        )
    model = build_model(payload.get("cfg", {"preset": "tiny"}), v.VOCAB_SIZE)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    logger.info("model on %s (%s)", device, ckpt_path)
    return model.logits_for


class GenerateRequest(BaseModel):
    prompt: str
    controls: Dict[str, Any] = Field(default_factory=dict)
    seed: int = 0
    tempo: float = 120.0
    ckpt: Optional[str] = "default_150m.pt"


class TrainRequest(BaseModel):
    preset: str = "tiny"
    steps: int = 100
    ckpt_name: str = "user_ckpt.pt"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("MidiNote Neural Host starting (USE_REAL=%s)", USE_REAL)
    CKPT_DIR.mkdir(exist_ok=True)
    CORPUS_DIR.mkdir(exist_ok=True)
    STATIC_DIR.mkdir(exist_ok=True)
    yield
    logger.info("MidiNote Neural Host shutting down")


app = FastAPI(title="MidiNote Neural Host", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    index = STATIC_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {"status": "ok", "use_real": USE_REAL}


# ─── SMF writer ──────────────────────────────────────────────────────────────

def write_multitrack_smf(
    tracks: List[Dict[str, Any]],
    ppq: int = 480,
    tempo: float = 120.0,
) -> bytes:
    def vlq(n: int) -> List[int]:
        out = [n & 127]
        n >>= 7
        while n:
            out.append(128 | (n & 127))
            n >>= 7
        return list(reversed(out))

    out = bytearray([0x4D, 0x54, 0x68, 0x64, 0, 0, 0, 6, 0, 1])
    out.extend(struct.pack(">H", len(tracks) + 1))
    out.extend(struct.pack(">H", ppq))

    def build_track(events: List) -> bytes:
        t = bytearray()
        for tick, status, *args in events:
            t.extend(vlq(tick))
            t.append(status)
            for a in args:
                if isinstance(a, bytes):
                    t.extend(a)
                else:
                    t.append(int(a))
        t.extend([0, 0xFF, 0x2F, 0])
        return b"MTrk" + struct.pack(">I", len(t)) + bytes(t)

    tempo_us = max(1, int(60_000_000 / tempo))
    meta: List = [
        (0, 0xFF, 0x51, 3, tempo_us.to_bytes(3, "big")),
        (0, 0xFF, 0x03, len(b"MidiNote Neural"), b"MidiNote Neural"),
    ]
    out.extend(build_track(meta))

    for i, trk in enumerate(tracks):
        ch = min(i, 15)
        if trk["name"].lower() == "drums":
            ch = 9

        events: List = []
        name = trk["name"].encode("utf-8")
        events.append((0, 0xFF, 0x03, len(name), name))
        if ch != 9:
            events.append((0, 0xC0 | ch, int(trk.get("program", 0))))

        note_evs = []
        for pitch, start, dur, vel in trk["notes"]:
            on = int(start * (ppq // 4))
            off = int((start + max(1, dur)) * (ppq // 4))
            note_evs.append((on, 0x90 | ch, int(pitch), max(1, min(127, int(vel)))))
            note_evs.append((off, 0x80 | ch, int(pitch), 0))
        note_evs.sort(key=lambda e: (e[0], e[1] & 0xF0))

        cursor = 0
        for tick, status, *rest in note_evs:
            events.append((tick - cursor, status, *rest))
            cursor = tick

        out.extend(build_track(events))

    return bytes(out)


# ─── Generate ────────────────────────────────────────────────────────────────

@app.post("/api/generate")
async def generate_midi(req: GenerateRequest):
    logger.info(
        "generate seed=%s tempo=%s ckpt=%s prompt=%r",
        req.seed, req.tempo, req.ckpt, req.prompt,
    )

    try:
        model_fn = get_cached_model(req.ckpt or "default_150m.pt")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}") from e

    accepted, reason, redirect = screen_prompt(req.prompt)
    if not accepted:
        raise HTTPException(status_code=403, detail=f"Refused: {reason}. {redirect}")

    v = load_vocab()
    key_ctx = None
    if USE_REAL and next_root is not None:
        key_ctx = next_root(None, None, req.prompt, req.seed)

    roles = [
        {"name": "Drums", "program": 0, "register": "low", "role": "drums"},
        {"name": "Bass", "program": 33, "register": "low", "role": "bass"},
        {"name": "Chords", "program": 0, "register": "mid", "role": "chords"},
        {"name": "Lead", "program": 81, "register": "high", "role": "melody"},
    ]
    muted = {m.lower() for m in req.controls.get("muted_roles", [])}
    tracks: List[Dict[str, Any]] = []
    context_ids: List[int] = [getattr(v, "BOS", 1)]

    for role in roles:
        if role["name"].lower() in muted:
            continue
        role_prompt = f"{req.prompt}, role: {role['role']}, register: {role['register']}"
        stem_seed = req.seed + (hash(role["name"]) % 997)

        try:
            if USE_REAL and next_seq is not None:
                # Offload CPU/GPU sampling so the event loop stays responsive
                ids, notes, stop = await run_in_threadpool(
                    next_seq,
                    model=model_fn,
                    ctrl={},
                    n_tokens=1000,
                    prompt_ids=context_ids,
                    temperature=0.85,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=1.15,
                    enforce_grammar=True,
                    enforce_scale=True,
                    enforce_register=True,
                    key=key_ctx,
                    register=role["register"],
                    seed=stem_seed,
                )
            else:
                import nm_stub as _s

                notes, ids = _s.generate_track(
                    prompt=role_prompt,
                    seed=stem_seed,
                    register=role["register"],
                    role=role["role"],
                    context_ids=context_ids,
                )

            tracks.append(
                {
                    "name": role["name"],
                    "program": role["program"],
                    "notes": notes,
                }
            )
            context_ids.extend(ids)
            logger.info(
                "%s → %d notes (context %d tokens)",
                role["name"], len(notes), len(context_ids),
            )
        except Exception as e:
            logger.error("track %s failed: %s", role["name"], e, exc_info=True)

    if not tracks:
        raise HTTPException(status_code=500, detail="No tracks generated")

    midi = write_multitrack_smf(tracks, tempo=req.tempo)
    return Response(
        content=midi,
        media_type="audio/midi",
        headers={
            "Content-Disposition": f'attachment; filename="MidiNote_Neural_seed{req.seed}.mid"'
        },
    )


@app.post("/api/upload-corpus")
async def upload_corpus(file: UploadFile = File(...)):
    name = file.filename or "upload.zip"
    if not name.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Must be a .zip file")

    CORPUS_DIR.mkdir(exist_ok=True)
    zip_path = CORPUS_DIR / Path(name).name
    try:
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(CORPUS_DIR)
        zip_path.unlink(missing_ok=True)
        mid_count = sum(
            1 for p in CORPUS_DIR.rglob("*") if p.suffix.lower() in {".mid", ".midi"}
        )
        return {
            "status": "success",
            "message": f"Corpus extracted ({mid_count} MIDI files).",
            "midi_files": mid_count,
        }
    except zipfile.BadZipFile as e:
        zip_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid ZIP: {e}") from e
    except Exception as e:
        zip_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to extract: {e}") from e


@app.post("/api/train")
async def train_model(req: TrainRequest, background_tasks: BackgroundTasks):
    def _run() -> None:
        try:
            args = argparse.Namespace(
                train=str(CORPUS_DIR),
                preset=req.preset,
                steps=req.steps,
                ckpt=str(CKPT_DIR / req.ckpt_name),
                device="auto",
            )
            logger.info("training %s for %s steps…", req.preset, req.steps)
            code = cmd_train(args)
            if code == 0:
                logger.info("training complete")
                get_cached_model.cache_clear()
            else:
                logger.error("training exit code %s", code)
        except Exception as e:
            logger.exception("training crashed: %s", e)

    background_tasks.add_task(_run)
    return {
        "status": "training_started",
        "message": f"Training {req.preset} for {req.steps} steps → checkpoints/{req.ckpt_name}",
    }


@app.get("/api/models")
async def list_models():
    found = sorted(p.name for p in CKPT_DIR.glob("*.pt"))
    return {"models": ["default_150m.pt"] + found, "use_real": USE_REAL}


if __name__ == "__main__":
    if uvicorn is None:
        raise SystemExit("pip install uvicorn")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
