# MidiNote Neural

**Text-to-MIDI studio** — prompt → multi-track Type-1 MIDI. Deterministic, safety-screened, trainable.

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Quick start

```bash
pip install -r requirements.txt
python app.py
# open http://localhost:8000
```

Runs in **stub mode** by default (no GPU required).

### Real NeuralMidi weights

```bash
export NEURALMIDI_REAL=1
export PYTHONPATH="$(pwd):${PYTHONPATH}"   # so Tools.NeuralMidi imports
python app.py
```

You should see: `⚡ RUNNING REAL NEURALMIDI CORE`

Bundled checkpoint: `checkpoints/default_150m.pt` (tiny ScoreNet; replace with a larger trained ckpt as needed).

## Features

- Genre / tempo / seed controls
- Sequential stems: Drums → Bass → Chords → Lead (`prompt_ids` context)
- Instant WebAudio + piano-roll feed (`html-midi-player`)
- ZIP corpus upload + background training
- Deterministic: same seed + prompt + ckpt → identical MIDI
- Safety screen (`screen_prompt`) + Type-1 SMF (drums ch 10)

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/generate` | Multi-track MIDI blob |
| `POST` | `/api/upload-corpus` | ZIP of training files |
| `POST` | `/api/train` | Background ScoreNet train |
| `GET`  | `/api/models` | Checkpoint list |

### Generate example

```bash
curl -s -X POST http://localhost:8000/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"sad piano in C minor","seed":42,"tempo":72,"ckpt":"default_150m.pt"}' \
  -o out.mid
```

## Layout

```
MidiNote-Neural/
├── app.py                 # FastAPI host (real / stub dual-path)
├── nm_stub.py             # Offline fallback engine
├── static/index.html      # UI
├── checkpoints/           # *.pt weights
├── corpus/                # training data
├── Tools/NeuralMidi/      # ScoreNet + next_seq + safety (GPL-3.0)
└── requirements.txt
```

## Train

Drop MIDI / note JSON into `corpus/`, then use the UI **Train ScoreNet** panel, or:

```bash
NEURALMIDI_REAL=1 PYTHONPATH=. python -c "
import argparse
from Tools.NeuralMidi.neuralmidi import cmd_train
cmd_train(argparse.Namespace(train='corpus/', preset='tiny', steps=200,
                             ckpt='checkpoints/user.pt', device='auto'))
"
```

## License

- **NeuralMidi core** (`Tools/NeuralMidi`): GPL-3.0-or-later  
- Host / UI: same terms unless otherwise noted  

## Related

Plugin host (BlackBox Studio / PolyChains) is a **separate** product and is not included in this release.
