# Hope — tiny language models trained from scratch

Hope is a character-level, decoder-only Transformer written in Python and PyTorch. This repository contains the complete model folder, two trained checkpoints, training code, source texts, provenance, tests, and recorded losses. **No demo website is included.**

## Download

- [Complete model folder](hope-model/): browse the source, data, and trained checkpoints.
- [Hope Stories v2 weights](hope-model/runs/hope-stories-v2/model.pt): 843,138 parameters, saved at 3,000 training steps.
- [Hope Original weights](hope-model/runs/hope/model.pt): 356,331 parameters, saved at 1,500 training steps.
- [Detailed model card](MODEL_CARD.md): architecture, data, results, limitations, and validation status.
- [SHA-256 checksums](SHA256.json): integrity checks for the downloadable files.

GitHub stores these files; it does not execute the model. Run the Python commands below on your computer.

## Setup and generation

Download this repository using **Code > Download ZIP**, extract it, open a terminal in `hope-model`, and install Python 3.10 or newer.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe generate.py --checkpoint runs/hope-stories-v2/model.pt --prompt "Once upon a time" --length 300 --temperature 0.7
```

### macOS / Linux

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python generate.py --checkpoint runs/hope-stories-v2/model.pt --prompt "Once upon a time" --length 300 --temperature 0.7
```

For the original model, change the checkpoint to `runs/hope/model.pt`. Prompts must use characters present in the selected model's training vocabulary. Length is measured in characters. The model sees only the most recent 64 or 128 characters, depending on the checkpoint.

## Train and test

With the virtual environment activated, run:

```sh
python -m unittest -v
python train.py --data data/stories/train.txt --val-data data/stories/validation.txt --out runs/new-stories --width 128 --layers 4 --context 128 --steps 4000 --eval-every 500
```

The training command starts a **new model from random weights**. It does not resume the included checkpoints. Use a new output directory to preserve existing weights. Checkpoints contain inference weights and vocabulary, not optimizer state for an exact restart. The trainer saves best validation weights as `model.pt`; completed runs also save `last.pt`.

## Folder contents

| Path | Purpose |
|---|---|
| `model.py` | Transformer layers, causal attention, and generation |
| `train.py` | Training, held-out evaluation, and checkpoint saving |
| `generate.py` | Command-line text generation |
| `test_model.py` | Causal masking, learning, and context-window tests |
| `prepare_data.py`, `prepare_stories.py` | Reproducible text preparation and downloads |
| `data/` | Training texts, original Gutenberg files, licenses, and source manifests |
| `runs/hope/` | Original checkpoint, metrics, and sample |
| `runs/hope-stories-v2/` | Saved 3,000-step checkpoint and metrics |
| `compare_models.py` | Optional same-window comparison script; no completed comparative report is claimed |

This is a learning project, not an instruction-tuned chatbot. It often generates invented words or incoherent sentences. Training has not established reliable factual knowledge or conversational ability.
