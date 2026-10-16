# Hope Original

Hope Original is a small experimental language model built from scratch in Python and PyTorch. It predicts the next character to continue a prompt. This repository distributes one model, trained on three storybooks.

## Model size and training

- **843,138 parameters**; checkpoint **3,387,008 bytes (about 3.39 MB)**.
- 4 Transformer layers, 4 attention heads, embedding width 128.
- Context: 128 characters; vocabulary: 130 characters.
- Saved after **3,000 training steps** on CPU. The planned 4,000-step session was interrupted.
- AdamW, learning rate 0.0003, batch size 16, seed 42.
- Recorded validation loss decreased from 5.0411 to 1.7293. This is not an accuracy percentage.

## Training data

The model learned from Project Gutenberg editions of *Alice's Adventures in Wonderland*, *Three Hundred Aesop's Fables*, and *Grimms' Fairy Tales*. Prepared data contains 817,109 training characters and 91,964 validation characters. Approximately the final 10% of each book was held out at a paragraph boundary. Original source files, notices, and provenance are included in `data/`.

It was not trained on the whole internet or on the Hands-On Machine Learning textbook.

## Download and run

- [Complete model folder](./)
- [Hope Original weights](runs/hope/model.pt)
- [Detailed model card](MODEL_CARD.md)
- [File checksums](../SHA256.json)

Choose **Code > Download ZIP**, extract the repository, and open a terminal in `hope-model`. Install Python 3.10 or newer.

Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe generate.py --prompt "Once upon a time" --length 300 --temperature 0.7
```

macOS / Linux:
```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python generate.py --prompt "Once upon a time" --length 300 --temperature 0.7
```

Generation uses `runs/hope/model.pt` by default. Length is measured in characters. Prompts must use characters in the model vocabulary. GitHub stores the files; it does not run the model.

## Training and tests

With the virtual environment activated:
```sh
python -m unittest -v
python train.py --data data/stories/train.txt --val-data data/stories/validation.txt --out runs/new-training --width 128 --heads 4 --layers 4 --context 128 --steps 4000 --eval-every 500
```

Training starts from random weights. It does not resume the included model. Use a separate output folder to preserve the checkpoint. Exact resume support requires optimizer and random-generator state, which are not saved.

## Files and limitations

`model.py` defines the Transformer; `train.py` trains it; `generate.py` generates text; `test_model.py` checks basic mechanics. Dataset preparation scripts and source manifests are included. `runs/hope/` contains the single checkpoint and its recorded metrics.

Hope Original is an educational text-completion experiment. It often produces invented words and incoherent sentences. It has no instruction tuning, browsing, external tools, or reliable factual-answering capability. No demo website is included.
