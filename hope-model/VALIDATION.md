# Hope: verified training run

Completed on 2026-10-13 using Python 3.12.14 and PyTorch 2.6.0 on CPU.

## Results

- All three unit tests passed: causal masking, loss reduction, and generation beyond the context window.
- Hope trained from random weights for 1,500 steps on `data/alice.txt`.
- Model: 356,331 parameters, 75-character vocabulary, 3 layers, 4 attention heads, width 96, context 64.
- Batch size: 16. AdamW learning rate: 0.0003. Random seed: 42.
- Training loss: 4.4853 initially, 1.9915 finally.
- Held-out validation loss: 4.4695 initially, 2.0054 finally. These are average next-character cross-entropy losses in natural-log units on fixed sampled evaluation windows, not accuracy percentages.
- Final weights were saved to `runs/hope/model.pt` and successfully reloaded by `generate.py` to generate 300 new characters.
- The original internet source and checksum are recorded in `data/source.json`.

## Limitations

This first small run learns fragments of spelling and prose patterns. Output still includes invented words and incoherent sentences. It is a text-completion learning experiment, not a question-answering assistant, and it has no live internet access.

The dataset is one book. The final 10% of its characters are held out; scores measure performance on the same book, not broad language ability. Vocabulary is derived from the entire book.

Windows prevented the package installer from creating optional command-line launchers. The Python libraries were installed and their imports, tests, training, and inference succeeded. PyTorch emits an optional NumPy warning in this environment; this project does not use NumPy interoperability.

## Reproduce

```powershell
python -m unittest -v
python train.py --data data/alice.txt --steps 1500 --out runs/hope --eval-every 250
python generate.py --prompt "Alice " --length 300 --temperature 0.7
```

Training again overwrites the named run. Use a different `--out` to preserve these weights. Random seeds are fixed, but exact outputs can differ across hardware or library versions.
