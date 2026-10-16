# Hope Original validation

The included checkpoint has 843,138 parameters and is saved at step 3,000. Its recorded validation loss decreases from 5.0411 to 1.7293; see `runs/hope/metrics.json`. The planned 4,000-step run was interrupted.

Three automated tests cover causal masking, learning a repeating pattern, and generation beyond the context window. Checkpoint loading and text generation are verified separately. These checks assess mechanics, not conversational or factual reliability. No standardized language benchmark has been completed.

Run `python -m unittest -v` and `python generate.py --prompt "Once upon a time" --length 100` from this folder after installing requirements. Training starts from scratch; use a separate output folder.
