# Hope Original — model card

## Identity

Hope Original is the single distributed model. The selected checkpoint is the larger three-book training run previously labelled Hope Stories v2. The public name has been corrected at the owner's request; this change does not merge weights or add training. The smaller earlier experiment is retained locally and in Git history, not distributed as a second current model. Checkpoint bytes are unchanged, including their internal name `Hope`.

## Architecture and size

Character-level decoder-only Transformer with learned token and positional embeddings, pre-normalized causal self-attention, GELU feedforward layers, and cross-entropy loss. It has 843,138 parameters, 4 blocks, width 128, 4 attention heads, a 128-character context and a 130-character vocabulary. The checkpoint is 3,387,008 bytes (3.39 decimal MB).

## Training data

Project Gutenberg sources:
- Alice's Adventures in Wonderland: https://www.gutenberg.org/ebooks/11
- Three Hundred Aesop's Fables: https://www.gutenberg.org/ebooks/21
- Grimms' Fairy Tales: https://www.gutenberg.org/ebooks/2591

Prepared files have 817,109 training and 91,964 validation characters, including separators. Approximately the last 10% of each book is held out at a paragraph boundary. Vocabulary is derived from both splits. Validation is within-book, not an independent-book benchmark. Source manifests are in `data/stories/sources.json`. Original texts retain Gutenberg notices; prepared texts omit Gutenberg headers and footers, but retain book front matter. These editions are listed as public domain in the USA; permissions elsewhere may differ. No additional code or weights license is declared.

## Training and results

Trained from scratch on CPU with Python 3.12.14 and PyTorch 2.6.0, AdamW, learning rate 0.0003, batch size 16, gradient clipping 1.0 and seed 42. Saved step: 3,000. The planned 4,000-step run was interrupted. Last recorded training loss: 1.6327. Validation loss: initially 5.0411, last recorded 1.7293. These are character cross-entropy losses, not accuracy percentages.

## Validation and limitations

Three model tests passed: causal masking, loss reduction on a simple repeating sequence, and generation beyond the context window. Checkpoint loading and text generation succeeded. No standardized QA, reasoning, or conversation benchmark has been completed. Output often includes nonsense, invented words and inconsistent sentences. The short context and small literary corpus limit capability; older texts can contribute biases. It is not an instruction-tuned assistant and has no browsing, tools or persistent memory.

## Further training

The current trainer starts from scratch. Checkpoints lack optimizer and RNG states needed for exact resumption. Renaming the model does not combine the two earlier training runs or count their training steps together.
