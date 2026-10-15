# Hope model card

## Checkpoints

| Property | Original Hope | Hope Stories v2 |
|---|---:|---:|
| Parameters | 356,331 | 843,138 |
| Transformer blocks | 3 | 4 |
| Embedding width | 96 | 128 |
| Attention heads | 4 | 4 |
| Context length | 64 characters | 128 characters |
| Vocabulary | 75 characters | 130 characters |
| Saved training step | 1,500 | 3,000 |
| Initial validation loss | 4.4695 | 5.0411 |
| Last recorded validation loss | 2.0054 | 1.7293 |

Stories v2 was planned for 4,000 steps, but the training session was interrupted. The included checkpoint is at step 3,000. Do not interpret it as a finished 4,000-step run. The validation datasets differ, so the loss values are not a controlled comparison between versions.

## Method

Character-level next-token prediction, learned token and positional embeddings, pre-normalized causal self-attention, GELU feedforward layers, and cross-entropy loss. Both checkpoints were trained from randomly initialized weights on CPU, using AdamW, learning rate 0.0003, batch size 16, gradient clipping at 1.0, and random seed 42. No pretrained external model or paid API was used.

The recorded environment was Python 3.12.14 and PyTorch 2.6.0. Exact generation may vary across hardware and software versions.

## Data

Original: 144,600 characters from Lewis Carroll's *Alice's Adventures in Wonderland*, with the first 90% used for training and the last 10% for validation.

Stories v2: *Alice's Adventures in Wonderland*, *Three Hundred Aesop's Fables*, and *Grimms' Fairy Tales*. The approximately final 10% of each book was held out at a paragraph boundary. Including separators, the files contain 817,109 training characters and 91,964 validation characters. Vocabulary is derived from both splits. This is within-book validation, not an independent-book benchmark.

Source catalogs:
- https://www.gutenberg.org/ebooks/11
- https://www.gutenberg.org/ebooks/21
- https://www.gutenberg.org/ebooks/2591

Project Gutenberg lists these editions as public domain in the USA. Original downloaded files retain their source notices and licenses. Gutenberg headers and footers are removed from the prepared training text; book front matter remains. Download URLs and original-file checksums are included in `data/source.json` and `data/stories/sources.json` inside `hope-model/`. Source-text permissions are separate from any license for the model code or weights; no additional code/weights license is declared here.

## Verification

Three automated model tests passed: causal masking prevents future-token leakage; optimization reduces loss on a simple repeating sequence; and generation works past the model context limit. Both saved checkpoints were successfully loaded and used for generation. The local inference integration also passed repeatability checks with fixed seeds and output-length checks. These checks verify mechanics, not language quality. No standardized QA, reasoning, or conversational benchmark has been completed.

## Limitations

Hope is a small experimental text-completion model. It may produce nonsense, invented words, inconsistent statements, and text reflecting its old literary training corpus. It is not suitable for reliable answers or consequential decisions. It has no browsing, instruction tuning, persistent memory, or tools. The short context limits its ability to maintain a story. Training repeatedly on the same texts may overfit.

## Continued training

The existing trainer starts from scratch. Included checkpoints do not save optimizer state or random-generator state for exact resumption. Further work should add resumable training, stronger evaluation, and carefully chosen data before claiming improved capability.
