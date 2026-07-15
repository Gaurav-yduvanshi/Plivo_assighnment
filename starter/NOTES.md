Best current configuration:

- Tokenizer: corpus-trained byte-fallback tokenizer saved as `tokenizer.json`.
- Tokenization change: attach leading whitespace to the following unit and learn frequent UTF-8 byte substrings from the provided corpus.
- Model: 4-layer GPT, 160 hidden size, 4 heads, tied input/output embeddings.
- Optimization: AdamW with weight decay 0.1, warmup fraction 0.03, cosine decay to 0.1 of peak LR, gradient clipping at 1.0.
- Final checkpoint: `ckpt.pt`.
- Dev score: 2.1009 bpb on `dev_eval.txt`.
- Parameter count: 1,913,280, safely under the 2,000,000 cap.
- The tokenizer change mattered most because it reduced sequence length on mixed English/Hindi text.
- The model and optimizer changes helped stabilize training under the fixed 2,000-step budget.
- The evaluator still loads the checkpoint unchanged.
