# RUNLOG

## Baseline
- Command: `python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt`
- Score: 2.3718 bpb on `../data/dev_eval.txt`
- Params: 1,339,840
- Notes: stock byte tokenizer, stock 4-layer/160-width GPT, constant LR Adam.

## Improved run
- Command: `python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt`
- Score: 2.1009 bpb on `../data/dev_eval.txt`
- Params: 1,913,280
- Notes: corpus-trained byte-fallback tokenizer with leading-space units, tied weights, AdamW, warmup + cosine decay, gradient clipping.
