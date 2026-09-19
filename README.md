# The Polyglot's Shorthand: An Efficient Intent Classifier for Romanized Code-Mixed Text

An intent-classification system for Latinized/Romanized Hinglish customer-support text, built under a 500M-parameter, single-digit-millisecond latency budget, and made robust to spelling drift, slang, typos, and emoji/punctuation as semantic signal.

**Full writeup: [`whitepaper.md`](./whitepaper.md)** — read this first. It documents the entire process end to end: problem understanding, dataset design, model selection, training, latency benchmarking, error analysis, a held-out robustness evaluation, and a secondary sentiment-analysis exploration. Every claim in it is backed by a script in this repo and is reproducible from the steps below.

**Recommended model: XLM-R-base**, fp32, GPU inference — 6.2ms mean p95 single-request latency, 91.67% accuracy on the held-out robustness set (Dataset B), full reasoning in whitepaper Sections 7, 9, 11.

---

## Repository structure

```
.
├── whitepaper.md                          # Full technical writeup — start here
├── src/                                   # All scripts, in the order used
│   ├── check_dataset.py                   # Validates Dataset A v2 (row counts, balance, leakage)
│   ├── tokenizer_audit.py                 # Token-length audit -> justifies MAX_LENGTH=64
│   ├── train.py                           # Trains intent classifier (ModernBERT/XLM-R/MuRIL/mDeBERTa)
│   ├── benchmark_latency.py               # Single-request + batched latency/throughput benchmark
│   ├── error_analysis.py                  # Confusion matrix, error breakdown by variant/domain/intent
│   ├── inspect_confusion.py               # Deep-dives a specific confusion pair across models
│   ├── patch_dataset.py                   # Documented, non-destructive fix for one ambiguous base utterance
│   ├── parse_whatsapp_export.py           # Keyword-candidate extraction (abandoned approach, kept for record)
│   ├── eval_dataset_B.py                  # Evaluates a trained model on the held-out Dataset B
│   ├── build_sentiment_dataset_v2.py      # Generates the sentiment dataset (secondary task)
│   └── train_sentiment.py                 # Trains + evaluates the sentiment classifier
├── data/
│   ├── dataset_A_v2_36class_final.csv             # Primary dataset (frozen), 21,600 rows, 36 intents
│   ├── dataset_A_v2_36class_final_patched.csv     # Same, with one documented text correction (see whitepaper 8.3)
│   ├── dataset_A_v2_patch_changelog.json          # What changed in the patch, and why
│   ├── dataset_A_v2_36class_taxonomy.json         # Intent/domain definitions
│   ├── dataset_A_v2_36class_final_audit.json      # Dataset integrity audit (row counts, split sizes, leakage checks)
│   ├── dataset_A_v2_36class_2160_semantic_bases.csv
│   ├── dataset_B_robustness.csv                   # Held-out, independently-authored robustness set (108 rows)
│   ├── dataset_sentiment_v2.csv                   # Secondary sentiment dataset (1,860 rows)
│   └── whatsapp_candidates.csv                    # Small filtered sample from the abandoned WhatsApp approach
└── outputs/                               # Generated at training time (see .gitignore — not committed)
```

**Note on `outputs/`:** trained model weights and checkpoints are excluded from this repo via `.gitignore` (see below) — they're large binary files unsuited to git, and every number in the whitepaper is reproducible from the scripts and data above. If you need the actual trained weights rather than reproducing them, ask separately (e.g. a release asset or external storage link) rather than expecting them in the git history.

---

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install torch transformers datasets scikit-learn pandas numpy accelerate sentencepiece protobuf
```

Requires a CUDA-capable GPU to reproduce the latency figures as reported (measured on an RTX 5060 Laptop GPU, ~8GB VRAM); training and evaluation will also run on CPU, but latency numbers won't match Section 7's GPU figures.

---

## Reproduction steps

Run everything from the repository root (not from inside `src/`).

**1. Validate the dataset**
```bash
python src/check_dataset.py
python src/tokenizer_audit.py
```

**2. Train the intent classifiers**
```bash
python src/train.py --model modernbert
python src/train.py --model xlmr
python src/train.py --model muril
python src/train.py --model mdeberta   # documented to fail — see whitepaper Section 5.1
```

**3. Benchmark latency** (repeat flag recommended for the models you care most about)
```bash
python src/benchmark_latency.py --model_dir outputs/xlmr/best_model --device cuda --repeats 3
python src/benchmark_latency.py --model_dir outputs/xlmr/best_model --device cuda --fp16 --repeats 3
python src/benchmark_latency.py --model_dir outputs/xlmr/best_model --device cpu
# repeat for outputs/modernbert/best_model and outputs/muril/best_model
```

**4. Error analysis on Dataset A's own test split**
```bash
python src/error_analysis.py --model_dir outputs/modernbert/best_model --device cuda
python src/error_analysis.py --model_dir outputs/xlmr/best_model --device cuda
python src/inspect_confusion.py   # reads the two CSVs above, no GPU needed
```

**5. (Optional) Apply and re-verify the documented dataset patch**
```bash
python src/patch_dataset.py
python src/error_analysis.py --model_dir outputs/modernbert/best_model --device cuda --data data/dataset_A_v2_36class_final_patched.csv
python src/error_analysis.py --model_dir outputs/xlmr/best_model --device cuda --data data/dataset_A_v2_36class_final_patched.csv
```

**6. Evaluate on the held-out robustness set (Dataset B)**
```bash
python src/eval_dataset_B.py --model_dir outputs/modernbert/best_model --device cuda
python src/eval_dataset_B.py --model_dir outputs/xlmr/best_model --device cuda
python src/eval_dataset_B.py --model_dir outputs/muril/best_model --device cuda
```

**7. (Optional) Secondary task: sentiment analysis**
```bash
python src/build_sentiment_dataset_v2.py   # regenerates data/dataset_sentiment_v2.csv, or just use the committed file
python src/train_sentiment.py
```

Every number reported in `whitepaper.md` was produced by one of the commands above — nothing in the writeup is estimated or hand-adjusted.

---

## Key findings (see whitepaper for full detail)

- **Parameter count does not predict latency.** XLM-R (278M params) and MuRIL (238M params) are both faster than ModernBERT (150M params) at the single-request serving pattern this task requires — ModernBERT's architecture is optimized for large-batch throughput, which doesn't match this workload. (Section 7)
- **In-distribution test accuracy is a poor proxy for real-world robustness.** All models scored >99.5% on Dataset A's own test split, but dropped to 76-92% on an independently-authored held-out set (Dataset B) — a gap that would have gone undetected without a deliberately separate, never-tuned-against benchmark. (Section 9)
- **The model correctly reads emoji as pragmatic/sentiment signal, not noise** — validated directly via an emoji-sarcasm-flip test (positive text + sarcastic emoji → correctly classified negative), 100% correct across both sentiment dataset versions. (Appendix A)
- One model (mDeBERTa-v3-base) failed to train in this environment despite three independent, correctly-diagnosed fix attempts; this is documented as a negative result rather than silently omitted. (Section 5.1)