# The Polyglot's Shorthand: An Efficient Intent Classifier for Romanized Code-Mixed Text

## Abstract

South Asian customer-support text is overwhelmingly Latinized and code-mixed — Hinglish typed in Roman script, full of phonetic spelling drift, slang, and emoji that carry real meaning rather than noise. Existing tooling handles this badly: native-script Indic models fragment on Romanized input, English models treat the syntax as malformed, and large LLMs are too slow and expensive for real-time routing. This paper builds and evaluates an intent classifier for this text under hard constraints — under 500M parameters, single-digit-millisecond latency — while staying robust to exactly the kind of noise this text actually contains.

We built an original 21,600-example dataset spanning 36 customer-support intents, generated with explicit surface-level variation (typos, shorthand, code-mixing, emoji, punctuation) and split by semantic base to avoid leakage. Four candidate encoder models were trained and compared: ModernBERT, XLM-R, MuRIL, and mDeBERTa (the last of which failed to train for reasons documented as a negative result). Latency was benchmarked properly — single-request, not batched — and a second, independently-authored held-out set (Dataset B) was built specifically to catch a problem the in-distribution test split couldn't: every model scored above 99.5% on Dataset A's own test split, but dropped to 76-92% on Dataset B, exposing how much of that headline accuracy was really just memorizing one dataset's writing style rather than the underlying task.

XLM-R came out ahead on the measures that actually mattered — 6.2ms median-case latency and the best generalization to genuinely unseen phrasing — and is the model we recommend for deployment. A smaller, secondary experiment in sentiment analysis (Appendix A) tests the emoji-as-signal requirement directly: the same sentence, praise on its own but sarcasm with the right emoji attached, gets classified correctly in both directions, which is a decent sign that emoji is being read for meaning rather than stripped as noise.

---

## 1. Problem Statement

Customer support and consumer-platform text traffic across South Asia is dominated by informal, Latinized/Romanized code-mixed language (Hinglish and similar), rather than formal native-script or pure-English text. Existing NLP infrastructure handles this poorly:

- Native-script Indic models (e.g. IndicBERT) expect native scripts and fragment Romanized words into out-of-vocabulary noise.
- English-only models treat the code-mixed syntax as ill-formed input.
- Large frontier LLMs can interpret the text correctly, but their latency and cost make them unsuitable for real-time, high-throughput routing.

This is compounded by conversational shorthand: phonetic spelling drift ("kya kar rahe ho" / "kya kr rhe ho" / "kya krre ho"), intra-sentence English-Indic code-mixing, and emojis/punctuation that carry or invert meaning.

**Deployment constraints given:**
- Maximum 500M parameters
- Single-digit-millisecond inference latency
- High throughput
- Robustness to phonetic spelling drift, slang, typos, emojis, and punctuation as semantic signal, rather than noise to be stripped

**Scope of this work:** Of the downstream tasks the problem statement allows (sentiment analysis, intent classification, summarization, question answering), we focus on **intent classification** for customer-support-style utterances, and build a purpose-built dataset and evaluation pipeline for it.

---

## 2. Dataset Strategy

Two datasets were planned:

- **Dataset A** — the primary training/development/test dataset, used for model selection and iteration.
- **Dataset B** — a separate, held-out robustness benchmark, not used for tuning, reserved for final evaluation only. This separation exists specifically so that we do not optimize against the benchmark we are ultimately judged on.

*(Dataset B design and results: see Section 9.)*

---

## 3. Dataset A v2

Dataset A v2 is an original, constructed dataset (no external licensed utterances) covering 36 customer-support intents across 8 domains.

| Property | Value |
|---|---|
| Total examples | 21,600 |
| Intents | 36 |
| Domains | 8 |
| Semantic base utterances | 2,160 |
| Bases per intent | 60 |
| Surface variants per base | 10 |
| Examples per intent | 600 |
| Source | original_constructed |
| External licensed utterances | None |

**Domain breakdown:**

| Domain | Intents |
|---|---|
| Orders | 5 |
| Delivery | 5 |
| Returns & Refunds | 5 |
| Subscriptions & Billing | 5 |
| Account & Support | 4 |
| Payments | 4 |
| Travel & Transport | 4 |
| General | 4 |

Each of the 36 intents is realized as **60 semantic base utterances**, each of which is expanded into **10 surface variants** covering: clean, shorthand, code-mix, typo, emoji, punctuation, shorthand+typo, shorthand+emoji, code-mix+emoji, and stress combinations. This directly targets the problem statement's robustness requirements (spelling drift, slang, typos, emoji/punctuation as semantic signal) rather than treating them as separate noise to filter out.

### 3.1 Splitting and leakage control

The dataset was split **by semantic base**, not by individual row, so that all 10 surface variants of a given base utterance stay together in one split. This prevents a model from seeing one paraphrase of a sentence in training and a near-duplicate in test.

| Split | Base utterances / intent | Rows |
|---|---|---|
| Train | 42 | 15,120 |
| Validation | 9 | 3,240 |
| Test | 9 | 3,240 |

Verified leakage control:
- Zero normalized-text overlap between train/validation/test (train-vs-validation, train-vs-test, validation-vs-test all measured at 0 overlapping normalized texts)
- No empty text fields
- Unique IDs
- Correct, exactly-uniform class distribution (600 rows/intent) and variant distribution (2,160 rows/variant type)

---

## 4. Tokenization Audit

Token-length statistics were measured on the train+validation portion for both candidate tokenizers, to set `MAX_LENGTH` empirically rather than arbitrarily.

| Percentile | ModernBERT | XLM-R |
|---|---|---|
| P50 | 28 | 23 |
| P90 | 38 | 31 |
| P95 | 41 | 34 |
| P99 | 47 | 39 |
| Max | 61 | 51 |

`MAX_LENGTH = 64` was locked as a result — 0% of observed sequences exceeded this in the audit, for either tokenizer. MuRIL and mDeBERTa (added later, Section 5.1) reused this same value without an independent tokenizer audit; given both use similar subword tokenization schemes to XLM-R and the underlying text is unchanged, this is a reasonable assumption but was not separately verified.

---

## 5. Model Selection

Two candidate architectures were selected first, chosen to represent genuinely different points in the design space rather than two arbitrary similarly-sized models:

- **ModernBERT-base** represents a modern, efficiency-oriented encoder architecture (alternating local/global attention, rotary position embeddings, unpadded sequence processing) explicitly designed for throughput at scale — a natural fit for the problem statement's "high throughput" requirement on paper.
- **XLM-R-base** represents a mature, extensively validated multilingual encoder with a strong track record on code-mixed and low-resource-language text specifically — directly relevant to the Latinized/Romanized Indic text this problem targets, independent of any architectural efficiency claims.

Comparing these two lets us separate two different bets: "newer architecture, optimized for scale" versus "proven multilingual robustness" — rather than assuming a priori which one would win on this task's actual serving pattern (short sequences, single-request latency). Section 7 shows this comparison was worth making: the two models trade places on latency in a way that would not have been predicted from parameter count or architectural novelty alone.

Following the results in Sections 7 and 9, two further candidates were added to widen the comparison:

- **MuRIL-base-cased** (`google/muril-base-cased`) — purpose-built by Google specifically for Indian languages and code-mixed/transliterated text, making it a direct domain match for this problem statement rather than a general-purpose multilingual model applied to this domain.
- **mDeBERTa-v3-base** (`microsoft/mdeberta-v3-base`) — DeBERTa's disentangled-attention mechanism has shown strong sample-efficiency and accuracy in published benchmarks, making it a reasonable candidate for squeezing more accuracy out of a similar parameter budget.

All four candidates are well under the 500M-parameter budget:

| Model | Model ID | Parameters |
|---|---|---|
| ModernBERT-base | `answerdotai/ModernBERT-base` | 149,632,548 |
| MuRIL-base-cased | `google/muril-base-cased` | 237,583,908 |
| XLM-R-base | `FacebookAI/xlm-roberta-base` | 278,071,332 |
| mDeBERTa-v3-base | `microsoft/mdeberta-v3-base` | 278,837,028 |

All are pretrained multilingual/general-purpose transformer encoders (MuRIL is multilingual across Indian languages specifically); none was shrunk artificially just to minimize the reported parameter count, since all were already comfortably within budget.

### 5.1 mDeBERTa-v3-base: training failure (documented negative result)

mDeBERTa-v3-base could not be successfully trained in this environment (`transformers==5.17.0`), despite three independent, methodical fix attempts:

1. **Initial fp16 run** crashed immediately with `ValueError: Attempting to unscale FP16 gradients` — a known DeBERTa-v3 issue where its embedding-sharing mechanism keeps some weights in fp16 even after a standard model load, breaking `GradScaler`'s fp16-unscale step. **Fix attempted:** switched to bf16 (no gradient scaler required).
2. **bf16 run** no longer crashed, but failed to learn: accuracy stayed at 2.78% (exactly 1/36, the random-guess baseline) across all epochs, with an oscillating, non-decreasing loss and an unstable grad-norm spike (43.4) at epoch 2. **Fix attempted:** DeBERTa-v3's documented sensitivity to AdamW's default `epsilon=1e-8` (its disentangled attention produces large embedding-scale values that cause a near-zero-denominator instability) was addressed by switching to full fp32 precision and raising `adam_epsilon` to `1e-6`, both standard, documented fixes for this exact architecture.
3. **fp32 + adam_epsilon=1e-6 run** eliminated the NaN/instability (loss and grad-norm became finite and smoothly decreasing in magnitude), but loss remained pinned at ln(36) ≈ 3.584 — the exact loss value of a model outputting uniform random probability across all 36 classes — for all three epochs before early stopping triggered. **Fix attempted:** a 5x learning-rate increase (2e-5 → 1e-4), to test whether the classifier head was simply undertrained.
4. **fp32 + adam_epsilon=1e-6 + learning_rate=1e-4** still produced loss pinned at 3.584-3.586 with no meaningful movement across epochs, while grad-norm shrank toward zero (1.28 → 0.36 → 0.31) — the signature of a model output that has collapsed to a constant the optimizer cannot escape, not a model learning slowly.

Ruling out a fp16-specific bug, an optimizer-epsilon instability, and an underfit learning rate leaves a fourth, unaddressed possibility: a version-specific incompatibility between `transformers==5.17.0` and this architecture's `DebertaV2ForSequenceClassification` implementation, given DeBERTa-v2/v3 is a comparatively less mainstream architecture than the BERT/RoBERTa-family models (ModernBERT, XLM-R, MuRIL) that all trained successfully in the same environment, on the same data, with the same pipeline. This was judged out of scope to debug further (see Section 12); **mDeBERTa-v3-base is excluded from all downstream comparisons in this whitepaper**, and MuRIL, ModernBERT, and XLM-R form the valid three-model comparison set.

---

## 6. Training Pipeline


A single, frozen training configuration was used across all four candidate models, for a fair comparison:

| Setting | Value |
|---|---|
| Seed | 42 |
| Max sequence length | 64 |
| Epochs | 5 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Warmup ratio | 10% |
| Train batch size | 8 |
| Eval batch size | 32 |
| Gradient accumulation | 1 |
| Precision | FP16 (CUDA) |
| Model-selection metric | Validation macro-F1 |
| Early stopping | Patience 2, on validation macro-F1 |

**Hardware/environment:** RTX 5060 Laptop GPU (~8GB VRAM), PyTorch 2.14.0+cu132, CUDA 13.2, driver 595.97, Python 3.11.9, Transformers 5.17.0.

An initial ModernBERT run at batch size 16 hit a CUDA CUBLAS internal error partway through training. This was diagnosed (standalone FP16 GEMM test, CUDA/VRAM/process checks) rather than assumed to be a hardware fault, and resolved by reducing the training batch size to 8, after which training completed cleanly. ModernBERT and XLM-R (the original two candidates) were trained under this batch-size-8 configuration; MuRIL and mDeBERTa, added later (Section 5.1), used the same batch size from the start.

### 6.1 Training results

| Metric | ModernBERT | XLM-R | MuRIL |
|---|---|---|---|
| Parameters | 149.6M | 278.1M | 237.6M |
| Best validation macro-F1 | 99.938% | 99.938% | 99.938% |
| Test accuracy | 99.568% | 99.599% | 99.599% |
| Test macro-F1 | 99.567% | 99.598% | 99.598% |
| Test weighted-F1 | 99.567% | 99.598% | 99.598% |

All three models converge to near-identical, very high accuracy on Dataset A v2's frozen test split — XLM-R and MuRIL are effectively tied. Parameter count alone does not separate them on accuracy; at this saturation level, Dataset A's own test split has stopped being able to discriminate between models. Sections 7 and 9 show that latency and held-out generalization (Dataset B) are where real differences actually emerge.

---

## 7. Latency & Throughput Benchmarking

Trainer's batched-evaluation runtime figures are not representative of production single-request latency (they reflect large-batch GPU evaluation, not the batch=1 request pattern of a real-time routing service). A dedicated benchmark was built to measure this directly.

**Method:**
- Real test-set utterances used (not synthetic strings), so sequence-length distribution matches production traffic.
- Single-example (batch=1) latency measured per-request, with 20 warmup requests discarded and 500 measured requests, `torch.cuda.synchronize()` called immediately before/after each timed forward pass.
- Batched throughput measured separately at batch size 32 (20 warmup + 50 measured batches).
- Each configuration repeated 3× for the top candidates to confirm run-to-run stability rather than reporting a single noisy run.
- Configurations tested: GPU fp32, GPU fp16 (via `torch.autocast`), and CPU fp32, for all three models.

### 7.1 Results

| Model | Config | p50 (ms) | p95 (ms) | p99 (ms) | Batch-32 throughput (ex/s) | Meets <10ms target? |
|---|---|---|---|---|---|---|
| ModernBERT | GPU fp32 | 11.3 | 12.0 | 12.6 | 670 | No |
| ModernBERT | GPU fp16 | 15.0 | 16.7 | 17.5 | 1,415 | No |
| ModernBERT | CPU fp32 | 27.0 | 33.0 | 35.0 | 58 | No |
| XLM-R | GPU fp32 | 5.7 | 6.2 | 6.8 | 1,146 | Yes |
| XLM-R | GPU fp16 | 8.1 | 8.7 | 9.2 | 2,862 | Yes (tighter margin) |
| XLM-R | CPU fp32 | 17.2 | 20.9 | 21.9 | 97 | No |
| **MuRIL** | **GPU fp32** | **5.3** | **5.6** | **5.9** | 1,158 | **Yes (best margin)** |

(XLM-R and MuRIL rows above are the mean of 3 repeated runs; std across repeats was small in both cases — e.g. p95 std of 0.18ms for XLM-R fp32 and 0.046ms for MuRIL fp32 — confirming these are stable measurements, not one-off noise.)

### 7.2 Discussion

The larger model (XLM-R, 278M params) is **faster** than the smaller model (ModernBERT, 149.6M params) at every tested configuration, and MuRIL (237.6M params, in between the other two) is faster still, meeting the latency target with the widest margin of any model tested — at batch=1, GPU fp32 MuRIL achieves 5.6ms p95 latency versus XLM-R's 6.2ms and ModernBERT's 12.0ms.

This is counter-intuitive if parameter count is treated as a latency proxy, but is explainable architecturally: ModernBERT's design (alternating local/global attention, rotary embeddings, unpadded/variable-length-sequence handling) is optimized for throughput at **large batch sizes and longer sequences** — exactly the opposite of this workload's short sequences (P50 ≈ 28 tokens) and single-request (batch=1) serving pattern. Those optimizations introduce fixed per-call kernel-launch and bookkeeping overhead that simpler, mature, heavily-optimized BERT-family architectures like XLM-R and MuRIL do not carry. This overhead dominates at batch=1 and is not amortized away, while it is a much smaller fraction of total time at batch=32 — which is also why ModernBERT's throughput numbers, while still behind the other two, are less dramatically behind than its latency numbers.

fp16 illustrates the same effect from a different angle: it *improves* batched throughput substantially (e.g. XLM-R: 1,146 → 2,862 ex/s) but *worsens* single-request latency (XLM-R: 6.2ms → 8.7ms p95). `torch.autocast`'s per-call casting overhead is fixed cost that isn't amortized at batch=1, so it adds latency there even though it reduces the model's per-example compute cost at scale.

**Conclusion:** parameter count is not a reliable predictor of deployment latency; architecture-workload fit (batch size and sequence length the architecture was designed around, versus the actual serving pattern) matters more. For this problem's single-request, short-sequence serving pattern, both MuRIL and XLM-R in fp32 comfortably clear the latency bar, with MuRIL holding a small but consistent edge; ModernBERT does not meet the requirement in any tested configuration.

**Caveats:** these figures were measured on a single consumer laptop GPU (RTX 5060 Laptop, ~8GB VRAM) using PyTorch eager-mode execution — no ONNX/TensorRT graph compilation and no production-grade serving infrastructure (e.g. continuous batching via Triton/vLLM-style serving) were used. This establishes a conservative baseline; server-grade hardware and a compiled inference runtime would likely improve these numbers further. Quantization/export was considered but not pursued — see Section 10.

---

## 8. Error Analysis

Error analysis was run on the full frozen test split (3,240 examples, 90 per intent) for ModernBERT and XLM-R — the two models trained before this analysis was performed; MuRIL was trained afterward and was not retroactively put through this same error-analysis pass — to characterize *how* the remaining ~0.4% of test errors are distributed — specifically whether they cluster in noisy surface-variant types (which would indicate a robustness failure) or elsewhere.

### 8.1 Initial results (pre-patch)

| Model | Test accuracy | Errors |
|---|---|---|
| ModernBERT | 99.568% | 14 / 3,240 |
| XLM-R | 99.599% | 13 / 3,240 |

**Error rate by surface-variant type** showed no systematic robustness failure: aside from a mild elevation on `typo` (1.2% ModernBERT / 0.9% XLM-R), every other variant type — including `clean` — sat at a single error out of 324 (0.31%). This indicates the models are **not** disproportionately failing on noisy, code-mixed, or emoji-laden input; errors were roughly uniform across surface-variant types.

**Error rate by intent**, however, showed one clear outlier: `delivery_tracking` at 11.1% (10/90 errors) for both models, dwarfing every other intent (next-highest was 3.3%). Critically, both independently-trained models produced the *same* confusion direction: `delivery_tracking → order_status`.

### 8.2 Root-cause investigation

Inspecting the actual misclassified rows showed that all 10 errors, for both models, traced back to a **single semantic base utterance** (`delivery_tracking_b027`) and its 10 surface variants — not 10 independently-confused sentences. The base text read:

> *"pls tracking update de do, tracking update nahi mil raha"*
> (roughly: "please give a tracking update, not getting a tracking update")

This phrasing sits right on the boundary between `order_status` ("check status of an order") and `delivery_tracking` ("track a shipment") — it contains no explicit shipment/courier/package cue that would disambiguate it from a generic order-status request. Both models converging on the same misreading, across every surface-variant transformation of the same sentence (typo, emoji, code-mix, stress, etc.), is strong evidence this is a **data ambiguity in one base utterance**, not a model robustness weakness — a real robustness failure would be expected to vary with the noise type, not stay invariant to it.

### 8.3 Case study: targeted, documented correction

Rather than leaving this uninvestigated or silently altering the frozen dataset, we applied a **minimal, fully-documented patch**:

- **Scope:** only `delivery_tracking_b027`'s 10 rows.
- **Label:** unchanged (`delivery_tracking` is the correct intent for this utterance).
- **Text:** rewritten to add an explicit courier/parcel cue, removing the overlap with generic order-status phrasing, e.g.:
  - Before: *"pls tracking update de do, tracking update nahi mil raha"*
  - After: *"pls courier ka tracking update de do, parcel ka tracking update nahi mil raha"*
- **Provenance:** written to a separate, versioned file (`dataset_A_v2_36class_final_patched.csv`); the original frozen dataset was left untouched. A full changelog recording every before/after text pair was generated alongside it.

Re-running the (unretrained, i.e. exact same model weights) models against this patched text confirmed the fix:

| Model | Test accuracy (original) | Test accuracy (patched text) | `delivery_tracking` errors |
|---|---|---|---|
| ModernBERT | 99.568% | 99.877% | 10/90 → 0/90 |
| XLM-R | 99.599% | 99.907% | 10/90 → 0/90 |

Both models correctly classify the disambiguated phrasing across all 10 of its surface variants, with **zero retraining** — confirming the original errors were driven by base-sentence ambiguity rather than a decision-boundary or robustness issue that the rewrite happened to dodge.

**Production models for all other reported results in this whitepaper continue to be the ones trained on the original, frozen Dataset A v2** (per Section 6.1); this patch is reported as a case study in the error-analysis methodology, not as a change to the primary training/evaluation pipeline.

### 8.4 Remaining errors

After accounting for the `delivery_tracking_b027` cluster, remaining test errors are low-count (4 for ModernBERT, 3 for XLM-R, out of 3,240) and scattered across unrelated intents with no shared base utterance or shared confusion direction — e.g. `order_modify → exchange_item`, `flight_booking → hotel_booking`, `human_agent → subscription_charged`, `return_item → refund_status`. These are plausible near-miss confusions between semantically adjacent intents rather than a systematic failure mode, and at this error count the analysis is close to the practical noise floor of a 3,240-example test set.

---

## 9. Dataset B: Held-Out Robustness Evaluation

### 9.1 Design

Per the dataset strategy in Section 2, Dataset B was built as a genuinely independent check, never used for any tuning or model-selection decision. It consists of 108 hand-authored examples (3 per intent, all 36 intents covered), written from scratch rather than derived from Dataset A's base-utterance + surface-variant generation pipeline — different author, different phrasing conventions, different sentence structure and word choices, no shared templates.

An initial attempt was made to source Dataset B from real WhatsApp chat exports rather than hand-authoring it, on the reasoning that real, unscripted user text would be a stronger robustness check than any synthetic approximation. A keyword-matching extraction script was built to surface customer-support-flavored candidate messages from a chat export (9,234 messages scanned, 98 keyword-matched candidates, covering only 7 of 36 intents). On manual review, most of these matches turned out to be false positives — casual chat that happened to contain a matched keyword (e.g. "thanks", "delay") without the message actually being a support-style request for that intent. Precision was too low to yield reliable labels without much more sophisticated filtering (e.g. embedding-based relevance scoring or LLM-assisted labeling), which was judged out of scope here. This path was abandoned in favor of the hand-authored set; see Section 12 for it as a noted direction for future work.

### 9.2 Results

| Model | Accuracy | Macro-F1 | Weighted-F1 | Errors |
|---|---|---|---|---|
| ModernBERT | 75.93% | 75.08% | 75.08% | 26 / 108 |
| **XLM-R** | **91.67%** | **91.23%** | **91.23%** | **9 / 108** |
| MuRIL | 89.81% | 89.84% | 89.84% | 11 / 108 |

For comparison, all three models scored above 99.5% on Dataset A's own frozen test split (Section 6.1). The drop on Dataset B is substantial for all three, and dramatically larger for ModernBERT:

| Model | Dataset A test | Dataset B | Drop |
|---|---|---|---|
| ModernBERT | 99.568% | 75.93% | −23.64 points |
| XLM-R | 99.599% | 91.67% | −7.93 points |
| MuRIL | 99.599% | 89.81% | −9.79 points |

### 9.3 Discussion

This gap is the most consequential finding in this whitepaper, and it is the reason the Dataset A / Dataset B split was designed the way it was. Dataset A's train, validation, and test splits are all drawn from the same 2,160-base-utterance generation process — split by base so there is zero literal text overlap between splits, but all three splits nonetheless share the same underlying authoring style, vocabulary choices, and sentence templates. A model can score extremely well on Dataset A's test split by fitting to that shared generative style, without that score being informative about true generalization to language the generation process didn't happen to produce. Dataset B, being independently authored, is the first evaluation in this project that actually tests generalization rather than in-distribution memorization of a particular synthetic style — and the results show the Dataset A test score was, on its own, a poor proxy for real-world robustness.

ModernBERT is not competitive with either of the other two models here. Its Dataset B errors frequently look like outright failures to generalize rather than plausible near-misses — for example, "cncl my ordr asap" (intended: `order_cancel`) was predicted as `password_reset`, and "wapas krna h ye item" (intended: `return_item`) was predicted as `thanks`. These are not semantically adjacent confusions; they suggest the model's decision boundary is more tightly fit to Dataset A's specific phrasing patterns than to the underlying intent semantics.

XLM-R and MuRIL, by contrast, both show mostly plausible near-miss errors between related intents — for XLM-R, `refund_missing` confused with `refund_status` or `payment_pending`, `order_modify` confused with `delivery_address_change` (both errors involve an address-change request, differing only in which system entity is being modified). MuRIL's errors follow a more specific pattern: three unrelated true intents (`order_cancel`, `order_status`, `order_modify`) were all misclassified as `greeting`, and all three source utterances share heavily typo'd English loanwords ("cncl", "ordr", "whr"). This suggests MuRIL's tokenizer/vocabulary — despite being purpose-built for Indian-language and code-mixed text — may handle heavily-typo'd English loanwords somewhat less robustly than XLM-R's broader, noisier multilingual pretraining corpus. Being built for the right *domain* (Indic/code-mixed) didn't guarantee the best fit for this specific *noise profile* (typo'd transliteration).

Combined with Section 7's latency results, XLM-R is the clear overall recommendation, though the margin over MuRIL is narrower than the margin over ModernBERT: **XLM-R and MuRIL both meet the single-digit-millisecond latency requirement (MuRIL is in fact marginally faster), but XLM-R holds a meaningful ~1.9-point edge in generalization to unseen phrasing**, which is the more decisive factor between two models that already clear the latency bar comfortably. ModernBERT's architectural throughput advantages (Section 5) don't translate into either a latency or a robustness win for this task's actual serving pattern and data characteristics, and it is excluded from deployment consideration on both grounds.

---

## 10. Quantization / Export

Quantization (e.g. dynamic/static INT8) and graph-compiled export (ONNX Runtime, TensorRT) were considered as a path to further reduce latency and footprint beyond the PyTorch-eager-mode results in Section 7, but were not pursued within the scope of this project. Given that XLM-R already meets the single-digit-millisecond latency target in fp32 with meaningful headroom (6.2ms mean p95 against a 10ms target), the immediate deployment requirement is satisfied without this additional step. It is retained as a natural next optimization — see Section 12.

---

## 11. Final Model Recommendation

**XLM-R-base is the recommended model for deployment**, ahead of both other candidates on the two axes that matter most beyond Dataset A's saturated test accuracy:

| Criterion | ModernBERT | XLM-R | MuRIL | Winner |
|---|---|---|---|---|
| Parameters | 149.6M | 278.1M | 237.6M | ModernBERT (smallest) |
| Dataset A test accuracy | 99.568% | 99.599% | 99.599% | Tie (all effectively saturated) |
| Single-request latency (GPU fp32, p95) | 12.0ms — fails target | 6.2ms — meets target | 5.6ms — meets target, best margin | MuRIL |
| Batched throughput (batch=32, fp16 for XLM-R; fp32 for MuRIL) | 1,415 ex/s | 2,862 ex/s | 1,158 ex/s (fp32; not benchmarked in fp16) | XLM-R |
| Dataset B (held-out) accuracy | 75.93% | **91.67%** | 89.81% | **XLM-R** |

ModernBERT is excluded from deployment consideration outright: it is slower than both other candidates at the batch=1 serving pattern this problem targets (Section 7.2 explains why — its architectural optimizations target large-batch, long-sequence workloads that don't match this task's short single requests), and it generalizes dramatically worse to language outside Dataset A's specific synthetic style (Section 9.3).

Between XLM-R and MuRIL, the choice is closer: MuRIL is marginally faster (5.6ms vs 6.2ms p95 — a difference unlikely to matter in practice, since both comfortably clear the 10ms requirement), but XLM-R holds a real ~1.9-point edge in Dataset B generalization (91.67% vs 89.81%), with MuRIL's specific error pattern (confusing typo'd English loanwords for `greeting`) suggesting a genuine, if narrow, robustness gap rather than noise. Since robustness to exactly this kind of spelling drift and noise is a core, explicitly-stated requirement of the problem statement, this tips the recommendation to XLM-R despite MuRIL's small latency edge and its closer domain match on paper.

**Recommended deployment configuration:** XLM-R-base, GPU inference, fp32 precision, for the primary single-request routing path (6.2ms mean p95 latency, comfortable margin under the 10ms requirement). If a secondary high-throughput batch path is needed (e.g. queue-draining or backfill classification), fp16 is a reasonable choice there specifically, trading single-request latency (8.7ms mean p95, still under target but with less margin) for roughly 2.5x batched throughput. MuRIL remains a credible fallback candidate if latency margin becomes the binding constraint in a future deployment context.

---

## 12. Limitations and Future Work

- **Dataset B scale.** At 108 examples, Dataset B is a first indicative check of generalization, not a comprehensive robustness benchmark. The gap it revealed (−7.9 to −23.6 points versus Dataset A test, across all three successfully-trained models) is large enough to be a clear signal, but a larger and more diverse held-out set would give tighter error bars on the true generalization accuracy of any candidate model.
- **Dataset B is still synthetic (hand-authored).** An initial attempt was made to build Dataset B from real WhatsApp chat data instead (Section 9.1), which would have been a stronger test of real-world robustness than any hand-authored text, including this project's own. This was abandoned because a keyword-matching extraction pipeline produced too many false positives to yield reliable labels efficiently. A follow-up attempt using embedding-based relevance filtering or LLM-assisted candidate labeling (with human verification) could make real conversational data usable at reasonable effort, and would be a natural next step to further validate the results in Section 9.
- **mDeBERTa-v3-base could not be evaluated.** As documented in Section 5.1, this candidate failed to train in this environment despite three independent, methodical debugging attempts addressing three distinct, real failure modes (fp16 gradient-unscaling crash, AdamW epsilon instability causing NaN loss, and a learning-rate check). The remaining hypothesis — a `transformers==5.17.0`-specific incompatibility with this architecture's implementation — was not pursued further, since it points to an environment/library issue rather than a property of the task or dataset. Re-attempting this model in a different `transformers` version (e.g. pinning to an older, more battle-tested release) is a natural follow-up if a fourth model candidate is wanted.
- **Hardware scope.** All latency figures (Section 7) were measured on a single consumer laptop GPU (RTX 5060 Laptop, ~8GB VRAM) using PyTorch eager-mode execution. Production deployment on server-grade hardware with a compiled inference runtime (ONNX Runtime, TensorRT) and proper request batching infrastructure (e.g. Triton, vLLM-style continuous batching) would likely improve on these numbers; this whitepaper's latency results should be read as a conservative baseline, not a ceiling.
- **Quantization unexplored.** As noted in Section 10, INT8 quantization and graph-compiled export were not attempted. Since XLM-R and MuRIL already meet the latency target in fp32, this was not required to satisfy the stated deployment constraints, but remains a natural avenue to widen the latency margin further or to make ModernBERT viable if its other advantages (fewer parameters, higher raw batched throughput per Section 7) become more relevant in a different deployment context (e.g. purely offline/batch classification with no single-request latency constraint).
- **Single ambiguous base utterance in Dataset A.** Section 8's error analysis found and patched one base utterance (`delivery_tracking_b027`) whose phrasing was genuinely ambiguous with `order_status`. It is plausible that other, less prominent ambiguities exist elsewhere in Dataset A's 2,160 base utterances that a 3-4-error test-set noise floor was too small to surface; a full manual audit of the base-utterance set was out of scope here.
- **Task scope.** This work focused on intent classification, one of several downstream tasks the original problem statement allows (sentiment analysis, summarization, and question answering were not pursued in the primary deliverable). A separate, smaller sentiment-analysis dataset and pipeline was prototyped alongside this work but is reported separately, not as part of this whitepaper's primary results. The dataset, tokenization audit, and latency/robustness benchmarking methodology developed here would transfer directly to those other tasks if pursued further.



---

## Appendix A: Sentiment Analysis (Secondary Task)

The problem statement lists sentiment analysis as one of several downstream tasks this project's approach could target (Section 1). Intent classification (Sections 1-12) is the primary deliverable; sentiment analysis was pursued as a smaller, secondary exploration to test the same architecture and methodology against a genuinely different task — one where the problem statement's emoji-as-semantic-signal requirement is most directly testable (its own example: "Bohot badhiya service" is praise, but the same text with a sarcastic emoji is a complaint).

### A.1 Task design

3-class sentiment (positive / negative / neutral) was chosen over binary or 5-class, as a reasonable middle ground for a secondary exploration. Rather than weak-labeling Dataset A's existing intent-classification text (rejected — see rationale below), a dedicated hand-authored dataset was built specifically to test the emoji-flip phenomenon.

**Why not reuse Dataset A's text:** Dataset A's emoji usage (🙏, 😕, 😭) was found to be uniform across all 36 intents rather than intent- or sentiment-specific, and deriving sentiment labels from intent categories alone would have produced labels that are really just a coarse relabeling of intent (e.g. `complaint`→negative, `thanks`→positive), with severe class imbalance (only ~600 of 21,600 rows would be genuinely positive). A dedicated dataset was built instead.

**Design pattern — flip triplets:** for each positive base sentence, three variants were generated: plain (positive), reinforced with an affirming emoji (positive), and the *same text* with a sarcastic emoji appended (negative). This directly tests whether a model reads emoji as semantic signal or merely strips it as noise, which a standard bag-of-positive/negative-words baseline would fail.

### A.2 Two dataset versions

| | v1 (prototype) | v2 (scaled) |
|---|---|---|
| Base sentences | 25 positive / 25 negative / 50 neutral | 60 positive / 60 negative / 60 neutral |
| Surface variants | 3 per positive base only (flip triplets); standalone negative/neutral not varied | 10 systematic variants per base (clean, shorthand, typo, punctuation, stress, shorthand+typo, code-mix, emoji-reinforce, shorthand+emoji, stress+emoji), same methodology as Dataset A |
| Flip-sarcasm rows | 25 (one per positive base, part of the 3-way flip triplet) | 60 (one per positive base, kept as a separate bonus slice so the flip test isn't diluted by the larger standalone-variant volume) |
| Total rows | 150 | 1,860 |
| Class balance | 50 / 50 / 50 | 600 positive / 660 negative (600 standalone + 60 flip) / 600 neutral |
| Generation method | Fully hand-authored | Hand-authored base sentences (180 total) + deterministic, seeded rule-based surface-variant generation |

v2's surface-variant transformations (shorthand substitution, character-swap typo injection, vowel-elongation stress, discourse-word code-mix insertion) are rule-based rather than hand-authored per row, trading some of v1's fully-human-authored realism for the ability to scale 12x while keeping the base sentences — the actual sentiment content — hand-written and reviewed.

### A.3 Training and results

Both versions were trained on XLM-R-base only (the winning model from the intent-classification comparison, Section 11) — no separate model comparison was run for this secondary task, and no dataset-B-style held-out generalization check was built for sentiment (see Limitations below).

| | v1 | v2 |
|---|---|---|
| Train / val / test split | 105 / 22 / 23 (stratified) | 1,302 / 279 / 279 (stratified) |
| Test accuracy | 95.65% | 98.57% |
| Test macro-F1 | 95.82% | 98.59% |
| Test errors | 1 / 23 | 4 / 279 |

Both versions converged cleanly with no training instabilities (unlike the mDeBERTa intent-classification runs, Section 5.1) — XLM-R trained without issue on both dataset sizes.

**The flip-sarcasm test passed at both scales:** all 4 flip_sarcasm rows in v1's test split and all 6 in v2's test split were classified correctly (predicted negative despite the underlying text being straightforwardly positive praise). This is the headline finding of the sentiment work: the model is reading emoji as a genuine pragmatic signal that overrides the literal positive wording, not stripping it as noise — directly validating the problem statement's stated requirement, on a task built specifically to test it.

**Error pattern:** in both versions, the small number of errors clustered on genuinely weak-signal, borderline sentences rather than on noisy surface variants (typo/shorthand/code-mix rows were classified correctly at a similar rate to clean rows). For example, v2's two error-producing base sentences — "refund amount bilkul sahi tha" ("the refund amount was correct") — are factual confirmations with no strong evaluative wording, arguably ambiguous between neutral and mildly positive even to a human reader. This is the right failure mode to have: errors driven by genuine label ambiguity rather than by noise sensitivity.

### A.4 Limitations (sentiment-specific)

- **No held-out robustness set.** Unlike the intent-classification task (Section 9), no Dataset-B-equivalent independent test set was built for sentiment; all reported numbers are in-distribution relative to the same base-sentence generation process (train/val/test are stratified splits of the same 180-base-sentence pool, not independently authored). Given how much the Dataset B result changed the intent-classification conclusions (Section 9.3), the same caveat likely applies here: these accuracy numbers may not fully reflect generalization to genuinely novel phrasing.
- **v2's surface variants are rule-based, not hand-authored**, trading realism for scale (Section A.2). The rule-based typo/stress transformations are simpler than the noise patterns real users produce (Dataset A's own audit and Dataset B's results suggest real-world noise is messier than any systematic generator, rule-based or template-based, fully captures).
- **Single model tested.** No architecture comparison (ModernBERT, MuRIL) was run for sentiment; XLM-R was carried over directly from the intent-classification recommendation without separately validating that choice for this task.
- **Small emoji vocabulary.** Only 3 reinforcing emoji per class and 3 sarcasm emoji were used; real-world emoji usage is far more varied, and this dataset does not test whether the flip-detection capability generalizes beyond this specific small set.
- **Task scope.** This remains a secondary, exploratory result reported separately from the primary intent-classification deliverable (Sections 1-12), not a fully-validated second production model.

---

*Numbers throughout this document come directly from experiment logs and script output — nothing here is estimated or rounded for effect. The code that produced them is in the accompanying repository.*