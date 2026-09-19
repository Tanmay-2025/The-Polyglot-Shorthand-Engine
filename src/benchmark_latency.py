"""
Production-style latency / throughput benchmark.

Trainer's batched eval numbers (Step 12 in the log) tell you nothing about
real-time single-request latency. This script measures that directly:

  - single-example (batch=1) latency distribution -- the number that maps
    to the "single-digit millisecond" requirement
  - batched throughput (configurable batch size) -- for queue-draining /
    high-throughput scenarios
  - CPU and GPU (fp32 and fp16 on GPU)

Usage:
    python benchmark_latency.py --model_dir outputs/modernbert/best_model --device cuda
    python benchmark_latency.py --model_dir outputs/modernbert/best_model --device cpu
    python benchmark_latency.py --model_dir outputs/xlmr/best_model --device cuda --fp16
"""

from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Paths are anchored to the project root (the parent of this file's
# folder, e.g. src/benchmark_latency.py -> project root is src/..), so
# this script works whether you run it from the project root or from
# inside src/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "dataset_A_v2_36class_final.csv"
MAX_LENGTH = 64

N_WARMUP = 20
N_SINGLE_RUNS = 500
BATCH_SIZE_FOR_THROUGHPUT = 32
N_BATCH_RUNS = 50


def load_sample_texts(n):
    """Pull real test-set text so sequence lengths reflect production traffic."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    test_df = df[df["split"] == "test"]

    if len(test_df) == 0:
        raise ValueError("No rows with split == 'test' found.")

    # Sample with replacement if the test set is smaller than n.
    replace = len(test_df) < n
    sample = test_df.sample(n=n, replace=replace, random_state=42)
    return sample["text"].astype(str).tolist()


def percentile_summary(latencies_ms):
    arr = np.asarray(latencies_ms)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(np.max(arr)),
        "min_ms": float(np.min(arr)),
    }


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark_single_example(model, tokenizer, texts, device, dtype):
    """Batch=1 latency: tokenize + forward pass, timed per example."""
    latencies_ms = []

    model.eval()
    with torch.no_grad():
        for i, text in enumerate(texts[: N_WARMUP + N_SINGLE_RUNS]):
            enc = tokenizer(
                text,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)

            sync_if_cuda(device)
            start = time.perf_counter()

            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=(dtype is not None),
            ):
                _ = model(**enc)

            sync_if_cuda(device)
            end = time.perf_counter()

            if i >= N_WARMUP:
                latencies_ms.append((end - start) * 1000.0)

    return percentile_summary(latencies_ms)


def benchmark_batched_throughput(model, tokenizer, texts, device, dtype, batch_size):
    """Fixed-batch-size throughput, examples/sec."""
    model.eval()

    batches = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        if len(chunk) == batch_size:
            batches.append(chunk)
        if len(batches) >= N_WARMUP + N_BATCH_RUNS:
            break

    if len(batches) < N_WARMUP + N_BATCH_RUNS:
        raise ValueError(
            "Not enough text to form the requested number of full batches; "
            "reduce N_BATCH_RUNS, N_WARMUP, or batch_size, or increase the "
            "sample pool size."
        )

    durations = []
    with torch.no_grad():
        for i, batch in enumerate(batches):
            enc = tokenizer(
                batch,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(device)

            sync_if_cuda(device)
            start = time.perf_counter()

            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=(dtype is not None),
            ):
                _ = model(**enc)

            sync_if_cuda(device)
            end = time.perf_counter()

            if i >= N_WARMUP:
                durations.append(end - start)

    total_examples = len(durations) * batch_size
    total_time = sum(durations)
    throughput = total_examples / total_time

    return {
        "batch_size": batch_size,
        "n_batches_measured": len(durations),
        "throughput_examples_per_sec": throughput,
        "mean_batch_latency_ms": float(np.mean(durations) * 1000.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=str)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use autocast fp16 (GPU only). Ignored on CPU.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of times to repeat the full benchmark, to check run-to-run stability.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this machine.")

    # Resolve --model_dir relative to the project root if a relative
    # path was given, so "outputs/modernbert/best_model" works no
    # matter which directory you launched the script from.
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = (PROJECT_ROOT / model_dir).resolve()
    args.model_dir = str(model_dir)

    dtype = None
    if args.fp16:
        if device.type != "cuda":
            print("Warning: --fp16 ignored on CPU (using fp32).")
        else:
            dtype = torch.float16

    print(f"Loading model from: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")
    print(f"Device: {device}, dtype: {dtype or 'fp32'}")

    print("\nLoading sample texts from test split...")
    # Pool must be large enough for both benchmarks: the single-example
    # run consumes (N_WARMUP + N_SINGLE_RUNS) texts one at a time, and the
    # batched run consumes (N_WARMUP + N_BATCH_RUNS) * batch_size texts to
    # form that many full batches. Take the max, with a little headroom.
    single_need = N_WARMUP + N_SINGLE_RUNS
    batch_need = (N_WARMUP + N_BATCH_RUNS) * BATCH_SIZE_FOR_THROUGHPUT
    pool_size = max(single_need, batch_need) + 100
    texts = load_sample_texts(n=pool_size)

    all_runs = []

    for run_idx in range(1, args.repeats + 1):
        print(f"\n{'=' * 60}")
        print(f"REPEAT {run_idx} / {args.repeats}")
        print(f"{'=' * 60}")

        print(f"\nRunning single-example (batch=1) benchmark "
              f"({N_WARMUP} warmup + {N_SINGLE_RUNS} measured runs)...")
        single_result = benchmark_single_example(model, tokenizer, texts, device, dtype)

        print("\nSingle-example latency (ms):")
        for k, v in single_result.items():
            print(f"  {k}: {v:.3f}")

        print(
            f"\nRunning batched throughput benchmark "
            f"(batch_size={BATCH_SIZE_FOR_THROUGHPUT}, "
            f"{N_WARMUP} warmup + {N_BATCH_RUNS} measured batches)..."
        )
        batch_result = benchmark_batched_throughput(
            model, tokenizer, texts, device, dtype, BATCH_SIZE_FOR_THROUGHPUT
        )

        print("\nBatched throughput:")
        for k, v in batch_result.items():
            print(f"  {k}: {v}")

        p95 = single_result["p95_ms"]
        verdict = "MEETS" if p95 < 10.0 else "DOES NOT MEET"
        print(
            f"\nRun {run_idx} check: p95 single-example latency "
            f"is {p95:.3f} ms -> {verdict} the <10ms target."
        )

        all_runs.append({
            "run": run_idx,
            "single_example_latency": single_result,
            "batched_throughput": batch_result,
        })

    # Stability summary across repeats.
    p50s = [r["single_example_latency"]["p50_ms"] for r in all_runs]
    p95s = [r["single_example_latency"]["p95_ms"] for r in all_runs]
    p99s = [r["single_example_latency"]["p99_ms"] for r in all_runs]
    throughputs = [r["batched_throughput"]["throughput_examples_per_sec"] for r in all_runs]

    stability = {
        "n_repeats": args.repeats,
        "p50_ms": {"mean": float(np.mean(p50s)), "std": float(np.std(p50s)), "values": p50s},
        "p95_ms": {"mean": float(np.mean(p95s)), "std": float(np.std(p95s)), "values": p95s},
        "p99_ms": {"mean": float(np.mean(p99s)), "std": float(np.std(p99s)), "values": p99s},
        "throughput_examples_per_sec": {
            "mean": float(np.mean(throughputs)),
            "std": float(np.std(throughputs)),
            "values": throughputs,
        },
    }

    print(f"\n{'=' * 60}")
    print(f"STABILITY SUMMARY ({args.repeats} repeats)")
    print(f"{'=' * 60}")
    for metric_name, stats in stability.items():
        if metric_name == "n_repeats":
            continue
        print(f"  {metric_name}: mean={stats['mean']:.3f}  std={stats['std']:.3f}  values={[round(v, 3) for v in stats['values']]}")

    result = {
        "model_dir": args.model_dir,
        "device": str(device),
        "dtype": str(dtype) if dtype else "fp32",
        "param_count": param_count,
        "max_length": MAX_LENGTH,
        "runs": all_runs,
        "stability": stability,
    }

    out_path = Path(args.model_dir).parent / "latency_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved results to: {out_path}")

    mean_p95 = stability["p95_ms"]["mean"]
    verdict = "MEETS" if mean_p95 < 10.0 else "DOES NOT MEET"
    print(
        f"\nFinal check (mean of {args.repeats} runs): mean p95 single-example "
        f"latency is {mean_p95:.3f} ms -> {verdict} the <10ms target."
    )


if __name__ == "__main__":
    main()