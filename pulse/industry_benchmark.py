"""
Industry-style benchmarks (MMLU, GSM8K, HellaSwag) via lm-evaluation-harness.

Like model cards from Qwen, Meta, Mistral — fixed public suites, exact-match scoring.

Quick smoke (Mac, ~10–20 min):
  python -m pulse.industry_benchmark --compare pulse,base --limit 50

Full run (Kaggle GPU, ~30–60 min):
  python -m pulse.industry_benchmark --compare pulse,base --device cuda

Outputs:
  pulse/output/industry-benchmark.json
  pulse/output/industry-benchmark.png
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PULSE_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PULSE_ROOT / "output" / "lattice-pulse"
OUT_JSON = PULSE_ROOT / "output" / "industry-benchmark.json"
OUT_PNG = PULSE_ROOT / "output" / "industry-benchmark.png"

# Standard suites on 1.5B model cards (Qwen, Meta, Mistral style)
TASKS = ["mmlu", "gsm8k", "hellaswag"]
TASK_LABELS = {
    "mmlu": "MMLU",
    "gsm8k": "GSM8K",
    "hellaswag": "HellaSwag",
}

# Published Qwen2.5-1.5B-Instruct (official blog, lm-eval-harness)
PUBLISHED_BASE = {
    "mmlu": 50.7,  # MMLU-redux in Qwen blog; lm-eval mmlu ~similar ballpark
    "gsm8k": 73.2,
    "hellaswag": 68.0,  # approximate from Qwen2.5-1.5B base table
}

ALIASES = {
    "pulse": str(DEFAULT_MODEL),
    "base": "Qwen/Qwen2.5-1.5B-Instruct",
    "hf": "oli-mebberson/lattice-pulse",
}


def _resolve_models(spec: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            name, path = part.split("=", 1)
            out.append((name.strip(), path.strip()))
        else:
            out.append((part, ALIASES.get(part, part)))
    return out


def _parse_stdout_table(text: str, tasks: list[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for task in tasks:
        # Match task row with strict-match filter (standard for GSM8K etc.)
        pattern = rf"\|{task}\|[^|]*\|strict-match[^|]*\|[^|]*\|exact_match\|↑\s*\|\s*([\d.]+)"
        m = re.search(pattern, text)
        if m:
            scores[task] = round(float(m.group(1)) * 100, 1)
            continue
        # HellaSwag / MMLU use acc_norm in stdout table
        pattern2 = rf"\|{task}\|[^|]*\|none\s*\|[^|]*\|acc_norm\|↑\s*\|\s*([\d.]+)"
        m2 = re.search(pattern2, text)
        if m2:
            scores[task] = round(float(m2.group(1)) * 100, 1)
            continue
        pattern3 = rf"\|{task}\|[^|]*\|none\s*\|[^|]*\|acc\|↑\s*\|\s*([\d.]+)"
        m3 = re.search(pattern3, text)
        if m3:
            scores[task] = round(float(m3.group(1)) * 100, 1)
    return scores


def _run_lm_eval(
    model_path: str,
    device: str,
    limit: int | None,
    batch_size: int,
    tasks: list[str] | None = None,
) -> dict[str, float]:
    task_list = tasks or TASKS
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "results.json"
        cmd = [
            sys.executable, "-m", "lm_eval", "run",
            "--model", "hf",
            "--model_args", f"pretrained={model_path},trust_remote_code=True",
            "--tasks", ",".join(task_list),
            "--device", device,
            "--batch_size", str(batch_size),
            "--output_path", str(out_path),
        ]
        if limit:
            cmd.extend(["--limit", str(limit)])

        print(f"\n>>> lm_eval {model_path}")
        print(" ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        combined = proc.stdout + "\n" + proc.stderr
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            raise RuntimeError(f"lm_eval failed for {model_path}")

        scores = _parse_stdout_table(combined, task_list)

        json_files = sorted(Path(tmp).rglob("*.json"), key=lambda p: p.stat().st_mtime)
        if json_files:
            data = json.loads(json_files[-1].read_text())
            for task in task_list:
                if task in scores:
                    continue
                block = data.get("results", {}).get(task, {})
                metric = (
                    block.get("acc_norm,none")
                    or block.get("acc,none")
                    or block.get("exact_match,strict-match")
                    or block.get("exact_match,flexible-extract")
                )
                if metric is not None:
                    scores[task] = round(float(metric) * 100, 1)

        if not scores:
            raise RuntimeError(f"Could not parse lm_eval scores for {model_path}")
        return scores


def _plot(results: dict[str, dict[str, float]], limit: int | None, task_list: list[str]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(results.keys())
    tasks = [t for t in task_list if any(t in results[n] for n in names)]
    labels = [TASK_LABELS.get(t, t.upper()) for t in tasks]
    x = np.arange(len(labels))
    width = 0.8 / max(len(names), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2563eb", "#16a34a", "#9333ea", "#ea580c"]

    for i, name in enumerate(names):
        vals = [results[name].get(t, 0) for t in tasks]
        offset = (i - len(names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=colors[i % len(colors)])
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    subtitle = f"lm-evaluation-harness · limit={limit}" if limit else "lm-evaluation-harness · full split"
    ax.set_title(f"Lattice Pulse — industry benchmarks\n{subtitle}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Benchmark suite")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.axhline(50, color="#94a3b8", linestyle=":", linewidth=1, alpha=0.7)
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=160, facecolor="white")
    print(f"Wrote {OUT_PNG}")


def main() -> None:
    p = argparse.ArgumentParser(description="Industry benchmarks via lm-eval")
    p.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    p.add_argument("--compare", type=str, help="e.g. pulse,base")
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--limit", type=int, default=None, help="Cap examples per task (smoke test)")
    p.add_argument("--tasks", type=str, default=",".join(TASKS))
    p.add_argument("--use-published-base", action="store_true",
                   help="Skip base lm_eval; use official Qwen2.5-1.5B numbers")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    args = p.parse_args()

    models = _resolve_models(args.compare) if args.compare else [("model", args.model)]
    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]

    results: dict[str, dict[str, float]] = {}
    for name, path in models:
        if args.use_published_base and name == "base":
            results[name] = {t: PUBLISHED_BASE.get(t, 0) for t in task_list}
            print(f"\n>>> {name}: using published Qwen2.5-1.5B-Instruct scores")
            continue
        results[name] = _run_lm_eval(path, args.device, args.limit, args.batch_size, task_list)

    payload = {
        "tasks": task_list,
        "task_labels": TASK_LABELS,
        "limit": args.limit,
        "device": args.device,
        "models": results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.json_out}")

    print("\n" + "=" * 50)
    print("INDUSTRY BENCHMARKS (accuracy %)")
    print("=" * 50)
    header = f"{'model':<10}" + "".join(f"{TASK_LABELS.get(t, t.upper()):>12}" for t in task_list)
    print(header)
    print("-" * len(header))
    for name, scores in results.items():
        row = f"{name:<10}" + "".join(f"{scores.get(t, 0):>12.1f}" for t in task_list)
        print(row)

    _plot(results, args.limit, task_list)


if __name__ == "__main__":
    main()
