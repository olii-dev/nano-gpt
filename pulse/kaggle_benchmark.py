"""
Run on Kaggle: GPU T4, Internet ON, clone nano-gpt or mount pulse/.

  !python pulse/kaggle_benchmark.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "pulse" / "output" / "benchmark-compare.json"
CHART = ROOT / "pulse" / "output" / "benchmark-compare.png"


def main() -> None:
    subprocess.check_call(
        [
            sys.executable, "-m", "pulse.benchmark",
            "--compare", "pulse,base",
            "--device", "cuda",
            "--json-out", str(OUT),
        ],
        cwd=ROOT,
    )

    import matplotlib.pyplot as plt

    data = json.loads(OUT.read_text())
    names = list(data.keys())
    suites = ["identity", "factual", "multi_turn"]
    labels = ["Identity", "Factual", "Multi-turn"]
    x = list(range(len(labels)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, name in enumerate(names):
        rates = [data[name]["summary"][s]["pass_rate"] for s in suites]
        offset = (i - len(names) / 2 + 0.5) * width
        ax.bar([xi + offset for xi in x], rates, width, label=name)

    ax.set_ylabel("Pass rate (%)")
    ax.set_title("Lattice Pulse vs Qwen2.5-1.5B-Instruct")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.axhline(80, color="green", linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    CHART.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(CHART, dpi=150)
    print(f"Saved {CHART}")


if __name__ == "__main__":
    main()
