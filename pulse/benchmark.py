"""Benchmark Lattice Pulse — identity, facts, multi-turn, contamination checks."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from pulse.chat import generate, load_model

PULSE_ROOT = Path(__file__).resolve().parent
IDENTITY_PATH = PULSE_ROOT / "data" / "lattice_custom.json"
DEFAULT_MODEL = PULSE_ROOT / "output" / "lattice-pulse"

BAD_BRANDS = re.compile(
    r"\b(smol\s*lm|smollm|luminous|lumina\s*labs?|lumo\s*labs?|liatech|openai|chatgpt)\b",
    re.I,
)

FACTUAL = [
    ("What is the capital of France?", ["paris"]),
    ("What is the capital of Australia?", ["canberra"]),
    ("What is 17 + 25?", ["42"]),
]

FACTUAL_PROMPTS = {q for q, _ in FACTUAL}


def _is_identity_prompt(prompt: str) -> bool:
    if prompt.strip() in FACTUAL_PROMPTS:
        return False
    lower = prompt.lower()
    if re.search(r"capital of|\d+\s*\+\s*\d+", lower):
        return False
    return True

MULTI_TURN = [
    {
        "name": "creator_followup",
        "turns": [
            "Who made you?",
            "What is the capital of Australia?",
        ],
        "checks": [
            {"must": ["lattice"], "must_not": []},
            {"must": ["canberra"], "must_not": []},
        ],
    },
    {
        "name": "identity_then_math",
        "turns": [
            "What's your name?",
            "What is 17 + 25?",
        ],
        "checks": [
            {"must": ["pulse"], "must_not": []},
            {"must": ["42"], "must_not": []},
        ],
    },
]


@dataclass
class CaseResult:
    prompt: str
    response: str
    passed: bool
    reason: str
    latency_s: float


@dataclass
class BenchReport:
    model: str
    device: str
    greedy: bool
    identity: list[CaseResult] = field(default_factory=list)
    factual: list[CaseResult] = field(default_factory=list)
    multi_turn: list[CaseResult] = field(default_factory=list)

    @property
    def all_results(self) -> list[CaseResult]:
        return self.identity + self.factual + self.multi_turn

    def summary(self) -> dict:
        total = len(self.all_results)
        passed = sum(1 for r in self.all_results if r.passed)
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(100 * passed / total, 1) if total else 0.0,
            "identity": _rate(self.identity),
            "factual": _rate(self.factual),
            "multi_turn": _rate(self.multi_turn),
            "avg_latency_s": round(
                sum(r.latency_s for r in self.all_results) / total, 2,
            ) if total else 0.0,
        }


def _rate(results: list[CaseResult]) -> dict:
    if not results:
        return {"passed": 0, "total": 0, "pass_rate": 0.0}
    p = sum(1 for r in results if r.passed)
    return {"passed": p, "total": len(results), "pass_rate": round(100 * p / len(results), 1)}


def _check_response(
    response: str,
    must: list[str] | None = None,
    must_not: list[str] | None = None,
    expect_lattice: bool = False,
) -> tuple[bool, str]:
    text = response.lower()
    if BAD_BRANDS.search(response):
        return False, f"bad brand mention: {BAD_BRANDS.search(response).group(0)}"

    if expect_lattice and "lattice" not in text:
        return False, "missing 'Lattice'"

    for token in must or []:
        if token.lower() not in text:
            return False, f"missing '{token}'"

    for token in must_not or []:
        if token.lower() in text:
            return False, f"forbidden '{token}'"

    return True, "ok"


def _run_single(
    model,
    tokenizer,
    device: str,
    prompt: str,
    history: list[dict[str, str]] | None,
    greedy: bool,
    must: list[str] | None = None,
    must_not: list[str] | None = None,
    expect_lattice: bool = False,
) -> CaseResult:
    t0 = time.perf_counter()
    response = generate(
        model, tokenizer, device, prompt,
        history=history,
        greedy=greedy,
    )
    latency = time.perf_counter() - t0
    ok, reason = _check_response(response, must, must_not, expect_lattice)
    return CaseResult(prompt, response, ok, reason, latency)


def run_benchmark(
    model_path: str | Path,
    device: str = "auto",
    greedy: bool = True,
    identity_limit: int | None = None,
) -> BenchReport:
    print(f"Loading {model_path} ...")
    model, tokenizer, device = load_model(model_path, device)
    print(f"Ready on {device} (greedy={greedy})\n")

    report = BenchReport(str(model_path), device, greedy)

    identity_data = json.loads(IDENTITY_PATH.read_text())
    if identity_limit:
        identity_data = identity_data[:identity_limit]

    print("=== Identity (training Q&As) ===")
    for row in identity_data:
        q = row["instruction"].strip()
        r = _run_single(
            model, tokenizer, device, q, None, greedy,
            expect_lattice=_is_identity_prompt(q),
        )
        report.identity.append(r)
        mark = "PASS" if r.passed else "FAIL"
        print(f"[{mark}] {q}")
        print(f"       → {r.response[:120]}{'…' if len(r.response) > 120 else ''}")
        if not r.passed:
            print(f"       ({r.reason})")

    print("\n=== Factual spot-checks ===")
    for q, must in FACTUAL:
        r = _run_single(model, tokenizer, device, q, None, greedy, must=must)
        report.factual.append(r)
        mark = "PASS" if r.passed else "FAIL"
        print(f"[{mark}] {q} → {r.response[:80]}")

    print("\n=== Multi-turn ===")
    for scenario in MULTI_TURN:
        history: list[dict[str, str]] = []
        for i, (turn, check) in enumerate(zip(scenario["turns"], scenario["checks"])):
            label = f"{scenario['name']} turn {i + 1}"
            r = _run_single(
                model, tokenizer, device, turn, list(history), greedy,
                must=check.get("must"),
                must_not=check.get("must_not"),
                expect_lattice=(i == 0 and "who" in turn.lower()),
            )
            r.prompt = label + ": " + turn
            report.multi_turn.append(r)
            mark = "PASS" if r.passed else "FAIL"
            print(f"[{mark}] {label}: {turn}")
            print(f"       → {r.response[:100]}")
            history.append({"role": "user", "content": turn})
            short = r.response if len(r.response) <= 200 else r.response[:200].rsplit(" ", 1)[0] + "..."
            history.append({"role": "assistant", "content": short})

    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark Lattice Pulse")
    p.add_argument("--model", "-m", default=str(DEFAULT_MODEL))
    p.add_argument(
        "--compare",
        nargs="*",
        metavar="SPEC",
        help='Compare models: --compare pulse,base OR pulse=path base=Qwen/...',
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--greedy", action="store_true", default=True)
    p.add_argument("--sample", action="store_true", help="Use sampling instead of greedy")
    p.add_argument("--identity-limit", type=int, default=None)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    greedy = not args.sample

    PRESETS = {
        "pulse": str(DEFAULT_MODEL),
        "base": "Qwen/Qwen2.5-1.5B-Instruct",
        "hf": "oli-mebberson/lattice-pulse",
    }

    if args.compare:
        models: dict[str, str] = {}
        specs: list[str] = []
        for spec in args.compare:
            specs.extend(s.strip() for s in spec.split(",") if s.strip())
        for spec in specs:
            if "=" in spec:
                name, path = spec.split("=", 1)
                models[name.strip()] = path.strip()
            elif spec in PRESETS:
                models[spec] = PRESETS[spec]
            else:
                models[spec] = spec

        all_reports: dict[str, dict] = {}
        print("MODEL COMPARISON")
        print("=" * 60)
        for name, path in models.items():
            print(f"\n>>> {name}: {path}\n")
            report = run_benchmark(path, args.device, greedy, args.identity_limit)
            summary = report.summary()
            all_reports[name] = {
                "model_path": path,
                "summary": summary,
                "results": [
                    {
                        "prompt": r.prompt,
                        "response": r.response,
                        "passed": r.passed,
                        "reason": r.reason,
                        "latency_s": r.latency_s,
                    }
                    for r in report.all_results
                ],
            }

        print("\n" + "=" * 60)
        print("COMPARISON TABLE (pass %)")
        print("=" * 60)
        header = f"{'model':<12} {'overall':>8} {'identity':>10} {'factual':>10} {'multi':>8} {'latency':>8}"
        print(header)
        print("-" * len(header))
        for name, data in all_reports.items():
            s = data["summary"]
            print(
                f"{name:<12} {s['pass_rate']:>7.1f}% "
                f"{s['identity']['pass_rate']:>9.1f}% "
                f"{s['factual']['pass_rate']:>9.1f}% "
                f"{s['multi_turn']['pass_rate']:>7.1f}% "
                f"{s['avg_latency_s']:>7.2f}s"
            )

        out = args.json_out or (PULSE_ROOT / "output" / "benchmark-compare.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(all_reports, indent=2))
        print(f"\nWrote {out}")
        return

    report = run_benchmark(args.model, args.device, greedy, args.identity_limit)
    summary = report.summary()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if args.json_out:
        payload = {
            "summary": summary,
            "results": [
                {
                    "prompt": r.prompt,
                    "response": r.response,
                    "passed": r.passed,
                    "reason": r.reason,
                    "latency_s": r.latency_s,
                }
                for r in report.all_results
            ],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
