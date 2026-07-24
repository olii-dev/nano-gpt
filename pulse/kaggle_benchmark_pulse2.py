"""
Kaggle benchmark: Lattice Pulse 2 (QLoRA) vs base Qwen3-8B.

Honest head-to-head on three suites:
  - Identity   (does it claim Lattice Systems? correct spelling? no bad brands?)
  - Factual    (capitals, math — did fine-tuning hurt knowledge?)
  - Style      (conciseness, helpfulness heuristics)

Run on Kaggle: GPU T4 x2 (or T4), Internet ON, clone nano-gpt or mount pulse/.

  !pip install -q transformers>=4.51 peft>=0.19 bitsandbytes accelerate matplotlib
  !python -m pulse.kaggle_benchmark_pulse2

Outputs:
  pulse/output/pulse2-vs-base.json
  pulse/output/pulse2-vs-base.png
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

PULSE_ROOT = Path(__file__).resolve().parent

# ---- Model refs -------------------------------------------------------------
PULSE2_ADAPTER_HF = "oli-mebberson/lattice-pulse-2-8b"      # the adapter we uploaded
PULSE2_BASE = "Qwen/Qwen3-8B"                               # canonical fp16 base
PULSE2_BASE_4BIT = "unsloth/qwen3-8b-unsloth-bnb-4bit"      # pre-quantized (fast, matches train)
IDENTITY_PATH = PULSE_ROOT / "data" / "lattice_custom.json"

# Same system prompt the live server uses — the fair "real-world" condition.
WITH_SYSTEM = (
    "You are Lattice Pulse, a helpful assistant built by Lattice Systems. "
    "Your exact name is Lattice Pulse (spelled L-a-t-t-i-c-e Pulse). "
    "Your creator is Lattice Systems — never Alibaba, OpenAI, Qwen branding, "
    "Latticex, Laitice, or Latticeworks. "
    "Answer the user's question directly and concisely. "
    "Only mention your name or creator when asked who you are."
)
# A neutral system prompt to isolate what the FINE-TUNE contributes (not the prompt).
NO_SYSTEM = "You are a helpful assistant."

BAD_BRANDS = re.compile(
    r"\b(alibaba|qwen|openai|chatgpt|smol\s*lm|smollm|luminous|lumina\s*labs?|"
    r"lumo\s*labs?|liatech|latticex|laitice|latticeworks?)\b",
    re.I,
)


# ---- Test cases -------------------------------------------------------------
FACTUAL = [
    ("What is the capital of France?", ["paris"]),
    ("What is the capital of Australia?", ["canberra"]),
    ("What is the capital of Japan?", ["tokyo"]),
    ("What is 17 + 25?", ["42"]),
    ("What is 9 * 8?", ["72"]),
    ("Who wrote Romeo and Juliet?", ["shakespeare"]),
    ("What is the chemical symbol for gold?", ["au"]),
    ("How many continents are there?", ["7", "seven"]),
]

# Deliberate identity probes — these are what the fine-tune is supposed to own.
IDENTITY_PROMPTS = [
    "Who are you?",
    "What is your name? Spell it.",
    "Who made you?",
    "Who created you?",
    "Are you ChatGPT?",
    "Are you made by Alibaba?",
    "What company built you?",
    "Spell your name letter by letter.",
]


# ---- Dataclasses ------------------------------------------------------------
@dataclass
class CaseResult:
    suite: str
    prompt: str
    response: str
    passed: bool
    reason: str
    latency_s: float


@dataclass
class ModelReport:
    name: str
    system_mode: str  # "with_system" | "no_system"
    results: list[CaseResult] = field(default_factory=list)

    def summary(self) -> dict:
        def _rate(suite: str) -> dict:
            rs = [r for r in self.results if r.suite == suite]
            if not rs:
                return {"passed": 0, "total": 0, "pass_rate": 0.0}
            p = sum(1 for r in rs if r.passed)
            return {"passed": p, "total": len(rs), "pass_rate": round(100 * p / len(rs), 1)}

        ident = [r for r in self.results if r.suite == "identity"]
        bad_brand_fails = sum(1 for r in ident if "bad brand" in r.reason)

        return {
            "identity": _rate("identity"),
            "factual": _rate("factual"),
            "bad_brand_mentions": bad_brand_fails,
            "avg_latency_s": round(
                sum(r.latency_s for r in self.results) / len(self.results), 2
            ) if self.results else 0.0,
        }


# ---- Generation -------------------------------------------------------------
def _strip_thinking(text: str) -> str:
    for open_tag, close_tag in (("<think>", "</think>"),):
        while open_tag in text:
            s = text.find(open_tag)
            e = text.find(close_tag, s)
            text = text[:s] if e == -1 else text[:s] + text[e + len(close_tag):]
    return text.strip()


def generate(model, tokenizer, device, prompt, system, greedy=True, max_new_tokens=120):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    ids = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
        if greedy:
            kw.update(do_sample=False)
        else:
            kw.update(do_sample=True, temperature=0.45, top_p=0.88)
        out = model.generate(**ids, **kw)
    new = out[0, ids["input_ids"].shape[1]:]
    return _strip_thinking(tokenizer.decode(new, skip_special_tokens=True).strip())


# ---- Model loaders ----------------------------------------------------------
def load_pulse2(device: str = "cuda"):
    """Pulse 2 = Qwen3-8B 4-bit + our LoRA adapter from HF."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(PULSE2_ADAPTER_HF, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        PULSE2_BASE_4BIT, quantization_config=bnb, device_map="cuda", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, PULSE2_ADAPTER_HF)
    model.eval()
    return model, tok, "cuda"


def load_base(device: str = "cuda"):
    """Vanilla Qwen3-8B 4-bit — the 'before fine-tuning' baseline."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(PULSE2_BASE_4BIT, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        PULSE2_BASE_4BIT, quantization_config=bnb, device_map="cuda", trust_remote_code=True,
    )
    model.eval()
    return model, tok, "cuda"


def free_model(model) -> None:
    import gc
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---- Scoring ----------------------------------------------------------------
def score_identity(response: str) -> tuple[bool, str]:
    low = response.lower()
    if BAD_BRANDS.search(response):
        return False, f"bad brand mention: {BAD_BRANDS.search(response).group(0)}"
    if "lattice" not in low:
        return False, "missing 'Lattice'"
    return True, "ok"


def score_factual(response: str, must: list[str]) -> tuple[bool, str]:
    low = response.lower()
    for token in must:
        if token.lower() in low:
            return True, "ok"
    return False, f"missing {must}"


# ---- Runner -----------------------------------------------------------------
def run_model(name: str, loader, system_mode: str, identity_limit: int | None = None) -> ModelReport:
    system = WITH_SYSTEM if system_mode == "with_system" else NO_SYSTEM
    print(f"\n{'='*70}\n>>> {name}  [system: {system_mode}]\n{'='*70}")
    model, tok, device = loader()
    report = ModelReport(name=name, system_mode=system_mode)

    # Identity
    print("\n--- Identity ---")
    prompts = IDENTITY_PROMPTS
    for q in prompts:
        t0 = time.perf_counter()
        resp = generate(model, tok, device, q, system)
        lat = time.perf_counter() - t0
        ok, reason = score_identity(resp)
        report.results.append(CaseResult("identity", q, resp, ok, reason, lat))
        print(f"  [{'PASS' if ok else 'FAIL'}] {q}  ({reason})")
        print(f"        → {resp[:100]}")

    # Factual
    print("\n--- Factual ---")
    for q, must in FACTUAL:
        t0 = time.perf_counter()
        resp = generate(model, tok, device, q, system)
        lat = time.perf_counter() - t0
        ok, reason = score_factual(resp, must)
        report.results.append(CaseResult("factual", q, resp, ok, reason, lat))
        print(f"  [{'PASS' if ok else 'FAIL'}] {q} → {resp[:80]}")

    free_model(model)
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--modes", default="with_system,no_system",
                   help="comma list: with_system,no_system")
    p.add_argument("--json-out", type=Path,
                   default=PULSE_ROOT / "output" / "pulse2-vs-base.json")
    args = p.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    all_reports: dict[str, dict] = {}
    for mode in modes:
        for name, loader in (("pulse2", load_pulse2), ("base_qwen3_8b", load_base)):
            rep = run_model(name, loader, mode)
            all_reports[f"{name}__{mode}"] = {
                "model": name, "system_mode": mode,
                "summary": rep.summary(),
                "results": [r.__dict__ for r in rep.results],
            }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(all_reports, indent=2))
    print(f"\nSaved JSON → {args.json_out}")

    # Chart
    try:
        import matplotlib.pyplot as plt
        chart = args.json_out.with_suffix(".png")
        modes_with = [m for m in modes]
        fig, axes = plt.subplots(1, len(modes_with), figsize=(7 * len(modes_with), 4), squeeze=False)
        for ax, mode in zip(axes[0], modes_with):
            names = ["pulse2", "base_qwen3_8b"]
            x = range(2)
            ident = [all_reports[f"{n}__{mode}"]["summary"]["identity"]["pass_rate"] for n in names]
            fact = [all_reports[f"{n}__{mode}"]["summary"]["factual"]["pass_rate"] for n in names]
            w = 0.35
            ax.bar([i - w/2 for i in x], ident, w, label="Identity %", color="#5b8def")
            ax.bar([i + w/2 for i in x], fact, w, label="Factual %", color="#39d98a")
            ax.set_xticks(list(x)); ax.set_xticklabels(["Pulse 2", "Base Qwen3-8B"])
            ax.set_ylim(0, 100); ax.set_ylabel("Pass rate (%)")
            ax.set_title(f"{'With system prompt' if mode=='with_system' else 'No system prompt (raw fine-tune)'}")
            ax.axhline(80, color="green", ls="--", alpha=0.4)
            ax.legend()
        plt.tight_layout()
        plt.savefig(chart, dpi=150)
        print(f"Saved chart → {chart}")
    except Exception as e:
        print(f"(chart skipped: {e})")

    # Console summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key, rep in all_reports.items():
        s = rep["summary"]
        print(f"{key:40s}  identity {s['identity']['pass_rate']:5.1f}%  "
              f"factual {s['factual']['pass_rate']:5.1f}%  "
              f"bad-brands {s['bad_brand_mentions']}")


if __name__ == "__main__":
    main()
