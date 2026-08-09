"""Benchmark: Lattice Spark vs base Qwen2.5-0.5B-Instruct.

Runs both models on the same prompts (no system prompt) and scores:
  - Identity: does it say "Lattice"? Does it avoid competitor brands?
  - Factual: does it answer capitals, math, trivia correctly?

Usage:  .venv/bin/python3 benchmark_spark.py
"""
import time
from mlx_lm import load, generate

PROMPTS = {
    "identity": [
        ("Who are you?", ["lattice"]),
        ("What is your name?", ["lattice"]),
        ("Who made you?", ["lattice"]),
        ("Are you ChatGPT?", ["lattice"]),
        ("Are you made by Alibaba?", ["lattice"]),
        ("What company built you?", ["lattice"]),
        ("Are you Qwen?", ["lattice"]),
        ("Introduce yourself.", ["lattice"]),
    ],
    "factual": [
        ("What is the capital of France?", ["paris"]),
        ("What is the capital of Australia?", ["canberra"]),
        ("What is the capital of Japan?", ["tokyo"]),
        ("What is 17 + 25?", ["42"]),
        ("What is 9 * 8?", ["72"]),
        ("Who wrote Romeo and Juliet?", ["shakespeare"]),
        ("What is the chemical symbol for gold?", ["au"]),
        ("How many continents are there?", ["seven", "7"]),
    ],
}

FORBIDDEN = ["alibaba", "qwen", "openai", "chatgpt", "gpt-4", "gpt-3",
             "anthropic", "claude", "google", "llama", "meta"]


def score(response: str, expected: list[str], is_identity: bool) -> tuple[bool, str]:
    low = response.lower()
    for token in expected:
        if token.lower() in low:
            if is_identity:
                for brand in FORBIDDEN:
                    idx = low.find(brand)
                    if idx >= 0:
                        prefix = low[max(0, idx - 15):idx]
                        if "not " not in prefix and "n't" not in prefix:
                            return False, f"forbidden brand: {brand}"
            return True, "ok"
    return False, f"missing {expected}"


def run_model(name: str, model, tokenizer, prompts):
    results = []
    for category, qs in prompts.items():
        for question, expected in qs:
            t0 = time.time()
            resp = generate(model, tokenizer, prompt=question, max_tokens=80, verbose=False)
            elapsed = time.time() - t0
            ok, reason = score(resp, expected, is_identity=(category == "identity"))
            results.append((category, question, resp.strip(), ok, reason, elapsed))
    return results


def print_results(name: str, results: list):
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")
    for category in ("identity", "factual"):
        cat_results = [r for r in results if r[0] == category]
        passed = sum(1 for r in cat_results if r[3])
        print(f"\n  {category.upper()}: {passed}/{len(cat_results)}")
        for cat, q, resp, ok, reason, t in cat_results:
            mark = "✓" if ok else "✗"
            short = resp[:70] + ("..." if len(resp) > 70 else "")
            print(f"    {mark} {q}")
            print(f"      → {short}")
            if not ok:
                print(f"      ({reason})")
    ident_pass = sum(1 for r in results if r[0] == "identity" and r[3])
    fact_pass = sum(1 for r in results if r[0] == "factual" and r[3])
    total = len(results)
    total_pass = ident_pass + fact_pass
    print(f"\n  TOTAL: {total_pass}/{total}  (identity {ident_pass}/8, factual {fact_pass}/8)")
    return total_pass, ident_pass, fact_pass


def main():
    print("Loading base Qwen2.5-0.5B-Instruct...")
    base_model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    base_results = run_model("Base Qwen2.5-0.5B", base_model, tokenizer, PROMPTS)
    base_pass, base_ident, base_fact = print_results("BASE: Qwen2.5-0.5B-Instruct (no fine-tune)", base_results)

    del base_model

    print("\n\nLoading Lattice Spark...")
    spark_model, tokenizer = load(
        "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        adapter_path="adapters/lattice_spark",
    )
    spark_results = run_model("Lattice Spark", spark_model, tokenizer, PROMPTS)
    spark_pass, spark_ident, spark_fact = print_results("LATTICE SPARK (fine-tuned)", spark_results)

    print(f"\n{'=' * 70}")
    print(f"  HEAD TO HEAD")
    print(f"{'=' * 70}")
    print(f"  {'':30s} {'Base':>8s} {'Spark':>8s} {'Delta':>8s}")
    print(f"  {'-' * 56}")
    print(f"  {'Identity (out of 8)':30s} {base_ident:>8d} {spark_ident:>8d} {spark_ident - base_ident:>+8d}")
    print(f"  {'Factual (out of 8)':30s} {base_fact:>8d} {spark_fact:>8d} {spark_fact - base_fact:>+8d}")
    print(f"  {'TOTAL (out of 16)':30s} {base_pass:>8d} {spark_pass:>8d} {spark_pass - base_pass:>+8d}")

    if spark_ident > base_ident and spark_fact >= base_fact - 1:
        print(f"\n  ✓ FINE-TUNE HELPED — identity improved without breaking knowledge")
    elif spark_ident > base_ident and spark_fact < base_fact - 1:
        print(f"\n  ⚠ FINE-TUNE TRADED KNOWLEDGE FOR IDENTITY (overfit risk)")
    else:
        print(f"\n  ⚠ FINE-TUNE DIDN'T HELP MUCH")


if __name__ == "__main__":
    main()
