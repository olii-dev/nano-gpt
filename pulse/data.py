"""Prepare instruction data for Lattice Pulse SFT."""

from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path
from typing import Literal

from datasets import Dataset, load_dataset

from pulse.config import ALPACA_URL, DATA_DIR, SYSTEM_PROMPT

DatasetSource = Literal["smol-smoltalk", "alpaca", "mix", "lattice-identity"]


def _download_alpaca(path: Path) -> list[dict]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading Alpaca from {ALPACA_URL} ...")
        urllib.request.urlretrieve(ALPACA_URL, path)
    return json.loads(path.read_text(encoding="utf-8"))


def _instruction_to_user(item: dict) -> str:
    inst = (item.get("instruction") or "").strip()
    inp = (item.get("input") or "").strip()
    if inp:
        return f"{inst}\n\n{inp}"
    return inst


def alpaca_to_messages(item: dict, system_prompt: str = SYSTEM_PROMPT) -> dict | None:
    user = _instruction_to_user(item)
    assistant = (item.get("output") or "").strip()
    if len(user) < 3 or len(assistant) < 3:
        return None
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def normalize_messages(
    messages: list[dict],
    system_prompt: str = SYSTEM_PROMPT,
) -> dict | None:
    """Ensure Lattice system prompt and valid user/assistant turns."""
    if not messages:
        return None
    msgs = [dict(m) for m in messages]
    # Drop empty turns
    msgs = [m for m in msgs if (m.get("content") or "").strip()]
    if len(msgs) < 2:
        return None
    if msgs[0].get("role") == "system":
        msgs[0] = {"role": "system", "content": system_prompt}
    else:
        msgs = [{"role": "system", "content": system_prompt}] + msgs
    # Must end with assistant for SFT
    if msgs[-1].get("role") != "assistant":
        return None
    return {"messages": msgs}


def load_custom_json(path: Path) -> list[dict]:
    """Optional branding examples: instruction / input / output fields."""
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("examples", [])
    rows: list[dict] = []
    for item in data:
        row = alpaca_to_messages(item)
        if row is not None:
            rows.append(row)
    return rows


def _load_alpaca_rows(max_examples: int | None) -> list[dict]:
    raw_path = DATA_DIR / "alpaca_data.json"
    items = _download_alpaca(raw_path)
    rows: list[dict] = []
    for item in items:
        row = alpaca_to_messages(item)
        if row is not None:
            rows.append(row)
    random.shuffle(rows)
    if max_examples is not None:
        rows = rows[:max_examples]
    return rows


def _load_smoltalk_rows(max_examples: int | None, seed: int) -> list[dict]:
    print("Loading HuggingFaceTB/smol-smoltalk (HF's SmolLM2 instruct data) ...")
    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
    ds = ds.shuffle(seed=seed)
    if max_examples is not None:
        n = min(max_examples, len(ds))
        ds = ds.select(range(n))
    rows: list[dict] = []
    for ex in ds:
        msgs = ex.get("messages")
        if msgs is None:
            continue
        row = normalize_messages(msgs)
        if row is not None:
            rows.append(row)
    print(f"  Loaded {len(rows):,} smol-smoltalk conversations")
    return rows


def build_sft_dataset(
    max_examples: int = 10000,
    val_ratio: float = 0.02,
    seed: int = 1337,
    custom_path: Path | None = None,
    dataset_source: DatasetSource = "smol-smoltalk",
) -> tuple[Dataset, Dataset]:
    if custom_path is None:
        custom_path = DATA_DIR / "lattice_custom.json"

    custom_rows = load_custom_json(custom_path)
    if custom_rows:
        print(f"  Custom Lattice examples: {len(custom_rows)}")

    rows: list[dict] = list(custom_rows)

    if dataset_source == "smol-smoltalk":
        budget = max(0, max_examples - len(rows)) if max_examples else None
        rows.extend(_load_smoltalk_rows(budget, seed))
    elif dataset_source == "alpaca":
        budget = max(0, max_examples - len(rows)) if max_examples else None
        rows.extend(_load_alpaca_rows(budget))
    elif dataset_source == "mix":
        half = max(1, (max_examples - len(rows)) // 2) if max_examples else 5000
        rows.extend(_load_smoltalk_rows(half, seed))
        rows.extend(_load_alpaca_rows(half))
    elif dataset_source == "lattice-identity":
        if not custom_rows:
            raise ValueError("lattice-identity requires pulse/data/lattice_custom.json")
        # Repeat branding examples so identity wins over smoltalk personas
        repeats = 20
        rows = custom_rows * repeats
        print(f"  Identity-only mode: {len(custom_rows)} examples × {repeats} = {len(rows)}")
    else:
        raise ValueError(f"Unknown dataset_source: {dataset_source}")

    random.Random(seed).shuffle(rows)
    if max_examples and len(rows) > max_examples:
        rows = rows[:max_examples]

    n_val = max(1, int(len(rows) * val_ratio))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    print(f"  Source: {dataset_source}")
    print(f"  Train: {len(train_rows):,}  |  Val: {len(val_rows):,}")
    return Dataset.from_list(train_rows), Dataset.from_list(val_rows)
