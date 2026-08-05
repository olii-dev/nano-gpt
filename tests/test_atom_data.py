"""Tests for Atom data pipeline."""
def test_smollm_registry_entry():
    """smollm_corpus_atom is in the registry with required fields."""
    from dataset import DATASET_REGISTRY
    entry = DATASET_REGISTRY["smollm_corpus_atom"]
    assert "source" in entry
    assert entry["source"].startswith("HuggingFaceTB/")
    assert "split" in entry
    assert "streamable" in entry and entry["streamable"] is True


def test_token_budget_mix():
    """The Cosmopedia-tilted mix adds to 35B tokens."""
    from atom.prepare_data import TOKEN_BUDGET, SUBSET_BUDGETS
    assert TOKEN_BUDGET == 35_000_000_000
    total = sum(SUBSET_BUDGETS.values())
    assert total == TOKEN_BUDGET
    # Cosmopedia-heaviest (50% of mix)
    assert SUBSET_BUDGETS["cosmopedia_v2"] == 17_500_000_000
    assert SUBSET_BUDGETS["fineweb_edu"] == 10_500_000_000
    assert SUBSET_BUDGETS["python_edu"] == 7_000_000_000


def test_atom_batch_sampler(tmp_path):
    """Batch iterator yields independent random chunks across multiple shards."""
    import numpy as np
    import torch
    from dataset import get_atom_batch_iterator

    # Write two fake shards with distinct token patterns so we can verify
    # both shards are sampled (not just the first).
    shard_a = np.arange(0, 5000, dtype=np.uint16)        # 0,1,2,...
    shard_b = np.arange(5000, 10000, dtype=np.uint16)    # 5000,5001,...
    (tmp_path / "train_a.bin").write_bytes(shard_a.tobytes())
    (tmp_path / "train_b.bin").write_bytes(shard_b.tobytes())
    (tmp_path / "val_a.bin").write_bytes(shard_a[:1000].tobytes())

    train_iter, val_iter = get_atom_batch_iterator(
        tmp_path, block_size=32, batch_size=8, device=torch.device("cpu"), seed=42,
    )
    x, y = next(train_iter)
    # Shape: (batch, block_size)
    assert x.shape == (8, 32)
    assert y.shape == (8, 32)
    # y is x shifted by one (next-token prediction)
    assert torch.equal(x[:, 1:], y[:, :-1])
    # All 8 chunks in the batch are NOT identical (independent sampling)
    unique_rows = torch.unique(x, dim=0).shape[0]
    assert unique_rows > 1, "batch rows are identical — sampler is repeating, not sampling"

    # Both shards should appear across many batches (cosmopedia-style mix)
    seen_above_5000 = False
    seen_below_5000 = False
    for _ in range(20):
        x, _ = next(train_iter)
        if x.max() >= 5000:
            seen_above_5000 = True
        if x.min() < 5000:
            seen_below_5000 = True
    assert seen_above_5000 and seen_below_5000, "sampler only hits one shard"

    # Val iterator works too
    x_val, _ = next(val_iter)
    assert x_val.shape == (8, 32)


def test_atom_batch_sampler_missing_data(tmp_path):
    """Helpful error when no .bin files exist."""
    import torch
    from dataset import get_atom_batch_iterator
    try:
        get_atom_batch_iterator(tmp_path, 32, 4, torch.device("cpu"))
        assert False, "should have raised"
    except FileNotFoundError as e:
        assert "prepare_data" in str(e)
