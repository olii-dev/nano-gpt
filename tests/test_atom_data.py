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
    """The Cosmopedia-tilted mix adds to 100B tokens."""
    from atom.prepare_data import TOKEN_BUDGET, SUBSET_BUDGETS
    assert TOKEN_BUDGET == 100_000_000_000
    total = sum(SUBSET_BUDGETS.values())
    assert total == TOKEN_BUDGET
    # Cosmopedia-heaviest
    assert SUBSET_BUDGETS["cosmopedia_v2"] == 50_000_000_000
    assert SUBSET_BUDGETS["fineweb_edu"] == 30_000_000_000
    assert SUBSET_BUDGETS["python_edu"] == 20_000_000_000
