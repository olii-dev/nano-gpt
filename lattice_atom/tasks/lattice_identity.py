"""Lattice identity SFT task — the brand layer for Atom.

Mirrors tasks/smoltalk.py structure. Serves the hand-written Lattice
identity Q&As (lattice_identity.jsonl) so the model learns to say
"Lattice Atom" when asked who it is — without a system prompt.

This is the lesson from Pulse 2: identity has to be baked into SFT data,
not just the system prompt, or the model won't own it.

Usage: add to the SFT mixture in chat_sft.py:
    from lattice_atom.tasks.lattice_identity import LatticeIdentity
    train_tasks = [
        SmolTalk(split="train"),
        *[LatticeIdentity() for _ in range(20)],  # oversample to weight identity
        ...
    ]
"""
from __future__ import annotations
import json
from pathlib import Path

from tasks.common import Task


class LatticeIdentity(Task):
    """Hand-written Lattice identity Q&As in chat-messages format."""

    def __init__(self, split: str = "train", **kwargs):
        super().__init__(**kwargs)
        # split is accepted for interface compatibility but we have one set
        data_path = Path(__file__).resolve().parent.parent / "data" / "lattice_identity.jsonl"
        self.examples = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))
        self.length = len(self.examples)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return self.length

    def get_example(self, index):
        """Return a conversation dict with 'messages' (matches SmolTalk format)."""
        return self.examples[index % self.length]
