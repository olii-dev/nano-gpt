"""
Shared instruction-tuning prompt format.

Used by finetune.py (training) and generate.py (inference) so the template
matches exactly between fine-tuning and chat.
"""

from __future__ import annotations

INSTRUCTION_HEADER = "### Instruction:"
RESPONSE_HEADER = "### Response:"


def format_instruct_prompt(instruction: str, response: str | None = None) -> str:
    """
    Build a single instruction example.

    Training: pass both instruction and response (full sequence).
    Inference: pass instruction only — model completes after ### Response:
    """
    instruction = instruction.strip()
    text = f"{INSTRUCTION_HEADER}\n{instruction}\n{RESPONSE_HEADER}\n"
    if response is not None:
        text += response.strip()
    return text


def extract_response_text(generated: str) -> str:
    """Keep only text after the Response header (for display / API)."""
    if RESPONSE_HEADER in generated:
        return generated.split(RESPONSE_HEADER, 1)[-1].strip()
    return generated.strip()


def is_instruct_checkpoint(state: dict) -> bool:
    """True if checkpoint was produced by finetune.py."""
    return bool(state.get("is_instruct") or state.get("finetune_config"))
