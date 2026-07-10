"""
Lattice Mini — Gradio demo for Hugging Face Spaces.

Works on CPU Basic (default) and ZeroGPU (requires @spaces.GPU).
"""

from __future__ import annotations

import gradio as gr

CHECKPOINT = "checkpoints/chat_best.pt"

try:
    import spaces

    ZERO_GPU = True
except ImportError:
    ZERO_GPU = False

engine = None


def _get_engine(device: str):
    global engine
    from generate import LMEngine

    if engine is None:
        print(f"Loading Lattice Mini on {device}...")
        engine = LMEngine(CHECKPOINT, device=device)
        print("Lattice Mini ready.")
    return engine


def _generate(message: str, temperature: float, max_new_tokens: float, device: str) -> str:
    eng = _get_engine(device)
    result = eng.generate(
        message.strip(),
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_k=40,
        top_p=0.9,
    )
    return result.completion


if ZERO_GPU:
    @spaces.GPU(duration=120)
    def generate_reply(message: str, temperature: float, max_new_tokens: float) -> str:
        return _generate(message, temperature, max_new_tokens, "cuda")

    print("ZeroGPU mode — model loads on first message.")
else:
    _get_engine("cpu")

    def generate_reply(message: str, temperature: float, max_new_tokens: float) -> str:
        return _generate(message, temperature, max_new_tokens, "cpu")


DESCRIPTION = """
**Lattice Mini** by **Lattice Systems** — a 42M-parameter language model built entirely from scratch.

This is a **research demo**, not a production assistant. Expect repetition, wrong facts, and odd phrasing.
Try **temperature 0.3–0.5** for slightly more focused replies.
"""


def chat(
    message: str,
    history: list,
    temperature: float,
    max_new_tokens: float,
) -> str:
    if not message or not message.strip():
        return ""
    return generate_reply(message.strip(), float(temperature), float(max_new_tokens))


demo = gr.ChatInterface(
    fn=chat,
    title="Lattice Mini",
    description=DESCRIPTION,
    additional_inputs=[
        gr.Slider(0.1, 1.0, value=0.4, step=0.05, label="Temperature"),
        gr.Slider(20, 120, value=60, step=5, label="Max new tokens"),
    ],
    examples=[
        ["What is the capital of France?", 0.4, 60],
        ["Write a short poem about the ocean.", 0.4, 60],
        ["Explain gravity in simple terms.", 0.4, 60],
    ],
)

if __name__ == "__main__":
    demo.queue().launch()
