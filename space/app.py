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


DISCLAIMER = """
**Lattice Mini** by **Lattice Systems** — a 42M-parameter language model built entirely from scratch.

Custom tokenizer → WikiText-2 pretraining → instruction fine-tuning on Alpaca.

This is a **research demo**, not a production assistant. Expect repetition, wrong facts, and odd phrasing.
Try **temperature 0.3–0.5** for slightly more focused replies.
"""


def respond(
    message: str,
    history: list[tuple[str, str]],
    temperature: float,
    max_new_tokens: float,
) -> tuple[list[tuple[str, str]], str]:
    if not message.strip():
        return history, ""

    completion = generate_reply(message.strip(), temperature, max_new_tokens)
    history = history or []
    history.append((message.strip(), completion))
    return history, ""


with gr.Blocks(title="Lattice Mini") as demo:
    gr.Markdown("# Lattice Mini")
    gr.Markdown("*by Lattice Systems*")
    gr.Markdown(DISCLAIMER)

    chatbot = gr.Chatbot(label="Chat", height=400)
    msg = gr.Textbox(label="Your message", placeholder="Ask Lattice Mini something...")
    with gr.Row():
        temperature = gr.Slider(0.1, 1.0, value=0.4, step=0.05, label="Temperature")
        max_new_tokens = gr.Slider(20, 120, value=60, step=5, label="Max new tokens")
    clear = gr.ClearButton([msg, chatbot])

    msg.submit(
        respond,
        inputs=[msg, chatbot, temperature, max_new_tokens],
        outputs=[chatbot, msg],
    )

if __name__ == "__main__":
    demo.launch()
