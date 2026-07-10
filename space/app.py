"""
Lattice Mini — corporate demo UI for Hugging Face Spaces.
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

CUSTOM_CSS = """
/* Layout */
.gradio-container {
    max-width: 820px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}

/* Header */
.lattice-header {
    text-align: center;
    padding: 2.25rem 1.5rem 1.75rem;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-color-primary);
}
.lattice-logo {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 48px;
    height: 48px;
    border-radius: 12px;
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
    color: #fff;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: -0.02em;
    margin-bottom: 1rem;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.25);
}
.lattice-header h1 {
    margin: 0 0 0.25rem;
    font-size: 1.75rem;
    font-weight: 600;
    letter-spacing: -0.03em;
    color: var(--body-text-color);
}
.lattice-byline {
    margin: 0 0 0.75rem;
    font-size: 0.875rem;
    font-weight: 500;
    color: #6366f1;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}
.lattice-sub {
    margin: 0;
    font-size: 0.8125rem;
    color: var(--body-text-color-subdued);
    line-height: 1.5;
}
.lattice-notice {
    margin: 1.25rem 1.5rem 0;
    padding: 0.875rem 1rem;
    border-radius: 8px;
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    font-size: 0.8125rem;
    line-height: 1.55;
    color: var(--body-text-color-subdued);
    text-align: center;
}
.lattice-footer {
    text-align: center;
    padding: 1.25rem;
    margin-top: 0.5rem;
    font-size: 0.75rem;
    color: var(--body-text-color-subdued);
    border-top: 1px solid var(--border-color-primary);
}
.lattice-footer a {
    color: #6366f1;
    text-decoration: none;
}
.lattice-footer a:hover {
    text-decoration: underline;
}

/* Chat panel */
#chat-panel {
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 12px !important;
    overflow: hidden;
}
#chat-panel .wrap {
    min-height: 380px;
}

/* Input row */
#input-row {
    gap: 0.75rem;
    align-items: flex-end;
}
#send-btn {
    min-width: 100px;
    background: #4f46e5 !important;
    border: none !important;
}
#send-btn:hover {
    background: #4338ca !important;
}

/* Advanced settings */
#advanced-panel {
    margin-top: 0.25rem;
}
"""

HEADER_HTML = """
<div class="lattice-header">
    <div class="lattice-logo">LS</div>
    <h1>Lattice Mini</h1>
    <p class="lattice-byline">Lattice Systems</p>
    <p class="lattice-sub">Compact language model · 42M parameters · Research preview</p>
</div>
"""

NOTICE_HTML = """
<div class="lattice-notice">
    Research demo — not intended for production use. Responses may be inaccurate or repetitive.
    For best results, use temperature <strong>0.3–0.5</strong>.
</div>
"""

FOOTER_HTML = """
<div class="lattice-footer">
    Lattice Mini · Built from scratch by Lattice Systems<br>
    <a href="https://github.com/olii-dev/nano-gpt" target="_blank" rel="noopener">View training pipeline</a>
</div>
"""


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


def respond(
    message: str,
    history: list[dict[str, str]],
    temperature: float,
    max_new_tokens: float,
) -> tuple[list[dict[str, str]], str]:
    if not message.strip():
        return history, ""

    completion = generate_reply(message.strip(), float(temperature), float(max_new_tokens))
    history = history or []
    history.append({"role": "user", "content": message.strip()})
    history.append({"role": "assistant", "content": completion})
    return history, ""


theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
).set(
    button_primary_background_fill="#4f46e5",
    button_primary_background_fill_hover="#4338ca",
    block_radius="10px",
    block_border_width="1px",
    input_radius="8px",
)

with gr.Blocks(title="Lattice Mini", theme=theme, css=CUSTOM_CSS) as demo:
    gr.HTML(HEADER_HTML)
    gr.HTML(NOTICE_HTML)

    chatbot = gr.Chatbot(
        label="Conversation",
        height=400,
        show_label=False,
        type="messages",
        elem_id="chat-panel",
        avatar_images=(None, None),
    )

    with gr.Row(elem_id="input-row"):
        msg = gr.Textbox(
            label="Message",
            placeholder="Ask Lattice Mini a question…",
            show_label=False,
            scale=5,
            container=False,
        )
        send = gr.Button("Send", variant="primary", elem_id="send-btn", scale=1)

    with gr.Accordion("Advanced settings", open=False, elem_id="advanced-panel"):
        with gr.Row():
            temperature = gr.Slider(
                0.1, 1.0, value=0.4, step=0.05, label="Temperature",
            )
            max_new_tokens = gr.Slider(
                20, 120, value=60, step=5, label="Max new tokens",
            )

    gr.HTML(FOOTER_HTML)

    inputs = [msg, chatbot, temperature, max_new_tokens]
    outputs = [chatbot, msg]

    msg.submit(respond, inputs=inputs, outputs=outputs)
    send.click(respond, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    demo.queue().launch()
