"""Chat with Lattice Spark — your fine-tuned model."""
from mlx_lm import load, generate
from mlx_lm.generate import make_sampler

model, tokenizer = load(
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    adapter_path="adapters/lattice_spark",
)

sampler = make_sampler(temp=0.7, top_p=0.9)

print("Lattice Spark is ready! Type a message (or 'quit' to exit).")
print()

while True:
    try:
        msg = input("You: ")
    except (EOFError, KeyboardInterrupt):
        break
    if msg.lower().strip() in ("quit", "exit", ""):
        break
    response = generate(model, tokenizer, prompt=msg, max_tokens=150, sampler=sampler)
    print(f"Spark: {response}")
    print()
