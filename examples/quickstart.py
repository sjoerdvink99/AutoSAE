"""
Quickstart: load a locally extracted card and steer GPT-2.

Run demo_gpt2.py first to extract and save the card, then:
    uv run python examples/quickstart.py

For larger models (requires GPU + HF access):
    steerer = Steerer("meta-llama/Llama-3.1-8B-Instruct", load_in_4bit=True)
    steerer.load_card(ConceptCard.from_registry("formality"), alpha=2.0)
"""

from pathlib import Path

from autosae import ConceptCard, Steerer

card_path = Path("./cards/formality_gpt2.safetensors")

if not card_path.exists():
    print("Card not found. Run examples/demo_gpt2.py first to extract it.")
    raise SystemExit(1)

steerer = Steerer("gpt2")
steerer.load_card(ConceptCard.load(card_path), alpha=2.0)

print(steerer.generate("Write an email about tomorrow's meeting.", max_new_tokens=60))
