"""
End-to-end AutoSAE demo using GPT-2 (no GPU, no HF token required).

Run:
    uv run python examples/demo_gpt2.py

What it does:
  1. Extracts a 'formality' Concept Card from GPT-2's residual stream
  2. Steers the model at alpha = -2, 0, +2 on a few prompts
  3. Saves the card to ./cards/formality_gpt2.safetensors for reuse
"""

from pathlib import Path

from autosae import ContrastiveDataset, Extractor, Steerer

PROMPT = "Write a short message to the team about"

ALPHAS = [
    (-5.0, "casual   "),
    (0.0,  "baseline "),
    (5.0,  "formal   "),
]

PROMPTS = [
    "Write a short message to the team about the upcoming deadline.",
    "Describe what happened at the meeting yesterday.",
    "Explain why the project is delayed.",
]

dataset = ContrastiveDataset(
    positive=[
        "I am writing to formally advise you that the aforementioned proposal has been reviewed.",
        "Please find enclosed the requisite documentation pertaining to your recent inquiry.",
        "The committee respectfully requests your attendance at the scheduled proceedings.",
        "We would like to extend our sincerest gratitude for your continued cooperation.",
        "Pursuant to our prior agreement, the deliverables shall be submitted accordingly.",
        "The undersigned hereby acknowledges receipt of the aforementioned correspondence.",
        "We regret to inform you that the request cannot be accommodated at this juncture.",
        "Kindly ensure that all necessary preparations are completed prior to the commencement.",
        "The board of directors has unanimously resolved to approve the proposed amendment.",
        "Your prompt attention to this matter would be greatly appreciated.",
    ],
    negative=[
        "Just a heads up — we looked at your idea and we're good with it.",
        "Hey, attached is the stuff you were asking about!",
        "Can you come to the meeting? It's super important.",
        "Thanks a ton for helping out, seriously means a lot.",
        "So per what we agreed on, here's the stuff.",
        "Got your message, thanks!",
        "Sorry but we can't do that right now.",
        "Make sure everything's ready before we kick things off.",
        "Everyone on the board said yes to the change.",
        "If you could get back to me soon that'd be great.",
    ],
)

print("=== AutoSAE Demo — GPT-2 Formality Steering ===\n")
print("Step 1: Extracting 'formality' concept card from GPT-2...")

extractor = Extractor("gpt2", layer_frac=0.75)
card = extractor.extract(dataset, concept="formality", default_alpha=1.5, description="Formal academic prose ↔ casual conversational text")
extractor.unload()

save_path = Path("./cards/formality_gpt2.safetensors")
save_path.parent.mkdir(exist_ok=True)
card.save(save_path)
print(f"  Saved to {save_path}")
print(f"  Layer: {card.meta.layer}/{card.meta.hidden_dim}d\n")

print("Step 2: Steering across alphas...\n")

steerer = Steerer("gpt2")
steerer.load_card(card)

for prompt in PROMPTS:
    print(f"Prompt: {prompt!r}")
    print("-" * 60)
    for alpha, label in ALPHAS:
        steerer.set_alpha("formality", alpha)
        output = steerer.generate(prompt, max_new_tokens=40, temperature=0.8)
        print(f"  α={alpha:+.1f} [{label}] {output.strip()}")
    print()

steerer.unload()
print(f"Card saved to {save_path} — load it later with:")
print(f"  ConceptCard.load('{save_path}')")
