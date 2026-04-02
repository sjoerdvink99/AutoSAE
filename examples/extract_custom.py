"""
Extract a custom Concept Card for any concept you can describe contrastively.

This example extracts a 'legal register' card from GPT-2.
Swap the model for a larger one (Llama, Mistral, Qwen) for better steering.

Run:
    uv run python examples/extract_custom.py
"""

from pathlib import Path

from autosae import ContrastiveDataset, Extractor, Steerer

dataset = ContrastiveDataset(
    positive=[
        "The defendant contends that the aforementioned clause is null and void.",
        "Pursuant to Article 4(b), the lessor hereby indemnifies the lessee.",
        "The court finds that the evidentiary standard has not been met.",
        "Notwithstanding the foregoing, the obligations herein shall remain binding.",
        "The parties hereby agree to submit all disputes to binding arbitration.",
        "The indemnifying party shall hold harmless the indemnified party.",
        "This agreement shall be governed by the laws of the applicable jurisdiction.",
        "The claimant asserts a breach of the implied covenant of good faith.",
        "Such representations and warranties shall survive the closing date.",
        "The licensor grants a non-exclusive, non-transferable right to use.",
    ],
    negative=[
        "They're saying that part of the contract doesn't count.",
        "Based on Article 4(b), the landlord covers the tenant.",
        "The judge says there's not enough evidence.",
        "Even with all that, they still have to do what they promised.",
        "Both sides agree to let a third party settle any fights.",
        "The one responsible covers the other one's losses.",
        "This deal follows the rules of the relevant place.",
        "The person suing says the other side broke their promise.",
        "Those promises still count after the deal is done.",
        "The owner lets you use it, but only you and you can't give it away.",
    ],
)

print("Extracting 'legalese' concept card from GPT-2...")

extractor = Extractor("gpt2", layer_frac=0.75)
card = extractor.extract(
    dataset,
    concept="legalese",
    default_alpha=1.5,
    description="Legal register ↔ plain language",
)
extractor.unload()

save_path = Path("./cards/legalese_gpt2.safetensors")
save_path.parent.mkdir(exist_ok=True)
card.save(save_path)
print(f"Saved: {card}")

steerer = Steerer("gpt2")
steerer.load_card(card)

prompt = "Explain what happens when someone breaks a contract."

print(f"\nPrompt: {prompt!r}\n")
for alpha in [-1.5, 0.0, 1.5]:
    steerer.set_alpha("legalese", alpha)
    output = steerer.generate(prompt, max_new_tokens=50)
    print(f"  α={alpha:+.1f}  {output.strip()}\n")

steerer.unload()
