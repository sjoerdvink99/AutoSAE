"""
Multi-concept steering demo using GPT-2.

Extracts two Concept Cards (formality + conciseness), loads both into
a Steerer simultaneously, and shows how they compose.

Run:
    uv run python examples/demo_multi_concept.py
"""

from pathlib import Path

from autosae import ConceptCard, ContrastiveDataset, Extractor, Steerer

CARDS_DIR = Path("./cards")
CARDS_DIR.mkdir(exist_ok=True)

FORMALITY_PATH = CARDS_DIR / "formality_gpt2.safetensors"
CONCISENESS_PATH = CARDS_DIR / "conciseness_gpt2.safetensors"

MODEL = "gpt2"


def extract_or_load(path: Path, concept: str, dataset: ContrastiveDataset) -> ConceptCard:
    if path.exists():
        print(f"  Loading cached '{concept}' card from {path}")
        return ConceptCard.load(path)
    print(f"  Extracting '{concept}' card (this takes ~30s)...")
    extractor = Extractor(MODEL, layer_frac=0.75)
    card = extractor.extract(dataset, concept=concept, default_alpha=1.5)
    extractor.unload()
    card.save(path)
    print(f"  Saved to {path}")
    return card


formality_dataset = ContrastiveDataset(
    positive=[
        "I am writing to formally advise you that the proposal has been reviewed.",
        "Please find enclosed the requisite documentation pertaining to your inquiry.",
        "The committee respectfully requests your attendance at the scheduled proceedings.",
        "We would like to extend our sincerest gratitude for your continued cooperation.",
        "Pursuant to our prior agreement, the deliverables shall be submitted accordingly.",
        "The undersigned hereby acknowledges receipt of the aforementioned correspondence.",
        "We regret to inform you that the request cannot be accommodated at this juncture.",
        "Kindly ensure all necessary preparations are completed prior to commencement.",
        "The board of directors has unanimously resolved to approve the proposed amendment.",
        "Your prompt attention to this matter would be greatly appreciated.",
    ],
    negative=[
        "Just a heads up — we looked at your idea and we're good with it.",
        "Hey, here's the stuff you were asking about!",
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

conciseness_dataset = ContrastiveDataset(
    positive=[
        "Done.",
        "See attached.",
        "No.",
        "Call at 3.",
        "Approved.",
        "Use the API.",
        "Rescheduled.",
        "Fixed.",
        "Yes.",
        "Sent.",
    ],
    negative=[
        "I have now completed the task that was assigned to me and everything should be in order.",
        "Please find attached the document you requested, which I have prepared for your review.",
        "I wanted to let you know that unfortunately the answer to your question is no.",
        "I just wanted to reach out and let you know that we should talk at 3 o'clock today.",
        "After careful consideration I have decided that this proposal is approved.",
        "The best way to solve this is probably to go ahead and just use the API endpoint.",
        "I wanted to let you know that the meeting has been moved to a different time.",
        "I went ahead and fixed the bug that was causing the issue you mentioned.",
        "Yes, that sounds good to me and I think we should proceed with that plan.",
        "I have sent the document over to you now and you should be receiving it shortly.",
    ],
)

print("=== AutoSAE Multi-Concept Demo — GPT-2 ===\n")
print("Step 1: Loading concept cards...\n")
formality_card = extract_or_load(FORMALITY_PATH, "formality", formality_dataset)
conciseness_card = extract_or_load(CONCISENESS_PATH, "conciseness", conciseness_dataset)

print("\nStep 2: Composing concepts...\n")

steerer = Steerer(MODEL)
steerer.load_card(formality_card)
steerer.load_card(conciseness_card)

prompt = "Write a response to a client asking about the project status."

configs = [
    {"formality": 0.0, "conciseness": 0.0, "label": "baseline"},
    {"formality": 2.0, "conciseness": 0.0, "label": "formal only"},
    {"formality": 0.0, "conciseness": 2.0, "label": "concise only"},
    {"formality": 2.0, "conciseness": 2.0, "label": "formal + concise"},
    {"formality": -2.0, "conciseness": -2.0, "label": "casual + verbose"},
]

print(f"Prompt: {prompt!r}\n")
print("-" * 70)

for cfg in configs:
    steerer.set_alpha("formality", cfg["formality"])
    steerer.set_alpha("conciseness", cfg["conciseness"])
    output = steerer.generate(prompt, max_new_tokens=50, temperature=0.8)
    label = cfg["label"]
    f_alpha = cfg["formality"]
    c_alpha = cfg["conciseness"]
    print(f"[{label:<22}] f={f_alpha:+.1f} c={c_alpha:+.1f}")
    print(f"  {output.strip()}\n")

steerer.unload()
