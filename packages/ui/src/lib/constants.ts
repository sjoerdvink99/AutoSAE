export const CONCEPT_COLORS: Record<string, string> = {
  formality: "#00e676",
  safety: "#00b0ff",
  reasoning: "#aa00ff",
  creativity: "#ff6d00",
  conciseness: "#ffea00",
  coding: "#40c4ff",
  empathy: "#f48fb1",
  certainty: "#b9f6ca",
};

export const CONCEPT_COLORS_LIGHT: Record<string, string> = {
  formality: "#059669",
  safety: "#2563eb",
  reasoning: "#7c3aed",
  creativity: "#ea580c",
  conciseness: "#b45309",
  coding: "#0891b2",
  empathy: "#db2777",
  certainty: "#16a34a",
};

export const CONCEPT_COLOR_PALETTE_DARK: string[] = [
  "#00e676",
  "#00b0ff",
  "#aa00ff",
  "#ff6d00",
  "#ffea00",
  "#f48fb1",
  "#b9f6ca",
  "#40c4ff",
  "#ff5252",
  "#64ffda",
];

export const CONCEPT_COLOR_PALETTE_LIGHT: string[] = [
  "#059669",
  "#2563eb",
  "#7c3aed",
  "#ea580c",
  "#b45309",
  "#db2777",
  "#16a34a",
  "#0891b2",
  "#dc2626",
  "#0d9488",
];

export const REGISTRY_CONCEPTS = [
  { name: "formality", description: "Formal ↔ casual register", model: "llama-3.1-8b" },
  { name: "safety", description: "Safe ↔ harmful intent", model: "llama-3.1-8b" },
  { name: "reasoning", description: "Structured ↔ intuitive", model: "llama-3.1-8b" },
  { name: "creativity", description: "Creative ↔ literal", model: "llama-3.1-8b" },
  { name: "conciseness", description: "Terse ↔ verbose", model: "llama-3.1-8b" },
  { name: "coding", description: "Code-focused ↔ prose-focused", model: "llama-3.1-8b" },
  { name: "empathy", description: "Empathetic ↔ detached", model: "llama-3.1-8b" },
  { name: "certainty", description: "Confident ↔ hedging", model: "llama-3.1-8b" },
] as const;

export const CONCEPT_ICONS: Record<string, string> = {
  formality: "🎩",
  safety: "🛡️",
  reasoning: "🧠",
  creativity: "🎨",
  conciseness: "✂️",
  coding: "💻",
  empathy: "💚",
  certainty: "🎯",
};

export const MAX_NEW_TOKENS = 512;

export function getConceptColor(
  concept: string,
  theme: "dark" | "light" = "dark",
  overrides?: Record<string, string>
): string {
  if (overrides?.[concept]) return overrides[concept]!;
  if (theme === "light") return CONCEPT_COLORS_LIGHT[concept] ?? "#888888";
  return CONCEPT_COLORS[concept] ?? "#888888";
}

export function getConceptIcon(concept: string): string | null {
  return CONCEPT_ICONS[concept] ?? null;
}
