import type { TokenChunk, TokenCluster } from "../types";

function cosineSimilarity(a: Record<string, number>, b: Record<string, number>): number {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (const k of keys) {
    const av = a[k] ?? 0;
    const bv = b[k] ?? 0;
    dot += av * bv;
    normA += av * av;
    normB += bv * bv;
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

export function clusterTokens(tokens: TokenChunk[], threshold = 0.85): TokenCluster[] {
  if (tokens.length === 0) return [];
  const clusters: TokenCluster[] = [];
  let start = 0;

  for (let i = 1; i < tokens.length; i++) {
    const sim = cosineSimilarity(tokens[i - 1]!.activations, tokens[i]!.activations);
    if (sim < threshold) {
      clusters.push(buildCluster(tokens, start, i));
      start = i;
    }
  }
  clusters.push(buildCluster(tokens, start, tokens.length));
  return clusters;
}

function buildCluster(tokens: TokenChunk[], start: number, end: number): TokenCluster {
  const slice = tokens.slice(start, end);
  const avgActivations: Record<string, number> = {};
  const concepts = new Set(slice.flatMap((t) => Object.keys(t.activations)));

  for (const concept of concepts) {
    avgActivations[concept] =
      slice.reduce((s, t) => s + (t.activations[concept] ?? 0), 0) / slice.length;
  }

  let dominantConcept: string | null = null;
  let maxAbs = 0.2;
  for (const [concept, val] of Object.entries(avgActivations)) {
    if (Math.abs(val) > maxAbs) {
      maxAbs = Math.abs(val);
      dominantConcept = concept;
    }
  }

  return { startIndex: start, endIndex: end, dominantConcept, avgActivations };
}
