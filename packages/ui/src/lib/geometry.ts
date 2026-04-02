export function clientInverseProject(
  jacobian: number[][],
  concepts: string[],
  delta: [number, number]
): Record<string, number> {
  const k = concepts.length;

  const a = jacobian[0] ?? [];
  const b = jacobian[1] ?? [];

  let aa = 0, ab = 0, bb = 0;
  for (let i = 0; i < k; i++) {
    aa += (a[i] ?? 0) * (a[i] ?? 0);
    ab += (a[i] ?? 0) * (b[i] ?? 0);
    bb += (b[i] ?? 0) * (b[i] ?? 0);
  }

  const det = aa * bb - ab * ab;
  if (Math.abs(det) < 1e-12) {
    return Object.fromEntries(concepts.map((c) => [c, 0]));
  }

  const v0 = (bb * delta[0] - ab * delta[1]) / det;
  const v1 = (aa * delta[1] - ab * delta[0]) / det;

  return Object.fromEntries(
    concepts.map((concept, i) => [concept, (a[i] ?? 0) * v0 + (b[i] ?? 0) * v1])
  );
}
