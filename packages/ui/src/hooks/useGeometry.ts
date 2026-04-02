import { useCallback, useEffect, useRef } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { geometryApi, cardsApi } from "../api/cards";
import { useStore } from "../stores/useStore";
import { clientInverseProject } from "../lib/geometry";

const ALPHA_PERSIST_DEBOUNCE_MS = 400;

export function useGeometry() {
  const cards = useStore((s) => s.cards);
  const setGeometry = useStore((s) => s.setGeometry);
  const updateCardAlpha = useStore((s) => s.updateCardAlpha);
  const queryClient = useQueryClient();

  const conceptKey = cards.map((c) => c.concept).join(",");

  const { data: queryGeometry, isLoading } = useQuery({
    queryKey: ["geometry", conceptKey],
    queryFn: () => geometryApi.get(),
    enabled: cards.length > 0,
    staleTime: 30000,
  });

  useEffect(() => {
    if (cards.length === 0) {
      setGeometry(null);
    } else if (queryGeometry !== undefined) {
      setGeometry(queryGeometry);
    }
  }, [cards.length, queryGeometry, setGeometry]);

  const geometry = useStore((s) => s.geometry);

  const persistTimersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());
  const abortControllersRef = useRef<Map<string, AbortController>>(new Map());
  const cardsRef = useRef(cards);
  useEffect(() => {
    cardsRef.current = cards;
  }, [cards]);

  const inverseProject = useCallback(
    (delta: [number, number], mode: "pca" | "concept_plane", anchors?: [string, string]) => {
      if (!geometry?.projection_jacobian) return;

      const applyDeltas = (deltas: Record<string, number>) => {
        for (const [concept, dAlpha] of Object.entries(deltas)) {
          const current = cardsRef.current.find((c) => c.concept === concept)?.alpha ?? 0;
          updateCardAlpha(concept, current + dAlpha);
        }

        for (const concept of Object.keys(deltas)) {
          const existing = persistTimersRef.current.get(concept);
          if (existing) clearTimeout(existing);

          const prevAbort = abortControllersRef.current.get(concept);
          prevAbort?.abort();
          const abort = new AbortController();
          abortControllersRef.current.set(concept, abort);

          persistTimersRef.current.set(
            concept,
            setTimeout(() => {
              const card = cardsRef.current.find((c) => c.concept === concept);
              if (card && !abort.signal.aborted) {
                cardsApi
                  .setAlpha(concept, card.alpha)
                  .then(() => queryClient.invalidateQueries({ queryKey: ["geometry"] }))
                  .catch(() => {});
              }
              persistTimersRef.current.delete(concept);
              abortControllersRef.current.delete(concept);
            }, ALPHA_PERSIST_DEBOUNCE_MS)
          );
        }
      };

      if (mode === "concept_plane" && anchors) {
        geometryApi
          .inverseProject(delta, mode, anchors)
          .then((res) => applyDeltas(res.alpha_deltas))
          .catch(() => {});
        return;
      }

      const deltas = clientInverseProject(geometry.projection_jacobian, geometry.concepts, delta);
      applyDeltas(deltas);
    },
    [geometry, updateCardAlpha]
  );

  const refetchGeometry = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ["geometry"] });
  }, [queryClient]);

  return { geometry, isLoading, inverseProject, refetchGeometry };
}
