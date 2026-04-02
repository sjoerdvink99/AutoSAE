import { useQuery } from "@tanstack/react-query";
import { conceptMapApi } from "../api/cards";
import { useStore } from "../stores/useStore";

export function useConceptMap(enabled = true) {
  const cards = useStore((s) => s.cards);
  return useQuery({
    queryKey: ["concept-map", cards.map((c) => c.concept).join(",")],
    queryFn: () => conceptMapApi.get(),
    staleTime: 30_000,
    enabled: enabled,
  });
}
