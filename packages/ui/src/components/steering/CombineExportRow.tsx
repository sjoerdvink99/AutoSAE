import { useState } from "react";
import { Download, Plus } from "lucide-react";
import { useStore } from "../../stores/useStore";
import { Button } from "../ui/Button";

export function CombineExportRow() {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const cards = useStore((s) => s.cards);

  const layerMismatch = new Set(cards.map((c) => c.layer)).size > 1;
  const canCombine = cards.length >= 2 && !layerMismatch && name.trim().length > 0;

  return (
    <details className="border-t border-bg-border shrink-0">
      <summary className="cursor-pointer px-3 py-2 font-mono text-xs font-semibold text-text-subtle hover:text-text select-none list-none flex items-center gap-1.5">
        <span className="text-text-muted">▸</span>
        Combine & Export
      </summary>
      <div className="flex flex-col gap-2 px-3 pb-3 pt-1">
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Name (required)"
          className="w-full rounded border border-bg-border bg-bg-elevated px-2 py-1 font-mono text-xs text-text outline-none focus:border-accent placeholder:text-text-muted"
        />
        <input
          type="text"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Description (auto-generated if blank)"
          className="w-full rounded border border-bg-border bg-bg-elevated px-2 py-1 font-mono text-xs text-text outline-none focus:border-accent placeholder:text-text-muted"
        />
        {layerMismatch && (
          <p className="font-mono text-[10px] text-red-400">Cards target different layers.</p>
        )}
        <div className="flex gap-1.5">
          <Button
            variant="ghost"
            size="sm"
            disabled={!canCombine}
          >
            <Download size={10} />
            Export & Download
          </Button>
          <Button
            variant="primary"
            size="sm"
            disabled={!canCombine}
          >
            <Plus size={10} />
            Load
          </Button>
        </div>
      </div>
    </details>
  );
}
