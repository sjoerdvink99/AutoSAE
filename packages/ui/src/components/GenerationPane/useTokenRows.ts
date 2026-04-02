import { useMemo } from "react";
import type { TokenChunk } from "../../types";

const CHAR_WIDTH = 7.2;
const NEWLINE_TOKEN_RE = /\n/;

export interface TokenRow {
  start: number;
  end: number;
}

function estimateWidth(token: string): number {
  return token.length * CHAR_WIDTH;
}

export function useTokenRows(tokens: TokenChunk[], containerWidth: number): TokenRow[] {
  return useMemo(() => {
    if (tokens.length === 0) return [];
    const rows: TokenRow[] = [];
    let rowStart = 0;
    let rowWidth = 0;
    const effective = containerWidth > 0 ? containerWidth : 600;

    for (let i = 0; i < tokens.length; i++) {
      const token = tokens[i]!;
      const hasNewline = NEWLINE_TOKEN_RE.test(token.token);
      const w = estimateWidth(token.token);

      if (rowWidth + w > effective && i > rowStart) {
        rows.push({ start: rowStart, end: i });
        rowStart = i;
        rowWidth = 0;
      }

      rowWidth += w;

      if (hasNewline) {
        rows.push({ start: rowStart, end: i + 1 });
        rowStart = i + 1;
        rowWidth = 0;
      }
    }

    if (rowStart < tokens.length) {
      rows.push({ start: rowStart, end: tokens.length });
    }

    return rows;
  }, [tokens, containerWidth]);
}
