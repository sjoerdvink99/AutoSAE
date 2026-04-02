import { useStore } from "../../stores/useStore";
import type { ConversationTurn } from "../../types";
import { TokenizedOutput } from "./TokenizedOutput";

interface UserMessageProps {
  content: string;
}

interface AssistantMessageProps {
  turn: ConversationTurn;
}

export function UserMessage({ content }: UserMessageProps) {
  return (
    <div className="flex justify-end mb-4">
      <div className="max-w-[85%] px-3 py-2 rounded-lg bg-bg-elevated border border-bg-border">
        <p className="font-mono text-sm text-text whitespace-pre-wrap">{content}</p>
      </div>
    </div>
  );
}

export function AssistantMessage({ turn }: AssistantMessageProps) {
  const tokens = useStore((s) => s.tokens);

  if (turn.isStreaming) {
    return (
      <div className="mb-4">
        <TokenizedOutput tokens={tokens} />
      </div>
    );
  }

  return (
    <div className="mb-4">
      <p className="font-mono text-sm text-text whitespace-pre-wrap leading-relaxed">
        {turn.content}
      </p>
    </div>
  );
}
