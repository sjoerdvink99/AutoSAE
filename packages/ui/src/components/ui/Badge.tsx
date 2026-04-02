import { cn } from "../../lib/cn";

interface BadgeProps {
  children: React.ReactNode;
  className?: string;
}

export function Badge({ children, className }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded px-1.5 py-0.5 font-mono text-xs border border-bg-border text-text-muted",
        className
      )}
    >
      {children}
    </span>
  );
}
