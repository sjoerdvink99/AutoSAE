import { cn } from "../../lib/cn";

interface ConceptDotProps {
  color: string;
  glow?: boolean;
  className?: string;
  style?: React.CSSProperties;
  icon?: string | null;
}

export function ConceptDot({ color, glow, className, style, icon }: ConceptDotProps) {
  if (icon) {
    return (
      <span
        className={cn("h-4 w-4 shrink-0 flex items-center justify-center text-[10px] leading-none", className)}
        style={style}
        title={icon}
      >
        {icon}
      </span>
    );
  }

  return (
    <span
      className={cn("h-1.5 w-1.5 rounded-full shrink-0", className)}
      style={{
        backgroundColor: color,
        boxShadow: glow ? `0 0 6px ${color}` : undefined,
        ...style,
      }}
    />
  );
}
