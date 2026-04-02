import { forwardRef } from "react";
import { Search } from "lucide-react";
import { cn } from "../../lib/cn";

interface InputProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, "size"> {
  size?: "sm" | "md";
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, size = "sm", ...props }, ref) => (
    <input
      ref={ref}
      className={cn(
        "rounded border border-bg-border bg-bg-elevated px-3 py-2 font-mono text-text placeholder:text-text-subtle focus:border-accent/50 focus:outline-none focus:ring-1 focus:ring-accent/30 transition-colors",
        size === "sm" && "text-xs",
        size === "md" && "text-sm",
        className
      )}
      {...props}
    />
  )
);
Input.displayName = "Input";

export function SearchInput({ className, size, ...props }: InputProps) {
  return (
    <div className="relative">
      <Search
        size={12}
        className="absolute left-3 top-1/2 -translate-y-1/2 text-text-subtle pointer-events-none"
      />
      <Input size={size} {...props} className={cn("pl-8", className)} />
    </div>
  );
}
