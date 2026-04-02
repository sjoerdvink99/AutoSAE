import { forwardRef } from "react";
import { cn } from "../../lib/cn";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "ghost" | "danger";
  size?: "sm" | "md";
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "ghost", size = "md", ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={cn(
          "inline-flex items-center justify-center gap-1.5 rounded font-mono text-sm font-medium transition-all focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent disabled:pointer-events-none disabled:opacity-40",
          variant === "primary" &&
            "bg-accent text-bg hover:bg-accent/90 shadow-glow-sm px-4 py-2",
          variant === "ghost" &&
            "border border-bg-border text-text-muted hover:text-text hover:border-accent/50 px-3 py-1.5",
          variant === "danger" &&
            "border border-danger/30 text-danger hover:bg-danger/10 px-3 py-1.5",
          size === "sm" && "text-xs px-2 py-1",
          className
        )}
        {...props}
      />
    );
  }
);

Button.displayName = "Button";
