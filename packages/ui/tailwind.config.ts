import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: {
          DEFAULT: "rgb(var(--color-bg-ch) / <alpha-value>)",
          surface: "rgb(var(--color-bg-surface-ch) / <alpha-value>)",
          elevated: "rgb(var(--color-bg-elevated-ch) / <alpha-value>)",
          border: "rgb(var(--color-bg-border-ch) / <alpha-value>)",
        },
        accent: {
          DEFAULT: "rgb(var(--color-accent-ch) / <alpha-value>)",
          dim: "rgb(var(--color-accent-ch) / 0.15)",
          glow: "rgb(var(--color-accent-ch) / 0.4)",
        },
        text: {
          DEFAULT: "rgb(var(--color-text-ch) / <alpha-value>)",
          muted: "rgb(var(--color-text-muted-ch) / <alpha-value>)",
          subtle: "rgb(var(--color-text-subtle-ch) / <alpha-value>)",
        },
        danger: "rgb(var(--color-danger-ch) / <alpha-value>)",
        warn: "rgb(var(--color-warn-ch) / <alpha-value>)",
      },
      fontFamily: {
        mono: ["'JetBrains Mono'", "'Fira Code'", "ui-monospace", "monospace"],
        sans: ["'Inter'", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      boxShadow: {
        glow: "var(--color-glow)",
        "glow-sm": "var(--color-glow-sm)",
        "glow-accent": "0 0 16px var(--color-accent-glow)",
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "gradient-shift": "gradient-shift 4s linear infinite",
      },
      keyframes: {
        "gradient-shift": {
          "0%": { backgroundPosition: "0% 50%" },
          "100%": { backgroundPosition: "300% 50%" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
