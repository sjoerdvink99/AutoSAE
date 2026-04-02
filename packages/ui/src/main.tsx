import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";
import { App } from "./App";

const stored = localStorage.getItem("autosae-theme");
const prefersLight = window.matchMedia("(prefers-color-scheme: light)").matches;
const theme = stored === "light" || stored === "dark" ? stored : prefersLight ? "light" : "dark";
document.documentElement.dataset.theme = theme;

const root = document.getElementById("root");
if (!root) throw new Error("Root element not found");

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>
);
