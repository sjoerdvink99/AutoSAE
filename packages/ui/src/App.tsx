import { useEffect, useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Group as PanelGroup, Panel, Separator as PanelResizeHandle } from "react-resizable-panels";
import { CanvasPanel } from "./components/canvas/CanvasPanel";
import { GenerationPane } from "./components/generation/GenerationPane";
import { Header } from "./components/layout/Header";
import { FooterBar } from "./components/layout/FooterBar";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { SteeringPanel } from "./components/steering/SteeringPanel";
import { healthApi } from "./api/cards";
import { useStore } from "./stores/useStore";
import { useAlphaHistory } from "./hooks/useAlphaHistory";
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, staleTime: 2000 },
  },
});

function Dashboard() {
  const setCapabilities = useStore((s) => s.setCapabilities);
  const [connected, setConnected] = useState<boolean | null>(null);

  useAlphaHistory();
  useKeyboardShortcuts();

  useEffect(() => {
    healthApi
      .get()
      .then((data) => {
        setCapabilities(data.capabilities);
        setConnected(true);
      })
      .catch(() => setConnected(false));
  }, [setCapabilities]);

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-bg text-text">
      <Header connected={connected} />
      <div className="flex flex-1 min-h-0">
        <PanelGroup orientation="horizontal" className="flex-1 min-h-0">
          <Panel defaultSize="22" minSize="18">
            <SteeringPanel />
          </Panel>
          <PanelResizeHandle className="w-px bg-bg-border hover:bg-accent/40 cursor-col-resize transition-colors shrink-0" />
          <Panel defaultSize="44" minSize="28">
            <CanvasPanel />
          </Panel>
          <PanelResizeHandle className="w-px bg-bg-border hover:bg-accent/40 cursor-col-resize transition-colors shrink-0" />
          <Panel defaultSize="34" minSize="24">
            <GenerationPane />
          </Panel>
        </PanelGroup>
      </div>
      <FooterBar />
    </div>
  );
}

export function App() {
  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <Dashboard />
      </QueryClientProvider>
    </ErrorBoundary>
  );
}
