import { Component, type ReactNode } from "react";
import { AlertTriangle, RotateCcw } from "lucide-react";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex h-screen items-center justify-center bg-bg text-text p-8">
          <div className="max-w-md text-center space-y-6">
            <div className="flex justify-center">
              <div className="p-4 rounded-full bg-danger/10 border border-danger/20">
                <AlertTriangle size={32} className="text-danger" />
              </div>
            </div>

            <div className="space-y-2">
              <h1 className="text-xl font-semibold text-text">
                Something went wrong
              </h1>
              <p className="text-sm text-text-muted">
                An unexpected error occurred in the application.
              </p>
            </div>

            {this.state.error && (
              <div className="p-3 rounded-lg bg-bg-elevated border border-bg-border text-left">
                <code className="font-mono text-xs text-danger break-all">
                  {this.state.error.message}
                </code>
              </div>
            )}

            <button
              onClick={this.handleReset}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-accent text-bg font-medium text-sm hover:bg-accent/90 transition-colors"
            >
              <RotateCcw size={14} />
              Reload Application
            </button>

            <p className="text-xs text-text-subtle">
              If this persists, check the browser console for details.
            </p>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
