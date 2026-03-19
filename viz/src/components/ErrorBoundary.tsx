import React from 'react';

interface Props {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export default class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('[ErrorBoundary] Caught error:', error, info.componentStack);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return <>{this.props.fallback}</>;
      return (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          width: '100%',
          height: '100%',
          background: '#0a0a0a',
          color: '#e0e0e0',
          fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
          gap: 16,
          padding: 32,
        }}>
          <div style={{ fontSize: 13, color: '#ff3333', letterSpacing: '0.1em', fontWeight: 700 }}>
            COMPONENT ERROR
          </div>
          <div style={{
            background: '#0d0d1a',
            border: '1px solid #ff333344',
            borderRadius: 4,
            padding: '12px 16px',
            fontSize: 11,
            color: '#ff6666',
            maxWidth: 480,
            wordBreak: 'break-word',
          }}>
            {this.state.error?.message ?? 'Unknown error'}
          </div>
          <button
            onClick={this.handleReset}
            style={{
              background: 'rgba(0,255,255,0.1)',
              border: '1px solid #00ffff44',
              color: '#00ffff',
              fontSize: 11,
              fontFamily: 'inherit',
              padding: '6px 16px',
              cursor: 'pointer',
              borderRadius: 2,
              letterSpacing: '0.08em',
            }}
          >
            RETRY
          </button>
        </div>
      );
    }

    return <>{this.props.children}</>;
  }
}
