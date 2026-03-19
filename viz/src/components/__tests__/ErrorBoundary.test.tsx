import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import ErrorBoundary from '../ErrorBoundary';

// Suppress console.error noise from intentional error throws
beforeEach(() => {
  vi.spyOn(console, 'error').mockImplementation(() => {});
});

// A component that throws when the `shouldThrow` prop is true
function BrokenChild({ shouldThrow }: { shouldThrow: boolean }) {
  if (shouldThrow) throw new Error('Test error message');
  return <div>Child rendered</div>;
}

describe('ErrorBoundary', () => {
  it('renders children normally when no error is thrown', () => {
    render(
      <ErrorBoundary>
        <BrokenChild shouldThrow={false} />
      </ErrorBoundary>
    );
    expect(screen.getByText('Child rendered')).toBeInTheDocument();
  });

  it('catches errors and shows fallback UI with error message', () => {
    render(
      <ErrorBoundary>
        <BrokenChild shouldThrow={true} />
      </ErrorBoundary>
    );
    expect(screen.getByText('COMPONENT ERROR')).toBeInTheDocument();
    expect(screen.getByText('Test error message')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument();
  });

  it('uses a custom fallback when provided', () => {
    render(
      <ErrorBoundary fallback={<div>Custom fallback</div>}>
        <BrokenChild shouldThrow={true} />
      </ErrorBoundary>
    );
    expect(screen.getByText('Custom fallback')).toBeInTheDocument();
    expect(screen.queryByText('COMPONENT ERROR')).not.toBeInTheDocument();
  });

  it('retry button resets the error state and re-renders children', async () => {
    const user = userEvent.setup();

    // Render with an error first, then switch the child to not throw
    // We do this by rendering the boundary with a stateful wrapper
    function Wrapper() {
      const [shouldThrow, setShouldThrow] = React.useState(true);
      return (
        <ErrorBoundary>
          {shouldThrow ? (
            <BrokenChild shouldThrow={true} />
          ) : (
            <div>Recovered</div>
          )}
          {/* Button outside boundary to control the state after reset */}
          <button onClick={() => setShouldThrow(false)} style={{ display: 'none' }}>
            fix
          </button>
        </ErrorBoundary>
      );
    }

    // Simpler approach: the retry resets hasError; children won't throw again
    // if the child component itself no longer errors. We'll use a ref-style trick.
    let throwError = true;
    function TogglableChild() {
      if (throwError) {
        throwError = false; // next render will not throw
        throw new Error('Thrown once');
      }
      return <div>Recovered</div>;
    }

    render(
      <ErrorBoundary>
        <TogglableChild />
      </ErrorBoundary>
    );

    expect(screen.getByText('COMPONENT ERROR')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /retry/i }));

    expect(screen.getByText('Recovered')).toBeInTheDocument();
    expect(screen.queryByText('COMPONENT ERROR')).not.toBeInTheDocument();
  });
});

// Import React for JSX inside the test (needed for the Wrapper component)
import React from 'react';
