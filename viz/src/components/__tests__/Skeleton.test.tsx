import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { SkeletonChart, SkeletonPanel } from '../Skeleton';

describe('SkeletonChart', () => {
  it('renders without crashing', () => {
    const { container } = render(<SkeletonChart />);
    expect(container.firstChild).not.toBeNull();
  });

  it('renders loading progressbar elements', () => {
    render(<SkeletonChart />);
    const bars = screen.getAllByRole('progressbar');
    expect(bars.length).toBeGreaterThan(0);
  });

  it('renders aria-busy elements', () => {
    render(<SkeletonChart />);
    const busy = screen.getAllByRole('progressbar').filter(el => el.getAttribute('aria-busy') === 'true');
    expect(busy.length).toBeGreaterThan(0);
  });
});

describe('SkeletonPanel', () => {
  it('renders without crashing', () => {
    const { container } = render(<SkeletonPanel />);
    expect(container.firstChild).not.toBeNull();
  });

  it('renders loading progressbar elements', () => {
    render(<SkeletonPanel />);
    const bars = screen.getAllByRole('progressbar');
    expect(bars.length).toBeGreaterThan(0);
  });
});
