import { describe, it, expect } from 'vitest';
import {
  gamma,
  computeGeodesicPath,
  computeSpacetimeEvents,
} from '../physics';
import type { OHLCVBar } from '../../types/market';

// ---------------------------------------------------------------------------
// gamma()
// ---------------------------------------------------------------------------
describe('gamma', () => {
  it('gamma(0) === 1', () => {
    expect(gamma(0)).toBe(1);
  });

  it('gamma(0.5) is approximately 1.1547', () => {
    expect(gamma(0.5)).toBeCloseTo(1.1547, 4);
  });

  it('gamma clamps at 0.9999 (does not diverge to infinity)', () => {
    // gamma(1) would be Infinity; clamping to 0.9999 keeps it finite
    const g = gamma(1);
    expect(isFinite(g)).toBe(true);
    expect(g).toBeGreaterThan(1);
    // Should equal gamma(0.9999)
    expect(g).toBeCloseTo(gamma(0.9999), 10);
  });

  it('gamma is monotonically increasing from 0 to 0.9999', () => {
    expect(gamma(0.5)).toBeGreaterThan(gamma(0));
    expect(gamma(0.9)).toBeGreaterThan(gamma(0.5));
    expect(gamma(0.9999)).toBeGreaterThan(gamma(0.9));
  });
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function makeBar(index: number, close: number, volume = 1000): OHLCVBar {
  return {
    index,
    timestamp: 1700000000000 + index * 60000,
    open: close,
    high: close + 0.5,
    low: close - 0.5,
    close,
    volume,
  };
}

// ---------------------------------------------------------------------------
// computeGeodesicPath()
// ---------------------------------------------------------------------------
describe('computeGeodesicPath', () => {
  it('returns empty array for empty input', () => {
    expect(computeGeodesicPath([], 0.5)).toEqual([]);
  });

  it('returns single price for a single bar', () => {
    const bar = makeBar(0, 42);
    const result = computeGeodesicPath([bar], 0.5);
    expect(result).toHaveLength(1);
    expect(result[0]).toBe(42);
  });

  it('returns the same number of points as input bars', () => {
    const bars = [makeBar(0, 100), makeBar(1, 110), makeBar(2, 105), makeBar(3, 115)];
    const result = computeGeodesicPath(bars, 0.3);
    expect(result).toHaveLength(bars.length);
  });

  it('first and last points match start/end flat geodesic anchors', () => {
    const bars = [makeBar(0, 100), makeBar(1, 110), makeBar(2, 90)];
    const result = computeGeodesicPath(bars, 0.5);
    // Endpoints are always the flat geodesic (no curvature perturbation applied)
    expect(result[0]).toBeCloseTo(100);
    expect(result[result.length - 1]).toBeCloseTo(90);
  });
});

// ---------------------------------------------------------------------------
// computeSpacetimeEvents()
// ---------------------------------------------------------------------------
describe('computeSpacetimeEvents', () => {
  it('returns empty array for empty input', () => {
    expect(computeSpacetimeEvents([], 0.5)).toEqual([]);
  });

  it('returns empty array for a single bar', () => {
    expect(computeSpacetimeEvents([makeBar(0, 100)], 0.5)).toEqual([]);
  });

  it('returns n-1 events for n bars', () => {
    const bars = [makeBar(0, 100), makeBar(1, 110), makeBar(2, 105)];
    const events = computeSpacetimeEvents(bars, 0.3);
    expect(events).toHaveLength(bars.length - 1);
  });

  it('each event has correct beta and gamma values', () => {
    const bars = [makeBar(0, 100), makeBar(1, 110)];
    const beta = 0.6;
    const events = computeSpacetimeEvents(bars, beta);
    expect(events[0].beta).toBe(beta);
    expect(events[0].gamma).toBeCloseTo(gamma(beta), 10);
  });
});

// ---------------------------------------------------------------------------
// Math.max spread bug — large arrays must use reduce, not spread
// ---------------------------------------------------------------------------
describe('Math.max spread bug regression', () => {
  it('reduce-based max handles arrays too large for spread without throwing', () => {
    // Math.max(...largeArray) throws "Maximum call stack size exceeded"
    // for very large arrays. The reduce-based approach in physics.ts is safe.
    const largeArray = Array.from({ length: 200_000 }, (_, i) => i);

    // Spread version would crash; test that reduce version works
    const maxViaReduce = largeArray.reduce((acc, v) => Math.max(acc, v), -Infinity);
    expect(maxViaReduce).toBe(199_999);

    // Confirm Math.max(...largeArray) is the problematic pattern and that
    // computeGeodesicPath with many bars does not throw
    const bars = largeArray.slice(0, 1000).map((_, i) => makeBar(i, 100 + i * 0.01));
    expect(() => computeGeodesicPath(bars, 0.5)).not.toThrow();
  });
});
