import type { OHLCVBar, SpacetimeEvent, Regime } from '../types/market';

// Seeded LCG random number generator — deterministic
function lcg(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };
}

/**
 * Lorentz factor γ = 1/√(1−β²).
 * Clamps β to [0, 0.9999] to avoid division by zero.
 */
export function gamma(beta: number): number {
  const b = Math.min(Math.max(beta, 0), 0.9999);
  return 1 / Math.sqrt(1 - b * b);
}

/**
 * Rapidity φ = atanh(β).
 * Rapidity is additive under boosts — unlike velocity.
 */
export function rapidity(beta: number): number {
  const b = Math.min(Math.max(beta, 0), 0.9999);
  return Math.atanh(b);
}

/**
 * Time dilation: given proper time τ and Lorentz factor γ,
 * returns coordinate time t = γ·τ.
 */
export function timeDilation(g: number, tau: number): number {
  return g * tau;
}

/**
 * Spacetime interval ds².
 * Sign convention: ds² = −c²·ΔT² + ΔP² + ΔV²
 *   TIMELIKE  → ds² < 0  (causal, trend-following regime)
 *   LIGHTLIKE → ds² ≈ 0  (transition boundary)
 *   SPACELIKE → ds² > 0  (non-causal, mean-reversion regime)
 *
 * c_market = 1 + β × 0.5 (scaled: as β→1, cone opens)
 */
export function ds2(deltaT: number, deltaP: number, deltaV: number, c: number): number {
  return -(c * c * deltaT * deltaT) + deltaP * deltaP + deltaV * deltaV;
}

/**
 * Classify a spacetime interval into a market regime.
 * Epsilon threshold separates LIGHTLIKE from the other two.
 */
export function classifyRegime(interval: number): Regime {
  const epsilon = 0.05;
  if (Math.abs(interval) < epsilon) return 'LIGHTLIKE';
  if (interval < 0) return 'TIMELIKE';
  return 'SPACELIKE';
}

/**
 * Generate count synthetic OHLCV bars using a seeded LCG random walk.
 * Price starts at 100, with realistic intra-bar OHLCV structure.
 */
export function generateOHLCV(count: number, seed = 42): OHLCVBar[] {
  const rand = lcg(seed);
  const bars: OHLCVBar[] = [];

  let price = 100;
  const baseTimestamp = 1700000000000; // arbitrary epoch
  const barDuration = 60_000; // 1 minute bars

  for (let i = 0; i < count; i++) {
    const open = price;

    // Drift: slight upward tendency with noise
    const drift = (rand() - 0.48) * 2.5;
    const volatility = 0.5 + rand() * 1.5;

    const close = Math.max(open + drift + (rand() - 0.5) * volatility, 1);

    // High / low encompass open and close, plus extra wick
    const highExtra = rand() * volatility * 0.8;
    const lowExtra = rand() * volatility * 0.8;
    const high = Math.max(open, close) + highExtra;
    const low = Math.min(open, close) - lowExtra;

    // Volume: log-normal style, inversely correlated with smoothness
    const baseVol = 500 + rand() * 4500;
    const volMultiplier = 1 + Math.abs(drift) / 2;
    const volume = Math.round(baseVol * volMultiplier);

    bars.push({
      index: i,
      timestamp: baseTimestamp + i * barDuration,
      open,
      high,
      low,
      close,
      volume,
    });

    price = close;
  }

  return bars;
}

/**
 * Compute SpacetimeEvent for each bar using relativistic kinematics.
 * Returns one event per bar (skipping the first bar which has no prior).
 */
export function computeSpacetimeEvents(bars: OHLCVBar[], beta: number): SpacetimeEvent[] {
  const g = gamma(beta);
  const c_market = 1 + beta * 0.5;
  const events: SpacetimeEvent[] = [];

  if (bars.length < 2) return events;

  // Compute normalization constants from full dataset
  const priceRange = Math.max(...bars.map(b => b.high)) - Math.min(...bars.map(b => b.low));
  const volumeRange = Math.max(...bars.map(b => b.volume)) - Math.min(...bars.map(b => b.volume));

  for (let i = 1; i < bars.length; i++) {
    const prev = bars[i - 1];
    const curr = bars[i];

    // Normalize displacements to [0, ~1] scale
    const deltaT = 1.0; // one bar = one unit of proper time
    const deltaP = priceRange > 0 ? Math.abs(curr.close - prev.close) / (priceRange * 0.1) : 0;
    const deltaV = volumeRange > 0 ? Math.abs(curr.volume - prev.volume) / (volumeRange * 0.1) : 0;

    const interval = ds2(deltaT, deltaP, deltaV, c_market);
    const regime = classifyRegime(interval);

    events.push({
      bar: curr,
      beta,
      gamma: g,
      ds2: interval,
      regime,
      deltaP: curr.close - prev.close,
      deltaV: curr.volume - prev.volume,
      deltaT,
    });
  }

  return events;
}

/**
 * Compute the geodesic path through market spacetime.
 * In flat Minkowski space, geodesics are straight lines (linear interpolation).
 * β-dependent curvature bends the path via local momentum and volume.
 */
export function computeGeodesicPath(bars: OHLCVBar[], beta: number): number[] {
  if (bars.length === 0) return [];

  const prices = bars.map(b => b.close);
  const volumes = bars.map(b => b.volume);
  const n = prices.length;

  const startPrice = prices[0];
  const endPrice = prices[n - 1];

  // Flat geodesic: linear interpolation start → end
  const flatGeodesic = prices.map((_, i) => startPrice + (endPrice - startPrice) * (i / (n - 1)));

  // Compute local momentum for curvature perturbation
  const maxVol = Math.max(...volumes);
  const g = gamma(beta);

  const geodesic = flatGeodesic.map((flatPrice, i) => {
    if (i === 0 || i === n - 1) return flatPrice;

    // Local momentum: price velocity * relativistic gamma
    const localMomentum = i > 0 ? (prices[i] - prices[i - 1]) * g : 0;

    // Volume-induced curvature (high volume = stronger gravitational well)
    const volFraction = maxVol > 0 ? volumes[i] / maxVol : 0;
    const curvatureTerm = beta * localMomentum * volFraction * 0.3;

    return flatPrice + curvatureTerm;
  });

  return geodesic;
}

/**
 * Compute deviation between actual price path and geodesic.
 * Positive deviation = actual above geodesic (potential SHORT).
 * Negative deviation = actual below geodesic (potential LONG).
 */
export function geodesicDeviation(actual: number[], geodesic: number[]): number[] {
  const len = Math.min(actual.length, geodesic.length);
  return Array.from({ length: len }, (_, i) => actual[i] - geodesic[i]);
}
