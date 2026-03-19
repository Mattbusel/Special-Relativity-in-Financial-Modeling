/**
 * Mock / demo data for development, Storybook stories, and fallback when the
 * live WebSocket feed is unavailable.  These are NOT production values —
 * they are synthetic figures generated with seeded random functions so that
 * stories and offline runs remain deterministic.
 *
 * Swap any of these exports for real API responses to move from demo → live.
 */

import { generateOHLCV } from '../utils/physics';

// ─── BacktestPanel demo stats ──────────────────────────────────────────────────

/** Relativistic strategy backtest statistics (demo / fallback values). */
export const HARDCODED_STATS = {
  sharpe: 1.84,
  sortino: 2.31,
  maxDrawdown: -0.127,
  winRate: 0.587,
  totalReturn: 2.847,
  calmar: 2.18,
  annualizedReturn: 0.847,
  volatility: 0.142,
};

/** Buy-and-hold benchmark statistics (demo / fallback values). */
export const BH_STATS = {
  sharpe: 0.67,
  sortino: 0.89,
  maxDrawdown: -0.341,
  winRate: 0.523,
  totalReturn: 0.612,
  calmar: 0.48,
  annualizedReturn: 0.198,
  volatility: 0.218,
};

// ─── GeodesicViz demo bars ─────────────────────────────────────────────────────

/**
 * 200 synthetic OHLCV bars generated with seed=42.
 * Used by GeodesicViz when no live data source is connected.
 */
export const DEMO_BARS = generateOHLCV(200, 42);
