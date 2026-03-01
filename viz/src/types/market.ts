export type Regime = 'TIMELIKE' | 'SPACELIKE' | 'LIGHTLIKE';

export interface OHLCVBar {
  index: number;
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface SpacetimeEvent {
  bar: OHLCVBar;
  beta: number;
  gamma: number;
  ds2: number;
  regime: Regime;
  deltaP: number;
  deltaV: number;
  deltaT: number;
}

export interface BacktestStats {
  sharpe: number;
  sortino: number;
  maxDrawdown: number;
  winRate: number;
  totalReturn: number;
  calmar: number;
  annualizedReturn: number;
  volatility: number;
}
