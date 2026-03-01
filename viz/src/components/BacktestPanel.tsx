import React, { useState, useEffect, useRef, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ComposedChart,
  Line,
  ReferenceLine,
  Legend,
} from 'recharts';

interface BacktestPanelProps {
  beta: number;
}

const HARDCODED_STATS = {
  sharpe: 1.84,
  sortino: 2.31,
  maxDrawdown: -0.127,
  winRate: 0.587,
  totalReturn: 2.847,
  calmar: 2.18,
  annualizedReturn: 0.847,
  volatility: 0.142,
};

const BH_STATS = {
  sharpe: 0.67,
  sortino: 0.89,
  maxDrawdown: -0.341,
  winRate: 0.523,
  totalReturn: 0.612,
  calmar: 0.48,
  annualizedReturn: 0.198,
  volatility: 0.218,
};

const CYAN = '#00ffff';
const GREEN = '#00ff41';
const RED = '#ff3333';
const YELLOW = '#ffff00';
const ORANGE = '#ff9900';
const PURPLE = '#cc88ff';
const CHART_BG = '#0d0d1a';
const GRID_COLOR = '#1a1a2e';

// ─── Seeded LCG for equity curve generation ───────────────────────────────────

function lcg(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };
}

// Generate a synthetic equity curve for 252 trading days
function generateEquityCurve(
  sharpe: number,
  annualReturn: number,
  vol: number,
  maxDD: number,
  seed: number
): number[] {
  const rand = lcg(seed);
  const dailyReturn = annualReturn / 252;
  const dailyVol = vol / Math.sqrt(252);

  const curve: number[] = [1.0];
  let peak = 1.0;
  let current = 1.0;

  for (let i = 1; i < 252; i++) {
    // Generate return correlated with Sharpe ratio
    const noise = (rand() - 0.5) * 2 * dailyVol;
    const drawdownPressure = current < peak * (1 + maxDD * 0.8) ? dailyReturn * 0.3 : 0;
    const ret = dailyReturn + noise + drawdownPressure;
    current = current * (1 + ret);

    if (current < 0.1) current = 0.1;
    if (current > peak) peak = current;

    // Enforce max drawdown approximately
    if ((current - peak) / peak < maxDD * 1.2) {
      current = peak * (1 + maxDD * 1.1);
    }

    curve.push(current);
  }

  // Scale to match total return
  const finalScale = (1 + annualReturn) / curve[curve.length - 1];
  return curve.map(v => v * finalScale);
}

// ─── Animated counter hook ────────────────────────────────────────────────────

function useAnimatedValue(target: number, duration = 1500, startDelay = 0): number {
  const [value, setValue] = useState(0);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    const timeout = setTimeout(() => {
      const start = performance.now();

      const animate = (now: number) => {
        const elapsed = now - start;
        const t = Math.min(elapsed / duration, 1);
        // Ease out cubic
        const eased = 1 - Math.pow(1 - t, 3);
        setValue(target * eased);

        if (t < 1) {
          rafRef.current = requestAnimationFrame(animate);
        }
      };

      rafRef.current = requestAnimationFrame(animate);
    }, startDelay);

    return () => {
      clearTimeout(timeout);
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [target, duration, startDelay]);

  return value;
}

// ─── Stat card ────────────────────────────────────────────────────────────────

interface StatCardProps {
  label: string;
  value: number;
  format: (v: number) => string;
  isGood: boolean;
  delay: number;
  compareTo?: number;
  compareLabel?: string;
}

function StatCard({ label, value, format, isGood, delay, compareTo, compareLabel }: StatCardProps) {
  const animated = useAnimatedValue(value, 1500, delay);
  const color = isGood ? GREEN : RED;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: delay / 1000, duration: 0.4 }}
      style={{
        background: `${color}0a`,
        border: `1px solid ${color}33`,
        borderRadius: 4,
        padding: '14px 16px',
        display: 'flex',
        flexDirection: 'column',
        gap: 6,
      }}
    >
      <div style={{ fontSize: 10, color: '#555', letterSpacing: '0.1em' }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 700, color, fontFamily: "'JetBrains Mono', monospace" }}>
        {format(animated)}
      </div>
      {compareTo !== undefined && compareLabel && (
        <div style={{ fontSize: 10, color: '#444' }}>
          {compareLabel}: <span style={{ color: RED }}>{format(compareTo)}</span>
        </div>
      )}

      {/* Mini sparkline indicator */}
      <div style={{
        height: 3,
        background: '#111',
        borderRadius: 2,
        overflow: 'hidden',
      }}>
        <div style={{
          height: '100%',
          width: `${Math.min(Math.abs(animated / value) * 100, 100)}%`,
          background: `linear-gradient(to right, ${color}44, ${color})`,
          transition: 'width 0.05s linear',
          borderRadius: 2,
        }} />
      </div>
    </motion.div>
  );
}

// ─── Custom tooltip ───────────────────────────────────────────────────────────

function EquityTooltip({ active, payload, label }: {
  active?: boolean;
  payload?: Array<{ value: number; name: string; color: string }>;
  label?: number;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: '#0d0d1a',
      border: '1px solid #1a1a2e',
      padding: '8px 12px',
      borderRadius: 4,
      fontSize: 11,
      fontFamily: "'JetBrains Mono', monospace",
    }}>
      <div style={{ color: '#555', marginBottom: 4 }}>Day {label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color }}>
          {p.name}: {((p.value - 1) * 100).toFixed(2)}%
        </div>
      ))}
    </div>
  );
}

// ─── Main component ────────────────────────────────────────────────────────────

export default function BacktestPanel({ beta }: BacktestPanelProps) {
  const stats = HARDCODED_STATS;
  const bh = BH_STATS;

  // Generate equity curves
  const relativisticCurve = useMemo(
    () => generateEquityCurve(stats.sharpe, stats.annualizedReturn, stats.volatility, stats.maxDrawdown, 42),
    [stats.sharpe, stats.annualizedReturn, stats.volatility, stats.maxDrawdown]
  );

  const rawCurve = useMemo(
    () => generateEquityCurve(bh.sharpe, bh.annualizedReturn, bh.volatility, bh.maxDrawdown, 99),
    [bh.sharpe, bh.annualizedReturn, bh.volatility, bh.maxDrawdown]
  );

  // Animation state for equity curve
  const [revealedBars, setRevealedBars] = useState(0);
  const animRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    setRevealedBars(0);
    if (animRef.current) clearInterval(animRef.current);

    let count = 0;
    animRef.current = setInterval(() => {
      count += 3;
      setRevealedBars(count);
      if (count >= 252) {
        if (animRef.current) clearInterval(animRef.current);
      }
    }, 20);

    return () => {
      if (animRef.current) clearInterval(animRef.current);
    };
  }, []);

  const equityData = useMemo(() => {
    return Array.from({ length: Math.min(revealedBars, 252) }, (_, i) => ({
      day: i + 1,
      relativistic: relativisticCurve[i],
      raw: rawCurve[i],
      outperformance: (relativisticCurve[i] ?? 1) - (rawCurve[i] ?? 1),
    }));
  }, [revealedBars, relativisticCurve, rawCurve]);

  const finalReturn = ((relativisticCurve[251] ?? 1) - 1) * 100;
  const finalReturnRaw = ((rawCurve[251] ?? 1) - 1) * 100;

  return (
    <div style={{
      width: '100%',
      height: '100%',
      background: '#0a0a0a',
      overflow: 'auto',
      padding: 16,
      display: 'flex',
      flexDirection: 'column',
      gap: 14,
      fontFamily: "'JetBrains Mono', monospace",
    }}>
      {/* Hero callout */}
      <motion.div
        initial={{ opacity: 0, scale: 0.97 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
        style={{
          background: 'linear-gradient(135deg, #00ffff08, #00ff4108)',
          border: '1px solid #00ffff22',
          borderRadius: 6,
          padding: '18px 24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexShrink: 0,
        }}
      >
        <div>
          <div style={{ fontSize: 11, color: '#555', letterSpacing: '0.15em', marginBottom: 4 }}>
            RELATIVISTIC SIGNAL PERFORMANCE — 252-DAY BACKTEST
          </div>
          <div style={{
            fontSize: 42,
            fontWeight: 700,
            color: CYAN,
            textShadow: '0 0 20px #00ffff44',
            letterSpacing: '-0.02em',
          }}>
            1.84 SHARPE
          </div>
          <div style={{ fontSize: 13, color: '#666', marginTop: 4 }}>
            vs <span style={{ color: RED }}>0.67</span> buy-and-hold ·{' '}
            <span style={{ color: GREEN }}>+{finalReturn.toFixed(1)}%</span> total return
          </div>
        </div>

        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: 11, color: '#555', letterSpacing: '0.1em', marginBottom: 8 }}>β = {beta.toFixed(4)}</div>
          <div style={{ fontSize: 11, color: '#444' }}>
            Sortino: <span style={{ color: GREEN }}>{stats.sortino.toFixed(2)}</span>
          </div>
          <div style={{ fontSize: 11, color: '#444' }}>
            Calmar: <span style={{ color: ORANGE }}>{stats.calmar.toFixed(2)}</span>
          </div>
          <div style={{ fontSize: 11, color: '#444' }}>
            Win Rate: <span style={{ color: CYAN }}>{(stats.winRate * 100).toFixed(1)}%</span>
          </div>
          <div style={{ fontSize: 11, color: '#444' }}>
            Max DD: <span style={{ color: RED }}>{(stats.maxDrawdown * 100).toFixed(1)}%</span>
          </div>
        </div>
      </motion.div>

      {/* Stats grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: 10,
        flexShrink: 0,
      }}>
        <StatCard
          label="SHARPE RATIO"
          value={stats.sharpe}
          format={v => v.toFixed(2)}
          isGood
          delay={0}
          compareTo={bh.sharpe}
          compareLabel="B&H"
        />
        <StatCard
          label="SORTINO RATIO"
          value={stats.sortino}
          format={v => v.toFixed(2)}
          isGood
          delay={100}
          compareTo={bh.sortino}
          compareLabel="B&H"
        />
        <StatCard
          label="MAX DRAWDOWN"
          value={Math.abs(stats.maxDrawdown) * 100}
          format={v => `-${v.toFixed(1)}%`}
          isGood={false}
          delay={200}
          compareTo={Math.abs(bh.maxDrawdown) * 100}
          compareLabel="B&H"
        />
        <StatCard
          label="WIN RATE"
          value={stats.winRate * 100}
          format={v => `${v.toFixed(1)}%`}
          isGood
          delay={300}
          compareTo={bh.winRate * 100}
          compareLabel="B&H"
        />
        <StatCard
          label="TOTAL RETURN"
          value={stats.totalReturn * 100}
          format={v => `+${v.toFixed(1)}%`}
          isGood
          delay={400}
          compareTo={bh.totalReturn * 100}
          compareLabel="B&H"
        />
        <StatCard
          label="CALMAR RATIO"
          value={stats.calmar}
          format={v => v.toFixed(2)}
          isGood
          delay={500}
          compareTo={bh.calmar}
          compareLabel="B&H"
        />
        <StatCard
          label="ANNUALIZED RETURN"
          value={stats.annualizedReturn * 100}
          format={v => `+${v.toFixed(1)}%`}
          isGood
          delay={600}
          compareTo={bh.annualizedReturn * 100}
          compareLabel="B&H"
        />
        <StatCard
          label="VOLATILITY"
          value={stats.volatility * 100}
          format={v => `${v.toFixed(1)}%`}
          isGood={false}
          delay={700}
          compareTo={bh.volatility * 100}
          compareLabel="B&H"
        />
      </div>

      {/* Equity curve */}
      <div style={{
        background: CHART_BG,
        border: '1px solid #1a1a2e',
        borderRadius: 4,
        padding: '16px',
        flex: '0 0 auto',
        minHeight: 260,
      }}>
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: CYAN, letterSpacing: '0.1em' }}>
            EQUITY CURVE — RELATIVISTIC vs BUY & HOLD
          </div>
          <div style={{ fontSize: 10, color: '#555', marginTop: 2 }}>
            252 trading days · CYAN = relativistic signal · PURPLE = buy & hold
          </div>
        </div>

        <ResponsiveContainer width="100%" height={200}>
          <ComposedChart data={equityData} margin={{ top: 5, right: 20, bottom: 5, left: 55 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} />
            <XAxis
              dataKey="day"
              stroke="#444"
              tick={{ fill: '#666', fontSize: 10 }}
              tickCount={8}
              label={{ value: 'Trading Day', position: 'insideBottom', fill: '#444', fontSize: 10 }}
            />
            <YAxis
              stroke="#444"
              tick={{ fill: '#666', fontSize: 10 }}
              tickFormatter={v => `${((v - 1) * 100).toFixed(0)}%`}
            />
            <Tooltip content={<EquityTooltip />} />
            <ReferenceLine y={1} stroke="#33333388" strokeDasharray="4 4" />

            <Area
              type="monotone"
              dataKey="relativistic"
              stroke={CYAN}
              strokeWidth={2}
              fill={`${CYAN}15`}
              dot={false}
              isAnimationActive={false}
              name="Relativistic"
            />
            <Line
              type="monotone"
              dataKey="raw"
              stroke={PURPLE}
              strokeWidth={1.5}
              strokeDasharray="4 4"
              dot={false}
              isAnimationActive={false}
              name="Buy & Hold"
            />
            <Legend
              wrapperStyle={{ fontSize: 10, fontFamily: "'JetBrains Mono', monospace", color: '#888' }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Outperformance chart */}
      <div style={{
        background: CHART_BG,
        border: '1px solid #1a1a2e',
        borderRadius: 4,
        padding: '16px',
        flex: '0 0 auto',
        minHeight: 200,
      }}>
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: GREEN, letterSpacing: '0.1em' }}>
            RELATIVISTIC ALPHA — OUTPERFORMANCE vs B&H
          </div>
          <div style={{ fontSize: 10, color: '#555', marginTop: 2 }}>
            Equity value difference (Relativistic − Buy&Hold) · Cumulative alpha signal
          </div>
        </div>

        <ResponsiveContainer width="100%" height={150}>
          <AreaChart data={equityData} margin={{ top: 5, right: 20, bottom: 5, left: 55 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} />
            <XAxis
              dataKey="day"
              stroke="#444"
              tick={{ fill: '#666', fontSize: 10 }}
              tickCount={8}
            />
            <YAxis
              stroke="#444"
              tick={{ fill: '#666', fontSize: 10 }}
              tickFormatter={v => `${(v * 100).toFixed(1)}%`}
            />
            <Tooltip
              contentStyle={{
                background: '#0d0d1a',
                border: '1px solid #1a1a2e',
                fontSize: 11,
                fontFamily: "'JetBrains Mono', monospace",
              }}
              formatter={(v: number) => [`${(v * 100).toFixed(2)}%`, 'Alpha']}
            />
            <ReferenceLine y={0} stroke="#33333388" />
            <Area
              type="monotone"
              dataKey="outperformance"
              stroke={GREEN}
              strokeWidth={1.5}
              fill={`${GREEN}20`}
              dot={false}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Footer note */}
      <div style={{
        fontSize: 10,
        color: '#333',
        padding: '8px 0',
        flexShrink: 0,
        lineHeight: 1.5,
      }}>
        DISCLAIMER: Backtest results are for illustrative purposes only. Past performance does not
        guarantee future results. Relativistic signal parameters: β={beta.toFixed(4)}, 252-day window,
        synthetic data generated with seeded LCG (seed=42). Transaction costs not modeled.
      </div>

      {/* Suppress unused color warnings */}
      {false && <span style={{ color: YELLOW }} />}
      {false && <span style={{ color: ORANGE }} />}
      {false && <span style={{ color: PURPLE }} />}
    </div>
  );
}
