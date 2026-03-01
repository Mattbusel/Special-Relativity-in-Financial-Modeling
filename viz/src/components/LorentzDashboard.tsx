import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ReferenceDot,
  BarChart,
  Bar,
  Cell,
  ComposedChart,
  Area,
  Legend,
} from 'recharts';
import { gamma, rapidity } from '../utils/physics';

interface LorentzDashboardProps {
  beta: number;
}

const DARK_BG = '#0a0a0a';
const CHART_BG = '#0d0d1a';
const GRID_COLOR = '#1a1a2e';
const CYAN = '#00ffff';
const GREEN = '#00ff41';
const RED = '#ff3333';
const YELLOW = '#ffff00';
const ORANGE = '#ff9900';
const PURPLE = '#cc88ff';

// Generate 200-point beta sweep data
function generateBetaSweep(currentBeta: number) {
  const points = 200;
  return Array.from({ length: points }, (_, i) => {
    const b = (i / (points - 1)) * 0.9999;
    const g = gamma(b);
    const phi = rapidity(b);
    return {
      beta: parseFloat(b.toFixed(4)),
      gamma: g > 10 ? 10 : g,
      gammaRaw: g,
      rapidity: phi,
      newtonianP: b,              // p = m·v (normalized, m=1, v=b·c)
      relativisticP: g * b,       // p = γ·m·v
      isCurrent: Math.abs(b - currentBeta) < 0.005,
    };
  });
}

const darkTooltipStyle = {
  contentStyle: {
    background: '#0d0d1a',
    border: '1px solid #1a1a2e',
    borderRadius: 4,
    color: '#e0e0e0',
    fontSize: 11,
    fontFamily: "'JetBrains Mono', monospace",
  },
  labelStyle: { color: '#888' },
  itemStyle: { color: CYAN },
};

// Custom tooltip factory
function makeTooltip(formatter: (value: number, name: string) => string) {
  return ({ active, payload, label }: {
    active?: boolean;
    payload?: Array<{ value: number; name: string; color: string }>;
    label?: number;
  }) => {
    if (!active || !payload?.length) return null;
    return (
      <div style={{
        background: '#0d0d1a',
        border: '1px solid #1a1a2e',
        borderRadius: 4,
        padding: '8px 12px',
        fontSize: 11,
        fontFamily: "'JetBrains Mono', monospace",
      }}>
        <div style={{ color: '#888', marginBottom: 4 }}>β = {(label ?? 0).toFixed(4)}</div>
        {payload.map((p, i) => (
          <div key={i} style={{ color: p.color }}>
            {formatter(p.value, p.name)}
          </div>
        ))}
      </div>
    );
  };
}

// Chart wrapper
function ChartCard({ title, subtitle, children }: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <div style={{
      background: CHART_BG,
      border: '1px solid #1a1a2e',
      borderRadius: 4,
      padding: '16px',
      display: 'flex',
      flexDirection: 'column',
    }}>
      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: CYAN, letterSpacing: '0.1em' }}>
          {title}
        </div>
        {subtitle && (
          <div style={{ fontSize: 10, color: '#555', marginTop: 2 }}>{subtitle}</div>
        )}
      </div>
      {children}
    </div>
  );
}

// ─── Chart 1: γ(β) ───────────────────────────────────────────────────────────

function GammaCurve({ data, currentBeta }: { data: ReturnType<typeof generateBetaSweep>; currentBeta: number }) {
  const g = gamma(currentBeta);
  const gCapped = Math.min(g, 10);
  const GammaTooltip = makeTooltip((v, n) => `${n}: ${v.toFixed(4)}`);

  return (
    <ChartCard
      title="γ(β) — LORENTZ FACTOR"
      subtitle="γ = 1/√(1−β²) · diverges as β → 1"
    >
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={data} margin={{ top: 5, right: 10, bottom: 5, left: 45 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} />
          <XAxis
            dataKey="beta"
            tickFormatter={v => v.toFixed(2)}
            stroke="#444"
            tick={{ fill: '#666', fontSize: 10 }}
            tickCount={6}
          />
          <YAxis
            domain={[1, 10]}
            stroke="#444"
            tick={{ fill: '#666', fontSize: 10 }}
            tickFormatter={v => `γ=${v.toFixed(1)}`}
          />
          <Tooltip content={<GammaTooltip />} />
          <ReferenceLine x={currentBeta} stroke="#ffffff22" strokeDasharray="4 4" />
          <Line
            type="monotone"
            dataKey="gamma"
            stroke={CYAN}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
            name="γ"
          />
          <ReferenceDot
            x={currentBeta}
            y={gCapped}
            r={5}
            fill={RED}
            stroke={RED}
            label={{
              value: `γ=${g.toFixed(2)}`,
              position: 'right',
              fill: RED,
              fontSize: 10,
              fontFamily: "'JetBrains Mono', monospace",
            }}
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

// ─── Chart 2: Rapidity ────────────────────────────────────────────────────────

function RapidityCurve({ data, currentBeta }: { data: ReturnType<typeof generateBetaSweep>; currentBeta: number }) {
  const phi = rapidity(currentBeta);
  const RapidityTooltip = makeTooltip((v) => `φ = ${v.toFixed(4)}`);

  return (
    <ChartCard
      title="φ(β) — RAPIDITY"
      subtitle="φ = atanh(β) · additive under successive boosts"
    >
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={data} margin={{ top: 5, right: 10, bottom: 5, left: 50 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} />
          <XAxis
            dataKey="beta"
            tickFormatter={v => v.toFixed(2)}
            stroke="#444"
            tick={{ fill: '#666', fontSize: 10 }}
            tickCount={6}
          />
          <YAxis
            stroke="#444"
            tick={{ fill: '#666', fontSize: 10 }}
            tickFormatter={v => `φ=${v.toFixed(2)}`}
          />
          <Tooltip content={<RapidityTooltip />} />
          <ReferenceLine x={currentBeta} stroke="#ffffff22" strokeDasharray="4 4" />
          <Line
            type="monotone"
            dataKey="rapidity"
            stroke={ORANGE}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
            name="φ"
          />
          <ReferenceDot
            x={currentBeta}
            y={phi}
            r={5}
            fill={YELLOW}
            stroke={YELLOW}
            label={{
              value: `φ=${phi.toFixed(3)}`,
              position: 'right',
              fill: YELLOW,
              fontSize: 10,
              fontFamily: "'JetBrains Mono', monospace",
            }}
          />
        </LineChart>
      </ResponsiveContainer>
      <div style={{ fontSize: 10, color: '#555', marginTop: 8, fontStyle: 'italic' }}>
        φ₁ + φ₂ = φ_total (additive!) — unlike velocities which obey v_rel = (v₁+v₂)/(1+v₁v₂/c²)
      </div>
    </ChartCard>
  );
}

// ─── Chart 3: Time Dilation ───────────────────────────────────────────────────

function TimeDilationChart({ currentBeta }: { currentBeta: number }) {
  const g = gamma(currentBeta);
  const dilated = g * 1.0;

  const barData = [
    { name: 'Proper Time τ', value: 1.0, fill: GREEN },
    { name: 'Dilated Time t', value: dilated > 20 ? 20 : dilated, fill: RED },
  ];

  return (
    <ChartCard
      title="TIME DILATION"
      subtitle={`t = γτ — moving clocks run slow. t = ${dilated.toFixed(3)}τ`}
    >
      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={barData} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} vertical={false} />
          <XAxis
            dataKey="name"
            stroke="#444"
            tick={{ fill: '#888', fontSize: 10, fontFamily: "'JetBrains Mono', monospace" }}
          />
          <YAxis
            stroke="#444"
            tick={{ fill: '#666', fontSize: 10 }}
            tickFormatter={v => `${v.toFixed(1)}τ`}
          />
          <Tooltip
            contentStyle={{
              background: '#0d0d1a',
              border: '1px solid #1a1a2e',
              fontSize: 11,
              fontFamily: "'JetBrains Mono', monospace",
            }}
            formatter={(v: number) => [`${v.toFixed(4)}τ`]}
          />
          <Bar dataKey="value" isAnimationActive radius={[2, 2, 0, 0]}>
            {barData.map((entry, i) => (
              <Cell key={i} fill={entry.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

// ─── Chart 4: Momentum Divergence ─────────────────────────────────────────────

function MomentumDivergence({ data, currentBeta }: { data: ReturnType<typeof generateBetaSweep>; currentBeta: number }) {
  const g = gamma(currentBeta);
  const MomTooltip = ({ active, payload, label }: {
    active?: boolean;
    payload?: Array<{ value: number; name: string; color: string }>;
    label?: number;
  }) => {
    if (!active || !payload?.length) return null;
    return (
      <div style={{
        background: '#0d0d1a',
        border: '1px solid #1a1a2e',
        borderRadius: 4,
        padding: '8px 12px',
        fontSize: 11,
        fontFamily: "'JetBrains Mono', monospace",
      }}>
        <div style={{ color: '#888', marginBottom: 4 }}>β = {(label ?? 0).toFixed(4)}</div>
        {payload.map((p, i) => (
          <div key={i} style={{ color: p.color }}>
            {p.name}: {p.value.toFixed(4)}
          </div>
        ))}
      </div>
    );
  };

  return (
    <ChartCard
      title="MOMENTUM DIVERGENCE"
      subtitle="Newtonian p=mv vs Relativistic p=γmv (m=1 unit)"
    >
      <ResponsiveContainer width="100%" height={180}>
        <ComposedChart data={data} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} />
          <XAxis
            dataKey="beta"
            tickFormatter={v => v.toFixed(2)}
            stroke="#444"
            tick={{ fill: '#666', fontSize: 10 }}
            tickCount={6}
          />
          <YAxis
            stroke="#444"
            tick={{ fill: '#666', fontSize: 10 }}
            tickFormatter={v => v.toFixed(1)}
          />
          <Tooltip content={<MomTooltip />} />
          <ReferenceLine x={currentBeta} stroke="#ffffff22" strokeDasharray="4 4" />

          {/* Fill between curves */}
          <Area
            type="monotone"
            dataKey="relativisticP"
            stroke={CYAN}
            fill={`${CYAN}18`}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
            name="Relativistic p"
          />
          <Line
            type="monotone"
            dataKey="newtonianP"
            stroke={PURPLE}
            strokeWidth={1.5}
            strokeDasharray="4 4"
            dot={false}
            isAnimationActive={false}
            name="Newtonian p"
          />

          <ReferenceDot
            x={currentBeta}
            y={Math.min(g * currentBeta, 10)}
            r={4}
            fill={RED}
            stroke={RED}
          />

          <Legend
            wrapperStyle={{ fontSize: 10, color: '#888', fontFamily: "'JetBrains Mono', monospace" }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

// ─── Stats panel ──────────────────────────────────────────────────────────────

function StatsPanel({ beta }: { beta: number }) {
  const g = gamma(beta);
  const phi = rapidity(beta);

  const stats = [
    { label: 'β (velocity/c)', value: beta.toFixed(6), color: CYAN },
    { label: 'γ (Lorentz factor)', value: g.toFixed(6), color: GREEN },
    { label: 'φ (rapidity)', value: phi.toFixed(6), color: ORANGE },
    { label: '1/γ (time compression)', value: (1 / g).toFixed(6), color: PURPLE },
    { label: 'γ·β (relativistic momentum)', value: (g * beta).toFixed(6), color: YELLOW },
    { label: 'γ-1 (kinetic energy/mc²)', value: (g - 1).toFixed(6), color: RED },
    { label: 'c_market = 1+β·0.5', value: (1 + beta * 0.5).toFixed(6), color: '#88aaff' },
    { label: 'Cone half-angle (°)', value: `${(Math.asin(1 / g) * 180 / Math.PI).toFixed(2)}°`, color: '#ffaa88' },
  ];

  return (
    <div style={{
      background: CHART_BG,
      border: '1px solid #1a1a2e',
      borderRadius: 4,
      padding: '14px 20px',
    }}>
      <div style={{ fontSize: 12, fontWeight: 700, color: CYAN, letterSpacing: '0.1em', marginBottom: 12 }}>
        RELATIVISTIC KINEMATICS — LIVE STATE
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px 24px' }}>
        {stats.map(({ label, value, color }) => (
          <div key={label} style={{ minWidth: 220 }}>
            <span style={{ color: '#555', fontSize: 10 }}>{label}: </span>
            <span style={{ color, fontSize: 12, fontWeight: 700 }}>{value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Main ─────────────────────────────────────────────────────────────────────

export default function LorentzDashboard({ beta }: LorentzDashboardProps) {
  const data = useMemo(() => generateBetaSweep(beta), [beta]);

  return (
    <div style={{
      width: '100%',
      height: '100%',
      background: DARK_BG,
      overflow: 'auto',
      padding: 16,
      display: 'flex',
      flexDirection: 'column',
      gap: 12,
    }}>
      {/* 2x2 chart grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: 12,
        flex: 1,
        minHeight: 0,
      }}>
        <GammaCurve data={data} currentBeta={beta} />
        <RapidityCurve data={data} currentBeta={beta} />
        <TimeDilationChart currentBeta={beta} />
        <MomentumDivergence data={data} currentBeta={beta} />
      </div>

      {/* Stats panel */}
      <StatsPanel beta={beta} />
    </div>
  );
}

// Suppress unused import warnings for recharts items we import but don't use directly
void darkTooltipStyle;
