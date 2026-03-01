import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
  ReferenceLine,
} from 'recharts';
import { generateOHLCV, computeGeodesicPath, geodesicDeviation } from '../utils/physics';

interface GeodesicVizProps {
  beta: number;
}

const DARK_BG = '#0a0a0a';
const CHART_BG = '#0d0d1a';
const GRID_COLOR = '#1a1a2e';
const CYAN = '#00ffff';
const WHITE = '#e0e0e0';
const RED = '#ff3333';

// ─── Chart helpers ─────────────────────────────────────────────────────────────

function GeoTooltip({ active, payload, label }: {
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
      <div style={{ color: '#555', marginBottom: 4 }}>Bar #{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color }}>{p.name}: {p.value.toFixed(4)}</div>
      ))}
    </div>
  );
}

// ─── Main component ────────────────────────────────────────────────────────────

export default function GeodesicViz({ beta }: GeodesicVizProps) {
  const [animatedBars, setAnimatedBars] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const [animSpeed, setAnimSpeed] = useState(80);
  const animRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const allBars = useMemo(() => generateOHLCV(200, 42), []);

  const geodesicPath = useMemo(() => computeGeodesicPath(allBars, beta), [allBars, beta]);

  const prices = useMemo(() => allBars.map(b => b.close), [allBars]);

  const deviation = useMemo(() => geodesicDeviation(prices, geodesicPath), [prices, geodesicPath]);

  // 75th percentile threshold for deviation signal
  const deviationThreshold = useMemo(() => {
    const sorted = [...deviation].map(Math.abs).sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length * 0.75)] ?? 0;
  }, [deviation]);

  const startAnimation = useCallback(() => {
    if (animRef.current) clearInterval(animRef.current);
    setAnimatedBars(0);
    setIsAnimating(true);

    let count = 0;
    animRef.current = setInterval(() => {
      count++;
      setAnimatedBars(count);
      if (count >= allBars.length) {
        setIsAnimating(false);
        if (animRef.current) clearInterval(animRef.current);
      }
    }, animSpeed);
  }, [allBars.length, animSpeed]);

  useEffect(() => {
    startAnimation();
    return () => {
      if (animRef.current) clearInterval(animRef.current);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // only on mount

  // Restart animation when beta changes
  useEffect(() => {
    startAnimation();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [beta]);

  // Build chart data up to animatedBars
  const chartData = useMemo(() => {
    return allBars.slice(0, Math.max(animatedBars, 1)).map((bar, i) => ({
      index: bar.index,
      actual: bar.close,
      geodesic: geodesicPath[i],
      deviation: deviation[i],
      deviationAbs: Math.abs(deviation[i]),
      isSignal: Math.abs(deviation[i]) > deviationThreshold,
    }));
  }, [allBars, animatedBars, geodesicPath, deviation, deviationThreshold]);

  // For deviation chart coloring
  const deviationData = useMemo(() => {
    return allBars.slice(0, Math.max(animatedBars, 1)).map((bar, i) => ({
      index: bar.index,
      deviation: deviation[i],
      isSignal: Math.abs(deviation[i]) > deviationThreshold,
    }));
  }, [allBars, animatedBars, deviation, deviationThreshold]);

  const signalCount = useMemo(
    () => deviationData.filter(d => d.isSignal).length,
    [deviationData]
  );

  return (
    <div style={{
      width: '100%',
      height: '100%',
      background: DARK_BG,
      display: 'flex',
      flexDirection: 'column',
      overflow: 'auto',
      padding: 16,
      gap: 12,
    }}>
      {/* Controls */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 16,
        padding: '10px 14px',
        background: CHART_BG,
        border: '1px solid #1a1a2e',
        borderRadius: 4,
        flexShrink: 0,
      }}>
        <button
          onClick={startAnimation}
          disabled={isAnimating}
          style={{
            background: isAnimating ? 'transparent' : 'rgba(0,255,255,0.15)',
            border: `1px solid ${isAnimating ? '#1a1a2e' : '#00ffff44'}`,
            color: isAnimating ? '#444' : CYAN,
            fontSize: 11,
            fontFamily: "'JetBrains Mono', monospace",
            padding: '5px 14px',
            cursor: isAnimating ? 'default' : 'pointer',
            borderRadius: 2,
          }}
        >
          {isAnimating ? '⟳ COMPUTING...' : '▶ ANIMATE GEODESIC'}
        </button>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11 }}>
          <span style={{ color: '#555' }}>SPEED:</span>
          <input
            type="range"
            min={50}
            max={500}
            step={10}
            value={animSpeed}
            onChange={e => setAnimSpeed(parseInt(e.target.value))}
            style={{ width: 100, accentColor: CYAN }}
          />
          <span style={{ color: '#00ffff', minWidth: 50 }}>{animSpeed}ms</span>
        </div>

        <div style={{ marginLeft: 'auto', display: 'flex', gap: 20, fontSize: 11 }}>
          <span style={{ color: '#555' }}>
            BARS: <span style={{ color: CYAN }}>{animatedBars}/{allBars.length}</span>
          </span>
          <span style={{ color: '#555' }}>
            SIGNALS: <span style={{ color: RED }}>{signalCount}</span>
          </span>
          <span style={{ color: '#555' }}>
            THRESHOLD: <span style={{ color: '#ffff00' }}>±{deviationThreshold.toFixed(3)}</span>
          </span>
        </div>
      </div>

      {/* Top chart: Price vs Geodesic */}
      <div style={{
        background: CHART_BG,
        border: '1px solid #1a1a2e',
        borderRadius: 4,
        padding: '16px 16px 8px',
        flex: '0 0 auto',
        minHeight: 260,
      }}>
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: CYAN, letterSpacing: '0.1em' }}>
            PRICE vs GEODESIC PATH
          </div>
          <div style={{ fontSize: 10, color: '#555', marginTop: 2 }}>
            WHITE: actual close · CYAN DASHED: flat-spacetime geodesic · RED FILL: deviation signal
          </div>
        </div>
        <ResponsiveContainer width="100%" height={200}>
          <ComposedChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 50 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} />
            <XAxis
              dataKey="index"
              stroke="#444"
              tick={{ fill: '#666', fontSize: 10 }}
              tickCount={8}
            />
            <YAxis
              stroke="#444"
              tick={{ fill: '#666', fontSize: 10 }}
              tickFormatter={v => v.toFixed(0)}
            />
            <Tooltip content={<GeoTooltip />} />

            {/* Area between actual and geodesic for signal zones */}
            <Area
              type="monotone"
              dataKey="actual"
              stroke="none"
              fill={`${RED}22`}
              isAnimationActive={false}
              name="actual"
            />
            <Area
              type="monotone"
              dataKey="geodesic"
              stroke="none"
              fill={DARK_BG}
              isAnimationActive={false}
              name="geodesic"
            />

            {/* Geodesic path */}
            <Line
              type="monotone"
              dataKey="geodesic"
              stroke={CYAN}
              strokeWidth={1.5}
              strokeDasharray="6 3"
              dot={false}
              isAnimationActive={false}
              name="Geodesic"
            />

            {/* Actual price */}
            <Line
              type="monotone"
              dataKey="actual"
              stroke={WHITE}
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
              name="Actual"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Bottom chart: Deviation signal */}
      <div style={{
        background: CHART_BG,
        border: '1px solid #1a1a2e',
        borderRadius: 4,
        padding: '16px 16px 8px',
        flex: '0 0 auto',
        minHeight: 220,
      }}>
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: CYAN, letterSpacing: '0.1em' }}>
            GEODESIC DEVIATION SIGNAL
          </div>
          <div style={{ fontSize: 10, color: '#555', marginTop: 2 }}>
            RED bars exceed 75th percentile threshold → potential alpha signal
          </div>
        </div>
        <ResponsiveContainer width="100%" height={160}>
          <BarChart data={deviationData} margin={{ top: 5, right: 20, bottom: 5, left: 50 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} vertical={false} />
            <XAxis
              dataKey="index"
              stroke="#444"
              tick={{ fill: '#666', fontSize: 10 }}
              tickCount={8}
            />
            <YAxis
              stroke="#444"
              tick={{ fill: '#666', fontSize: 10 }}
              tickFormatter={v => v.toFixed(2)}
            />
            <Tooltip
              contentStyle={{
                background: '#0d0d1a',
                border: '1px solid #1a1a2e',
                fontSize: 11,
                fontFamily: "'JetBrains Mono', monospace",
              }}
              formatter={(v: number) => [v.toFixed(4), 'deviation']}
            />
            <ReferenceLine y={deviationThreshold} stroke="#ffff0066" strokeDasharray="4 4" />
            <ReferenceLine y={-deviationThreshold} stroke="#ffff0066" strokeDasharray="4 4" />
            <ReferenceLine y={0} stroke="#33333388" />
            <Bar dataKey="deviation" isAnimationActive={false} maxBarSize={6}>
              {deviationData.map((entry, i) => (
                <Cell
                  key={i}
                  fill={entry.isSignal ? RED : '#00ffff44'}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Explanation panel */}
      <div style={{
        background: CHART_BG,
        border: '1px solid #1a1a2e',
        borderRadius: 4,
        padding: '14px 16px',
        fontSize: 11,
        color: '#888',
        lineHeight: 1.7,
        flexShrink: 0,
      }}>
        <div style={{ color: CYAN, fontWeight: 700, fontSize: 12, marginBottom: 8, letterSpacing: '0.1em' }}>
          THEORETICAL BASIS
        </div>
        <p>
          In flat <strong style={{ color: WHITE }}>Minkowski spacetime</strong>, geodesics are straight lines
          — the shortest paths between two events. In the market spacetime defined by the metric
          ds² = −c²ΔT² + ΔP² + ΔV², the geodesic represents the &quot;natural&quot; price trajectory.
        </p>
        <p style={{ marginTop: 8 }}>
          <strong style={{ color: CYAN }}>Market curvature</strong> bends the geodesic path via local
          momentum (price velocity × γ) and volume-induced gravitational wells. When β increases,
          the Lorentz factor γ amplifies momentum effects, creating larger deviations between
          the flat-space geodesic and the curved path.
        </p>
        <p style={{ marginTop: 8 }}>
          <strong style={{ color: RED }}>Deviation signals</strong> above the 75th percentile
          threshold indicate regions where the market has curved significantly away from its
          geodesic — potential mean-reversion alpha opportunities (SPACELIKE regime).
        </p>
        <p style={{ marginTop: 8, color: '#555' }}>
          Current β = {beta.toFixed(4)} → curvature coefficient = {(beta * 0.3).toFixed(4)} ·
          geodesic drawn over {animatedBars}/{allBars.length} bars
        </p>
      </div>
    </div>
  );
}
