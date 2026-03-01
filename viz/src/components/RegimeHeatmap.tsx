import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { generateOHLCV, computeSpacetimeEvents, gamma } from '../utils/physics';
import type { SpacetimeEvent } from '../types/market';

interface RegimeHeatmapProps {
  beta: number;
}

type PlaybackSpeed = 1 | 5 | 10;

const TIMELIKE_BG = 'rgba(0,255,255,0.12)';
const SPACELIKE_BG = 'rgba(255,51,51,0.12)';
const LIGHTLIKE_BG = 'rgba(255,255,0,0.12)';
const TIMELIKE_COLOR = '#00ffff';
const SPACELIKE_COLOR = '#ff3333';
const LIGHTLIKE_COLOR = '#ffff00';

function regimeBg(regime: string): string {
  switch (regime) {
    case 'TIMELIKE': return TIMELIKE_BG;
    case 'SPACELIKE': return SPACELIKE_BG;
    case 'LIGHTLIKE': return LIGHTLIKE_BG;
    default: return 'transparent';
  }
}

function regimeColor(regime: string): string {
  switch (regime) {
    case 'TIMELIKE': return TIMELIKE_COLOR;
    case 'SPACELIKE': return SPACELIKE_COLOR;
    case 'LIGHTLIKE': return LIGHTLIKE_COLOR;
    default: return '#e0e0e0';
  }
}

// ─── Mini distribution bar ────────────────────────────────────────────────────

function DistBar({ events }: { events: SpacetimeEvent[] }) {
  const total = events.length;
  if (total === 0) return null;

  const tl = events.filter(e => e.regime === 'TIMELIKE').length;
  const sl = events.filter(e => e.regime === 'SPACELIKE').length;
  const ll = events.filter(e => e.regime === 'LIGHTLIKE').length;

  return (
    <div style={{ display: 'flex', width: 120, height: 8, borderRadius: 2, overflow: 'hidden' }}>
      <div style={{ flex: tl, background: TIMELIKE_COLOR }} />
      <div style={{ flex: sl, background: SPACELIKE_COLOR }} />
      <div style={{ flex: ll, background: LIGHTLIKE_COLOR }} />
    </div>
  );
}

// ─── Event detail side panel ──────────────────────────────────────────────────

function EventDetail({ event, onClose }: { event: SpacetimeEvent; onClose: () => void }) {
  const b = event.bar;
  return (
    <div style={{
      position: 'absolute',
      top: 0,
      right: 0,
      width: 320,
      height: '100%',
      background: '#0d0d1a',
      borderLeft: '1px solid #1a1a2e',
      padding: 20,
      overflowY: 'auto',
      zIndex: 20,
      fontFamily: "'JetBrains Mono', monospace",
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
        <div style={{ color: '#00ffff', fontSize: 12, fontWeight: 700, letterSpacing: '0.1em' }}>
          BAR #{b.index} DETAIL
        </div>
        <button
          onClick={onClose}
          style={{
            background: 'transparent',
            border: '1px solid #333',
            color: '#888',
            cursor: 'pointer',
            fontFamily: 'inherit',
            fontSize: 11,
            padding: '2px 8px',
            borderRadius: 2,
          }}
        >
          ✕
        </button>
      </div>

      <Section title="OHLCV">
        <Row label="Open" value={b.open.toFixed(4)} color="#aaa" />
        <Row label="High" value={b.high.toFixed(4)} color="#00ff41" />
        <Row label="Low" value={b.low.toFixed(4)} color="#ff3333" />
        <Row label="Close" value={b.close.toFixed(4)} color="#00ffff" />
        <Row label="Volume" value={b.volume.toLocaleString()} color="#888" />
        <Row label="Timestamp" value={new Date(b.timestamp).toISOString().slice(0, 19)} color="#666" />
      </Section>

      <Section title="SPACETIME KINEMATICS">
        <Row label="β" value={event.beta.toFixed(6)} color="#00ffff" />
        <Row label="γ" value={event.gamma.toFixed(6)} color="#00ff41" />
        <Row label="ds²" value={event.ds2.toFixed(6)} color={regimeColor(event.regime)} />
        <Row label="ΔP (price)" value={event.deltaP.toFixed(4)} color="#aaa" />
        <Row label="ΔV (volume)" value={event.deltaV.toFixed(0)} color="#aaa" />
        <Row label="ΔT (time)" value={event.deltaT.toFixed(4)} color="#aaa" />
      </Section>

      <Section title={`REGIME: ${event.regime}`}>
        <div style={{
          background: regimeBg(event.regime),
          border: `1px solid ${regimeColor(event.regime)}44`,
          borderRadius: 4,
          padding: '10px 12px',
          fontSize: 11,
          color: '#aaa',
          lineHeight: 1.6,
        }}>
          {event.regime === 'TIMELIKE' && (
            <>
              <strong style={{ color: TIMELIKE_COLOR }}>TIMELIKE</strong>: ds² &lt; 0. The
              spacetime interval is causal — price displacement is dominated by time flow.
              Analogous to a trend-following regime where momentum carries price information
              across time.
            </>
          )}
          {event.regime === 'SPACELIKE' && (
            <>
              <strong style={{ color: SPACELIKE_COLOR }}>SPACELIKE</strong>: ds² &gt; 0. The
              spacetime interval is non-causal — price displacement exceeds the market
              speed of causality. Analogous to a mean-reversion regime where price has moved
              too far, too fast.
            </>
          )}
          {event.regime === 'LIGHTLIKE' && (
            <>
              <strong style={{ color: LIGHTLIKE_COLOR }}>LIGHTLIKE</strong>: ds² ≈ 0. The
              event lies on the light cone boundary — transitioning between causal and
              non-causal dynamics. High uncertainty, potential regime change.
            </>
          )}
        </div>
      </Section>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 10, color: '#555', letterSpacing: '0.1em', marginBottom: 8 }}>
        {title}
      </div>
      {children}
    </div>
  );
}

function Row({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4, fontSize: 11 }}>
      <span style={{ color: '#666' }}>{label}</span>
      <span style={{ color }}>{value}</span>
    </div>
  );
}

// ─── Main component ────────────────────────────────────────────────────────────

export default function RegimeHeatmap({ beta }: RegimeHeatmapProps) {
  const [visibleCount, setVisibleCount] = useState(50);
  const [playbackSpeed, setPlaybackSpeed] = useState<PlaybackSpeed>(1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedEvent, setSelectedEvent] = useState<SpacetimeEvent | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const tableRef = useRef<HTMLDivElement>(null);

  const allBars = useMemo(() => generateOHLCV(500, 42), []);
  const allEvents = useMemo(() => computeSpacetimeEvents(allBars, beta), [allBars, beta]);

  const visibleEvents = useMemo(() => allEvents.slice(0, visibleCount), [allEvents, visibleCount]);

  const counts = useMemo(() => ({
    TIMELIKE: visibleEvents.filter(e => e.regime === 'TIMELIKE').length,
    SPACELIKE: visibleEvents.filter(e => e.regime === 'SPACELIKE').length,
    LIGHTLIKE: visibleEvents.filter(e => e.regime === 'LIGHTLIKE').length,
  }), [visibleEvents]);

  const g = gamma(beta);

  const stopPlayback = useCallback(() => {
    setIsPlaying(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  const startPlayback = useCallback(() => {
    setIsPlaying(true);
    if (intervalRef.current) clearInterval(intervalRef.current);

    intervalRef.current = setInterval(() => {
      setVisibleCount(prev => {
        const next = prev + playbackSpeed * 2;
        if (next >= allEvents.length) {
          stopPlayback();
          return allEvents.length;
        }
        return next;
      });
    }, 100);
  }, [playbackSpeed, allEvents.length, stopPlayback]);

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  // Restart playback when speed changes
  useEffect(() => {
    if (isPlaying) {
      if (intervalRef.current) clearInterval(intervalRef.current);
      intervalRef.current = setInterval(() => {
        setVisibleCount(prev => {
          const next = prev + playbackSpeed * 2;
          if (next >= allEvents.length) {
            stopPlayback();
            return allEvents.length;
          }
          return next;
        });
      }, 100);
    }
  }, [playbackSpeed, isPlaying, allEvents.length, stopPlayback]);

  const handleReset = () => {
    stopPlayback();
    setVisibleCount(50);
    setSelectedEvent(null);
  };

  const progress = (visibleCount / allEvents.length) * 100;

  return (
    <div style={{
      width: '100%',
      height: '100%',
      background: '#0a0a0a',
      display: 'flex',
      flexDirection: 'column',
      fontFamily: "'JetBrains Mono', monospace",
      position: 'relative',
      overflow: 'hidden',
    }}>
      {/* Stats bar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 24,
        padding: '10px 16px',
        background: '#0d0d1a',
        borderBottom: '1px solid #1a1a2e',
        flexShrink: 0,
      }}>
        <div style={{ fontSize: 11 }}>
          <span style={{ color: '#555' }}>TIMELIKE: </span>
          <span style={{ color: TIMELIKE_COLOR, fontWeight: 700 }}>{counts.TIMELIKE}</span>
        </div>
        <div style={{ fontSize: 11 }}>
          <span style={{ color: '#555' }}>SPACELIKE: </span>
          <span style={{ color: SPACELIKE_COLOR, fontWeight: 700 }}>{counts.SPACELIKE}</span>
        </div>
        <div style={{ fontSize: 11 }}>
          <span style={{ color: '#555' }}>LIGHTLIKE: </span>
          <span style={{ color: LIGHTLIKE_COLOR, fontWeight: 700 }}>{counts.LIGHTLIKE}</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 10, color: '#444' }}>DIST:</span>
          <DistBar events={visibleEvents} />
        </div>
        <div style={{ marginLeft: 'auto', fontSize: 11 }}>
          <span style={{ color: '#555' }}>β=</span>
          <span style={{ color: TIMELIKE_COLOR }}>{beta.toFixed(4)}</span>
          <span style={{ color: '#555', marginLeft: 12 }}>γ=</span>
          <span style={{ color: '#00ff41' }}>{g.toFixed(4)}</span>
        </div>
      </div>

      {/* Playback controls */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        padding: '8px 16px',
        background: '#080810',
        borderBottom: '1px solid #1a1a2e',
        flexShrink: 0,
      }}>
        <button
          onClick={isPlaying ? stopPlayback : startPlayback}
          disabled={visibleCount >= allEvents.length}
          style={{
            background: isPlaying ? 'rgba(255,51,51,0.15)' : 'rgba(0,255,65,0.15)',
            border: `1px solid ${isPlaying ? '#ff333344' : '#00ff4144'}`,
            color: isPlaying ? '#ff3333' : '#00ff41',
            fontSize: 11,
            fontFamily: 'inherit',
            padding: '4px 14px',
            cursor: 'pointer',
            borderRadius: 2,
            minWidth: 70,
          }}
        >
          {isPlaying ? '⏸ PAUSE' : '▶ PLAY'}
        </button>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11 }}>
          <span style={{ color: '#555' }}>SPEED:</span>
          {([1, 5, 10] as PlaybackSpeed[]).map(s => (
            <button
              key={s}
              onClick={() => setPlaybackSpeed(s)}
              style={{
                background: playbackSpeed === s ? 'rgba(0,255,255,0.2)' : 'transparent',
                border: `1px solid ${playbackSpeed === s ? '#00ffff44' : '#1a1a2e'}`,
                color: playbackSpeed === s ? TIMELIKE_COLOR : '#555',
                fontSize: 10,
                fontFamily: 'inherit',
                padding: '2px 10px',
                cursor: 'pointer',
                borderRadius: 2,
              }}
            >
              {s}x
            </button>
          ))}
        </div>

        {/* Progress bar */}
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{
            flex: 1,
            height: 4,
            background: '#1a1a2e',
            borderRadius: 2,
            overflow: 'hidden',
          }}>
            <div style={{
              width: `${progress}%`,
              height: '100%',
              background: `linear-gradient(to right, #00ffff44, #00ffff)`,
              transition: 'width 0.1s linear',
            }} />
          </div>
          <span style={{ fontSize: 10, color: '#555', whiteSpace: 'nowrap' }}>
            {visibleCount}/{allEvents.length}
          </span>
        </div>

        <button
          onClick={handleReset}
          style={{
            background: 'transparent',
            border: '1px solid #1a1a2e',
            color: '#555',
            fontSize: 10,
            fontFamily: 'inherit',
            padding: '4px 10px',
            cursor: 'pointer',
            borderRadius: 2,
          }}
        >
          ↺ RESET
        </button>
      </div>

      {/* Table area */}
      <div style={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
        <div ref={tableRef} style={{ width: '100%', height: '100%', overflowY: 'auto', overflowX: 'auto' }}>
          <table style={{
            width: '100%',
            borderCollapse: 'collapse',
            fontSize: 11,
          }}>
            <thead style={{
              position: 'sticky',
              top: 0,
              background: '#0d0d1a',
              zIndex: 10,
            }}>
              <tr>
                {['#', 'TIMESTAMP', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'β', 'γ', 'ds²', 'REGIME'].map(col => (
                  <th
                    key={col}
                    style={{
                      padding: '8px 12px',
                      textAlign: 'left',
                      color: '#555',
                      fontWeight: 400,
                      fontSize: 10,
                      letterSpacing: '0.1em',
                      borderBottom: '1px solid #1a1a2e',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {visibleEvents.map((evt, i) => (
                <tr
                  key={evt.bar.index}
                  onClick={() => setSelectedEvent(prev => prev?.bar.index === evt.bar.index ? null : evt)}
                  style={{
                    background: selectedEvent?.bar.index === evt.bar.index
                      ? `${regimeColor(evt.regime)}22`
                      : i % 2 === 0 ? regimeBg(evt.regime) : 'transparent',
                    cursor: 'pointer',
                    transition: 'background 0.1s',
                    borderBottom: '1px solid #0d0d1a',
                  }}
                >
                  <td style={{ padding: '5px 12px', color: '#555' }}>{evt.bar.index}</td>
                  <td style={{ padding: '5px 12px', color: '#444', whiteSpace: 'nowrap' }}>
                    {new Date(evt.bar.timestamp).toISOString().slice(11, 19)}
                  </td>
                  <td style={{ padding: '5px 12px', color: '#aaa' }}>{evt.bar.open.toFixed(2)}</td>
                  <td style={{ padding: '5px 12px', color: '#00ff41' }}>{evt.bar.high.toFixed(2)}</td>
                  <td style={{ padding: '5px 12px', color: '#ff3333' }}>{evt.bar.low.toFixed(2)}</td>
                  <td style={{ padding: '5px 12px', color: '#e0e0e0', fontWeight: 600 }}>{evt.bar.close.toFixed(2)}</td>
                  <td style={{ padding: '5px 12px', color: '#666' }}>{evt.bar.volume.toLocaleString()}</td>
                  <td style={{ padding: '5px 12px', color: '#00ffff' }}>{evt.beta.toFixed(4)}</td>
                  <td style={{ padding: '5px 12px', color: '#00ff41' }}>{evt.gamma.toFixed(4)}</td>
                  <td style={{
                    padding: '5px 12px',
                    color: regimeColor(evt.regime),
                    fontWeight: 600,
                  }}>
                    {evt.ds2.toFixed(4)}
                  </td>
                  <td style={{ padding: '5px 12px' }}>
                    <span style={{
                      background: regimeBg(evt.regime),
                      color: regimeColor(evt.regime),
                      padding: '2px 8px',
                      borderRadius: 2,
                      fontSize: 10,
                      letterSpacing: '0.08em',
                      fontWeight: 700,
                      border: `1px solid ${regimeColor(evt.regime)}33`,
                    }}>
                      {evt.regime}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Selected event detail panel */}
        {selectedEvent && (
          <EventDetail event={selectedEvent} onClose={() => setSelectedEvent(null)} />
        )}
      </div>
    </div>
  );
}
