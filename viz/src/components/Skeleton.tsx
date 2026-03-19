import React from 'react';

const shimmerStyle = `
  @keyframes srfm-shimmer {
    0%   { background-position: -600px 0; }
    100% { background-position:  600px 0; }
  }
  .srfm-skeleton {
    background: linear-gradient(
      90deg,
      #0d0d1a 25%,
      #1a1a2e 50%,
      #0d0d1a 75%
    );
    background-size: 600px 100%;
    animation: srfm-shimmer 1.6s infinite linear;
    border-radius: 4px;
  }
`;

interface SkeletonBlockProps {
  width?: string | number;
  height?: string | number;
  style?: React.CSSProperties;
}

function SkeletonBlock({ width = '100%', height = 16, style }: SkeletonBlockProps) {
  return (
    <div
      className="srfm-skeleton"
      role="progressbar"
      aria-busy="true"
      aria-label="Loading"
      style={{ width, height, ...style }}
    />
  );
}

/** Full-panel skeleton for chart tabs */
export function SkeletonChart() {
  return (
    <>
      <style>{shimmerStyle}</style>
      <div style={{
        width: '100%',
        height: '100%',
        background: '#0a0a0a',
        padding: 16,
        display: 'flex',
        flexDirection: 'column',
        gap: 12,
      }}>
        {/* Header row */}
        <SkeletonBlock height={36} width="60%" />
        {/* Stat cards */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 10 }}>
          {Array.from({ length: 4 }, (_, i) => (
            <SkeletonBlock key={i} height={90} />
          ))}
        </div>
        {/* Main chart area */}
        <SkeletonBlock height={240} />
        {/* Secondary chart */}
        <SkeletonBlock height={180} />
      </div>
    </>
  );
}

/** Compact skeleton for smaller panels */
export function SkeletonPanel() {
  return (
    <>
      <style>{shimmerStyle}</style>
      <div style={{
        width: '100%',
        height: '100%',
        background: '#0a0a0a',
        padding: 16,
        display: 'flex',
        flexDirection: 'column',
        gap: 10,
      }}>
        <SkeletonBlock height={24} width="40%" />
        <SkeletonBlock height={160} />
        <SkeletonBlock height={12} width="70%" />
        <SkeletonBlock height={12} width="55%" />
      </div>
    </>
  );
}

export default SkeletonBlock;
