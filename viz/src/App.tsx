import React, { useState, useCallback } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { gamma, rapidity } from './utils/physics';
import LightCone3D from './components/LightCone3D';
import LorentzDashboard from './components/LorentzDashboard';
import RegimeHeatmap from './components/RegimeHeatmap';
import GeodesicViz from './components/GeodesicViz';
import BacktestPanel from './components/BacktestPanel';

type TabId = 'lightcone' | 'lorentz' | 'heatmap' | 'geodesic' | 'backtest';

interface Tab {
  id: TabId;
  label: string;
}

const TABS: Tab[] = [
  { id: 'lightcone', label: 'Light Cone 3D' },
  { id: 'lorentz', label: 'Lorentz Dashboard' },
  { id: 'heatmap', label: 'Regime Heatmap' },
  { id: 'geodesic', label: 'Geodesic' },
  { id: 'backtest', label: 'Backtest' },
];

const styles: Record<string, React.CSSProperties> = {
  root: {
    display: 'flex',
    flexDirection: 'column',
    width: '100vw',
    height: '100vh',
    background: '#0a0a0a',
    color: '#e0e0e0',
    fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
    overflow: 'hidden',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '10px 20px',
    borderBottom: '1px solid #1a1a2e',
    background: '#050508',
    flexShrink: 0,
  },
  headerTitle: {
    fontSize: '13px',
    fontWeight: 700,
    letterSpacing: '0.15em',
    color: '#00ffff',
    textShadow: '0 0 12px #00ffff88',
  },
  headerRight: {
    display: 'flex',
    alignItems: 'center',
    gap: '20px',
  },
  githubLink: {
    color: '#00ff41',
    textDecoration: 'none',
    fontSize: '11px',
    letterSpacing: '0.1em',
    opacity: 0.8,
    transition: 'opacity 0.2s',
  },
  sliderBar: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    padding: '8px 20px',
    background: '#0d0d1a',
    borderBottom: '1px solid #1a1a2e',
    flexShrink: 0,
  },
  sliderLabel: {
    fontSize: '11px',
    color: '#888',
    letterSpacing: '0.05em',
    whiteSpace: 'nowrap',
  },
  sliderValue: {
    fontSize: '12px',
    color: '#00ffff',
    fontWeight: 700,
    minWidth: '60px',
  },
  gammaValue: {
    fontSize: '12px',
    color: '#00ff41',
    fontWeight: 700,
    minWidth: '80px',
  },
  tabBar: {
    display: 'flex',
    alignItems: 'center',
    padding: '0 20px',
    background: '#050508',
    borderBottom: '1px solid #1a1a2e',
    flexShrink: 0,
    gap: '2px',
  },
  tabContent: {
    flex: 1,
    overflow: 'hidden',
    position: 'relative',
  },
};

const sliderTrackStyle = `
  .beta-slider {
    -webkit-appearance: none;
    appearance: none;
    width: 240px;
    height: 4px;
    border-radius: 2px;
    background: linear-gradient(to right, #1a1a2e, #00ffff);
    outline: none;
    cursor: pointer;
  }
  .beta-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #00ffff;
    box-shadow: 0 0 8px #00ffff;
    cursor: pointer;
  }
  .beta-slider::-moz-range-thumb {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #00ffff;
    box-shadow: 0 0 8px #00ffff;
    cursor: pointer;
    border: none;
  }
`;

export default function App() {
  const [beta, setBeta] = useState(0.72);
  const [activeTab, setActiveTab] = useState<TabId>('lightcone');

  const g = gamma(beta);
  const phi = rapidity(beta);

  const handleBetaChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setBeta(parseFloat(e.target.value));
  }, []);

  const renderTab = () => {
    switch (activeTab) {
      case 'lightcone':
        return <LightCone3D beta={beta} />;
      case 'lorentz':
        return <LorentzDashboard beta={beta} />;
      case 'heatmap':
        return <RegimeHeatmap beta={beta} />;
      case 'geodesic':
        return <GeodesicViz beta={beta} />;
      case 'backtest':
        return <BacktestPanel beta={beta} />;
    }
  };

  return (
    <div style={styles.root}>
      <style>{sliderTrackStyle}</style>

      {/* Header */}
      <div style={styles.header}>
        <div style={styles.headerTitle}>
          TOKIO PROMPT // RELATIVISTIC MARKET INTELLIGENCE
        </div>
        <div style={styles.headerRight}>
          <a
            href="https://github.com/tokio-prompt"
            target="_blank"
            rel="noopener noreferrer"
            style={styles.githubLink}
          >
            [GITHUB]
          </a>
          <span style={{ fontSize: '11px', color: '#444' }}>v1.0.0</span>
        </div>
      </div>

      {/* β slider bar */}
      <div style={styles.sliderBar}>
        <span style={styles.sliderLabel}>β (velocity/c):</span>
        <input
          type="range"
          className="beta-slider"
          min={0}
          max={0.9999}
          step={0.0001}
          value={beta}
          onChange={handleBetaChange}
        />
        <span style={styles.sliderValue}>β = {beta.toFixed(4)}</span>
        <span style={{ fontSize: '11px', color: '#555', margin: '0 4px' }}>|</span>
        <span style={styles.sliderLabel}>γ (Lorentz factor):</span>
        <span style={styles.gammaValue}>γ = {g.toFixed(4)}</span>
        <span style={{ fontSize: '11px', color: '#555', margin: '0 4px' }}>|</span>
        <span style={styles.sliderLabel}>φ (rapidity):</span>
        <span style={{ fontSize: '12px', color: '#ff9900', fontWeight: 700, minWidth: '80px' }}>
          φ = {phi.toFixed(4)}
        </span>
        <span style={{ fontSize: '11px', color: '#555', margin: '0 4px' }}>|</span>
        <span style={styles.sliderLabel}>1/γ:</span>
        <span style={{ fontSize: '12px', color: '#cc88ff', fontWeight: 700, minWidth: '70px' }}>
          {(1 / g).toFixed(4)}
        </span>
      </div>

      {/* Tab bar */}
      <div style={styles.tabBar}>
        {TABS.map(tab => (
          <TabButton
            key={tab.id}
            tab={tab}
            active={activeTab === tab.id}
            onClick={() => setActiveTab(tab.id)}
          />
        ))}
      </div>

      {/* Tab content */}
      <div style={styles.tabContent}>
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.2 }}
            style={{ width: '100%', height: '100%' }}
          >
            {renderTab()}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}

interface TabButtonProps {
  tab: Tab;
  active: boolean;
  onClick: () => void;
}

function TabButton({ tab, active, onClick }: TabButtonProps) {
  return (
    <button
      onClick={onClick}
      style={{
        background: active ? '#00ffff18' : 'transparent',
        border: 'none',
        borderBottom: active ? '2px solid #00ffff' : '2px solid transparent',
        color: active ? '#00ffff' : '#666',
        padding: '10px 16px',
        fontSize: '11px',
        fontFamily: 'inherit',
        letterSpacing: '0.1em',
        cursor: 'pointer',
        transition: 'all 0.2s',
        fontWeight: active ? 700 : 400,
        textShadow: active ? '0 0 8px #00ffff66' : 'none',
        whiteSpace: 'nowrap',
      }}
    >
      {tab.label.toUpperCase()}
    </button>
  );
}
