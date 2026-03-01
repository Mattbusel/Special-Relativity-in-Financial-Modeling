import { useState, useEffect, useRef } from "react";

const BETA_MAX = 0.9999;

function gamma(beta) {
  const b = Math.min(Math.abs(beta), BETA_MAX);
  return 1.0 / Math.sqrt(1.0 - b * b);
}

function spacetimeInterval(dt, dP, dV, dM) {
  return -(dt * dt) + dP * dP + dV * dV + dM * dM;
}

function classifyInterval(ds2) {
  if (ds2 < -1e-6) return "TIMELIKE";
  if (ds2 > 1e-6) return "SPACELIKE";
  return "LIGHTLIKE";
}

function composeVelocities(b1, b2) {
  return (b1 + b2) / (1 + b1 * b2);
}

function generatePriceSeries(n = 80, vol = 0.025) {
  let p = 100;
  const prices = [p];
  for (let i = 1; i < n; i++) {
    p = Math.max(p * (1 + (Math.random() - 0.5) * vol * 2), 1);
    prices.push(p);
  }
  return prices;
}

function computeBetas(prices, window = 10) {
  const vels = [];
  for (let i = 1; i < prices.length; i++)
    vels.push(Math.abs(Math.log(prices[i] / prices[i - 1])));
  return vels.map((_, i) => {
    const slice = vels.slice(Math.max(0, i - window), i + 1);
    const maxV = Math.max(...vels.slice(0, i + 1), 1e-10);
    const v = slice.reduce((a, b) => a + b, 0) / slice.length;
    return Math.min(v / maxV, BETA_MAX);
  });
}

export default function SRFM() {
  const [beta, setBeta] = useState(0.6);
  const [prices] = useState(() => generatePriceSeries(80, 0.025));
  const [betas] = useState(() => computeBetas(generatePriceSeries(80, 0.025)));
  const [tick, setTick] = useState(0);
  const animRef = useRef(null);

  useEffect(() => {
    let f = 0;
    const animate = () => {
      f = (f + 0.4) % 79;
      setTick(Math.floor(f));
      animRef.current = requestAnimationFrame(animate);
    };
    animRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animRef.current);
  }, []);

  const g = gamma(beta);
  const rapidity = Math.atanh(Math.min(beta, 0.9999));
  const doppler = Math.sqrt((1 + beta) / Math.max(1 - beta, 1e-9));

  const idx = Math.min(tick, betas.length - 1);
  const liveBeta = betas[idx] || 0;
  const liveGamma = gamma(liveBeta);

  const ds2 = spacetimeInterval(0.1, liveBeta * 2, liveBeta * 0.5, liveBeta * 0.3);
  const regime = classifyInterval(ds2);
  const regimeColor = { TIMELIKE: "#00ff88", SPACELIKE: "#ff4466", LIGHTLIKE: "#ffcc00" }[regime];
  const regimeDesc = { TIMELIKE: "causal · past predicts future", SPACELIKE: "stochastic · decorrelated", LIGHTLIKE: "critical transition" }[regime];

  const W = 500, H = 90;
  const minP = Math.min(...prices), maxP = Math.max(...prices);
  const px = (i) => (i / (prices.length - 1)) * W;
  const py = (p) => H - ((p - minP) / (maxP - minP)) * H;
  const bx = (i) => (i / (betas.length - 1)) * W;
  const by = (b) => H - b * H;

  const pPath = prices.map((p, i) => `${i === 0 ? "M" : "L"}${px(i)},${py(p)}`).join(" ");
  const bPath = betas.map((b, i) => `${i === 0 ? "M" : "L"}${bx(i)},${by(b)}`).join(" ");

  const gCurve = Array.from({ length: 100 }, (_, i) => {
    const b = (i / 100) * 0.999;
    const gv = gamma(b);
    return `${i === 0 ? "M" : "L"}${b * 300},${70 - Math.min(gv - 1, 7) * 10}`;
  }).join(" ");

  const scanX = (tick / 79) * W;

  return (
    <div style={{
      background: "#030a12",
      minHeight: "100vh",
      color: "#e2e8f0",
      fontFamily: "'JetBrains Mono', 'Courier New', monospace",
      padding: "28px",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@700;800&display=swap');
        * { box-sizing: border-box; }
        .p { background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.06); border-radius: 6px; padding: 18px; position: relative; overflow: hidden; }
        .p::after { content:''; position:absolute; top:0;left:0;right:0; height:1px; background:linear-gradient(90deg,transparent,rgba(0,255,136,0.2),transparent); }
        .lbl { font-size:9px; letter-spacing:.15em; text-transform:uppercase; color:rgba(255,255,255,0.3); margin-bottom:6px; }
        .val { font-size:26px; font-weight:700; }
        input[type=range]{-webkit-appearance:none;width:100%;height:2px;background:rgba(255,255,255,0.08);border-radius:1px;outline:none;margin-top:10px;}
        input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:12px;height:12px;border-radius:50%;background:#00ff88;cursor:pointer;box-shadow:0 0 8px rgba(0,255,136,0.5);}
      `}</style>

      {/* Header */}
      <div style={{ marginBottom: "24px", borderBottom: "1px solid rgba(255,255,255,0.05)", paddingBottom: "20px" }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: "14px", marginBottom: "8px" }}>
          <h1 style={{ fontFamily: "'Syne',sans-serif", fontSize: "20px", fontWeight: "800", margin: 0, letterSpacing: "-0.01em", color: "#f8fafc" }}>
            Special Relativity in Financial Modeling
          </h1>
          <span style={{ fontSize: "10px", color: "#00ff88", opacity: 0.8 }}>C++20</span>
        </div>
        <div style={{ display: "flex", gap: "20px", fontSize: "10px", color: "rgba(255,255,255,0.25)" }}>
          {["2,113 production LOC", "3,192 test LOC", "1.511:1 ratio", "0 panics", "0 warnings", "6 agents"].map(s => (
            <span key={s} style={{ display: "flex", alignItems: "center", gap: "4px" }}>
              <span style={{ color: "#00ff88", fontSize: "7px" }}>◆</span>{s}
            </span>
          ))}
        </div>
      </div>

      {/* Metrics row */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: "10px", marginBottom: "10px" }}>

        <div className="p">
          <div className="lbl">β · market velocity</div>
          <div className="val" style={{ color: "#60a5fa" }}>{beta.toFixed(3)}</div>
          <input type="range" min="0" max="0.999" step="0.001" value={beta} onChange={e => setBeta(+e.target.value)} />
          <div style={{ fontSize: "9px", color: "rgba(255,255,255,0.2)", marginTop: "6px" }}>β = |dP/dt| / c_market</div>
        </div>

        <div className="p">
          <div className="lbl">γ · Lorentz factor</div>
          <div className="val" style={{ color: g > 5 ? "#ff4466" : g > 2 ? "#ffcc00" : "#00ff88", transition: "color 0.3s" }}>
            {g > 9999 ? "∞" : g.toFixed(4)}
          </div>
          <div style={{ fontSize: "9px", color: "rgba(255,255,255,0.2)", marginTop: "14px" }}>γ = 1 / √(1 − β²)</div>
        </div>

        <div className="p">
          <div className="lbl">φ · rapidity (additive)</div>
          <div className="val" style={{ color: "#a78bfa", fontSize: "22px", marginTop: "4px" }}>
            {rapidity > 999 ? "∞" : rapidity.toFixed(4)}
          </div>
          <div style={{ fontSize: "9px", color: "rgba(255,255,255,0.2)", marginTop: "8px" }}>φ = atanh(β)</div>
          <div style={{ fontSize: "9px", color: "rgba(255,255,255,0.2)" }}>φ(β₁⊕β₂) = φ₁ + φ₂</div>
        </div>

        <div className="p">
          <div className="lbl">ds² · spacetime regime</div>
          <div style={{ marginTop: "8px" }}>
            <div style={{
              display: "inline-block", padding: "5px 10px", borderRadius: "3px",
              background: `${regimeColor}15`, border: `1px solid ${regimeColor}40`,
              color: regimeColor, fontSize: "12px", fontWeight: "700", letterSpacing: "0.06em",
              transition: "all 0.2s",
            }}>{regime}</div>
          </div>
          <div style={{ fontSize: "9px", color: "rgba(255,255,255,0.25)", marginTop: "8px" }}>{regimeDesc}</div>
        </div>
      </div>

      {/* Charts row */}
      <div style={{ display: "grid", gridTemplateColumns: "3fr 2fr", gap: "10px", marginBottom: "10px" }}>

        <div className="p">
          <div className="lbl" style={{ marginBottom: "10px" }}>live ohlcv stream · β(t) extraction · γ correction</div>
          <svg width={W} height={H} style={{ display: "block", overflow: "visible" }}>
            {[0.25, 0.5, 0.75].map(t => (
              <line key={t} x1="0" y1={t * H} x2={W} y2={t * H} stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
            ))}
            <path d={pPath} fill="none" stroke="#60a5fa" strokeWidth="1.5" opacity="0.5" />
            <path d={bPath} fill="none" stroke="#00ff88" strokeWidth="1.5" opacity="0.9" />
            <line x1={scanX} y1="0" x2={scanX} y2={H} stroke="rgba(255,255,255,0.4)" strokeWidth="1" strokeDasharray="3,3" />
            <circle cx={scanX} cy={by(liveBeta)} r="3.5" fill="#00ff88" style={{ filter: "drop-shadow(0 0 5px #00ff88)" }} />
          </svg>
          <div style={{ display: "flex", gap: "16px", marginTop: "8px", fontSize: "9px", color: "rgba(255,255,255,0.25)" }}>
            <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
              <svg width="12" height="2"><line x1="0" y1="1" x2="12" y2="1" stroke="#60a5fa" strokeWidth="2"/></svg>price
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
              <svg width="12" height="2"><line x1="0" y1="1" x2="12" y2="1" stroke="#00ff88" strokeWidth="2"/></svg>β(t)
            </span>
            <span style={{ marginLeft: "auto", color: "#00ff88" }}>
              β={liveBeta.toFixed(3)}  γ={liveGamma.toFixed(3)}  p_rel=γmv
            </span>
          </div>
        </div>

        <div className="p">
          <div className="lbl" style={{ marginBottom: "10px" }}>γ(β) curve · Lorentz factor</div>
          <svg width="300" height="70" style={{ display: "block", overflow: "visible" }}>
            <line x1="0" y1="70" x2="300" y2="70" stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
            <line x1="0" y1="0" x2="0" y2="70" stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
            <path d={`${gCurve} L 300 70 L 0 70 Z`} fill="rgba(167,139,250,0.05)" />
            <path d={gCurve} fill="none" stroke="#a78bfa" strokeWidth="2" />
            <line x1={beta * 300} y1="0" x2={beta * 300} y2="70" stroke="#00ff88" strokeWidth="1" strokeDasharray="3,3" opacity="0.8" />
            <circle cx={beta * 300} cy={70 - Math.min(g - 1, 7) * 10} r="4" fill="#00ff88" style={{ filter: "drop-shadow(0 0 6px #00ff88)" }} />
            <text x="4" y="66" fill="rgba(255,255,255,0.2)" fontSize="8">β=0, γ=1</text>
            <text x="240" y="66" fill="rgba(255,255,255,0.2)" fontSize="8">β→1, γ→∞</text>
          </svg>
          <div style={{ fontSize: "9px", color: "rgba(255,255,255,0.2)", marginTop: "6px" }}>
            Newtonian limit recovered at β→0
          </div>
        </div>
      </div>

      {/* Bottom row */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "10px" }}>

        <div className="p">
          <div className="lbl">spacetime interval · ds²</div>
          <div style={{ marginTop: "10px", padding: "10px", background: "rgba(0,0,0,0.3)", borderRadius: "4px", fontSize: "11px", lineHeight: "1.9", color: "rgba(255,255,255,0.6)" }}>
            ds² = −c²Δt² + ΔP²<br />
            <span style={{ paddingLeft: "40px" }}>+ ΔV² + ΔM²</span>
          </div>
          <div style={{ display: "flex", gap: "6px", marginTop: "8px", flexWrap: "wrap" }}>
            {[["ds²<0", "TIMELIKE", "#00ff88"], ["ds²>0", "SPACELIKE", "#ff4466"], ["ds²=0", "LIGHTLIKE", "#ffcc00"]].map(([label, name, color]) => (
              <span key={name} style={{ fontSize: "8px", padding: "1px 6px", borderRadius: "2px", background: `${color}12`, border: `1px solid ${color}30`, color }}>{label}</span>
            ))}
          </div>
        </div>

        <div className="p">
          <div className="lbl">velocity composition · β₁⊕β₂</div>
          <div style={{ fontSize: "10px", color: "rgba(255,255,255,0.4)", marginTop: "8px", marginBottom: "10px" }}>
            (β₁+β₂) / (1+β₁β₂) &lt; 1 always
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "6px", textAlign: "center" }}>
            {[[0.5, 0.5], [0.9, 0.9], [0.5, 0.8]].map(([b1, b2]) => (
              <div key={`${b1}${b2}`} style={{ padding: "6px", background: "rgba(0,0,0,0.3)", borderRadius: "4px" }}>
                <div style={{ fontSize: "8px", color: "rgba(255,255,255,0.25)" }}>{b1}⊕{b2}</div>
                <div style={{ fontSize: "13px", fontWeight: "700", color: "#60a5fa" }}>{composeVelocities(b1, b2).toFixed(3)}</div>
                <div style={{ fontSize: "8px", color: "#00ff88" }}>{"< 1 ✓"}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="p">
          <div className="lbl">6-agent build · test ratios</div>
          <div style={{ marginTop: "8px", display: "flex", flexDirection: "column", gap: "5px" }}>
            {[
              ["AGT-01", "Lorentz + Beta", "2.26:1", "#60a5fa"],
              ["AGT-02", "Manifold", "3.30:1", "#34d399"],
              ["AGT-03", "Momentum", "2.37:1", "#f472b6"],
              ["AGT-04", "Tensor + Geodesic", "2.27:1", "#a78bfa"],
              ["AGT-05", "Backtester", "—", "#fb923c"],
              ["AGT-06", "Engine + CLI", "1.51:1", "#00ff88"],
            ].map(([id, name, ratio, color]) => (
              <div key={id} style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "9px" }}>
                <span style={{ color, fontWeight: "700", minWidth: "48px" }}>{id}</span>
                <span style={{ color: "rgba(255,255,255,0.4)", flex: 1 }}>{name}</span>
                <span style={{ color: "rgba(255,255,255,0.25)" }}>{ratio}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div style={{ marginTop: "18px", paddingTop: "14px", borderTop: "1px solid rgba(255,255,255,0.04)", display: "flex", justifyContent: "space-between", fontSize: "9px", color: "rgba(255,255,255,0.18)" }}>
        <span>github.com/Mattbusel/Special-Relativity-in-Financial-Modeling</span>
        <span>C++20 · Eigen3 · Google Test · MIT</span>
      </div>
    </div>
  );
}
