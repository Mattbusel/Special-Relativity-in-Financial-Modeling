# DASHBOARD.md — SRFM Interactive Visualization

## What the Dashboard Shows

The dashboard is a single-page React application that makes every core
transform from the C++ library visually interactive in real time.

### Panels

| Panel | What it visualizes |
|-------|--------------------|
| **β · market velocity** | Drag the slider to set β ∈ [0, 0.999]. All other panels update instantly. |
| **γ · Lorentz factor** | γ = 1/√(1−β²). Color-coded: green (Newtonian), yellow (relativistic), red (ultra-relativistic). |
| **φ · rapidity** | φ = atanh(β). Displayed to 4 decimal places. Additivity property labeled: φ(β₁⊕β₂) = φ₁ + φ₂. |
| **ds² · spacetime regime** | Classifies the live price bar as TIMELIKE (ds² < 0, causal), SPACELIKE (ds² > 0, stochastic), or LIGHTLIKE (ds² ≈ 0, critical transition). |
| **Live OHLCV stream** | Animated synthetic price series (blue) with extracted β(t) overlay (green). A scan line sweeps left-to-right at ~24 fps. The live β and γ values update on each frame. |
| **γ(β) curve** | Plots the full Lorentz factor curve from β = 0 to β → 1. A dashed cursor tracks the current slider position. |
| **Spacetime interval formula** | ds² = −c²Δt² + ΔP² + ΔV² + ΔM² with labeled regime tags. |
| **Velocity composition** | Shows β₁ ⊕ β₂ = (β₁+β₂)/(1+β₁β₂) for three example pairs, confirming the result is always sub-luminal. |
| **6-agent build map** | Per-module test ratios for all six agents (AGT-01 through AGT-06). |

---

## Running the Dashboard

### Prerequisites

- Node.js 18 or later
- npm 9 or later

### Dev server (hot reload)

```bash
cd viz
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

### Production build (static files)

```bash
cd viz
npm run build
```

Output is written to `viz/dist/`. The contents can be hosted on any static
server (GitHub Pages, Netlify, S3, nginx, etc.).

```bash
# Preview the production build locally
npm run preview
```

---

## Screenshot Instructions

For the best visual state before taking a screenshot:

1. Set β slider to **0.866** (γ = 2.000 — a clean round number).
2. Wait for the scan line to reach roughly the **center** of the price chart
   so the animated dot is visible.
3. The ds² regime tile will cycle through all three states as the scan line
   moves — capture it when **TIMELIKE** is shown (green badge) for maximum
   contrast against the dark background.
4. Recommended resolution: **1440 × 900** or wider. The grid is 4-column at
   full width.
5. Browser: Chrome or Firefox. The JetBrains Mono and Syne fonts are loaded
   from Google Fonts — ensure network access for the first load.

---

## Technology

| Layer | Library | Version |
|-------|---------|---------|
| UI framework | React | 18 |
| Build tool | Vite | 5 |
| Fonts | JetBrains Mono, Syne | Google Fonts |
| Math | Inline JS (mirrors C++ impl) | — |

No external charting library is used. All SVG paths are computed inline from
the same formulas as the C++ backend (γ, ds², β-composition).
