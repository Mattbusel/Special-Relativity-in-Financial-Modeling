"""Build the README / site figures from real program output.

Inputs (produced by the C++ build, see README "Reproduce the figures"):
  <results>/<TICKER>_regime.csv   written by build/regime_validator
  <q1>/Q1_RESULTS_RAW.csv          written by validation/analyze_q1.py
  validation/data/<TICKER>_1m.csv  committed daily OHLCV bars

Outputs: standalone HTML pages (inline SVG, no external assets) that are
rendered to PNG with a headless browser at 2x, one light and one dark.

    python scripts/figures/make_figures.py --results out --q1 q1 --out figs
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

THEMES = {
    "light": dict(bg="#F5F2EA", card="#FBF9F4", ink="#1A1D24", ink2="#4A4B50",
                  muted="#6F6C64", grid="#DDD7C9", rule="#C9C2B1", time="#00809A",
                  space="#D0402A", well="#C9A200", cone_a=0.10, term="#16191F"),
    "dark": dict(bg="#0E1116", card="#141820", ink="#ECE7DC", ink2="#B9B4A9",
                 muted="#8F8B82", grid="#232831", rule="#343A45", time="#35A9C6",
                 space="#EE5E3F", well="#E0B43A", cone_a=0.13, term="#0A0C10"),
}
SERIF = "'Iowan Old Style','Palatino Linotype',Palatino,'Book Antiqua',Georgia,serif"
MONO = "ui-monospace,SFMono-Regular,'Cascadia Mono',Consolas,'DejaVu Sans Mono',Menlo,monospace"

TICKERS = ["AAPL", "BTC_USD", "GLD", "GS", "JPM", "META", "NVDA", "QQQ", "SPY", "TSLA"]


def read_csv(p: Path) -> list[dict]:
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def fmt_int(n: int) -> str:
    return f"{n:,}"


# ── Hero: SPY worldline coloured by the validator's interval labels ──────────

def worldline_svg(t: dict, closes, dates, labels, w, h, pad=(34, 18, 30, 54), cones=True):
    """labels[i] is the class of the interval (bar i-1 -> bar i)."""
    l, tp, r, b = pad
    n = len(closes)
    lo, hi = min(closes), max(closes)
    span = hi - lo
    lo -= span * 0.08
    hi += span * 0.08
    X = lambda i: l + (w - l - r) * i / (n - 1)
    Y = lambda v: tp + (h - tp - b) * (1 - (v - lo) / (hi - lo))
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">']
    # grid
    step = 10 if span < 120 else 20
    g0 = math.ceil(lo / step) * step
    v = g0
    while v < hi:
        y = Y(v)
        out.append(f'<line x1="{l}" x2="{w - r}" y1="{y:.1f}" y2="{y:.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
        out.append(f'<text x="{w - r + 6}" y="{y + 4:.1f}" font-family="{MONO}" font-size="11" fill="{t["muted"]}">{v:.0f}</text>')
        v += step
    # light cones on timelike bars that follow a run (sparse, for legibility)
    if cones:
        cone_len = (w - l - r) * 7 / (n - 1)
        # a 45-degree cone in chart units: unit time = one bar, unit price chosen so c = 1
        last = -99
        for i in range(1, n):
            if labels[i] == "Timelike" and i - last > 14 and i < n - 4:
                x0, y0 = X(i), Y(closes[i])
                out.append(
                    f'<path d="M{x0:.1f},{y0:.1f} L{x0 + cone_len:.1f},{y0 - cone_len * .62:.1f} '
                    f'L{x0 + cone_len:.1f},{y0 + cone_len * .62:.1f} Z" fill="{t["time"]}" '
                    f'fill-opacity="{t["cone_a"]}" stroke="{t["time"]}" stroke-opacity=".35" stroke-width="1"/>')
                last = i
    # worldline segments
    for i in range(1, n):
        x1, y1, x2, y2 = X(i - 1), Y(closes[i - 1]), X(i), Y(closes[i])
        if labels[i] == "Timelike":
            out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{t["time"]}" stroke-width="2.6" stroke-linecap="round"/>')
        else:
            out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{t["space"]}" stroke-width="1.7" stroke-dasharray="3 3"/>')
    for i in range(n):
        c = t["time"] if labels[i] == "Timelike" else t["space"]
        out.append(f'<circle cx="{X(i):.1f}" cy="{Y(closes[i]):.1f}" r="2" fill="{t["card"]}" stroke="{c}" stroke-width="1.2"/>')
    # regime strip
    sy = h - b + 12
    bw = (w - l - r) / (n - 1)
    for i in range(1, n):
        c = t["time"] if labels[i] == "Timelike" else t["space"]
        op = "1" if labels[i] == "Timelike" else ".35"
        out.append(f'<rect x="{X(i) - bw / 2:.1f}" y="{sy}" width="{bw * .8:.1f}" height="10" fill="{c}" fill-opacity="{op}"/>')
    # date ticks
    for i in [0, n // 3, 2 * n // 3, n - 1]:
        anchor = "start" if i == 0 else "end" if i == n - 1 else "middle"
        out.append(f'<text x="{X(i):.1f}" y="{h - 8}" text-anchor="{anchor}" font-family="{MONO}" font-size="11" fill="{t["muted"]}">{dates[i]}</text>')
    out.append("</svg>")
    return "\n".join(out)


def hero_html(theme: str, results: Path, q1: Path, social: bool = False) -> str:
    t = THEMES[theme]
    px = read_csv(ROOT / "validation/data/SPY_1m.csv")
    rg = {int(r["bar_index"]): r["interval_type"] for r in read_csv(results / "SPY_regime.csv")}
    N = 110
    idx = list(range(len(px) - N - 1, len(px) - 1))
    closes = [float(px[i]["close"]) for i in idx]
    dates = [px[i]["timestamp"][:10] for i in idx]
    labels = [rg.get(i, "Spacelike") for i in idx]
    n_tl_win = sum(1 for x in labels[1:] if x == "Timelike")
    svg = worldline_svg(t, closes, dates, labels, 700, 380)

    allrg = read_csv(results / "SPY_regime.csv")
    tl = sum(1 for r in allrg if r["interval_type"] == "Timelike")
    sl = len(allrg) - tl
    pooled = next(r for r in read_csv(q1 / "Q1_RESULTS_RAW.csv") if r["group"] == "POOLED")
    ratio = float(pooled["variance_ratio"])
    lev = float(pooled["levene_p"])

    extra = ""
    if social:  # 1280x640 card for the GitHub social preview
        extra = (f"body{{height:640px!important}} .card{{height:520px!important;top:48px!important}}"
                 f".stats{{bottom:84px!important}} .legend{{bottom:92px!important}}"
                 f".url{{position:absolute;left:48px;bottom:30px;font-family:{MONO};font-size:14px;color:{t['muted']}}}")
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0}}
body{{width:1280px;height:560px;background:{t['bg']};color:{t['ink']};font-family:{SERIF};overflow:hidden}}
.wrap{{position:relative;height:100%;padding:44px 48px}}
.grid{{position:absolute;inset:0;background-image:linear-gradient({t['grid']}55 1px,transparent 1px),linear-gradient(90deg,{t['grid']}55 1px,transparent 1px);background-size:40px 40px}}
.eyebrow{{font-family:{MONO};font-size:12.5px;letter-spacing:.14em;text-transform:uppercase;color:{t['muted']}}}
h1{{font-weight:600;font-size:46px;line-height:1.04;letter-spacing:-.015em;margin:14px 0 16px}}
p.lede{{font-size:18.5px;line-height:1.45;color:{t['ink2']};max-width:390px}}
.left{{position:absolute;left:48px;top:48px;width:430px}}
.card{{position:absolute;right:40px;top:40px;width:722px;height:480px;background:{t['card']};border:1px solid {t['rule']};border-radius:6px;padding:14px 10px 0 10px}}
.cap{{font-family:{MONO};font-size:11.5px;color:{t['muted']};display:flex;justify-content:space-between;padding:0 8px 4px 24px}}
.stats{{position:absolute;left:48px;bottom:44px;width:440px;font-family:{MONO};font-size:12.5px;color:{t['ink2']};border-top:1px solid {t['rule']};padding-top:14px;display:grid;grid-template-columns:auto 1fr;gap:6px 16px}}
.k{{color:{t['muted']}}}
.tl{{color:{t['time']};font-weight:600}} .sl{{color:{t['space']};font-weight:600}}
.legend{{position:absolute;right:56px;bottom:52px;font-family:{MONO};font-size:11.5px;color:{t['ink2']};display:flex;gap:18px}}
.sw{{display:inline-block;width:22px;height:0;vertical-align:middle;margin-right:6px}}
code{{font-family:{MONO};font-size:.86em}}
{extra}</style></head><body><div class="wrap"><div class="grid"></div>
<div class="left">
<div class="eyebrow">SRFM &middot; C++20 core</div>
<h1>Every bar is an event in spacetime.</h1>
<p class="lede">A C++20 library that gives each OHLCV bar a price velocity &beta;, a Lorentz factor &gamma; and a spacetime interval, then labels it timelike or spacelike.</p>
</div>
<div class="stats">
<span class="k">input</span><span>SPY daily bars, {fmt_int(len(allrg))} labelled</span>
<span class="k">labels</span><span><span class="tl">{fmt_int(tl)} timelike</span> &middot; <span class="sl">{fmt_int(sl)} spacelike</span></span>
<span class="k">10 tickers</span><span>variance SL / TL = {ratio:.2f}, Levene p = {lev:.3f}</span>
</div>
<div class="card">
<div class="cap"><span>SPY close, last {N} bars &middot; labels from build/regime_validator</span><span>{n_tl_win} TL / {N - 1 - n_tl_win} SL</span></div>
{svg}
</div>
<div class="legend"><span><span class="sw" style="border-top:3px solid {t['time']}"></span>timelike, ds&sup2; &lt; 0</span><span><span class="sw" style="border-top:2px dashed {t['space']}"></span>spacelike, ds&sup2; &gt; 0</span><span><span class="sw" style="height:10px;width:14px;background:{t['time']};opacity:.25;border:1px solid {t['time']}"></span>light cone</span></div>
{'<div class="url">github.com/Mattbusel/Special-Relativity-in-Financial-Modeling &middot; research code, not financial advice</div>' if social else ''}
</div></body></html>"""


# ── Ten-ticker regime split and variance ratio ────────────────────────────────

def regimes_html(theme: str, q1: Path) -> str:
    t = THEMES[theme]
    rows = {r["group"]: r for r in read_csv(q1 / "Q1_RESULTS_RAW.csv")}
    W, rowh, top, left = 1200, 40, 70, 120
    H = top + rowh * (len(TICKERS) + 1) + 70
    bar_w = 560
    rx0 = left + bar_w + 120
    rscale = 150  # px per 1.0 of variance ratio
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
         f'<rect width="{W}" height="{H}" fill="{t["bg"]}"/>',
         f'<text x="{left}" y="34" font-family="{MONO}" font-size="12" letter-spacing="1.5" fill="{t["muted"]}">SHARE OF BARS BY INTERVAL CLASS</text>',
         f'<text x="{rx0}" y="34" font-family="{MONO}" font-size="12" letter-spacing="1.5" fill="{t["muted"]}">NEXT-BAR VARIANCE, SPACELIKE / TIMELIKE</text>']
    x1 = rx0 + rscale * 1.0
    s.append(f'<line x1="{x1}" x2="{x1}" y1="{top - 14}" y2="{top + rowh * (len(TICKERS) + 1) - 8}" stroke="{t["rule"]}" stroke-dasharray="2 3"/>')
    s.append(f'<text x="{x1}" y="{top + rowh * (len(TICKERS) + 1) + 10}" text-anchor="middle" font-family="{MONO}" font-size="11" fill="{t["muted"]}">1.0 = no difference</text>')
    for k, name in enumerate(TICKERS + ["POOLED"]):
        r = rows[name]
        y = top + k * rowh
        ntl, nsl = int(r["n_timelike"]), int(r["n_spacelike"])
        f = ntl / (ntl + nsl)
        bold = name == "POOLED"
        if bold:
            s.append(f'<line x1="{left - 100}" x2="{W - 30}" y1="{y - 12}" y2="{y - 12}" stroke="{t["rule"]}"/>')
        s.append(f'<text x="{left - 16}" y="{y + 15}" text-anchor="end" font-family="{MONO}" font-size="13" font-weight="{700 if bold else 400}" fill="{t["ink"]}">{name.replace("_", "-") if name != "POOLED" else "all 10"}</text>')
        s.append(f'<rect x="{left}" y="{y}" width="{bar_w * f:.1f}" height="20" fill="{t["time"]}"/>')
        s.append(f'<rect x="{left + bar_w * f:.1f}" y="{y}" width="{bar_w * (1 - f):.1f}" height="20" fill="{t["space"]}" fill-opacity=".28"/>')
        s.append(f'<text x="{left + 8}" y="{y + 14.5}" font-family="{MONO}" font-size="11.5" fill="{t["card"]}">{f * 100:.0f}%</text>')
        s.append(f'<text x="{left + bar_w - 8}" y="{y + 14.5}" text-anchor="end" font-family="{MONO}" font-size="11.5" fill="{t["space"]}">{(1 - f) * 100:.0f}%</text>')
        vr = float(r["variance_ratio"])
        xv = rx0 + rscale * vr
        s.append(f'<line x1="{x1}" x2="{xv:.1f}" y1="{y + 10}" y2="{y + 10}" stroke="{t["rule"]}" stroke-width="1.5"/>')
        s.append(f'<circle cx="{xv:.1f}" cy="{y + 10}" r="6" fill="{t["well"] if vr > 1 else t["card"]}" stroke="{t["well"]}" stroke-width="1.5"/>')
        s.append(f'<text x="{xv + 12:.1f}" y="{y + 14.5}" font-family="{MONO}" font-size="12" fill="{t["ink2"]}">{vr:.2f}</text>')
    s.append(f'<text x="{left}" y="{H - 22}" font-family="{MONO}" font-size="11.5" fill="{t["muted"]}">'
             f'<tspan fill="{t["time"]}">&#9632;</tspan> timelike<tspan dx="14" fill="{t["space"]}" fill-opacity=".6">&#9632;</tspan> spacelike<tspan dx="28">'
             f'Daily bars 2021-03 to 2026-02. Pooled Levene p = {float(rows["POOLED"]["levene_p"]):.3f}; Cohen d = {float(rows["POOLED"]["cohens_d"]):.3f}. Small effect, not a trading edge.</tspan></text>')
    s.append("</svg>")
    return f'<!doctype html><html><head><meta charset="utf-8"><style>*{{margin:0}}body{{background:{t["bg"]};width:{W}px}}</style></head><body>{"".join(s)}</body></html>'


def site_data(results: Path, q1: Path) -> dict:
    """Compact JSON for the project site: SPY bars with labels, per-ticker stats."""
    px = read_csv(ROOT / "validation/data/SPY_1m.csv")
    rg = {int(r["bar_index"]): r for r in read_csv(results / "SPY_regime.csv")}
    bars = []
    for i, row in enumerate(px):
        if i in rg:
            bars.append([row["timestamp"][:10], round(float(row["close"]), 2),
                         1 if rg[i]["interval_type"] == "Timelike" else 0,
                         round(float(rg[i]["beta"]), 4)])
    stats = []
    for r in read_csv(q1 / "Q1_RESULTS_RAW.csv"):
        stats.append({k: r[k] for k in ("group", "n_timelike", "n_spacelike", "levene_p",
                                         "bartlett_p", "cohens_d", "variance_ratio")})
    return {"spy": bars, "q1": stats}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--q1", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        (a.out / f"hero-{theme}.html").write_text(hero_html(theme, a.results, a.q1), encoding="utf-8")
        (a.out / f"social-{theme}.html").write_text(hero_html(theme, a.results, a.q1, social=True), encoding="utf-8")
        (a.out / f"regimes-{theme}.html").write_text(regimes_html(theme, a.q1), encoding="utf-8")
    import json
    (a.out / "site-data.json").write_text(json.dumps(site_data(a.results, a.q1), separators=(",", ":")), encoding="utf-8")
    print("wrote", sorted(p.name for p in a.out.iterdir()))


if __name__ == "__main__":
    main()
