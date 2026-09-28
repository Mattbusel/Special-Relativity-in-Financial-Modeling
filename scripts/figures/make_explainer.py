"""Build the explainer figures in assets/, each as -light.svg and -dark.svg:

  how-it-works    the four steps, as cards
  lorentz-factor  gamma against beta, with the real SPY betas
  spy-last-60     the last 60 SPY bars, labelled

The numbers come from real data.

It recomputes, in plain Python, exactly what build/regime_validator does to each
bar of validation/data/SPY_1m.csv:

  1. event = (time = bar index, price = close, volume, momentum = simple return)
  2. rolling z-score of price, volume and momentum (srfm::CoordinateNormalizer,
     window 20; regime_validator feeds each bar to it twice, reproduced here)
  3. ds^2 = -c^2 dt^2 + dP^2 + dV^2 + dM^2 with c = SPEED_OF_INFORMATION = 1
     (SpacetimeInterval::compute), timelike if ds^2 < 0, spacelike if > 0
  4. beta = |step| / (c dt), clamped below 0.9999; gamma = 1 / sqrt(1 - beta^2)

If --regime points at the validator's output CSV, every label is checked
against it and the script refuses to draw on any mismatch.

    ./build/regime_validator --input validation/data/SPY_1m.csv --output spy_regime.csv --ticker SPY
    python scripts/figures/make_explainer.py --regime spy_regime.csv
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import deque
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "validation" / "data" / "SPY_1m.csv"

THEMES = {
    "light": dict(bg="#F5F2EA", card="#FBF9F4", ink="#1A1D24", ink2="#4A4B50",
                  muted="#6F6C64", grid="#DDD7C9", rule="#C9C2B1", time="#00809A",
                  space="#D0402A", well="#C9A200", cone_a=0.12),
    "dark": dict(bg="#0E1116", card="#141820", ink="#ECE7DC", ink2="#B9B4A9",
                 muted="#8F8B82", grid="#232831", rule="#343A45", time="#35A9C6",
                 space="#EE5E3F", well="#E0B43A", cone_a=0.16),
}
SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,'Cascadia Mono',Consolas,'DejaVu Sans Mono',Menlo,monospace"
BETA_MAX_SAFE = 0.9999


# ── the pipeline, mirrored from the C++ ─────────────────────────────────────

class Normalizer:
    """srfm::CoordinateNormalizer: rolling sample z-score, window 20, warm after 20 samples."""

    def __init__(self, window: int = 20, min_samples: int = 20):
        self.window, self.min_samples, self.total = window, min_samples, 0
        self.bufs = [deque(), deque(), deque()]

    @staticmethod
    def _z(v: float, buf: deque) -> float:
        k = len(buf)
        if k < 2:
            return 0.0
        mu = sum(buf) / k
        sd = math.sqrt(sum((x - mu) ** 2 for x in buf) / (k - 1))
        return 0.0 if sd < 1e-9 else (v - mu) / sd

    def normalize(self, ev):
        for j in range(3):
            self.bufs[j].append(ev[j + 1])
            if len(self.bufs[j]) > self.window:
                self.bufs[j].popleft()
        self.total += 1
        if self.total < self.min_samples:
            return None
        return (ev[0],) + tuple(self._z(ev[j + 1], self.bufs[j]) for j in range(3))


def classify_spy():
    with open(DATA, newline="") as f:
        rows = list(csv.DictReader(f))
    close = [float(r["close"]) for r in rows]
    vol = [float(r["volume"]) for r in rows]
    dates = [r["timestamp"][:10] for r in rows]
    n = len(close)
    raw = [(float(i), close[i], vol[i], (close[i] - close[i - 1]) / close[i - 1] if i else 0.0)
           for i in range(n)]
    norm = Normalizer()
    prev = norm.normalize(raw[0]) or (0.0, 0.0, 0.0, 0.0)
    out = []
    for i in range(1, n - 1):
        cur = norm.normalize(raw[i])          # MarketManifold::process
        again = norm.normalize(raw[i])        # regime_validator's second call
        if cur is not None:
            dt = cur[0] - prev[0]
            dp, dv, dm = (cur[j] - prev[j] for j in (1, 2, 3))
            step = math.sqrt(dp * dp + dv * dv + dm * dm)
            ds2 = step * step - dt * dt
            label = "Lightlike" if abs(ds2) <= 1e-12 else ("Timelike" if ds2 < 0 else "Spacelike")
            beta = min(step / abs(dt), BETA_MAX_SAFE - 1e-15) if dt else None
            out.append(dict(i=i, date=dates[i], close=close[i], dt=dt, dp=dp, dv=dv, dm=dm,
                            step=step, ds2=ds2, label=label, beta=beta))
        prev = again or (0.0, 0.0, 0.0, 0.0)
    return out


def check_against(bars, regime_csv: Path) -> int:
    with open(regime_csv, newline="") as f:
        got = {int(r["bar_index"]): r["interval_type"] for r in csv.DictReader(f)}
    bad = [b["i"] for b in bars if got.get(b["i"]) != b["label"]]
    if bad or len(got) != len(bars):
        raise SystemExit(f"labels differ from {regime_csv} at bars {bad[:10]} "
                         f"({len(got)} rows vs {len(bars)} recomputed)")
    return len(bars)


# ── drawing helpers ─────────────────────────────────────────────────────────

def text(x, y, s, t, size=16, fill=None, family=SANS, anchor="start", weight=400, italic=False,
         xml_space=False):
    it = ' font-style="italic"' if italic else ""
    it += ' xml:space="preserve"' if xml_space else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
            f'fill="{fill or t["ink"]}" text-anchor="{anchor}" font-weight="{weight}"{it}>{escape(s)}</text>')


def rich(x, y, parts, t, size=16, family=SANS, anchor="start"):
    """One line of text with differently coloured spans: parts = [(str, colour or None, weight)]."""
    spans = "".join(f'<tspan fill="{c or t["ink"]}" font-weight="{w}">{escape(s)}</tspan>' for s, c, w in parts)
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
            f'text-anchor="{anchor}" xml:space="preserve">{spans}</text>')


def arrow(x1, y1, x2, y2, col, w=2.4, dash=None):
    ang = math.atan2(y2 - y1, x2 - x1)
    hx1, hy1 = x2 - 11 * math.cos(ang - 0.42), y2 - 11 * math.sin(ang - 0.42)
    hx2, hy2 = x2 - 11 * math.cos(ang + 0.42), y2 - 11 * math.sin(ang + 0.42)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" '
            f'stroke-width="{w}" stroke-linecap="round"{d}/>'
            f'<path d="M{hx1:.1f},{hy1:.1f} L{x2:.1f},{y2:.1f} L{hx2:.1f},{hy2:.1f}" fill="none" '
            f'stroke="{col}" stroke-width="{w}" stroke-linecap="round" stroke-linejoin="round"/>')


# ── the figures ─────────────────────────────────────────────────────────────
# Three SVGs, each 900 wide so the text stays readable at GitHub's 830 px column.

W = 900


def open_svg(H, t, title, desc):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
            f'role="img" aria-labelledby="t d">',
            f'<title id="t">{escape(title)}</title>', f'<desc id="d">{escape(desc)}</desc>',
            f'<rect width="{W}" height="{H}" rx="14" fill="{t["bg"]}"/>']


def fig_steps(theme: str) -> str:
    """The four steps regime_validator runs on every bar, as a 2 x 2 grid of cards."""
    t = THEMES[theme]
    cw, ch, gx, gy, x0, y0 = 396, 228, 20, 20, 44, 118
    ky = y0 + 2 * ch + gy + 44
    H = ky + 92
    s = open_svg(H, t, "How SRFM labels a price bar timelike or spacelike",
                 "Four steps: take a bar, rescale price, volume and momentum to standard deviations, "
                 "measure the step from the previous bar, and compare it to a speed limit of one per bar. "
                 "Under the limit is timelike, over it is spacelike.")
    s.append(text(x0, 58, "How a price bar becomes timelike or spacelike", t, size=28, weight=700))
    s.append(text(x0, 90, "The four steps regime_validator runs on every bar. No physics needed.",
                  t, size=18, fill=t["ink2"]))
    cards = [
        ("1", "Take one bar", ["A day's close, its volume and its", "momentum (the day's % change)."],
         ["event = (time, price,", "         volume, momentum)"], "SpacetimeEvent"),
        ("2", "Put it on one scale", ["Rescale each number to \"standard",
                                      "deviations from the last 20 bars\",",
                                      "so dollars and shares compare."],
         ["z = (value - mean) / std dev"], "CoordinateNormalizer(20)"),
        ("3", "Measure the step", ["How far did the bar move from the", "one before, in all three at once?"],
         ["step = √(ΔP² + ΔV² + ΔM²)", "ds²  = step² − (c·Δt)²"], "SpacetimeInterval::compute"),
        ("4", "Compare to the speed limit", ["c = 1 standard deviation per bar."], [], ""),
    ]
    for k, (num, head, body, code, fn) in enumerate(cards):
        x, y = x0 + (k % 2) * (cw + gx), y0 + (k // 2) * (ch + gy)
        s.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="10" fill="{t["card"]}" stroke="{t["rule"]}"/>')
        s.append(f'<circle cx="{x + 32}" cy="{y + 34}" r="17" fill="{t["time"]}"/>')
        s.append(text(x + 32, y + 40, num, t, size=18, fill=t["card"], anchor="middle", weight=700))
        s.append(text(x + 60, y + 41, head, t, size=21, weight=700))
        for j, ln in enumerate(body):
            s.append(text(x + 20, y + 80 + j * 24, ln, t, size=17, fill=t["ink2"]))
        cy0 = y + 80 + len(body) * 24 + 14
        for j, ln in enumerate(code):
            s.append(text(x + 20, cy0 + j * 24, ln, t, size=17, family=MONO, xml_space=True))
        if fn:
            s.append(text(x + 20, y + ch - 20, fn, t, size=15, family=MONO, fill=t["muted"]))
    # card 4: the verdict and a light cone (time runs up)
    x, y = x0 + cw + gx, y0 + ch + gy
    s.append(text(x + 20, y + 118, "step < 1", t, size=19, family=MONO, fill=t["time"], weight=700))
    s.append(text(x + 20, y + 142, "timelike", t, size=17, fill=t["time"], weight=700))
    s.append(text(x + 20, y + 182, "step > 1", t, size=19, family=MONO, fill=t["space"], weight=700))
    s.append(text(x + 20, y + 206, "spacelike", t, size=17, fill=t["space"], weight=700))
    ox, oy, half = x + 262, y + 204, 92
    s.append(f'<path d="M{ox:.1f},{oy} L{ox + half:.1f},{oy - half} L{ox - half:.1f},{oy - half} Z" '
             f'fill="{t["time"]}" fill-opacity="{t["cone_a"] * 1.6}" stroke="{t["time"]}" stroke-opacity="0.6"/>')
    s.append(arrow(ox, oy, ox - 22, oy - 80, t["time"], w=3.2))
    s.append(arrow(ox, oy, ox + 118, oy - 46, t["space"], w=3.2, dash="7 5"))
    s.append(f'<circle cx="{ox:.1f}" cy="{oy}" r="5" fill="{t["ink"]}"/>')
    s.append(text(ox - 12, oy + 4, "now", t, size=15, family=MONO, fill=t["muted"], anchor="end"))
    s.append(text(ox, oy - half - 8, "light cone", t, size=15, family=MONO, fill=t["muted"], anchor="middle"))
    for k in range(2):  # arrows 1 -> 2 and 3 -> 4
        ay = y0 + k * (ch + gy) + 34
        s.append(arrow(x0 + cw + 2, ay, x0 + cw + gx - 2, ay, t["ink2"], w=2))
    s.append(rich(x0, ky, [("Timelike", t["time"], 700),
                           (" (inside the cone): the bar moved less than the speed limit. An ordinary step.",
                            t["ink2"], 400)], t, size=17))
    s.append(rich(x0, ky + 30, [("Spacelike", t["space"], 700),
                                (" (outside the cone): it moved further than c allows in one bar. A jump.",
                                 t["ink2"], 400)], t, size=17))
    s.append(text(x0, ky + 60, "In the code c is SPEED_OF_INFORMATION = 1 and Δt is one bar, so the test is step < 1.",
                  t, size=15, fill=t["muted"]))
    s.append("</svg>")
    return "\n".join(s)


def fig_lorentz(theme: str, bars) -> str:
    """Lorentz factor gamma against beta, with the real timelike SPY bars as a rug."""
    t = THEMES[theme]
    H = 600
    s = open_svg(H, t, "The Lorentz factor gamma, explained",
                 "Beta is a bar's step as a fraction of the speed limit. Gamma = 1 / sqrt(1 - beta squared) "
                 "stays near 1 for slow bars and shoots up near the limit. Marked: the values printed by "
                 "the lorentz_basics example and the betas of the real timelike SPY bars.")
    x0 = 44
    s.append(text(x0, 58, "The Lorentz factor γ: how close a bar came to the limit", t, size=26, weight=700))
    s.append(rich(x0, 92, [("β", t["ink"], 700), (" = step ÷ c, a speed from 0 to 1.   ", t["ink2"], 400),
                           ("γ", t["ink"], 700), (" = 1 / √(1 − β²).", t["ink2"], 400)], t, size=18))
    s.append(text(x0, 118, "Slow bars get γ ≈ 1. Near the limit γ shoots up, like a particle near light speed.",
                  t, size=17, fill=t["ink2"]))
    px0, px1, py0, py1 = 100, W - 50, 196, 460
    gmax = 8.0
    X = lambda b: px0 + b * (px1 - px0)  # noqa: E731
    Y = lambda g: py1 - (min(g, gmax) - 1) / (gmax - 1) * (py1 - py0)  # noqa: E731
    s.append(f'<rect x="{px0}" y="{Y(3):.1f}" width="{px1 - px0}" height="{Y(1) - Y(3):.1f}" '
             f'fill="{t["well"]}" fill-opacity="0.14"/>')
    s.append(text(px0 + 12, Y(3) + 24, "srfm --backtest sizes each position by γ, clamped to 1x to 3x",
                  t, size=15, fill=t["ink2"]))
    for g in range(1, int(gmax) + 1):
        s.append(f'<line x1="{px0}" x2="{px1}" y1="{Y(g):.1f}" y2="{Y(g):.1f}" stroke="{t["grid"]}"/>')
        s.append(text(px0 - 12, Y(g) + 5, f"{g}", t, size=15, family=MONO, fill=t["muted"], anchor="end"))
    for b10 in range(0, 11, 2):
        b = b10 / 10
        s.append(text(X(b), py1 + 26, f"{b:.1f}", t, size=15, family=MONO, fill=t["muted"], anchor="middle"))
    s.append(text((px0 + px1) / 2, py1 + 54, "β (step as a fraction of the speed limit)", t, size=16,
                  fill=t["muted"], anchor="middle"))
    s.append(text(px0 - 44, (py0 + py1) / 2, "γ", t, size=20, fill=t["muted"], anchor="middle"))
    pts, b = [], 0.0
    while b < 0.9915:
        pts.append(f"{X(b):.1f},{Y(1 / math.sqrt(1 - b * b)):.1f}")
        b += 0.0025
    s.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{t["ink"]}" stroke-width="3"/>')
    s.append(f'<line x1="{X(1):.1f}" x2="{X(1):.1f}" y1="{py0}" y2="{py1}" stroke="{t["space"]}" '
             f'stroke-width="2" stroke-dasharray="7 5"/>')
    s.append(text(X(1), py0 - 34, "β = 1: the limit", t, size=16, fill=t["space"], anchor="end"))
    s.append(text(X(1), py0 - 14, "(code caps β at 0.9999, so γ ≤ 70.7)", t, size=15, fill=t["muted"],
                  anchor="end"))
    # examples/lorentz_basics prints these (beta, gamma) pairs
    for b, anchor, dx, dy in ((0.5, "middle", 0, -14), (0.9, "end", -14, -14), (0.99, "end", -16, 6)):
        g = 1 / math.sqrt(1 - b * b)
        s.append(f'<circle cx="{X(b):.1f}" cy="{Y(g):.1f}" r="6" fill="{t["bg"]}" stroke="{t["ink"]}" stroke-width="2.5"/>')
        s.append(text(X(b) + dx, Y(g) + dy, f"β {b:g} → γ {g:.2f}", t, size=15, family=MONO, anchor=anchor))
    # rug: every real timelike SPY bar's beta from step 3
    tl = [r["beta"] for r in bars if r["label"] == "Timelike"]
    for b in tl:
        s.append(f'<line x1="{X(b):.1f}" x2="{X(b):.1f}" y1="{py1 - 16}" y2="{py1}" stroke="{t["time"]}" '
                 f'stroke-width="1.4" stroke-opacity="0.55"/>')
    med = sorted(tl)[len(tl) // 2]
    s.append(text(x0, py1 + 94, f"Teal ticks: β of the {len(tl)} timelike SPY daily bars (median β {med:.2f}, "
                  f"γ {1 / math.sqrt(1 - med * med):.2f}).", t, size=15, fill=t["muted"]))
    s.append(text(x0, py1 + 116, "Circles: what examples/lorentz_basics prints.", t, size=15, fill=t["muted"]))
    s.append("</svg>")
    return "\n".join(s)


def fig_chart(theme: str, bars, verified: int | None, last: int = 60) -> str:
    """Real chart: the step size of the last `last` SPY bars against the speed limit."""
    t = THEMES[theme]
    py0, py1 = 140, 400
    H = py1 + 84
    win = bars[-last:]
    n_t = sum(b["label"] == "Timelike" for b in win)
    s = open_svg(H, t, f"The last {last} SPY daily bars, labelled",
                 f"Bar chart of each bar's step size against the speed limit of 1. {n_t} of {last} bars "
                 "stay under it and are timelike; the rest are spacelike.")
    s.append(text(44, 50, f"Real data: the last {last} SPY daily bars", t, size=24, weight=700))
    s.append(text(44, 78, f"Bar height = the step from step 3. {n_t} of {last} stay under the speed limit.",
                  t, size=17, fill=t["ink2"]))
    ly = 112
    s.append(f'<rect x="44" y="{ly - 14}" width="17" height="17" rx="2" fill="{t["time"]}"/>')
    s.append(text(68, ly, "timelike", t, size=16))
    s.append(f'<rect x="156" y="{ly - 14}" width="17" height="17" rx="2" fill="{t["space"]}"/>')
    s.append(text(180, ly, "spacelike", t, size=16))
    s.append(f'<line x1="282" x2="322" y1="{ly - 5}" y2="{ly - 5}" stroke="{t["ink"]}" stroke-width="2" stroke-dasharray="7 5"/>')
    s.append(text(330, ly, "speed limit c = 1", t, size=16))
    px0, px1 = 84, W - 36
    ymax = float(max(4, math.ceil(max(b["step"] for b in win))))
    Y = lambda v: py1 - (min(v, ymax) / ymax) * (py1 - py0)  # noqa: E731
    for v in range(int(ymax) + 1):
        s.append(f'<line x1="{px0}" x2="{px1}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{t["grid"]}"/>')
        s.append(text(px0 - 12, Y(v) + 5, f"{v}", t, size=15, family=MONO, fill=t["muted"], anchor="end"))
    bw = (px1 - px0) / last
    for k, b in enumerate(win):
        col = t["time"] if b["label"] == "Timelike" else t["space"]
        h = py1 - Y(b["step"])
        s.append(f'<rect x="{px0 + k * bw + bw * 0.16:.1f}" y="{py1 - h:.1f}" width="{bw * 0.68:.1f}" '
                 f'height="{max(h, 1):.1f}" rx="1.5" fill="{col}"><title>{b["date"]}: step {b["step"]:.2f}, '
                 f'{b["label"].lower()}</title></rect>')
    s.append(f'<line x1="{px0}" x2="{px1}" y1="{Y(1):.1f}" y2="{Y(1):.1f}" stroke="{t["ink"]}" '
             f'stroke-width="2" stroke-dasharray="7 5"/>')
    for k in range(0, last, 15):
        s.append(text(px0 + k * bw + bw / 2, py1 + 24, win[k]["date"], t, size=14, family=MONO,
                      fill=t["muted"], anchor="middle"))
    tl = [b for b in win if b["label"] == "Timelike"]
    if tl:
        b = max(tl, key=lambda r: r["step"])
        k = win.index(b)
        g = 1 / math.sqrt(1 - b["beta"] ** 2)
        bx = px0 + k * bw + bw / 2
        s.append(f'<line x1="{bx:.1f}" y1="{Y(b["step"]) - 4:.1f}" x2="{bx:.1f}" y2="{ly + 8:.1f}" '
                 f'stroke="{t["time"]}" stroke-width="1.2" stroke-dasharray="2 3"/>')
        s.append(text(min(bx + 8, px1 - 290), ly, f'{b["date"]}: β {b["beta"]:.2f}, γ {g:.2f}',
                      t, size=15, family=MONO, fill=t["time"]))
    foot = (f"validation/data/SPY_1m.csv. Recomputed from the C++ formulas; labels match "
            f"regime_validator on all {verified:,} bars." if verified else
            "validation/data/SPY_1m.csv, recomputed from the C++ formulas.")
    s.append(text(44, py1 + 60, foot, t, size=14, fill=t["muted"]))
    s.append("</svg>")
    return "\n".join(s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", type=Path, help="regime_validator output for SPY, to verify labels")
    ap.add_argument("--out", type=Path, default=ROOT / "assets")
    a = ap.parse_args()
    bars = classify_spy()
    verified = check_against(bars, a.regime) if a.regime else None
    a.out.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        for name, svg in (("how-it-works", fig_steps(theme)), ("lorentz-factor", fig_lorentz(theme, bars)),
                          ("spy-last-60", fig_chart(theme, bars, verified))):
            (a.out / f"{name}-{theme}.svg").write_text(svg, encoding="utf-8")
    n_t = sum(b["label"] == "Timelike" for b in bars)
    print(f"{len(bars)} bars, {n_t} timelike, {len(bars) - n_t} spacelike"
          + (f", all {verified} labels match {a.regime}" if verified else ""))
    print(f"wrote how-it-works, lorentz-factor and spy-last-60 (-light/-dark).svg to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
