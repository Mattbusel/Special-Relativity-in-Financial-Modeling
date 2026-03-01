with open('sections/06_empirical.tex', 'rb') as f:
    raw = f.read()
c = raw.decode('utf-8')

# Fix cross-sectional paragraph (broken sentence)
import re
c = re.sub(
    r'\\paragraph\{Cross-sectional consistency\.\}.*?effect\.',
    r'\\paragraph{Cross-sectional consistency.} The pooled Bartlett test (less sensitive to non-normality) is significant at $p = 6.0 \\times 10^{-16}$, consistent with a systematic variance separation effect. Per-asset Levene tests do not reach significance after Bonferroni correction (Cohen\'s $d = 0.037$ pooled).',
    c, flags=re.DOTALL
)

# Fix Table 5 - replace entire table with honest finding
c = re.sub(
    r'\\textbf\{Results\.\}.*?\\end\{table\}',
    r'\\textbf{Results.} The geodesic deviation backtest produced no completed trades in Q1 2025. The strongly trending market resulted in fewer than two downside return observations, causing the Sortino ratio to return \\texttt{nullopt} for all 10 assets. This is an empirical finding: the signal is inactive during strongly directional regimes, consistent with its design as a regime-transition detector rather than a trend-following signal.',
    c, flags=re.DOTALL
)

# Fix Figure 5 duplicate sentence
c = c.replace(
    'Vertical dashed lines mark earnings announcements. Vertical dashed lines mark earnings\nannouncements (synthetic placeholder data).',
    'Vertical dashed lines mark earnings announcements (synthetic placeholder data).'
)

with open('sections/06_empirical.tex', 'wb') as f:
    f.write(c.encode('utf-8'))
print("06 done")

with open('sections/08_conclusion.tex', 'rb') as f:
    c = f.read().decode('utf-8')

c = re.sub(
    r'Backtested over Q1 2025.*?buy-and-hold~\\cite\{AGT07results\}\.',
    'The Q1 2025 backtest produced no completed trades in the strongly trending market; this motivates evaluation over a longer multi-regime period.',
    c, flags=re.DOTALL
)

with open('sections/08_conclusion.tex', 'wb') as f:
    f.write(c.encode('utf-8'))
print("08 done")
