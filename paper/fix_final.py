with open('sections/08_conclusion.tex', 'r', encoding='utf-8') as f:
    c = f.read()

c = c.replace(
    'Backtested over Q1 2025, the signal generates a Sharpe ratio\n  improvement of \\cite{AGT07results} relative to buy-and-hold~\\cite{AGT07results}.',
    'The Q1 2025 backtest produced no completed trades in the strongly trending market; this motivates evaluation over a longer multi-regime period.'
)

with open('sections/08_conclusion.tex', 'w', encoding='utf-8') as f:
    f.write(c)

with open('sections/06_empirical.tex', 'r', encoding='utf-8') as f:
    c = f.read()

c = c.replace(
    '  The geodesic deviation spike precedes the regime transition in\n  \\cite{AGT07results} out of \\cite{AGT07results} annotated events.',
    '  Vertical dashed lines mark earnings announcements (synthetic placeholder data).'
)

c = c.replace(
    'the pooled Bartlett test (less sensitive to non-normality)\nis significant at $p = 6.0 \\times 10^{-16}$. show statistically significant regime separation at the 5\\% level after\nBonferroni correction, consistent with a systematic (not idiosyncratic)\neffect.',
    'The pooled Bartlett test is significant at $p = 6.0 \\times 10^{-16}$, consistent with a systematic variance separation effect. Per-asset Levene tests do not reach significance after Bonferroni correction (Cohen\'s $d = 0.037$ pooled).'
)

with open('sections/06_empirical.tex', 'w', encoding='utf-8') as f:
    f.write(c)

print("Done")
