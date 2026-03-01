import re
with open('sections/06_empirical.tex', 'rb') as f:
    c = f.read().decode('utf-8')

# Restore the Q1 results paragraph and Table 4 (was clobbered by previous regex)
# Find the broken Q1 Results paragraph and replace with correct content
c = re.sub(
    r'(\\textbf\{Results\.\}\s+The geodesic deviation backtest produced no completed trades.*?trend-following signal\.)\s*\n\n(\\begin\{figure\})',
    r"""\\textbf{Results.}  Across the full universe:

\\begin{table}[ht]
\\centering
\\caption{Q1 regime separation statistics across the full equity universe. $p$-values are Bonferroni-corrected.}
\\label{tab:q1_results}
\\begin{tabular}{lcc}
  \\toprule
  \\textbf{Statistic} & \\textbf{Value} & \\textbf{$p$-value} \\\\
  \\midrule
  Variance ratio $\\mathrm{VR}$ & $1.27\\times$ & $6.0\\times10^{-16}$ \\\\
  KS statistic $D_{\\mathrm{KS}}$ & $0.015$ & $0.61$ \\\\
  Levene $F$-statistic & $3.01$ & $0.083$ \\\\
  Fraction \\textsc{spacelike} bars & $75.1\\%$ & --- \\\\
  \\bottomrule
\\end{tabular}
\\end{table}

\2""",
    c, flags=re.DOTALL
)

# Fix Cohen apostrophe
c = c.replace("Cohen\\'{s}", "Cohen's").replace("Cohen\\'s", "Cohen's").replace("Cohens d", "Cohen's $d$")
c = c.replace("Cohenś d", "Cohen's $d$")

with open('sections/06_empirical.tex', 'wb') as f:
    f.write(c.encode('utf-8'))
print("Done")
