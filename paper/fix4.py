with open('sections/06_empirical.tex', 'rb') as f:
    lines = f.read().decode('utf-8').split('\n')

# Find the line index of the bad Results paragraph in section 5.2
# It's the FIRST occurrence of "Results.  The geodesic deviation backtest"
target = "Results.  The geodesic deviation backtest produced no completed trades in Q1 2025.  The"
idx = next(i for i,l in enumerate(lines) if target in l)

# Replace lines idx through idx+4 with Table 4
replacement = r"""\textbf{Results.}  Across the full universe:

\begin{table}[ht]
\centering
\caption{Q1 regime separation statistics. $p$-values Bonferroni-corrected.}
\label{tab:q1_results}
\begin{tabular}{lcc}
  \toprule
  \textbf{Statistic} & \textbf{Value} & \textbf{$p$-value} \\
  \midrule
  Variance ratio $\mathrm{VR}$ & $1.27\times$ & $6.0\times10^{-16}$ \\
  KS statistic $D_{\mathrm{KS}}$ & $0.015$ & $0.61$ \\
  Levene $F$-statistic & $3.01$ & $0.083$ \\
  Fraction \textsc{spacelike} bars & $75.1\%$ & --- \\
  \bottomrule
\end{tabular}
\end{table}""".split('\n')

# Count how many lines to remove (up to the next blank line after the bad paragraph)
end = idx + 1
while end < len(lines) and lines[end].strip():
    end += 1

lines[idx:end] = replacement
with open('sections/06_empirical.tex', 'wb') as f:
    f.write('\n'.join(lines).encode('utf-8'))
print(f"Replaced lines {idx}-{end} with Table 4")
