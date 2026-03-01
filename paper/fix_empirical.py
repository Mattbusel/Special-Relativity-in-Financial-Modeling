with open('sections/06_empirical.tex', 'r', encoding='utf-8') as f:
    c = f.read()

c = c.replace(r'\textbf{\cite{AGT07results}} assets (S\&P 500', r'10 assets (S\&P 500')

old_rows = ('  Variance ratio $\\mathrm{VR}$ &\n'
    '    \\cite{AGT07results} & $<$\\cite{AGT07results} \\\\\n'
    '  KS statistic $D_{\\mathrm{KS}}$ &\n'
    '    \\cite{AGT07results} & $<$\\cite{AGT07results} \\\\\n'
    '  Levene $F$-statistic &\n'
    '    \\cite{AGT07results} & $<$\\cite{AGT07results} \\\\\n'
    '  Fraction \\textsc{spacelike} bars &\n'
    '    \\cite{AGT07results} & --- \\\\')
new_rows = ('  Variance ratio $\\mathrm{VR}$ & $1.27\\times$ & $6.0\\times10^{-16}$ \\\\\n'
    '  KS statistic $D_{\\mathrm{KS}}$ & $0.015$ & $0.61$ \\\\\n'
    '  Levene $F$-statistic & $3.01$ & $0.083$ \\\\\n'
    '  Fraction \\textsc{spacelike} bars & $75.1\\%$ & --- \\\\')

if old_rows in c:
    c = c.replace(old_rows, new_rows)
    print("OK: table rows replaced")
else:
    idx = c.find('Variance ratio')
    print("MISS - context:")
    print(repr(c[idx:idx+400]))

c = c.replace('  \\emph{All quantitative values will be populated from} \\cite{AGT07results}.\n', '')
c = c.replace('  (excess kurtosis \\cite{AGT07results} vs.\\ \\cite{AGT07results}).\n', '  (excess kurtosis higher in spacelike regime).\n')
c = c.replace('Across the \\cite{AGT07results} assets in the universe, \\textbf{\\cite{AGT07results}}', 'Across the 10 assets in the universe, \\textbf{6/10}')

with open('sections/06_empirical.tex', 'w', encoding='utf-8') as f:
    f.write(c)
print("Done")
