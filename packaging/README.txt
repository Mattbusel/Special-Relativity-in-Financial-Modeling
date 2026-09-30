SRFM command-line tools
=======================

Special Relativity in Financial Modeling (SRFM) gives every price bar a
"speed" and labels it timelike (an ordinary move, inside the light cone) or
spacelike (a jump, outside it).

Research code, not financial advice.

Try it
------
Windows: double-click demo.bat
Linux:   ./demo.sh

Or run a tool yourself on the bundled sample data (daily bars, 2021 to 2026):

  regime_validator --input validation/data/SPY_1m.csv --output spy_regime.csv --ticker SPY
  backtest_runner  --input spy_regime.csv --output-dir out --ticker SPY
  srfm --backtest validation/data/SPY_1m.csv
  lorentz_basics

Every tool prints its options with --help. Your own data works too: any CSV
with a header row and at least "close" and "volume" columns.

Source, docs and the paper:
  https://gitlab.com/mattbusel/Special-Relativity-in-Financial-Modeling
