#!/usr/bin/env sh
# SRFM demo: label the bundled SPY daily bars and print the library example.
set -e
cd "$(dirname "$0")"
echo "SRFM demo: label 1,256 SPY daily bars as timelike or spacelike"
echo
./regime_validator --input validation/data/SPY_1m.csv --output spy_regime.csv --ticker SPY
echo
echo "The first rows of spy_regime.csv:"
head -6 spy_regime.csv
echo
echo "The library example: beta, gamma and two interval labels"
./lorentz_basics
echo
echo "Next: run any tool with --help, e.g.  ./regime_validator --help"
