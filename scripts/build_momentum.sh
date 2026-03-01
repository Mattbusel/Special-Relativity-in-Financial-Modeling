#!/usr/bin/env bash
# build_momentum.sh — bash wrapper around build_momentum.bat
# Converts the Unix path to a Windows path and calls cmd.exe.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAT="$(cygpath -w "$SCRIPT_DIR/build_momentum.bat" 2>/dev/null || echo "$SCRIPT_DIR\\build_momentum.bat")"
cmd.exe /c "$BAT"
