#!/usr/bin/env bash
# build_stream.sh — bash wrapper around build_stream.bat
# Converts the Unix path to a Windows path and calls cmd.exe.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAT="$(cygpath -w "$SCRIPT_DIR/build_stream.bat" 2>/dev/null || echo "$SCRIPT_DIR\\build_stream.bat")"

TARGET="${1:-all}"
cmd.exe /c "$BAT" "$TARGET"
