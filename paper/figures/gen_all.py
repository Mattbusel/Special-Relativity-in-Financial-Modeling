#!/usr/bin/env python3
"""
gen_all.py
==========
Master figure generation script.
Runs all individual gen_*.py scripts in sequence.

Usage:
    python gen_all.py [--data-dir /path/to/agt07_results]

If --data-dir is provided, each script is called with the corresponding
empirical data file.  Otherwise, all scripts use synthetic placeholder data.
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

SCRIPTS = [
    "gen_q1_regime_distributions.py",
    "gen_q1_variance_ratio_heatmap.py",
    "gen_q2_cumulative_pnl.py",
    "gen_q2_geodesic_deviation_timeseries.py",
    "gen_lorentz_factor_surface.py",
    "gen_spacetime_diagram.py",
    "gen_covariance_manifold.py",
    "gen_module_pipeline.py",
]

DATA_FILES = {
    "gen_q1_regime_distributions.py":       "q1_regime_returns.npz",
    "gen_q1_variance_ratio_heatmap.py":      "q1_vr_sweep.npz",
    "gen_q2_cumulative_pnl.py":              "q2_backtest.npz",
    "gen_q2_geodesic_deviation_timeseries.py": "q2_deviation_ts.npz",
}


def run_script(script: str, data_dir: Path | None) -> bool:
    cmd = [sys.executable, str(HERE / script)]

    if data_dir and script in DATA_FILES:
        data_path = data_dir / DATA_FILES[script]
        if data_path.exists():
            cmd += ["--data-file", str(data_path)]
            print(f"[srfm] Using empirical data: {data_path}")
        else:
            print(f"[srfm] Data file not found ({data_path}); using synthetic placeholder.")

    print(f"[srfm] Running: {script} ...")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"[srfm] ERROR: {script} failed with exit code {result.returncode}",
              file=sys.stderr)
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all SRFM paper figures."
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Directory containing AGT-07 empirical .npz data files."
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else None

    failures = []
    for script in SCRIPTS:
        ok = run_script(script, data_dir)
        if not ok:
            failures.append(script)

    print()
    if failures:
        print(f"[srfm] {len(failures)} script(s) FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"[srfm] All {len(SCRIPTS)} figures generated successfully.")
        print(f"[srfm] Output directory: {HERE}")


if __name__ == "__main__":
    main()
