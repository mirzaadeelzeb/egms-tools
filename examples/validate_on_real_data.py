"""Validate egms-tools against a real EGMS product.

The test suite runs on synthetic data so that it needs no downloads. This script
is the complement: point it at an actual EGMS file and it checks that the reader
copes with the real column layout, and that a velocity fitted from the time
series agrees with the ``mean_velocity`` EGMS publishes alongside it.

Agreement is the meaningful check. EGMS derives its velocity from the same
series, so a correlation well below ~0.99 means something is being misread —
wrong columns picked up, a unit problem, or dates out of order.

Usage
-----
    python examples/validate_on_real_data.py EGMS_L2b_117_0239_IW3_VV_2019_2023_1.csv

Download a tile from https://egms.land.copernicus.eu/ (free, registration required).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

import egmstools as egms


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("path", help="an EGMS CSV, shapefile or GeoPackage")
    parser.add_argument("--n-points", type=int, default=500,
                        help="how many points to fit (default: 500)")
    parser.add_argument("--velocity-column", default=None,
                        help="override the published velocity column to compare against")
    args = parser.parse_args(argv)

    print(f"reading {args.path} ...")
    points = egms.read_egms(args.path)

    print("\n--- summarise() ---")
    print(json.dumps(egms.summarise(points), indent=2))

    columns = egms.date_columns(points)
    if not columns:
        print("\nNo acquisition columns found. Nothing further to check.", file=sys.stderr)
        return 1

    sample = points.iloc[: args.n_points]
    print(f"\n--- fitting velocities for {len(sample):,} points ---")
    fits = egms.fit_velocities(sample)
    print(f"  fitted mean velocity  : {fits['velocity'].mean():+.3f} mm/yr")
    print(f"  median R^2            : {fits['r_squared'].median():.3f}")
    print(f"  median acquisitions   : {int(fits['n'].median())}")

    column = args.velocity_column or _find_velocity_column(points)
    if column is None:
        print("\nNo published velocity column found — skipping the cross-check.")
        return 0

    published = np.asarray(sample[column], dtype=float)
    fitted = np.asarray(fits["velocity"], dtype=float)
    mask = np.isfinite(published) & np.isfinite(fitted)

    if mask.sum() < 10:
        print("\nToo few comparable points for a cross-check.", file=sys.stderr)
        return 1

    correlation = float(np.corrcoef(published[mask], fitted[mask])[0, 1])
    mean_abs = float(np.mean(np.abs(published[mask] - fitted[mask])))

    print(f"\n--- cross-check against published '{column}' (n={mask.sum():,}) ---")
    print(f"  correlation           : {correlation:.5f}")
    print(f"  mean abs difference   : {mean_abs:.4f} mm/yr")
    print(f"  max abs difference    : {np.max(np.abs(published[mask] - fitted[mask])):.4f} mm/yr")

    if correlation > 0.99:
        print("\nPASS — fitted velocities reproduce the published ones.")
        return 0
    print("\nFAIL — the fit disagrees with EGMS. Check column detection and date ordering.",
          file=sys.stderr)
    return 1


def _find_velocity_column(points) -> str | None:
    for column in points.columns:
        lowered = column.lower()
        if "veloc" in lowered and "std" not in lowered and "acc" not in lowered:
            return column
    return None


if __name__ == "__main__":
    raise SystemExit(main())
