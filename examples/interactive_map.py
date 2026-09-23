"""Turn an EGMS tile into an interactive web map.

Produces a single self-contained HTML file: pan, zoom, switch basemaps, and
click any point to read its velocity. It needs no server and nothing installed
on the other end, so it can be emailed as an attachment or dropped into a
report as a link.

Requires `leafmap`, which is not a dependency of this package:

    pip install leafmap

Usage
-----
    python examples/interactive_map.py EGMS_L2b_044_0240_IW2_VV_2020_2024_1.csv \\
        --bbox 14.19 40.78 14.47 40.93 \\
        --vmax 6 \\
        -o naples.html

Why it subsamples
-----------------
A browser will not draw a million markers; it hangs. The default budget of
12,000 goes two thirds to the points that are actually moving — keeping the
fastest — and one third to stable ground picked at random. Dropping the stable
points entirely would make a quiet city look like a sinking one, which is the
kind of map that loses you a client the first time someone checks.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import egmstools as egms

DEFAULT_MAX_POINTS = 12_000
DEFAULT_STRONG = 3.0


def build_map(path, bbox=None, vmax: float | None = None, max_points: int = DEFAULT_MAX_POINTS,
              strong: float = DEFAULT_STRONG, velocity_column: str = "mean_velocity",
              basemap: str = "SATELLITE", quiet: bool = False):
    """Build the leafmap object. Returns ``(map, points_shown)``."""
    try:
        import folium
        import leafmap.foliumap as leafmap
    except ImportError:  # pragma: no cover - depends on the user's environment
        raise SystemExit("this example needs leafmap:  pip install leafmap")

    import matplotlib

    points = egms.read_egms(path, columns=["latitude", "longitude", velocity_column])
    if not quiet:
        print(f"  {len(points):,} points in the tile")

    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        points = points[points.longitude.between(min_lon, max_lon)
                        & points.latitude.between(min_lat, max_lat)]
        if not quiet:
            print(f"  {len(points):,} inside the area of interest")
    if points.empty:
        raise SystemExit("no points inside the bbox — check the order: min_lon min_lat max_lon max_lat")

    frame = pd.DataFrame({
        "lat": points.latitude.to_numpy(),
        "lon": points.longitude.to_numpy(),
        "v": points[velocity_column].to_numpy(dtype=float),
    }).dropna()

    moving_budget = int(max_points * 2 / 3)
    moving = frame[frame.v.abs() >= strong]
    if len(moving) > moving_budget:
        moving = moving.reindex(moving.v.abs().sort_values(ascending=False).index).head(moving_budget)

    stable = frame[frame.v.abs() < strong]
    room = max(max_points - len(moving), 0)
    if len(stable) > room:
        stable = stable.sample(room, random_state=0)

    shown = pd.concat([stable, moving])          # moving drawn last, so on top
    if not quiet:
        print(f"  showing {len(shown):,} ({len(moving):,} moving >= {strong} mm/yr, "
              f"{len(stable):,} stable context)")

    if vmax is None:
        vmax = float(np.ceil(np.percentile(np.abs(frame.v), 98))) or 1.0

    cmap = matplotlib.colormaps["RdBu"]
    norm = matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax)

    m = leafmap.Map(center=[float(frame.lat.mean()), float(frame.lon.mean())], zoom=12)
    m.add_basemap(basemap)

    layer = folium.FeatureGroup(name=f"EGMS velocity ({len(shown):,} points)")
    for lat, lon, v in shown.itertuples(index=False):
        folium.CircleMarker(
            location=[lat, lon], radius=2, color=None, weight=0, fill=True,
            fill_color=matplotlib.colors.to_hex(cmap(norm(v))), fill_opacity=0.85,
            popup=folium.Popup(f"{v:+.1f} mm/year", max_width=150),
        ).add_to(layer)
    layer.add_to(m)

    m.add_colorbar(
        colors=[matplotlib.colors.to_hex(cmap(norm(v))) for v in np.linspace(-vmax, vmax, 11)],
        vmin=-vmax, vmax=vmax,
        caption=f"Velocity along satellite line of sight (mm/year) — red = moving away "
                f"(scale ±{vmax:g})",
    )
    folium.LayerControl().add_to(m)
    return m, len(shown)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("path", help="an EGMS CSV, shapefile or GeoPackage")
    parser.add_argument("-o", "--output", default="egms_map.html")
    parser.add_argument("--bbox", type=float, nargs=4,
                        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    parser.add_argument("--vmax", type=float,
                        help="colour scale -VMAX..+VMAX (default: 98th percentile)")
    parser.add_argument("--max-points", type=int, default=DEFAULT_MAX_POINTS,
                        help=f"markers to draw (default {DEFAULT_MAX_POINTS:,}); "
                             "lower it if the file is too big to email")
    parser.add_argument("--strong", type=float, default=DEFAULT_STRONG,
                        help="mm/yr above which a point counts as moving")
    parser.add_argument("--velocity-column", default="mean_velocity")
    parser.add_argument("--basemap", default="SATELLITE")
    args = parser.parse_args(argv)

    print(f"reading {args.path} ...")
    m, _ = build_map(args.path, bbox=args.bbox, vmax=args.vmax, max_points=args.max_points,
                     strong=args.strong, velocity_column=args.velocity_column,
                     basemap=args.basemap)
    m.to_html(args.output)
    print(f"\nwrote {args.output}")
    print("Open it in any browser. It is self-contained — nothing to install on the other end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
