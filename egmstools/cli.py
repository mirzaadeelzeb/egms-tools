"""Command-line interface: ``egms <command> ...``.

Enough to answer the routine questions without opening a notebook — what is in
this file, what are the velocities, what is moving inside my polygons.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__


def _cmd_info(args) -> int:
    from .io import read_egms, summarise

    points = read_egms(args.path, crs=None if args.keep_crs else 3035)
    print(json.dumps(summarise(points), indent=2))
    return 0


def _cmd_velocity(args) -> int:
    from .io import read_egms
    from .timeseries import fit_velocities

    from .io import date_columns

    points = read_egms(args.path)
    fits = fit_velocities(points, min_acquisitions=args.min_acquisitions)

    # Carry the metadata across but drop the acquisition columns: they are the
    # input to the fit, they dominate the file size, and nobody wants 300 extra
    # columns next to the answer. --keep-series puts them back.
    drop = {points.geometry.name}
    if not args.keep_series:
        drop |= set(date_columns(points))
    metadata = points.drop(columns=[c for c in points.columns if c in drop])
    result = metadata.join(fits, rsuffix="_fitted")

    valid = fits["velocity"].notna().sum()
    print(f"fitted {valid:,} of {len(fits):,} points", file=sys.stderr)
    _write_table(result, args.output)
    return 0


def _cmd_aggregate(args) -> int:
    import geopandas as gpd

    from .aggregate import aggregate_velocity
    from .io import read_egms

    points = read_egms(args.path)
    polygons = gpd.read_file(args.polygons)
    table = aggregate_velocity(
        points, polygons, args.id_column,
        velocity_column=args.velocity_column, buffer=args.buffer,
    )

    empty = int((table["n_points"] == 0).sum())
    print(f"{len(table):,} polygons, {empty:,} with no measurement points", file=sys.stderr)
    _write_table(table, args.output)
    return 0


def _cmd_decompose(args) -> int:
    import numpy as np
    import pandas as pd

    from .decompose import decompose, vertical_fraction
    from .io import read_egms

    ascending = read_egms(args.ascending)
    descending = read_egms(args.descending)

    if len(ascending) != len(descending):
        print(
            f"error: {len(ascending):,} ascending points vs {len(descending):,} descending. "
            "Both geometries must be sampled at the same locations — resample them onto a "
            "common grid or aggregate them onto shared polygons first.",
            file=sys.stderr,
        )
        return 2

    up, east = decompose(
        np.vstack([ascending[args.velocity_column], descending[args.velocity_column]]),
        [args.asc_incidence, args.dsc_incidence],
        [args.asc_heading, args.dsc_heading],
    )
    table = pd.DataFrame({
        "v_up": up,
        "v_east": east,
        "vertical_fraction": vertical_fraction(up, east),
    })
    _write_table(table, args.output)
    return 0


def _cmd_map(args) -> int:
    import matplotlib
    matplotlib.use("Agg")

    from .io import read_egms
    from .plot import publication_map

    points = read_egms(args.path, columns=["latitude", "longitude", args.velocity_column])

    labels = []
    for spec in args.label or []:
        try:
            name, coords = spec.split(":", 1)
            lon, lat = (float(v) for v in coords.split(","))
        except ValueError:
            print(f"error: --label expects 'Name:lon,lat', got {spec!r}", file=sys.stderr)
            return 2
        labels.append((name, lon, lat))

    try:
        fig = publication_map(
            points, column=args.velocity_column, bbox=args.bbox, vmax=args.vmax,
            title=args.title, subtitle=args.subtitle, labels=labels, credit=args.credit,
            point_size=args.point_size, dpi=args.dpi,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    fig.savefig(args.output, facecolor="white")
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


def _write_table(table, output) -> None:
    if output:
        table.to_csv(output, index=False)
        print(f"wrote {output}", file=sys.stderr)
    else:
        table.to_csv(sys.stdout, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="egms", description="Utilities for European Ground Motion Service data."
    )
    parser.add_argument("--version", action="version", version=f"egms-tools {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="describe an EGMS file")
    info.add_argument("path")
    info.add_argument("--keep-crs", action="store_true", help="do not reproject to EPSG:3035")
    info.set_defaults(func=_cmd_info)

    velocity = subparsers.add_parser("velocity", help="fit a linear velocity per point")
    velocity.add_argument("path")
    velocity.add_argument("-o", "--output", help="CSV to write (default: stdout)")
    velocity.add_argument("--min-acquisitions", type=int, default=10)
    velocity.add_argument("--keep-series", action="store_true",
                          help="also write the acquisition columns through to the output")
    velocity.set_defaults(func=_cmd_velocity)

    aggregate = subparsers.add_parser("aggregate", help="summarise velocities over polygons")
    aggregate.add_argument("path")
    aggregate.add_argument("polygons", help="any vector file GeoPandas can read")
    aggregate.add_argument("id_column", help="column identifying each polygon")
    aggregate.add_argument("-o", "--output")
    aggregate.add_argument("--velocity-column", default="mean_velocity")
    aggregate.add_argument("--buffer", type=float, default=0.0, help="dilate polygons, in CRS units")
    aggregate.set_defaults(func=_cmd_aggregate)

    decompose_cmd = subparsers.add_parser(
        "decompose", help="combine ascending and descending velocities into up/east"
    )
    decompose_cmd.add_argument("ascending")
    decompose_cmd.add_argument("descending")
    decompose_cmd.add_argument("-o", "--output")
    decompose_cmd.add_argument("--velocity-column", default="mean_velocity")
    decompose_cmd.add_argument("--asc-incidence", type=float, default=39.0)
    decompose_cmd.add_argument("--asc-heading", type=float, default=347.0)
    decompose_cmd.add_argument("--dsc-incidence", type=float, default=39.0)
    decompose_cmd.add_argument("--dsc-heading", type=float, default=193.0)
    decompose_cmd.set_defaults(func=_cmd_decompose)

    map_cmd = subparsers.add_parser("map", help="render a finished velocity map as an image")
    map_cmd.add_argument("path")
    map_cmd.add_argument("-o", "--output", default="velocity_map.png")
    map_cmd.add_argument("--velocity-column", default="mean_velocity")
    map_cmd.add_argument("--bbox", type=float, nargs=4,
                         metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
                         help="crop to this window")
    map_cmd.add_argument("--vmax", type=float,
                         help="colour scale runs -VMAX..+VMAX (default: 98th percentile)")
    map_cmd.add_argument("--title")
    map_cmd.add_argument("--subtitle")
    map_cmd.add_argument("--label", action="append", metavar="NAME:LON,LAT",
                         help="mark a place; repeat for several")
    map_cmd.add_argument("--credit", help="footer line (keep the EGMS attribution)")
    map_cmd.add_argument("--point-size", type=float, default=0.5)
    map_cmd.add_argument("--dpi", type=int, default=200)
    map_cmd.set_defaults(func=_cmd_map)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
