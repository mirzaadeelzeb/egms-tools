"""Synthetic EGMS-shaped data, so the tests run without downloading anything.

Real EGMS tiles are hundreds of megabytes and carry their own licence terms, so
they do not belong in a repository. These generators reproduce the *structure*
of an EGMS product — column naming, CRS, 6-day revisit, millimetre units — with
known ground truth, which is what the tests actually need.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

EGMS_CRS = 3035


def acquisition_dates(start: str = "2019-01-05", n: int = 200, revisit_days: int = 6):
    """EGMS-style ``YYYYMMDD`` column names at a fixed revisit interval."""
    stamps = pd.date_range(start=start, periods=n, freq=f"{revisit_days}D")
    return [s.strftime("%Y%m%d") for s in stamps]


def make_points(n_points: int = 200, velocity: float = -5.0, noise_mm: float = 1.5,
                n_dates: int = 200, origin=(4_500_000.0, 2_200_000.0), spread: float = 500.0,
                seed: int = 42) -> gpd.GeoDataFrame:
    """A patch of measurement points deforming at a known linear rate.

    Parameters
    ----------
    velocity : float
        True rate in mm/yr, shared by every point. Negative means moving away
        from the sensor, which is what subsidence looks like in EGMS.
    noise_mm : float
        Standard deviation of the white noise added to each acquisition.
    """
    rng = np.random.default_rng(seed)
    dates = acquisition_dates(n=n_dates)
    t = np.arange(n_dates) * 6 / 365.25

    easting = origin[0] + rng.uniform(-spread, spread, n_points)
    northing = origin[1] + rng.uniform(-spread, spread, n_points)

    signal = velocity * t
    displacement = signal[None, :] + rng.normal(0, noise_mm, size=(n_points, n_dates))

    frame = pd.DataFrame(displacement, columns=dates)
    frame.insert(0, "pid", [f"PS_{i:05d}" for i in range(n_points)])
    frame.insert(1, "easting", easting)
    frame.insert(2, "northing", northing)
    frame.insert(3, "mean_velocity", velocity + rng.normal(0, 0.3, n_points))

    return gpd.GeoDataFrame(
        frame, geometry=gpd.points_from_xy(easting, northing), crs=EGMS_CRS
    )


def make_polygons(origin=(4_500_000.0, 2_200_000.0), size: float = 400.0,
                  ids=("A", "B")) -> gpd.GeoDataFrame:
    """Two adjacent square polygons, the first centred on :func:`make_points`."""
    x0, y0 = origin
    geoms, names = [], []
    for i, name in enumerate(ids):
        cx = x0 + i * 2 * size
        geoms.append(
            Polygon([
                (cx - size / 2, y0 - size / 2), (cx + size / 2, y0 - size / 2),
                (cx + size / 2, y0 + size / 2), (cx - size / 2, y0 + size / 2),
            ])
        )
        names.append(name)
    return gpd.GeoDataFrame({"site_id": names, "geometry": geoms}, crs=EGMS_CRS)


def write_csv(gdf: gpd.GeoDataFrame, path) -> str:
    """Write a GeoDataFrame back out in EGMS CSV form (no geometry column)."""
    gdf.drop(columns="geometry").to_csv(path, index=False)
    return str(path)
