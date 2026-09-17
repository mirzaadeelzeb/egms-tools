"""Aggregating scattered EGMS points onto polygons.

Most applied questions are asked about an object, not about a scatterer: is
*this* landslide body moving, is *this* viaduct settling, is *this* district
subsiding. That means summarising the points that fall inside a polygon — and
being honest about how few of them there sometimes are.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from .io import date_columns, dates_as_years, displacement_matrix
from .timeseries import fit_velocity


def points_in_polygons(points: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame, id_column: str,
                       buffer: float = 0.0, predicate: str = "within") -> gpd.GeoDataFrame:
    """Join EGMS points to the polygon containing them.

    Parameters
    ----------
    points, polygons : GeoDataFrame
        Reprojected to the polygons' CRS automatically.
    id_column : str
        Column of *polygons* identifying each feature; carried onto the points.
    buffer : float, default 0
        Dilate the polygons by this distance (in CRS units) before joining.
        Useful when a structure is narrower than the positioning accuracy of the
        scatterers — a bridge deck, for instance.
    predicate : str
        Spatial predicate passed to :func:`geopandas.sjoin`.

    Returns
    -------
    GeoDataFrame
        The points that fell inside a polygon, with *id_column* attached.
    """
    if id_column not in polygons.columns:
        raise KeyError(f"{id_column!r} not found in polygons: {list(polygons.columns)}")

    target = polygons[[id_column, "geometry"]].copy()
    if buffer:
        target["geometry"] = target.geometry.buffer(buffer)

    pts = points.to_crs(target.crs) if points.crs != target.crs else points
    joined = gpd.sjoin(pts, target, predicate=predicate, how="inner")
    return joined.drop(columns=[c for c in ("index_right",) if c in joined.columns])


def aggregate_velocity(points: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame, id_column: str,
                       velocity_column: str = "mean_velocity", buffer: float = 0.0,
                       min_points: int = 1) -> pd.DataFrame:
    """Summarise a per-point velocity field over each polygon.

    Returns one row per polygon — including polygons with no points at all,
    which come back with ``n_points = 0`` and ``NaN`` statistics. Empty
    polygons are the interesting ones in a coverage assessment, so they are
    never silently dropped.

    The ``point_density`` column is in points per square kilometre.
    """
    joined = points_in_polygons(points, polygons, id_column, buffer=buffer)

    if velocity_column not in joined.columns:
        raise KeyError(
            f"{velocity_column!r} not found. Available: "
            f"{[c for c in joined.columns if c != 'geometry']}"
        )

    grouped = joined.groupby(id_column)[velocity_column].agg(
        n_points="count", velocity_mean="mean", velocity_median="median",
        velocity_std="std", velocity_min="min", velocity_max="max",
    )

    out = polygons[[id_column]].copy()
    out = out.merge(grouped, on=id_column, how="left")
    out["n_points"] = out["n_points"].fillna(0).astype(int)

    area_km2 = polygons.geometry.area.to_numpy() / 1e6
    with np.errstate(invalid="ignore", divide="ignore"):
        out["area_km2"] = area_km2
        out["point_density"] = np.where(area_km2 > 0, out["n_points"] / area_km2, np.nan)

    stat_columns = ["velocity_mean", "velocity_median", "velocity_std", "velocity_min", "velocity_max"]
    out.loc[out["n_points"] < min_points, stat_columns] = np.nan
    return out


def aggregate_timeseries(points: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame, id_column: str,
                         buffer: float = 0.0, min_points: int = 1):
    """Average the displacement time series of the points inside each polygon.

    Averaging suppresses the noise of individual scatterers and yields one
    representative series per object, which is what you want before fitting a
    velocity for a landslide body or a structure.

    Returns
    -------
    (summary, series) : (DataFrame, DataFrame)
        *summary* has one row per polygon with the fitted velocity and its
        diagnostics; *series* holds the averaged displacement, polygons as rows
        and acquisition dates as columns.
    """
    columns = date_columns(points)
    if not columns:
        raise ValueError("no acquisition columns found — did you read the full product?")

    joined = points_in_polygons(points, polygons, id_column, buffer=buffer)
    t = dates_as_years(columns)

    series_rows, summary_rows = {}, []
    for feature_id, group in joined.groupby(id_column):
        if len(group) < min_points:
            continue
        mean_series = np.nanmean(displacement_matrix(group, columns), axis=0)
        series_rows[feature_id] = mean_series
        fit = fit_velocity(t, mean_series)
        summary_rows.append({id_column: feature_id, "n_points": len(group), **fit})

    summary = pd.DataFrame(summary_rows)
    series = pd.DataFrame.from_dict(series_rows, orient="index", columns=columns)
    series.index.name = id_column

    if not summary.empty:
        summary = polygons[[id_column]].merge(summary, on=id_column, how="left")
        summary["n_points"] = summary["n_points"].fillna(0).astype(int)
    return summary, series
