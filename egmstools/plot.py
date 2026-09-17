"""Standard figures for EGMS datasets.

Deformation velocity is a diverging quantity: the sign matters as much as the
magnitude, and zero is a meaningful midpoint. These helpers default to a
symmetric colour scale around zero so that a map of subsidence never reads as a
map of uplift because of an autoscaled colour bar.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .io import date_columns, dates_as_datetimes, displacement_matrix
from .timeseries import fit_velocity, dates_as_years

#: Diverging colormap, reversed so that negative (moving away / subsiding) is warm.
VELOCITY_CMAP = "RdBu"


def symmetric_limits(values, percentile: float = 98.0) -> tuple[float, float]:
    """Colour limits centred on zero, clipped to a percentile of ``|values|``.

    A handful of unstable scatterers would otherwise stretch the scale until
    every real signal renders as the same pale colour.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    limit = float(np.percentile(np.abs(finite), percentile))
    limit = limit or float(np.max(np.abs(finite))) or 1.0
    return -limit, limit


def plot_velocity_map(gdf, column: str = "mean_velocity", ax=None, basemap=None,
                      markersize: float = 2.0, percentile: float = 98.0,
                      title: str | None = None, cbar_label: str = "LOS velocity (mm/yr)"):
    """Scatter the measurement points, coloured by velocity.

    Parameters
    ----------
    gdf : GeoDataFrame
        EGMS points.
    column : str
        Velocity column to map.
    basemap : GeoDataFrame, optional
        Context geometry (coastline, province boundary, building footprints)
        drawn underneath in grey.
    percentile : float
        Passed to :func:`symmetric_limits`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    if basemap is not None:
        basemap.to_crs(gdf.crs).plot(ax=ax, facecolor="#f2f2f2", edgecolor="#9a9a9a", linewidth=0.6)

    vmin, vmax = symmetric_limits(gdf[column], percentile)
    gdf.plot(ax=ax, column=column, cmap=VELOCITY_CMAP, vmin=vmin, vmax=vmax,
             markersize=markersize, legend=True,
             legend_kwds={"label": cbar_label, "shrink": 0.6})

    ax.set_title(title or f"{column} ({len(gdf):,} points)")
    ax.set_axis_off()
    ax.set_aspect("equal")
    return ax


def plot_timeseries(series, dates=None, ax=None, label: str | None = None,
                    show_fit: bool = True, color: str = "#0072B2"):
    """Plot one displacement time series, optionally with its linear fit.

    *series* may be a 1-D array of displacements with *dates* supplied
    separately, or a pandas Series whose index holds the acquisition dates.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    if dates is None:
        if not hasattr(series, "index"):
            raise ValueError("pass `dates`, or a pandas Series indexed by date")
        dates = dates_as_datetimes(series.index)

    values = np.asarray(series, dtype=float)
    ax.plot(dates, values, ".", color=color, markersize=4, alpha=0.75, label=label)

    if show_fit:
        t = np.array([(d - dates[0]).days / 365.25 for d in dates], dtype=float)
        fit = fit_velocity(t, values)
        if np.isfinite(fit["velocity"]):
            ax.plot(dates, fit["velocity"] * t + fit["intercept"], "-", color="#D55E00", linewidth=1.8,
                    label=f"{fit['velocity']:+.1f} ± {fit['velocity_std']:.1f} mm/yr")

    ax.axhline(0, color="#999999", linewidth=0.8, zorder=0)
    ax.set_xlabel("Date")
    ax.set_ylabel("LOS displacement (mm)")
    ax.grid(alpha=0.25)
    if label or show_fit:
        ax.legend(frameon=False, fontsize=9)
    return ax


def plot_timeseries_envelope(gdf, ax=None, color: str = "#0072B2", label: str | None = None):
    """Plot the mean displacement of a group of points with a ±1 sigma band.

    Use this for an aggregated object — a landslide body, a building — where the
    spread between scatterers is itself informative: a tight band means the
    object is moving coherently, a wide one means it is deforming internally or
    the points are noisy.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    columns = date_columns(gdf)
    matrix = displacement_matrix(gdf, columns)
    dates = dates_as_datetimes(columns)

    mean = np.nanmean(matrix, axis=0)
    spread = np.nanstd(matrix, axis=0)

    ax.fill_between(dates, mean - spread, mean + spread, color=color, alpha=0.2, linewidth=0)
    ax.plot(dates, mean, "-", color=color, linewidth=1.6,
            label=label or f"mean of {len(gdf)} points")
    ax.axhline(0, color="#999999", linewidth=0.8, zorder=0)
    ax.set_xlabel("Date")
    ax.set_ylabel("LOS displacement (mm)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    return ax


def plot_velocity_histogram(gdf, column: str = "mean_velocity", ax=None, bins: int = 60):
    """Histogram of the velocity field, with the stable-ground peak marked.

    A well-calibrated EGMS tile has its mode at approximately zero. A clearly
    offset peak usually points to a reference-frame problem rather than to
    regional ground motion, and is worth checking before interpreting anything.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    values = np.asarray(gdf[column], dtype=float)
    values = values[np.isfinite(values)]
    counts, edges, _ = ax.hist(values, bins=bins, color="#0072B2", alpha=0.85)

    mode = 0.5 * (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1])
    ax.axvline(0, color="#999999", linewidth=1.0)
    ax.axvline(mode, color="#D55E00", linewidth=1.4, linestyle="--", label=f"mode {mode:+.2f} mm/yr")

    ax.set_xlabel(f"{column} (mm/yr)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    return ax
