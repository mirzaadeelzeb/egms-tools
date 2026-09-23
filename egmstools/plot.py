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


def publication_map(gdf, column: str = "mean_velocity", bbox=None, vmax: float | None = None,
                    title: str | None = None, subtitle: str | None = None,
                    labels=None, credit: str | None = None, figsize=(12, 8.4), dpi: int = 200,
                    point_size: float = 0.5, cmap: str = "RdBu"):
    """Build a finished, client-ready velocity map.

    :func:`plot_velocity_map` draws points onto an axis you own. This assembles a
    whole figure — title block, land-mark labels, a colour bar that says which
    way is down, and a data credit line — so that the output can be handed over
    without further editing.

    Parameters
    ----------
    gdf : GeoDataFrame
        EGMS points. Needs ``latitude``/``longitude`` columns, which every EGMS
        product carries.
    bbox : tuple, optional
        ``(min_lon, min_lat, max_lon, max_lat)`` to crop to.
    vmax : float, optional
        Colour scale runs from ``-vmax`` to ``+vmax``. Left out, it is taken
        from the 98th percentile of ``|values|``, rounded up to a whole mm.

        Choose this deliberately. A scale much wider than the data flattens
        everything to one colour; a scale much tighter makes near-stable ground
        look alarming. State the value in the caption either way.
    labels : sequence, optional
        ``(name, lon, lat)`` triples, or ``(name, lon, lat, below)`` to put the
        text under the marker instead of above it.
    credit : str, optional
        Footer line. Defaults to the EGMS attribution, which the licence expects
        you to keep.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import numpy as np

    data = gdf
    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        data = data[data.longitude.between(min_lon, max_lon)
                    & data.latitude.between(min_lat, max_lat)]
    if data.empty:
        raise ValueError("no points inside bbox — check the order: (min_lon, min_lat, max_lon, max_lat)")

    values = np.asarray(data[column], dtype=float)
    if vmax is None:
        finite = values[np.isfinite(values)]
        vmax = float(np.ceil(np.percentile(np.abs(finite), 98))) if finite.size else 1.0
        vmax = max(vmax, 1.0)

    # Draw the strongest signals last so that a dense stable background cannot
    # bury the very thing the map exists to show.
    order = np.argsort(np.abs(values))
    data = data.iloc[order]

    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.075, 0.115, 0.80, 0.755])
    ax.set_facecolor("#efefef")

    sc = ax.scatter(data.longitude, data.latitude, c=data[column], cmap=cmap,
                    vmin=-vmax, vmax=vmax, s=point_size, linewidths=0, rasterized=True)

    cax = fig.add_axes([0.885, 0.20, 0.019, 0.58])
    cb = fig.colorbar(sc, cax=cax, extend="both")
    cb.set_label("Velocity along satellite line of sight  (mm/year)\n"
                 "red = moving away        white = stable        blue = moving towards",
                 fontsize=9.8, labelpad=12)
    cb.ax.tick_params(labelsize=9.5)

    for label in labels or []:
        name, lon, lat = label[0], label[1], label[2]
        below = label[3] if len(label) > 3 else False
        ax.plot(lon, lat, "o", ms=5, mfc="none", mec="black", mew=1.3, zorder=5)
        ax.annotate(name, (lon, lat), xytext=(0, -18 if below else 12),
                    textcoords="offset points", ha="center", fontsize=11,
                    fontweight="bold", zorder=6)

    if title:
        fig.text(0.075, 0.945, title, fontsize=15, fontweight="bold", ha="left")
    if subtitle:
        fig.text(0.075, 0.905, subtitle, fontsize=10.2, color="#4a4a4a", ha="left")

    ax.set_xlabel("Longitude (\u00b0E)", fontsize=10.5)
    ax.set_ylabel("Latitude (\u00b0N)", fontsize=10.5)
    ax.tick_params(labelsize=9.5)
    # Degrees of longitude shrink with latitude; without this the map is stretched.
    ax.set_aspect(1 / np.cos(np.radians(float(data.latitude.mean()))))
    ax.grid(alpha=0.14, linewidth=0.5)

    fig.text(0.075, 0.038,
             credit or "Data: Copernicus European Ground Motion Service (EGMS) \u2014 free and open",
             fontsize=8.8, color="#555555", ha="left")
    return fig
