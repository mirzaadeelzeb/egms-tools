"""Velocity fitting and time-series characterisation for EGMS displacement data.

EGMS already publishes a ``mean_velocity`` per point, but a fitted velocity is
still worth computing: it gives you the uncertainty alongside the slope, it can
be run over an arbitrary sub-period, and it can be applied to a series you have
aggregated yourself (a whole landslide body, a bridge deck, a building block)
rather than to a single scatterer.
"""

from __future__ import annotations

import numpy as np

from .io import dates_as_years, date_columns, displacement_matrix

#: Minimum number of valid acquisitions before a velocity is considered meaningful.
MIN_ACQUISITIONS = 10


def fit_velocity(t, y, min_acquisitions: int = MIN_ACQUISITIONS) -> dict:
    """Fit a linear displacement rate to one time series.

    Parameters
    ----------
    t : array_like
        Time in decimal years (see :func:`egmstools.io.dates_as_years`).
    y : array_like
        Displacement in millimetres. ``NaN`` entries are ignored.
    min_acquisitions : int
        Series with fewer valid samples return ``NaN`` results instead of a
        slope fitted through almost nothing.

    Returns
    -------
    dict
        ``velocity`` (mm/yr), ``velocity_std`` (1-sigma of the slope),
        ``intercept`` (mm), ``rmse`` (mm), ``r_squared`` and ``n`` — the number
        of acquisitions actually used.

    Notes
    -----
    The standard error assumes independent, identically distributed residuals.
    InSAR time series are usually autocorrelated, so treat ``velocity_std`` as
    an optimistic lower bound rather than a calibrated uncertainty.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(t) & np.isfinite(y)
    n = int(valid.sum())

    nan_result = {
        "velocity": np.nan,
        "velocity_std": np.nan,
        "intercept": np.nan,
        "rmse": np.nan,
        "r_squared": np.nan,
        "n": n,
    }
    if n < min_acquisitions:
        return nan_result

    tv, yv = t[valid], y[valid]
    if np.ptp(tv) == 0:
        return nan_result

    design = np.vstack([tv, np.ones_like(tv)]).T
    (slope, intercept), residuals, *_ = np.linalg.lstsq(design, yv, rcond=None)

    fitted = slope * tv + intercept
    resid = yv - fitted
    dof = n - 2
    rmse = float(np.sqrt(np.sum(resid**2) / dof)) if dof > 0 else np.nan

    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    r_squared = float(1 - np.sum(resid**2) / ss_tot) if ss_tot > 0 else np.nan

    if dof > 0:
        sxx = float(np.sum((tv - tv.mean()) ** 2))
        slope_std = float(rmse / np.sqrt(sxx)) if sxx > 0 else np.nan
    else:
        slope_std = np.nan

    return {
        "velocity": float(slope),
        "velocity_std": slope_std,
        "intercept": float(intercept),
        "rmse": rmse,
        "r_squared": r_squared,
        "n": n,
    }


def fit_velocities(gdf, columns=None, min_acquisitions: int = MIN_ACQUISITIONS):
    """Fit a velocity for every point in an EGMS GeoDataFrame.

    Returns a :class:`~pandas.DataFrame` indexed like *gdf* with the columns
    produced by :func:`fit_velocity`.
    """
    import pandas as pd

    columns = columns or date_columns(gdf)
    t = dates_as_years(columns)
    matrix = displacement_matrix(gdf, columns)
    rows = [fit_velocity(t, series, min_acquisitions) for series in matrix]
    return pd.DataFrame(rows, index=gdf.index)


def cumulative_displacement(y) -> float:
    """Total displacement between the first and last valid acquisition, in mm."""
    y = np.asarray(y, dtype=float)
    valid = np.flatnonzero(np.isfinite(y))
    if valid.size < 2:
        return np.nan
    return float(y[valid[-1]] - y[valid[0]])


def detect_breakpoint(t, y, min_segment: int = 8) -> dict:
    """Locate the single best two-segment split of a displacement series.

    A landslide that reactivates, or a structure that starts settling after a
    nearby excavation, produces a time series with two distinct rates. This
    scans every admissible split point and keeps the one minimising the summed
    squared residuals of two independent linear fits.

    Returns a dict with ``index``, ``time``, ``velocity_before``,
    ``velocity_after``, ``velocity_change`` and ``rss``, or ``NaN`` entries if
    the series is too short to split.

    This is a descriptive screening tool, not a significance test — it always
    finds *a* best split. Compare ``rss`` against the single-segment fit before
    concluding that a real change occurred.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(t) & np.isfinite(y)
    tv, yv = t[valid], y[valid]

    empty = {
        "index": -1,
        "time": np.nan,
        "velocity_before": np.nan,
        "velocity_after": np.nan,
        "velocity_change": np.nan,
        "rss": np.nan,
    }
    if tv.size < 2 * min_segment:
        return empty

    best = None
    for split in range(min_segment, tv.size - min_segment + 1):
        rss = _segment_rss(tv[:split], yv[:split]) + _segment_rss(tv[split:], yv[split:])
        if best is None or rss < best[0]:
            best = (rss, split)

    rss, split = best
    before = fit_velocity(tv[:split], yv[:split], min_acquisitions=min_segment)
    after = fit_velocity(tv[split:], yv[split:], min_acquisitions=min_segment)
    return {
        "index": int(split),
        "time": float(tv[split]),
        "velocity_before": before["velocity"],
        "velocity_after": after["velocity"],
        "velocity_change": after["velocity"] - before["velocity"],
        "rss": float(rss),
    }


def _segment_rss(t, y) -> float:
    design = np.vstack([t, np.ones_like(t)]).T
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coeffs
    return float(np.sum(resid**2))
