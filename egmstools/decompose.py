"""Decomposition of line-of-sight rates into vertical and east-west components.

A single SAR geometry only measures the projection of ground motion onto the
line of sight. Combining an ascending and a descending track recovers two of the
three displacement components. The north-south component stays unresolved:
near-polar orbits are almost insensitive to it, and including it makes the
system ill-conditioned, so it is conventionally assumed to be zero.

This is the same assumption behind the EGMS Level 3 Ortho product. Use this
module when you need the decomposition on your own footprint — a sub-period, a
set of aggregated polygons, or tracks that L3 does not cover.

Sign convention
---------------
Displacements are positive **towards the satellite**, which is how EGMS
distributes them. The unit vector points from the ground target to the sensor::

    e_east  = -sin(theta) * cos(alpha)
    e_north =  sin(theta) * sin(alpha)
    e_up    =  cos(theta)

with ``theta`` the incidence angle and ``alpha`` the satellite heading, measured
clockwise from north, for a right-looking sensor. For Sentinel-1 this gives an
ascending track negative east-sensitivity and a descending track positive
east-sensitivity, since the ascending swath lies east of its ground track.

If your metadata uses a different heading convention, check the sign of
``e_east`` for a known geometry before trusting the output.
"""

from __future__ import annotations

import numpy as np

#: Nominal Sentinel-1 geometry, useful for quick looks when per-point angles
#: are unavailable. Always prefer the real per-point values when you have them.
SENTINEL1_ASCENDING = {"incidence": 39.0, "heading": 347.0}
SENTINEL1_DESCENDING = {"incidence": 39.0, "heading": 193.0}


def los_unit_vector(incidence_deg, heading_deg):
    """Unit vector from ground target to satellite, as ``(east, north, up)``.

    Accepts scalars or arrays; returns arrays broadcast to a common shape.
    """
    theta = np.radians(np.asarray(incidence_deg, dtype=float))
    alpha = np.radians(np.asarray(heading_deg, dtype=float))
    east = -np.sin(theta) * np.cos(alpha)
    north = np.sin(theta) * np.sin(alpha)
    up = np.cos(theta) * np.ones_like(east)
    return east, north, up


def project_to_los(up, east, incidence_deg, heading_deg, north=0.0):
    """Project a 3-D displacement onto one line of sight.

    The inverse of :func:`decompose`, and the basis of its round-trip test.
    """
    e_e, e_n, e_u = los_unit_vector(incidence_deg, heading_deg)
    return np.asarray(up) * e_u + np.asarray(east) * e_e + np.asarray(north) * e_n


def decompose(los_values, incidences, headings, min_conditioning: float = 0.1):
    """Solve for vertical and east-west motion from two or more geometries.

    Parameters
    ----------
    los_values : array_like, shape (n_geometries,) or (n_geometries, n_points)
        Line-of-sight displacement or velocity, positive towards the satellite.
        Every geometry must be sampled at the same locations.
    incidences, headings : array_like, shape (n_geometries,)
        Incidence and heading angles in degrees, one per geometry.
    min_conditioning : float
        Reject the solution where ``|det|`` of the 2x2 design matrix falls below
        this value. Two geometries that are too similar — two ascending tracks,
        say — cannot separate the components, and the naive solution explodes.
        Such points come back as ``NaN`` rather than as large spurious motion.

    Returns
    -------
    (up, east) : tuple of ndarray
        Vertical (positive up) and east-west (positive east) components, in the
        units of *los_values*.

    Raises
    ------
    ValueError
        If fewer than two geometries are supplied, or the shapes disagree.
    """
    los = np.asarray(los_values, dtype=float)
    incidences = np.atleast_1d(np.asarray(incidences, dtype=float))
    headings = np.atleast_1d(np.asarray(headings, dtype=float))

    # A 1-D input is one value per geometry, i.e. a single location — not a
    # single geometry with several points. Make that an (n_geometries, 1) column
    # so the solve below is shape-agnostic.
    single_location = los.ndim == 1
    if single_location:
        los = los[:, None]

    n_geom = los.shape[0]
    if n_geom < 2:
        raise ValueError("decomposition needs at least two geometries")
    if incidences.size != n_geom or headings.size != n_geom:
        raise ValueError(
            f"got {n_geom} LOS rows but {incidences.size} incidences and "
            f"{headings.size} headings — one angle per geometry is required"
        )

    e_e, _, e_u = los_unit_vector(incidences, headings)
    design = np.column_stack([e_u, e_e])  # columns: up, east

    if n_geom == 2:
        det = design[0, 0] * design[1, 1] - design[0, 1] * design[1, 0]
        if abs(det) < min_conditioning:
            nan = np.full(los.shape[1:], np.nan)
            return (float("nan"), float("nan")) if single_location else (nan, nan)
        solution = np.linalg.solve(design, los)
    else:
        solution, *_ = np.linalg.lstsq(design, los, rcond=None)

    up, east = solution[0], solution[1]
    if single_location:
        return float(up[0]), float(east[0])
    return up, east


def horizontal_magnitude(east, north=0.0):
    """Magnitude of the horizontal motion.

    With only ascending and descending geometries *north* is unresolved and left
    at zero, so this reduces to ``|east|``. It is kept as a separate function so
    that three-geometry solutions get the right answer without a code change.
    """
    return np.hypot(np.asarray(east, dtype=float), np.asarray(north, dtype=float))


def vertical_fraction(up, east, north=0.0):
    """Share of the total motion carried by the vertical component, in ``[0, 1]``.

    Values near 1 indicate settlement or uplift; values near 0 indicate motion
    that is essentially horizontal, which on a slope usually means sliding
    rather than subsidence. Returns ``NaN`` where the total magnitude is zero.
    """
    up = np.asarray(up, dtype=float)
    total = np.sqrt(up**2 + np.asarray(east, dtype=float) ** 2 + np.asarray(north, dtype=float) ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, np.abs(up) / total, np.nan)
