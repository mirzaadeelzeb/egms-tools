"""egms-tools — a small toolkit for working with European Ground Motion Service data.

EGMS publishes InSAR ground-motion measurements for the whole of Europe, free to
use. The data are easy to download and surprisingly fiddly to work with: one
column per acquisition date, a projected CRS most desktop GIS does not default
to, and line-of-sight values that mean little until you combine two geometries.

This package handles that groundwork so the analysis can start sooner.

    >>> import egmstools as egms
    >>> points = egms.read_egms("EGMS_L2b_146_0296_IW2_VV_2019_2023_1.csv")
    >>> summary, series = egms.aggregate_timeseries(points, landslides, "id")

See the README for a worked example.
"""

from .aggregate import aggregate_timeseries, aggregate_velocity, points_in_polygons
from .decompose import (
    SENTINEL1_ASCENDING,
    SENTINEL1_DESCENDING,
    decompose,
    horizontal_magnitude,
    los_unit_vector,
    project_to_los,
    vertical_fraction,
)
from .io import (
    EGMS_CRS,
    date_columns,
    dates_as_datetimes,
    dates_as_years,
    displacement_matrix,
    read_egms,
    summarise,
)
from .plot import (
    plot_timeseries,
    plot_timeseries_envelope,
    plot_velocity_histogram,
    plot_velocity_map,
    publication_map,
)
from .timeseries import cumulative_displacement, detect_breakpoint, fit_velocities, fit_velocity

__version__ = "0.1.0"

__all__ = [
    "EGMS_CRS",
    "SENTINEL1_ASCENDING",
    "SENTINEL1_DESCENDING",
    "aggregate_timeseries",
    "aggregate_velocity",
    "cumulative_displacement",
    "date_columns",
    "dates_as_datetimes",
    "dates_as_years",
    "decompose",
    "detect_breakpoint",
    "displacement_matrix",
    "fit_velocities",
    "fit_velocity",
    "horizontal_magnitude",
    "los_unit_vector",
    "plot_timeseries",
    "plot_timeseries_envelope",
    "plot_velocity_histogram",
    "plot_velocity_map",
    "publication_map",
    "points_in_polygons",
    "project_to_los",
    "read_egms",
    "summarise",
    "vertical_fraction",
]
