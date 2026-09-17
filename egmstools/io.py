"""Reading European Ground Motion Service (EGMS) products.

EGMS ships three product levels, all of which share the same broad shape: one
row per measurement point, a handful of metadata columns, and one column per
SAR acquisition date named ``YYYYMMDD`` holding displacement in millimetres.

============  =====================================  ==========================
Level         Content                                Typical geometry columns
============  =====================================  ==========================
L2a / L2b     Calibrated line-of-sight (per track)   ``easting``/``northing``
L3            Ortho-rectified Up and East components ``easting``/``northing``
============  =====================================  ==========================

Files are distributed as CSV, but users often convert them to shapefiles or
GeoPackages first. :func:`read_egms` accepts any of these and always returns a
:class:`geopandas.GeoDataFrame` in EPSG:3035 (ETRS89-LAEA), the EGMS native
grid, unless a different ``crs`` is requested.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

EGMS_CRS = 3035

#: Columns EGMS uses for point coordinates, in the order they are looked for.
_COORD_CANDIDATES = [("easting", "northing"), ("x", "y"), ("lon", "lat"), ("longitude", "latitude")]

_DATE_RE = re.compile(r"^\d{8}$")


def is_date_column(name: str) -> bool:
    """Return ``True`` if *name* looks like an EGMS acquisition column.

    EGMS names these ``YYYYMMDD``. Shapefile writers sometimes prefix them with
    a letter because DBF field names may not start with a digit, so ``D20190105``
    is accepted too.
    """
    name = str(name)
    if _DATE_RE.match(name):
        return True
    return len(name) == 9 and name[0].isalpha() and _DATE_RE.match(name[1:]) is not None


def date_columns(df) -> list[str]:
    """List the acquisition columns of *df*, sorted chronologically."""
    cols = [c for c in df.columns if is_date_column(c)]
    return sorted(cols, key=lambda c: _parse_date(c))


def _parse_date(name: str) -> datetime:
    digits = str(name)[-8:]
    return datetime.strptime(digits, "%Y%m%d")


def dates_as_datetimes(columns) -> pd.DatetimeIndex:
    """Convert acquisition column names to a :class:`~pandas.DatetimeIndex`."""
    return pd.DatetimeIndex([_parse_date(c) for c in columns])


def dates_as_years(columns) -> np.ndarray:
    """Convert acquisition columns to decimal years since the first acquisition.

    This is the x-axis used for velocity fitting, so that fitted slopes come out
    in millimetres per year.
    """
    stamps = dates_as_datetimes(columns)
    origin = stamps[0]
    return np.array([(s - origin).days / 365.25 for s in stamps], dtype=float)


def read_egms(path, columns=None, crs=EGMS_CRS, bbox=None) -> gpd.GeoDataFrame:
    """Read an EGMS product into a :class:`~geopandas.GeoDataFrame`.

    Parameters
    ----------
    path : str or Path
        A ``.csv`` file as downloaded from EGMS, or any vector format that
        GeoPandas can open (``.shp``, ``.gpkg``, ``.parquet`` ...).
    columns : sequence of str, optional
        Restrict the read to these columns. Acquisition columns are large and
        numerous, so passing ``["pid", "mean_velocity"]`` is much faster when
        the time series are not needed. Ignored for CSV input, which pandas
        reads in full.
    crs : int or str, default 3035
        Target CRS. EGMS coordinates are EPSG:3035; anything else triggers a
        reprojection.
    bbox : tuple, optional
        ``(minx, miny, maxx, maxy)`` in *crs* units, applied after reprojection.

    Returns
    -------
    geopandas.GeoDataFrame
        Point geometries with all requested attribute columns preserved.
    """
    path = Path(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        gdf = _read_egms_csv(path, columns=columns)
    else:
        gdf = gpd.read_file(path, columns=columns) if columns else gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs(EGMS_CRS)

    if crs is not None and gdf.crs is not None:
        gdf = gdf.to_crs(crs)
    if bbox is not None:
        minx, miny, maxx, maxy = bbox
        gdf = gdf.cx[minx:maxx, miny:maxy]
    return gdf


def _read_egms_csv(path: Path, columns=None) -> gpd.GeoDataFrame:
    df = pd.read_csv(path, usecols=columns) if columns else pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}
    for cx, cy in _COORD_CANDIDATES:
        if cx in lower and cy in lower:
            xcol, ycol = lower[cx], lower[cy]
            break
    else:
        raise ValueError(
            f"{path.name}: no coordinate columns found. Looked for "
            + ", ".join("/".join(c) for c in _COORD_CANDIDATES)
        )

    geometry = gpd.points_from_xy(df[xcol], df[ycol])
    src_crs = 4326 if xcol.lower() in {"lon", "longitude"} else EGMS_CRS
    return gpd.GeoDataFrame(df, geometry=geometry, crs=src_crs)


def displacement_matrix(gdf, columns=None) -> np.ndarray:
    """Extract the displacement time series as a ``(n_points, n_dates)`` array.

    Non-numeric entries become ``NaN`` rather than raising, because EGMS CSVs
    occasionally carry empty strings for missing acquisitions.
    """
    columns = columns or date_columns(gdf)
    if not columns:
        raise ValueError("no acquisition columns found — is this an EGMS product?")
    block = gdf[list(columns)].apply(pd.to_numeric, errors="coerce")
    return block.to_numpy(dtype=float)


def summarise(gdf) -> dict:
    """Return a short description of an EGMS dataset, for logging and CLI output."""
    cols = date_columns(gdf)
    out = {
        "n_points": int(len(gdf)),
        "crs": str(gdf.crs),
        "n_acquisitions": len(cols),
        "columns": [c for c in gdf.columns if not is_date_column(c) and c != gdf.geometry.name],
    }
    if cols:
        stamps = dates_as_datetimes(cols)
        out["first_acquisition"] = stamps[0].date().isoformat()
        out["last_acquisition"] = stamps[-1].date().isoformat()
        # Convert through timedelta64[h] rather than dividing raw integers: the
        # backing resolution of a DatetimeIndex is not guaranteed to be nanoseconds.
        revisits = np.diff(stamps.to_numpy()).astype("timedelta64[h]").astype(float) / 24.0
        out["median_revisit_days"] = float(np.median(revisits)) if len(revisits) else None
    return out
