"""Tests for egms-tools.

The decomposition tests matter most: a sign error there is invisible in a plot
but silently inverts every conclusion about whether ground is rising or sinking.
"""

from __future__ import annotations

import numpy as np
import pytest

import egmstools as egms
from tests import synthetic

# --------------------------------------------------------------------------- io


def test_date_columns_found_and_sorted():
    points = synthetic.make_points(n_points=5, n_dates=20)
    cols = egms.date_columns(points)
    assert len(cols) == 20
    assert cols == sorted(cols)
    assert "pid" not in cols and "mean_velocity" not in cols


def test_shapefile_style_date_columns_are_recognised():
    from egmstools.io import is_date_column

    assert is_date_column("20190105")
    assert is_date_column("D20190105")   # DBF-safe variant
    assert not is_date_column("mean_velocity")
    assert not is_date_column("2019010")  # too short


def test_dates_as_years_starts_at_zero_and_matches_revisit():
    points = synthetic.make_points(n_points=3, n_dates=10)
    years = egms.dates_as_years(egms.date_columns(points))
    assert years[0] == 0.0
    assert np.isclose(years[1], 6 / 365.25)
    assert np.all(np.diff(years) > 0)


def test_read_csv_roundtrip(tmp_path):
    original = synthetic.make_points(n_points=25, n_dates=12)
    path = synthetic.write_csv(original, tmp_path / "EGMS_test.csv")

    loaded = egms.read_egms(path)
    assert len(loaded) == 25
    assert loaded.crs.to_epsg() == egms.EGMS_CRS
    assert len(egms.date_columns(loaded)) == 12
    assert np.allclose(loaded.geometry.x, original.geometry.x)


def test_read_csv_without_coordinates_raises(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("pid,mean_velocity\nPS_1,-2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no coordinate columns"):
        egms.read_egms(path)


def test_summarise_reports_period_and_revisit():
    points = synthetic.make_points(n_points=10, n_dates=50)
    info = egms.summarise(points)
    assert info["n_points"] == 10
    assert info["n_acquisitions"] == 50
    assert info["first_acquisition"] == "2019-01-05"
    assert np.isclose(info["median_revisit_days"], 6.0)


# ------------------------------------------------------------------- timeseries


def test_fit_velocity_recovers_known_rate():
    t = np.linspace(0, 4, 200)
    y = -7.5 * t + np.random.default_rng(0).normal(0, 1.0, t.size)
    fit = egms.fit_velocity(t, y)
    assert fit["velocity"] == pytest.approx(-7.5, abs=0.2)
    assert fit["n"] == 200
    assert fit["r_squared"] > 0.95


def test_fit_velocity_ignores_nans_but_respects_minimum():
    t = np.linspace(0, 4, 50)
    y = 3.0 * t
    y[:45] = np.nan
    assert np.isnan(egms.fit_velocity(t, y)["velocity"])       # only 5 valid
    assert egms.fit_velocity(t, y, min_acquisitions=3)["velocity"] == pytest.approx(3.0)


def test_fit_velocities_over_a_dataset():
    points = synthetic.make_points(n_points=40, velocity=-4.0, noise_mm=1.0)
    fits = egms.fit_velocities(points)
    assert len(fits) == 40
    assert fits["velocity"].mean() == pytest.approx(-4.0, abs=0.3)


def test_cumulative_displacement():
    assert egms.cumulative_displacement([0.0, 1.0, np.nan, 5.0]) == pytest.approx(5.0)
    assert np.isnan(egms.cumulative_displacement([np.nan, 2.0]))


def test_detect_breakpoint_finds_a_rate_change():
    t = np.linspace(0, 4, 120)
    y = np.where(t < 2, -1.0 * t, -1.0 * 2 + -9.0 * (t - 2))
    result = egms.detect_breakpoint(t, y)
    assert result["time"] == pytest.approx(2.0, abs=0.15)
    assert result["velocity_before"] == pytest.approx(-1.0, abs=0.3)
    assert result["velocity_after"] == pytest.approx(-9.0, abs=0.3)
    assert result["velocity_change"] < 0


def test_detect_breakpoint_on_short_series_returns_nan():
    result = egms.detect_breakpoint(np.arange(5.0), np.arange(5.0))
    assert result["index"] == -1
    assert np.isnan(result["time"])


# ------------------------------------------------------------------ decompose


def test_los_unit_vector_is_normalised():
    east, north, up = egms.los_unit_vector(39.0, 347.0)
    assert np.hypot(np.hypot(east, north), up) == pytest.approx(1.0)


def test_sentinel1_east_sensitivity_has_opposite_signs():
    """Ascending looks east of its track, descending west — so the east
    sensitivities must have opposite sign, or decomposition cannot work."""
    asc_e, _, _ = egms.los_unit_vector(**{"incidence_deg": 39.0, "heading_deg": 347.0})
    dsc_e, _, _ = egms.los_unit_vector(incidence_deg=39.0, heading_deg=193.0)
    assert asc_e * dsc_e < 0
    assert asc_e == pytest.approx(-dsc_e, abs=1e-6)


def test_decompose_roundtrip_recovers_components():
    """Project a known (up, east) motion onto both geometries, then invert it."""
    up_true, east_true = -6.0, 3.5
    asc, dsc = egms.SENTINEL1_ASCENDING, egms.SENTINEL1_DESCENDING

    los_asc = egms.project_to_los(up_true, east_true, asc["incidence"], asc["heading"])
    los_dsc = egms.project_to_los(up_true, east_true, dsc["incidence"], dsc["heading"])

    up, east = egms.decompose(
        [los_asc, los_dsc],
        [asc["incidence"], dsc["incidence"]],
        [asc["heading"], dsc["heading"]],
    )
    assert up == pytest.approx(up_true, abs=1e-8)
    assert east == pytest.approx(east_true, abs=1e-8)


def test_decompose_roundtrip_vectorised():
    rng = np.random.default_rng(7)
    up_true = rng.normal(0, 5, 500)
    east_true = rng.normal(0, 5, 500)
    asc, dsc = egms.SENTINEL1_ASCENDING, egms.SENTINEL1_DESCENDING

    los = np.vstack([
        egms.project_to_los(up_true, east_true, asc["incidence"], asc["heading"]),
        egms.project_to_los(up_true, east_true, dsc["incidence"], dsc["heading"]),
    ])
    up, east = egms.decompose(los, [asc["incidence"], dsc["incidence"]],
                              [asc["heading"], dsc["heading"]])
    assert np.allclose(up, up_true)
    assert np.allclose(east, east_true)


def test_pure_subsidence_decomposes_to_vertical_only():
    asc, dsc = egms.SENTINEL1_ASCENDING, egms.SENTINEL1_DESCENDING
    los = [egms.project_to_los(-10.0, 0.0, asc["incidence"], asc["heading"]),
           egms.project_to_los(-10.0, 0.0, dsc["incidence"], dsc["heading"])]
    up, east = egms.decompose(los, [asc["incidence"], dsc["incidence"]],
                              [asc["heading"], dsc["heading"]])
    assert up == pytest.approx(-10.0)
    assert east == pytest.approx(0.0, abs=1e-9)
    assert egms.vertical_fraction(up, east) == pytest.approx(1.0)


def test_two_similar_geometries_return_nan():
    """Two ascending tracks cannot separate up from east; that must fail loudly
    as NaN rather than quietly as a huge number."""
    up, east = egms.decompose([-5.0, -5.1], [39.0, 39.5], [347.0, 347.5])
    assert np.isnan(up) and np.isnan(east)


def test_decompose_rejects_single_geometry():
    with pytest.raises(ValueError, match="at least two geometries"):
        egms.decompose([1.0], [39.0], [347.0])


def test_decompose_rejects_mismatched_angles():
    with pytest.raises(ValueError, match="one angle per geometry"):
        egms.decompose([[1.0], [2.0]], [39.0], [347.0, 193.0])


def test_vertical_fraction_of_horizontal_motion_is_zero():
    assert egms.vertical_fraction(0.0, 4.0) == pytest.approx(0.0)
    assert egms.horizontal_magnitude(3.0, 4.0) == pytest.approx(5.0)


# ------------------------------------------------------------------ aggregate


def test_aggregate_velocity_covers_every_polygon():
    points = synthetic.make_points(n_points=300, velocity=-6.0, spread=150.0)
    polygons = synthetic.make_polygons()

    result = egms.aggregate_velocity(points, polygons, "site_id")

    assert list(result["site_id"]) == ["A", "B"]
    site_a = result.set_index("site_id").loc["A"]
    site_b = result.set_index("site_id").loc["B"]

    assert site_a["n_points"] > 0
    assert site_a["velocity_mean"] == pytest.approx(-6.0, abs=0.3)
    assert site_a["point_density"] > 0

    # Polygon B is empty — it must still appear, with zero points.
    assert site_b["n_points"] == 0
    assert np.isnan(site_b["velocity_mean"])


def test_aggregate_velocity_missing_column_raises():
    points = synthetic.make_points(n_points=20)
    polygons = synthetic.make_polygons()
    with pytest.raises(KeyError, match="not found"):
        egms.aggregate_velocity(points, polygons, "site_id", velocity_column="v_los")


def test_aggregate_timeseries_averages_and_fits():
    points = synthetic.make_points(n_points=200, velocity=-8.0, noise_mm=2.0, spread=150.0)
    polygons = synthetic.make_polygons()

    summary, series = egms.aggregate_timeseries(points, polygons, "site_id")

    row = summary.set_index("site_id").loc["A"]
    assert row["n_points"] > 50
    assert row["velocity"] == pytest.approx(-8.0, abs=0.4)
    assert series.loc["A"].notna().all()


def test_buffer_captures_points_outside_a_thin_polygon():
    points = synthetic.make_points(n_points=200, spread=300.0)
    polygons = synthetic.make_polygons(size=20.0)  # far too small to catch much

    without = egms.aggregate_velocity(points, polygons, "site_id")
    with_buffer = egms.aggregate_velocity(points, polygons, "site_id", buffer=200.0)

    assert with_buffer.iloc[0]["n_points"] > without.iloc[0]["n_points"]
