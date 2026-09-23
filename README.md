# egms-tools

**Reading, analysing and decomposing European Ground Motion Service (EGMS) InSAR data.**

[EGMS](https://egms.land.copernicus.eu/) publishes millimetre-accuracy ground-motion measurements for the whole of Europe, updated annually and free to use. The data are easy to download and surprisingly fiddly to work with: one column per acquisition date, a projected CRS most desktop GIS does not default to, and line-of-sight values that mean little on their own until two viewing geometries are combined.

This package handles that groundwork, so the analysis can start sooner.

```python
import egmstools as egms

points = egms.read_egms("EGMS_L2b_117_0239_IW3_VV_2019_2023_1.csv")
summary, series = egms.aggregate_timeseries(points, landslides, "landslide_id")

summary.sort_values("velocity").head()
#    landslide_id  n_points  velocity  velocity_std  r_squared   n
#    F_0421              47     -18.3           1.1       0.94  243
```

---

## Why this exists

Every EGMS project starts by rewriting the same four things: work out which columns are dates, get the points into the right projection, average them over something meaningful, and turn two line-of-sight rates into vertical and horizontal motion. That last step is where sign conventions quietly go wrong — a flipped heading turns subsidence into uplift, and nothing in the output looks unusual.

This library does those four things once, with tests.

---

## Installation

```bash
pip install git+https://github.com/mirzaadeelzeb/egms-tools.git
```

Or for development:

```bash
git clone https://github.com/mirzaadeelzeb/egms-tools.git
cd egms-tools
pip install -e ".[test]"
pytest
```

Requires Python 3.9+, NumPy, pandas, GeoPandas, Shapely and Matplotlib.

---

## What it does

### Reading

`read_egms()` accepts EGMS CSVs as downloaded, or shapefiles and GeoPackages if you have already converted them. It finds the coordinate columns, builds point geometries, and returns a GeoDataFrame in EPSG:3035 — the EGMS native grid.

```python
points = egms.read_egms("EGMS_L3_E46N20_100km_U_2019_2023_1.csv")
egms.summarise(points)
# {'n_points': 412_883, 'crs': 'EPSG:3035', 'n_acquisitions': 307,
#  'first_acquisition': '2019-01-05', 'last_acquisition': '2023-12-26',
#  'median_revisit_days': 6.0, 'columns': ['pid', 'easting', ...]}
```

Acquisition columns are detected by name (`20190105`), including the `D20190105` form that shapefile writers produce because DBF field names cannot start with a digit.

### Velocities

EGMS ships a `mean_velocity` per point, but fitting your own gives you the uncertainty alongside the slope, lets you restrict the analysis to a sub-period, and works on a series you have aggregated yourself.

```python
fits = egms.fit_velocities(points)
# velocity, velocity_std, intercept, rmse, r_squared, n
```

`detect_breakpoint()` screens a series for a two-segment split — the signature of a landslide reactivating, or a structure that starts settling after nearby works.

```python
egms.detect_breakpoint(t, displacement)
# {'time': 2.09, 'velocity_before': -1.2, 'velocity_after': -9.4,
#  'velocity_change': -8.2, ...}
```

### Aggregating onto objects

Applied questions are asked about an object, not a scatterer: is *this* landslide moving, is *this* viaduct settling.

```python
table = egms.aggregate_velocity(points, bridges, "bridge_id", buffer=30)
```

Polygons containing no measurement points come back with `n_points = 0` rather than being dropped. In a coverage assessment those are the interesting rows, and silently losing them is how a study ends up describing only the places that happened to be observable.

The `buffer` argument dilates the polygons first, which matters for anything narrower than the positioning accuracy of the scatterers — a bridge deck, for instance.

### Decomposition

A single geometry measures only the projection of motion onto the line of sight. Combining ascending and descending tracks recovers the vertical and east-west components; the north-south component stays unresolved, because near-polar orbits are nearly insensitive to it. This is the same assumption behind the EGMS Level 3 Ortho product — use this when you need it on your own footprint, sub-period or track pair.

```python
up, east = egms.decompose(
    np.vstack([v_ascending, v_descending]),
    incidences=[39.0, 39.0],
    headings=[347.0, 193.0],
)
egms.vertical_fraction(up, east)   # 1.0 = pure settlement, 0.0 = pure horizontal
```

Two geometries that are too similar cannot separate the components. Rather than returning a large spurious number, the solver checks the conditioning of the system and returns `NaN`:

```python
egms.decompose([-5.0, -5.1], [39.0, 39.5], [347.0, 347.5])
# (nan, nan)  — two ascending tracks cannot resolve this
```

**Sign convention.** Displacements are positive towards the satellite, as EGMS distributes them. The unit vector from ground to sensor is `(-sinθ·cosα, sinθ·sinα, cosθ)` for a right-looking sensor, with `α` the heading clockwise from north. For Sentinel-1 this makes ascending negatively sensitive to eastward motion and descending positively so, because the ascending swath lies east of its ground track. If your metadata uses a different heading convention, check the sign of `e_east` for a known geometry first.

### Plotting

```python
egms.plot_velocity_map(points, basemap=province)
egms.plot_timeseries_envelope(points_on_landslide)
egms.plot_velocity_histogram(points)
```

Velocity is a diverging quantity where zero is meaningful, so the colour scale is symmetric about zero by default and clipped at the 98th percentile of `|v|` — otherwise a handful of unstable scatterers stretch the scale until every real signal renders the same pale colour.

`plot_velocity_histogram` marks the mode of the distribution. A well-referenced tile peaks near zero; a clearly offset peak usually means a reference-frame problem rather than regional ground motion, and is worth resolving before interpreting anything.

---

## Command line

```bash
egms info EGMS_L2b_117_0239_IW3_VV_2019_2023_1.csv
egms velocity tile.csv -o velocities.csv
egms aggregate tile.csv bridges.gpkg bridge_id --buffer 30 -o bridges_motion.csv
egms decompose ascending.csv descending.csv -o decomposed.csv
```

`egms map` renders a finished figure — title block, place labels, a colour bar
that says which way is down, and the data credit — ready to hand to a client
without further editing:

```bash
egms map tile.csv -o naples.png   --bbox 14.19 40.78 14.47 40.93   --vmax 6   --title "Ground motion around Naples, 2020-2024"   --subtitle "Sentinel-1 ascending track 044, 211 passes"   --label "NAPLES:14.268,40.851" --label "VESUVIUS:14.426,40.821"
```

Set `--vmax` deliberately. A scale much wider than the data flattens everything
to one colour; a scale much tighter makes near-stable ground look alarming.
Whatever you choose, state it in the caption.

---

## Testing

```bash
pytest
```

25 tests, no downloads required: `tests/synthetic.py` generates data with EGMS's structure — column naming, EPSG:3035, 6-day revisit, millimetre units — and known ground truth.

The decomposition tests matter most, because a sign error there is invisible in a plot but inverts every conclusion. They work by round trip: take a known vertical and east-west motion, project it onto both geometries, decompose it back, and require the original values to return.

**Validation against real EGMS products** is a separate step, since the files are large and licensed and so do not belong in a repository. `examples/validate_on_real_data.py` runs that check — it fits velocities from the time series and compares them against the `mean_velocity` EGMS publishes alongside:

```bash
python examples/validate_on_real_data.py EGMS_L2b_044_0240_IW2_VV_2020_2024_1.csv
```

This has been run. On EGMS L2b ascending track 044 (swath IW2, 2020–2024, 1,770,196 points, 211 acquisitions over the Naples area), velocities fitted from the time series reproduce the published `mean_velocity` with a **correlation of 0.998** and a **mean absolute difference of 0.06 mm/yr**.

The decomposition geometry was checked against the same product. EGMS ships per-point `los_east`, `los_north` and `los_up` components, so `los_unit_vector()` can be compared directly against the authoritative values rather than argued about:

| component | `los_unit_vector()` | EGMS published | max abs difference |
|---|---|---|---|
| east | −0.5889 | −0.5889 | 0.0006 |
| north | −0.1025 | −0.1025 | 0.0005 |
| up | +0.8016 | +0.8016 | 0.0006 |

The residual is the rounding in EGMS's own three-decimal fields. The sign convention documented above is therefore confirmed, not assumed — including the negative east sensitivity of an ascending track.

---

## Scope

This is deliberately a toolkit, not a framework. It does not process interferograms, run a PS-InSAR chain, or implement any particular hazard-assessment method — it handles the layer between an EGMS download and whatever you are actually trying to find out.

Related work it pairs with: landslide susceptibility modelling in [caserta-landslide-ml](https://github.com/mirzaadeelzeb/caserta-landslide-ml).

---

## Author

**Mirza Adeel Zeb** — PhD researcher, Department of Architecture and Industrial Design, University of Campania "Luigi Vanvitelli", Aversa, Italy.

Research focus: PS-InSAR and EGMS ground-motion analysis, landslide susceptibility, and infrastructure risk assessment. Peer-reviewed work using these methods:

- Zeb, M.A., Bencivenga, P., Zizi, M., De Matteis, G. (2026). Impact of land subsidence on bridges and viaducts: a comprehensive evaluation using remote sensing, GIS, and PS-InSAR techniques. *Procedia Structural Integrity*, 84, 256–263. [doi:10.1016/j.prostr.2026.06.034](https://doi.org/10.1016/j.prostr.2026.06.034)
- Zeb, M.A., Bencivenga, P., Zizi, M., De Matteis, G. (2026). Assessing landslide susceptibility for Italian bridges: a case study in the Caserta province. *Procedia Structural Integrity*, 84, 248–255. [doi:10.1016/j.prostr.2026.06.033](https://doi.org/10.1016/j.prostr.2026.06.033)

[ResearchGate](https://www.researchgate.net/profile/Mirza-Zeb) · [LinkedIn](https://linkedin.com/in/mirza-adeel-zeb-98958a153)

---

## License

MIT. EGMS data itself is distributed by Copernicus under its own terms — see the [EGMS portal](https://egms.land.copernicus.eu/).
