# compass-web API Reference

Complete function and usage reference for all public symbols in the `compass_web` library.

---

## Table of Contents

- [Module Overview](#module-overview)
- [Data Files](#data-files)
- [compass_web.config](#compass_webconfigpy)
- [compass_web.smoothing](#compass_websmoothingpy)
- [compass_web.pipeline](#compass_webpipelinepy)
- [compass_web.visualization](#compass_webvisualizationpy)
- [compass_web.lofted_surface_voronoi](#compass_weblofted_surface_voronoipy)
- [CLI — `materialize`](#cli--materialize)
- [Config File Format](#config-file-format)
- [Constants Reference](#constants-reference)

---

## Module Overview

```
src/compass_web/
├── config.py                  # PipelineConfig dataclass and JSON I/O
├── smoothing.py               # Radii/spacing smoothing to prevent degenerate lofts
├── pipeline.py                # End-to-end orchestrated pipeline
├── visualization.py           # PyVista scene helpers and screenshot utilities
├── lofted_surface_voronoi.py  # Core geometry: loft, voronoi, intersect, align, extrude
└── cli.py                     # Typer CLI (entrypoint: materialize)
```

---

## Data Files

Two JSON files drive default inputs. These live in `data/` and are loaded by `load_pipeline_config()`.

### `data/lofted_surface_inputs.json`

Controls the lofted surface shape.

| Key | Type | Description |
|-----|------|-------------|
| `radii` | `list[float]` | Exactly 8 circle radii, each 5–75 units, max 3 decimal places |
| `z_increment` | `float` | Default Z spacing between rings (mm-scale) |
| `circle_resolution` | `int` | Vertices per circle; default `120` (min 8) |
| `bbox_padding` | `float` | Padding around the half-surface bbox for voronoi seeds; default `4.0` |
| `line_tolerance` | `float` | Snap/merge tolerance for polyline operations; default `0.001` |

### `data/voronoi_points_inputs.json`

Controls voronoi seed point generation.

| Key | Type | Description |
|-----|------|-------------|
| `seed_count` | `int` | Number of voronoi seed points (≥ 2) |
| `random_seed` | `int` | RNG seed for reproducible layouts |

---

## `compass_web/config.py`

Pipeline configuration: frozen dataclass, JSON I/O, and geometry limit validation.

### Constants

| Name | Value | Meaning |
|------|-------|---------|
| `MAX_MODEL_SPAN` | `150.0` | Maximum allowed width **and** height in any axis (units) |
| `MIN_RADIUS` | `5.0` | Minimum allowed circle radius |
| `MAX_RADIUS` | `70.0` | Maximum allowed radius for smoothing/CLI validation |
| `MAX_Z_INCREMENT` | `≈ 21.43` | `MAX_MODEL_SPAN / 7` — implied max spacing |
| `SMALL_CELL_EXTRUSION_FACTOR` | `0.1` | Extrusion scale applied to small classified cells |

> **Note:** `lofted_surface_voronoi.py` independently defines `MAX_RADIUS = 75.0`, which is the value enforced when loading `lofted_surface_inputs.json`. The `config.py` value of `70.0` is used by the smoothing module and CLI parameter validation.

---

### `PipelineConfig`

```python
@dataclass(frozen=True)
class PipelineConfig:
    radii: tuple[float, ...]
    z_increment: float
    seed_count: int
    random_seed: int
    extrusion_multiplier: float
    scale_x: float
    scale_y: float
    circle_resolution: int = 120
    bbox_padding: float = 4.0
    line_tolerance: float = 0.001
    z_levels: tuple[float, ...] | None = None
```

Frozen (immutable) dataclass holding all parameters for a single pipeline run.

| Field | Description |
|-------|-------------|
| `radii` | Tuple of 8 circle radii for the loft rings |
| `z_increment` | Uniform Z spacing between rings (used when `z_levels` is `None`) |
| `seed_count` | Number of voronoi seed points |
| `random_seed` | RNG seed; change this to get different cell layouts |
| `extrusion_multiplier` | Signed multiplier for wall thickness offset (negative = inward) |
| `scale_x` | Non-uniform X-scale applied to the inner (scaled) surface (0.1–1.5) |
| `scale_y` | Non-uniform Y-scale applied to the inner (scaled) surface (0.1–1.5) |
| `circle_resolution` | Vertices per ring circle; default 120 |
| `bbox_padding` | Extra space added around the half-surface bbox for seed placement |
| `line_tolerance` | Snap/merge tolerance for all polyline operations |
| `z_levels` | When set (e.g. after smoothing), explicit non-uniform Z positions override `z_increment` |

**Computed properties:**

```python
config.effective_extrusion  # -> float: 5.0 * extrusion_multiplier
```

**Methods:**

```python
config.to_surface_config() -> LoftedVoronoiConfig
# Build the LoftedVoronoiConfig used by the geometry module.
# Respects z_levels if set, otherwise derives from z_increment.

config.to_point_config() -> VoronoiPointConfig
# Build the VoronoiPointConfig used for seed generation.

config.with_seed(new_seed: int) -> PipelineConfig
# Return a new frozen copy with random_seed replaced.

config.to_dict() -> dict
# Serialize to a plain dict (used for JSON saving).
```

---

### `validate_geometry_limits`

```python
def validate_geometry_limits(
    radii: tuple[float, ...],
    z_increment: float,
    *,
    z_levels: tuple[float, ...] | None = None,
) -> tuple[float, float]
```

Checks that the geometry stays within `MAX_MODEL_SPAN` (150 units) in both width (2 × max radius) and height. Raises `ValueError` with a descriptive message if either limit is exceeded.

Returns `(max_width, max_height)` on success.

```python
from compass_web.config import validate_geometry_limits

max_w, max_h = validate_geometry_limits((10.0, 20.0, 30.0, 25.0, 20.0, 15.0, 10.0, 12.0), 13.38)
```

---

### `load_pipeline_config`

```python
def load_pipeline_config(
    surface_path: str | Path,
    point_path: str | Path,
    *,
    extrusion_multiplier: float = -0.2,
    scale_x: float = 0.5,
    scale_y: float = 0.5,
) -> PipelineConfig
```

Build a `PipelineConfig` from the two standard JSON input files. Reads geometry parameters from `surface_path` and seed parameters from `point_path`. The three keyword arguments can override defaults from the JSON.

```python
from compass_web.config import load_pipeline_config

config = load_pipeline_config(
    "data/lofted_surface_inputs.json",
    "data/voronoi_points_inputs.json",
    extrusion_multiplier=-0.3,
    scale_x=0.6,
    scale_y=0.6,
)
```

---

### `load_pipeline_config_from_saved`

```python
def load_pipeline_config_from_saved(path: str | Path) -> PipelineConfig
```

Load a `PipelineConfig` from a previously saved config snapshot JSON file (e.g. from `configs/20260407_134445.json`). Handles both old configs (without `circle_resolution`) and new configs (with `z_levels`).

```python
from compass_web.config import load_pipeline_config_from_saved

config = load_pipeline_config_from_saved("configs/20260407_134445.json")
```

---

### `save_pipeline_config`

```python
def save_pipeline_config(
    config: PipelineConfig,
    configs_dir: str | Path,
    *,
    allow_duplicates: bool = False,
) -> Path | None
```

Serialize a `PipelineConfig` to a timestamped JSON file inside `configs_dir`. Creates the directory if it does not exist.

If `allow_duplicates=False` (default), scans existing files; if an identical config already exists, returns `None` without writing. Returns the `Path` of the file written, or `None` on skip.

```python
from compass_web.config import save_pipeline_config

saved_path = save_pipeline_config(config, "configs")
if saved_path:
    print(f"Saved to {saved_path}")
else:
    print("Identical config already on disk — skipped.")
```

---

### `list_saved_configs`

```python
def list_saved_configs(configs_dir: str | Path) -> list[str]
```

Return a sorted list of config stems (filenames without `.json`) from `configs_dir`, newest first (reverse alphabetical by timestamp).

```python
from compass_web.config import list_saved_configs

for name in list_saved_configs("configs"):
    print(name)
```

---

## `compass_web/smoothing.py`

Radii and spacing smoothing to prevent steep loft slopes that would produce degenerate voronoi intersections.

**When smoothing is needed:** if consecutive radii differ by more than `MAX_CONSECUTIVE_RATIO` (2.5×), the loft surface has a near-vertical slope. Smoothing increases the smaller radius until the ratio is at most `TARGET_RATIO` (2.0×), then adjusts Z spacing to satisfy angle constraints.

### Constants

| Name | Value | Meaning |
|------|-------|---------|
| `MAX_CONSECUTIVE_RATIO` | `2.5` | Trigger threshold: larger/smaller ratio |
| `TARGET_RATIO` | `2.0` | Target ratio after adjustment |
| `MIN_ANGLE_FROM_VERTICAL_DEG` | `25.0` | Minimum loft surface angle from vertical (degrees) |
| `MIN_ANGLE_BETWEEN_SEGMENTS_DEG` | `50.0` | Minimum turn angle between consecutive loft segments |
| `VERTEX_WIDEN_FACTOR` | `1.03` | Factor applied per spacing-widening step (1.03 = 3%) |
| `HEIGHT_LIMIT_FACTOR` | `2.0` | Max allowed Z span relative to `(n-1) × z_increment` |
| `MAX_HEIGHT_NUDGE_ROUNDS` | `120` | Limit on height correction iterations |

---

### `SmoothingResult`

```python
@dataclass(frozen=True)
class SmoothingResult:
    original_radii: tuple[float, ...]
    original_z_increment: float
    adjusted_radii: tuple[float, ...]
    adjusted_z_levels: tuple[float, ...]
    adjusted_z_increment: float
    adjustments: tuple[str, ...]
    was_adjusted: bool
```

| Field | Description |
|-------|-------------|
| `original_radii` | Input radii before any adjustment |
| `original_z_increment` | Input Z spacing before any adjustment |
| `adjusted_radii` | Radii after smoothing (unchanged if `was_adjusted=False`) |
| `adjusted_z_levels` | Non-uniform Z positions derived from adjusted spacings |
| `adjusted_z_increment` | Maximum per-segment spacing (informational) |
| `adjustments` | Human-readable log of every change made |
| `was_adjusted` | `True` if any adjustment was made |

---

### `smooth_radii_and_spacing`

```python
def smooth_radii_and_spacing(
    radii: tuple[float, ...],
    z_increment: float,
    *,
    max_ratio: float = MAX_CONSECUTIVE_RATIO,
    target_ratio: float = TARGET_RATIO,
) -> SmoothingResult
```

Core smoothing algorithm. Runs three passes:

1. **Ratio pass** — finds consecutive pairs exceeding `max_ratio` and increases the smaller radius to `larger / target_ratio`.
2. **Geometry pass** — for steep intervals, caps per-segment spacing to satisfy the `MIN_ANGLE_FROM_VERTICAL_DEG` constraint; widens spacing iteratively to meet `MIN_ANGLE_BETWEEN_SEGMENTS_DEG`.
3. **Height limit loop** — if the total Z span exceeds `HEIGHT_LIMIT_FACTOR × (n-1) × z_increment`, nudges adjusted radii back toward the originals by 2% per round and re-runs passes 1–2.

```python
from compass_web.smoothing import smooth_radii_and_spacing

result = smooth_radii_and_spacing((8.0, 45.0, 10.0, 12.0, 8.0, 11.0, 9.0, 14.0), 13.38)
if result.was_adjusted:
    for msg in result.adjustments:
        print(msg)
print("Z positions:", result.adjusted_z_levels)
```

---

### `apply_smoothing_to_config`

```python
def apply_smoothing_to_config(
    config: PipelineConfig,
) -> tuple[PipelineConfig, SmoothingResult]
```

Convenience wrapper: calls `smooth_radii_and_spacing` on `config.radii` and `config.z_increment`, then returns a new frozen `PipelineConfig` with `radii`, `z_increment`, and `z_levels` replaced, plus the `SmoothingResult`. If no adjustment is needed, returns `(config, result)` unchanged.

```python
from compass_web.smoothing import apply_smoothing_to_config

adjusted_config, result = apply_smoothing_to_config(config)
```

---

## `compass_web/pipeline.py`

End-to-end voronoi shell pipeline that stitches together geometry, analysis, and export.

### Constants

| Name | Value | Meaning |
|------|-------|---------|
| `ELONGATED_WIDTH_RATIO_THRESHOLD` | `0.20` | OBB width ratio below which a cell is removed as too elongated |
| `BOUNDARY_VERTEX_PLANE_TOL_FACTOR` | `50` | Tolerance multiplier for detecting cutting-plane boundary vertices |

---

### `PipelineResult`

```python
@dataclass
class PipelineResult:
    trimesh_result: trimesh.Trimesh
    cell_solids: list[pv.PolyData]
    generated_surface: pv.PolyData
    is_valid_volume: bool
    stats: dict
```

| Field | Description |
|-------|-------------|
| `trimesh_result` | Final assembled mesh, rotated for 3D printing (Z-up, sliced face on XY plane) |
| `cell_solids` | Individual watertight per-cell PyVista meshes before assembly |
| `generated_surface` | Merged surface (PyVista) of all cell solids, useful for preview |
| `is_valid_volume` | `True` if `trimesh_result.is_volume` — suitable for 3D printing |
| `stats` | Dict with keys: `polyline_count`, `elongated_removed`, `alignment_applied`, `alignment_max_shift`, `overlap_points_relocated`, `curve_count`, `cell_patch_count`, `cell_solid_count`, `face_count`, `is_watertight`, `is_volume` |

---

### `run_pipeline`

```python
def run_pipeline(
    config: PipelineConfig,
    *,
    verbose: bool = True,
    apply_smoothing: bool = True,
) -> PipelineResult
```

Execute the complete voronoi shell pipeline for a given config. Set `apply_smoothing=False` to use `config` as-is (e.g. when replaying a saved config that already has `z_levels`).

**Pipeline stages, in order:**

1. Optionally smooth radii/spacing (`apply_smoothing_to_config`)
2. Build full lofted surface (`build_lofted_surface`)
3. Clip to half-space at YZ plane (`clip_surface_in_half`)
4. Generate random voronoi seed points in padded bounds
5. Build bounded voronoi cells (`build_bounded_voronoi_cells`)
6. Intersect cells with surface → raw polylines (`intersect_cells_with_surface`)
7. Filter isolated polylines (no shared vertices)
8. Compact polyline shapes (tail removal)
9. Rebuild from discontinuities and cross-polyline intersections
10. Filter extreme elongated cells (OBB width ratio < 0.20)
11. Align shared edges between neighboring polylines
12. Fix polyline surface overlaps
13. Close free vertices (snap interior cell free vertices to neighbor edges)
14. Classify cells → analyze surfaces → build cell patches
15. Remove cells with naked edges
16. Build watertight cell solids (scale + loft walls + plane caps)
17. Filter disconnected cells (keep largest connected region)
18. Assemble trimesh and rotate for printing

```python
from compass_web.pipeline import run_pipeline

result = run_pipeline(config, verbose=True)
print(f"Valid volume: {result.is_valid_volume}")
print(f"Faces: {result.stats['face_count']}")
```

---

### `run_pipeline_with_retry`

```python
def run_pipeline_with_retry(
    config: PipelineConfig,
    *,
    max_attempts: int = 10,
    verbose: bool = True,
    apply_smoothing: bool = True,
) -> tuple[PipelineResult, PipelineConfig]
```

Run the pipeline; if the result is not a valid volume, automatically retry with different random seeds (`base_seed + attempt × 7`) up to `max_attempts` times. Returns `(result, config_used)` so the caller knows which seed succeeded.

```python
from compass_web.pipeline import run_pipeline_with_retry

result, used_config = run_pipeline_with_retry(config, max_attempts=10)
if result.is_valid_volume:
    print(f"Success with seed {used_config.random_seed}")
```

---

### `export_stl`

```python
def export_stl(
    result: PipelineResult,
    export_dir: str | Path,
    *,
    suffix: str = "",
) -> Path
```

Write `result.trimesh_result` to an STL file with a timestamped name (`voronoi_shell_YYYYMMDD_HHMMSS{suffix}.stl`). Creates `export_dir` if it does not exist. Returns the full `Path` of the written file.

```python
from compass_web.pipeline import export_stl

stl_path = export_stl(result, "exports", suffix="_v2")
print(f"Written: {stl_path}")
```

---

### `filter_isolated_polylines`

```python
def filter_isolated_polylines(
    polylines: list[np.ndarray],
    tolerance: float,
) -> tuple[list[np.ndarray], list[int], list[int]]
```

Remove polylines that share no boundary points with any other polyline (isolated islands). Returns `(kept_polylines, kept_indices, discarded_indices)`.

---

### `build_export_trimesh`

```python
def build_export_trimesh(solids_list: list[pv.PolyData]) -> trimesh.Trimesh
```

Convert a list of PyVista cell solids into a single `trimesh.Trimesh`. Applies outward-normal orientation and winding fixes per cell, concatenates, repairs normals globally, then rotates −90° around Y so the sliced face lies on the XY plane (Z-up for printing). Also translates so the bottom face sits at Z = 0.

---

### `restore_boundary_vertices`

```python
def restore_boundary_vertices(
    polylines: list[np.ndarray],
    original_polylines: list[np.ndarray],
    plane_axis: int,
    plane_coord: float,
    tolerance: float,
) -> list[np.ndarray]
```

After neighbour alignment, restore the exact cutting-plane-axis coordinate for vertices that were sitting on the cutting plane. Only the `plane_axis` coordinate is restored; the other two axes keep their aligned values so tiling isn't broken.

---

## `compass_web/visualization.py`

PyVista scene construction, camera control, static PNG rendering, and interactive viewer helpers.

### `distinct_colors`

```python
def distinct_colors(count: int) -> list[str]
```

Return `count` perceptually-distinct hex color strings (e.g. `"#ff6b6b"`) spaced evenly in HSV hue, at 70% saturation and full value. Useful for coloring individual voronoi cells.

```python
from compass_web.visualization import distinct_colors

colors = distinct_colors(12)  # 12 hex strings
```

---

### `camera_position_from_bounds`

```python
def camera_position_from_bounds(
    bounds: tuple[float, float, float, float, float, float],
    target: np.ndarray | list[float] | tuple[float, float, float],
) -> list[list[float]]
```

Compute a camera position that frames a scene from an oblique angle. The camera is placed at `target + [1.05, -1.45, 0.78] × max_span` and returns the PyVista-compatible `[camera_pos, focal_point, up_vector]` list.

---

### `merge_bounds`

```python
def merge_bounds(
    bounds_list: list[tuple[float, float, float, float, float, float]],
) -> tuple[float, float, float, float, float, float]
```

Compute the axis-aligned union of a list of `(xmin, xmax, ymin, ymax, zmin, zmax)` bounds. Raises `ValueError` if `bounds_list` is empty.

---

### `bounds_from_points`

```python
def bounds_from_points(
    points: np.ndarray,
) -> tuple[float, float, float, float, float, float] | None
```

Compute tight bounds from a point array. Returns `None` for empty arrays.

---

### `padded_scene_bounds`

```python
def padded_scene_bounds(
    bounds: tuple[float, float, float, float, float, float],
    padding_fraction: float = 0.22,
    min_padding: float = 1.0,
) -> tuple[float, float, float, float, float, float]
```

Expand bounds by `padding_fraction × span` in each axis (with a floor of `min_padding`). Used to add visual breathing room in screenshots.

---

### `add_scene_content`

```python
def add_scene_content(
    plotter: pv.Plotter,
    *,
    meshes: list[tuple[pv.DataSet, dict]] | None = None,
    line_meshes: list[tuple[pv.PolyData, dict]] | None = None,
    point_sets: list[tuple[np.ndarray, dict]] | None = None,
    label_sets: list[tuple[np.ndarray, list[str], dict]] | None = None,
) -> None
```

Add multiple heterogeneous scene elements to a `pv.Plotter`. Each element is a `(object, kwargs)` pair; `kwargs` are forwarded directly to the corresponding PyVista add method. Empty objects are silently skipped. Line meshes are rendered as tubes automatically.

```python
from compass_web.visualization import add_scene_content
import pyvista as pv

plotter = pv.Plotter()
add_scene_content(
    plotter,
    meshes=[(my_mesh, {"color": "#9b8cff", "opacity": 0.8}),
            (another_mesh, {"color": "red"})],
    point_sets=[(seed_pts, {"point_size": 6, "color": "yellow"})],
)
```

---

### `render_static_scene`

```python
def render_static_scene(
    *,
    title: str,
    bounds: tuple[float, float, float, float, float, float],
    target: np.ndarray | list[float] | tuple[float, float, float],
    meshes: list[tuple[pv.DataSet, dict]] | None = None,
    line_meshes: list[tuple[pv.PolyData, dict]] | None = None,
    point_sets: list[tuple[np.ndarray, dict]] | None = None,
    label_sets: list[tuple[np.ndarray, list[str], dict]] | None = None,
    fit_bounds: tuple[...] | None = None,
    fit_target: np.ndarray | None = None,
    zoom_factor: float = 1.1,
    window_size: tuple[int, int] = (1100, 820),
) -> bytes
```

Render a scene off-screen to a PNG and return the raw bytes. Uses a dark navy background (`#1a1a2e`). Pass `fit_bounds`/`fit_target` to frame the camera on a sub-region while showing the full scene.

```python
from compass_web.visualization import render_static_scene
from pathlib import Path

png_bytes = render_static_scene(
    title="My Shell",
    bounds=result.generated_surface.bounds,
    target=result.generated_surface.center,
    meshes=[(result.generated_surface, {"color": "#9b8cff"})],
)
Path("preview.png").write_bytes(png_bytes)
```

---

### `display_static_scene`

```python
def display_static_scene(**kwargs) -> None
```

Same as `render_static_scene` but displays the image inline in a Jupyter notebook using `IPython.display`. Accepts the same keyword arguments.

---

### `display_interactive_scene`

```python
def display_interactive_scene(
    *,
    title: str,
    bounds: tuple[...],
    target: np.ndarray | list[float] | tuple[float, float, float],
    meshes: list[tuple[pv.DataSet, dict]] | None = None,
    line_meshes: list[tuple[pv.PolyData, dict]] | None = None,
    point_sets: list[tuple[np.ndarray, dict]] | None = None,
    label_sets: list[tuple[np.ndarray, list[str], dict]] | None = None,
    fit_bounds: tuple[...] | None = None,
    fit_target: np.ndarray | None = None,
    zoom_factor: float = 1.0,
    window_size: tuple[int, int] = (1200, 900),
) -> None
```

Open a native VTK interactive window (orbit, pan, zoom). Does **not** use the trame backend. Blocks until the window is closed.

---

### `show_mesh_interactive`

```python
def show_mesh_interactive(
    mesh: pv.PolyData,
    *,
    title: str = "Voronoi Shell Preview",
    color: str = "#9b8cff",
    zoom: float = 1.8,
    window_size: tuple[int, int] = (1300, 950),
) -> None
```

Quick single-mesh interactive viewer. Wraps `display_interactive_scene` with sensible defaults. The default color is a muted purple (`#9b8cff`).

```python
from compass_web.visualization import show_mesh_interactive

show_mesh_interactive(result.generated_surface, title="My Shell")
```

---

### `save_screenshot`

```python
def save_screenshot(
    mesh: pv.PolyData,
    path: str | Path,
    *,
    title: str = "Voronoi Shell",
    color: str = "#9b8cff",
    zoom: float = 1.8,
    window_size: tuple[int, int] = (1300, 950),
) -> Path
```

Render `mesh` off-screen and write a PNG to `path`. Returns the resolved `Path`.

```python
from compass_web.visualization import save_screenshot

save_screenshot(result.generated_surface, "exports/preview.png")
```

---

## `compass_web/lofted_surface_voronoi.py`

Core geometry engine. Contains all low-level surface, voronoi, polyline, and mesh operations.

### Dataclasses

#### `LoftedVoronoiConfig`

Internal config used by `build_lofted_surface` and related functions. Built via `PipelineConfig.to_surface_config()`.

| Field | Description |
|-------|-------------|
| `radii` | Tuple of 8 radii |
| `z_levels` | Explicit Z positions for each ring |
| `z_increment` | Nominal Z spacing (informational) |
| `circle_resolution` | Vertices per ring |
| `slice_normal` | Half-space clip plane normal; default `(1, 0, 0)` |
| `slice_origin` | Clip plane origin; default `(0, 0, 0)` |
| `bbox_padding` | Voronoi seed bbox padding |
| `line_tolerance` | Snap/merge tolerance |

#### `VoronoiPointConfig`

Seed point config: `seed_count: int`, `random_seed: int`.

#### `CurveAnalysis`

Per-cell analysis result holding the original/followup polylines, fitted plane, bounding box, cell size ratio, planarity ratio, extrusion direction, etc. Produced by `analyze_and_generate_surfaces`.

#### `SurfaceGenerationResult`

Result of `analyze_and_generate_surfaces`: holds all `CurveAnalysis` objects, average ratio, and generated mesh surfaces.

#### `MeshCleanupResult`

Result of `clean_meshes_without_naked_edges`: partitions meshes into kept/removed and stores naked edge loops.

#### `MeshPrintabilityReport`

Counts of mesh topology statistics: `point_count`, `face_count`, `connected_region_count`, `boundary_edge_count`, `boundary_loop_count`, `non_manifold_edge_count`, `is_closed`, `is_printable`.

#### `MeshPreparationResult`

Result of `prepare_mesh_for_export`: holds the cleaned mesh, initial and final printability reports, and whether repair was attempted.

#### `AnalysisOutputMeshes`

Output of `build_analysis_output_meshes`: `preview_meshes`, `output_meshes`, `output_modes` (cell classification), `removed_by_retained_volume_indices`.

---

### Config Loaders

#### `load_generation_config`

```python
def load_generation_config(path: str | Path) -> LoftedVoronoiConfig
```

Read `lofted_surface_inputs.json` and return a `LoftedVoronoiConfig`. Validates exactly 8 radii in [5, 75], ≤ 3 decimal places, and max width ≤ 150. Raises `ValueError` on any violation.

#### `load_voronoi_point_config`

```python
def load_voronoi_point_config(path: str | Path) -> VoronoiPointConfig
```

Read `voronoi_points_inputs.json`. Validates `seed_count ≥ 2`.

---

### Surface Construction

#### `build_lofted_surface`

```python
def build_lofted_surface(config: LoftedVoronoiConfig) -> pv.PolyData
```

Build a triangulated lofted surface by connecting consecutive rings of points. Each ring is a circle sampled at `circle_resolution` points at the given radius and Z level. Returns a cleaned, triangulated `pv.PolyData`.

#### `clip_surface_in_half`

```python
def clip_surface_in_half(
    surface: pv.PolyData,
    normal: tuple[float, float, float],
    origin: tuple[float, float, float] | None = None,
) -> pv.PolyData
```

Clip a mesh to the half-space defined by `normal` (keeps the side the normal points toward). Default origin is `(0, 0, 0)`. Returns a cleaned, triangulated half-surface.

#### `pad_bounds`

```python
def pad_bounds(
    bounds: tuple[float, float, float, float, float, float],
    padding: float,
) -> tuple[float, float, float, float, float, float]
```

Expand each axis of an AABB by `padding` units in both directions.

---

### Voronoi Generation

#### `random_points_in_bounds`

```python
def random_points_in_bounds(
    bounds: tuple[float, float, float, float, float, float],
    count: int,
    seed: int,
) -> np.ndarray
```

Generate `count` uniformly random 3D points inside `bounds` using `numpy.random.default_rng(seed)`. Adds a small margin (0.1% of span) so no point lands exactly on a face.

#### `build_bounded_voronoi_cells`

```python
def build_bounded_voronoi_cells(
    seed_points: np.ndarray,
    bounds: tuple[float, float, float, float, float, float],
) -> list[pv.PolyData]
```

Build one bounded convex Voronoi cell (`pv.PolyData`) per seed point, clipped to the given bounding box. Uses `scipy.spatial.HalfspaceIntersection` and `ConvexHull`.

#### `intersect_cells_with_surface`

```python
def intersect_cells_with_surface(
    surface: pv.PolyData,
    cells: list[pv.PolyData],
    tolerance: float,
) -> list[np.ndarray]
```

Intersect each voronoi cell with the half-surface and extract the resulting boundary polylines. Deduplicates by canonical key. Also detects uncovered surface patches and generates polylines for them.

---

### Mesh Utilities

#### `weld_mesh_vertices`

```python
def weld_mesh_vertices(mesh: pv.PolyData, tolerance: float | None = None) -> pv.PolyData
```

Merge coincident vertices and triangulate. Wrapper around `pv.PolyData.clean()`.

#### `build_polyline_mesh`

```python
def build_polyline_mesh(polylines: list[np.ndarray]) -> pv.PolyData
```

Pack a list of polyline arrays into a single `pv.PolyData` line mesh for visualization.

#### `scale_points_in_xy`

```python
def scale_points_in_xy(
    points: np.ndarray,
    center: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> np.ndarray
```

Non-uniform scale of a point array in XY only (Z unchanged), pivoting around `center`.

#### `scale_polydata_in_xy`

```python
def scale_polydata_in_xy(
    mesh: pv.PolyData,
    center: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> pv.PolyData
```

Same as `scale_points_in_xy` but operates on a `pv.PolyData` mesh; returns a deep copy with scaled points.

#### `count_connected_regions`

```python
def count_connected_regions(mesh: pv.PolyData) -> int
```

Return the number of distinct connected components in `mesh` using VTK connectivity analysis.

#### `extract_surface_mesh`

```python
def extract_surface_mesh(mesh: pv.PolyData) -> pv.PolyData
```

Extract the outer surface (boundary) of a mesh, clean and triangulate.

#### `unify_mesh_normals`

```python
def unify_mesh_normals(mesh: pv.PolyData) -> pv.PolyData
```

Make all face normals consistently oriented using VTK's normal consistency filter.

#### `remove_closed_regions`

```python
def remove_closed_regions(mesh: pv.PolyData) -> pv.PolyData
```

Remove faces that belong to fully closed sub-regions (no boundary edges), retaining only open-boundary surfaces.

#### `resolve_non_manifold_faces`

```python
def resolve_non_manifold_faces(surface: pv.PolyData) -> pv.PolyData
```

Detect and remove faces that share an edge with more than one other face (non-manifold topology), producing a cleaner manifold surface.

#### `extract_naked_edge_loops`

```python
def extract_naked_edge_loops(
    surface: pv.PolyData,
    tolerance: float,
) -> tuple[pv.PolyData, list[np.ndarray]]
```

Extract boundary (naked) edges from `surface` and trace them into closed loop arrays. Returns `(boundary_edge_mesh, loops)`.

#### `clean_meshes_without_naked_edges`

```python
def clean_meshes_without_naked_edges(
    meshes: list[pv.PolyData],
    tolerance: float,
) -> MeshCleanupResult
```

Partition a list of meshes into those that have acceptable naked-edge topology (kept) vs. those that are too degenerate (removed). Returns a `MeshCleanupResult`.

#### `join_two_point_segments_into_polylines`

```python
def join_two_point_segments_into_polylines(
    segments: list[np.ndarray],
    tolerance: float,
) -> list[np.ndarray]
```

Stitch a flat list of two-point line segments into multi-vertex polylines by chaining endpoints within `tolerance`.

---

### Mesh Analysis and Export

#### `build_analysis_output_meshes`

```python
def build_analysis_output_meshes(
    analyses: tuple[CurveAnalysis, ...],
    average_ratio: float,
    loft_bounds: tuple[float, ...],
    tolerance: float,
    extrusion_multiplier: float,
    small_cell_extrusion_factor: float,
    slice_plane_x: float,
) -> AnalysisOutputMeshes
```

From a sequence of `CurveAnalysis` objects, build the actual per-cell surface meshes that will be turned into solids. Classifies cells as large/small/extreme and applies appropriate extrusion parameters.

#### `build_mesh_printability_report`

```python
def build_mesh_printability_report(
    mesh: pv.PolyData,
    tolerance: float,
) -> MeshPrintabilityReport
```

Analyse a mesh and return a `MeshPrintabilityReport` with counts of connected regions, boundary edges, boundary loops, non-manifold edges, and overall printability verdict.

#### `prepare_mesh_for_export`

```python
def prepare_mesh_for_export(
    mesh: pv.PolyData,
    tolerance: float,
    attempt_repair: bool = True,
) -> MeshPreparationResult
```

Attempt to repair a mesh for STL export (close holes, fix normals, remove non-manifold faces). Returns a `MeshPreparationResult` with before/after reports.

#### `export_mesh_to_stl`

```python
def export_mesh_to_stl(
    mesh: pv.PolyData,
    output_path: str | Path,
    tolerance: float,
    attempt_repair: bool = True,
) -> MeshPreparationResult
```

Prepare a mesh and write it directly to an STL file. Returns the `MeshPreparationResult`.

---

### Polyline Operations

#### `compact_polyline_shapes`

```python
def compact_polyline_shapes(
    polylines: list[np.ndarray],
    tolerance: float,
    *,
    tail_distance_ratio: float = 2.2,
    min_area_fraction: float = 0.4,
    min_vertices: int = 4,
) -> tuple[list[np.ndarray], int, list[str]]
```

Remove "tail" vertices — points that extend far from the cell's bulk centroid and barely contribute area. For each polyline:
1. Projects to a best-fit 2D plane.
2. Identifies vertices farther than `tail_distance_ratio × median_distance` from the centroid.
3. Removes a candidate only if the result keeps ≥ `min_vertices` vertices and retains ≥ `min_area_fraction` of the original area.

Returns `(compacted_polylines, total_removed_vertices, log_messages)`.

#### `rebuild_polylines_from_discontinuities`

```python
def rebuild_polylines_from_discontinuities(
    polylines: list[np.ndarray],
    tolerance: float,
    discontinuity_angle_degrees: float,
    neighbor_snap_tolerance: float,
) -> list[np.ndarray]
```

Re-trace polylines through vertices where the direction changes sharply (angle > `discontinuity_angle_degrees`), splicing in cross-polyline intersection points. Fixes cell boundaries that have jagged kinks or missed vertices where cells meet.

#### `find_polyline_neighbours`

```python
def find_polyline_neighbours(
    polylines: list[np.ndarray],
    tolerance: float,
) -> dict[int, list[int]]
```

Return a mapping of polyline index → sorted list of neighbor indices. Two polylines are neighbors if they share at least one vertex within `tolerance`. Uses a spatial grid for efficient O(n) lookup.

#### `align_neighbouring_polylines`

```python
def align_neighbouring_polylines(
    polylines: list[np.ndarray],
    tolerance: float,
    snap_tolerance: float | None = None,
    slice_plane_x: float | None = None,
) -> list[np.ndarray]
```

Align shared-edge segments between neighboring polylines to eliminate gaps and crossings that would produce overlapping cell walls. For each neighbor pair:
1. Finds shared vertices (Voronoi face corners).
2. Detects edge–edge crossings not at shared vertices.
3. Extracts sub-paths between consecutive shared vertices from both sides.
4. Averages matched positions (or resamples the shorter path via arc-length interpolation if vertex counts differ).

Vertices near the cutting plane are only moved 20% toward the midpoint to preserve boundary cell extensions.

#### `fix_polyline_surface_overlaps`

```python
def fix_polyline_surface_overlaps(
    polylines: list[np.ndarray],
    tolerance: float,
    snap_tolerance: float | None = None,
    max_iterations: int = 5,
) -> tuple[list[np.ndarray], int, list[str]]
```

Fix overlapping neighbor fan surfaces. For each pair, non-shared vertices of polyline A that lie inside polyline B's polygon are relocated to the nearest shared-edge segment. Runs up to `max_iterations` passes. Returns `(fixed_polylines, total_relocated, log_messages)`.

#### `resolve_pocket_cells`

```python
def resolve_pocket_cells(
    polylines: list[np.ndarray],
    tolerance: float,
    snap_tolerance: float | None = None,
) -> tuple[list[np.ndarray], int, list[str]]
```

Remove "pocket" cells — polylines that are fully enclosed by a single larger neighbor polygon. Returns `(filtered_polylines, removed_count, log_messages)`.

#### `close_free_vertices`

```python
def close_free_vertices(
    polylines: list[np.ndarray],
    surface: pv.PolyData,
    tolerance: float,
    snap_tolerance: float | None = None,
) -> tuple[list[np.ndarray], int, list[str]]
```

Snap free (unshared) vertices of interior cells to their neighbor polyline edges and insert them into the neighbor so the tiling closes completely. Only operates on non-boundary cells (those not near the cutting plane). Returns `(closed_polylines, snapped_count, log_messages)`.

---

### Geometry Helpers

#### `intersect_mesh_with_plane`

```python
def intersect_mesh_with_plane(
    mesh: pv.PolyData,
    normal: tuple[float, float, float],
    origin: tuple[float, float, float] | None = None,
    tolerance: float = 1e-4,
) -> list[np.ndarray]
```

Slice a mesh with a plane and return the resulting intersection polylines.

#### `filter_segments_against_curves`

```python
def filter_segments_against_curves(
    segments: list[np.ndarray],
    reference_curves: list[np.ndarray],
    tolerance: float,
) -> list[np.ndarray]
```

Remove two-point segments whose edge is already present in any reference curve (deduplication by canonical key).

#### `filter_naked_loops_against_base_polylines`

```python
def filter_naked_loops_against_base_polylines(
    naked_loops: list[np.ndarray],
    base_polylines: list[np.ndarray],
    tolerance: float,
    overlap_threshold: float = 0.3,
) -> list[np.ndarray]
```

Remove naked loops whose points overlap ≥ `overlap_threshold` (30%) with the base polyline point set. Keeps loops that are genuinely new boundaries.

#### `make_bounding_box`

```python
def make_bounding_box(bounds: tuple[float, ...]) -> pv.PolyData
```

Create a wireframe bounding box mesh from a `(xmin, xmax, ymin, ymax, zmin, zmax)` tuple.

#### `close_mesh_boundaries`

```python
def close_mesh_boundaries(
    mesh: pv.PolyData,
    tolerance: float,
    max_iterations: int = 10,
) -> pv.PolyData
```

Iteratively close open boundary loops in a mesh using fan-triangulation from the loop centroid. Runs up to `max_iterations` passes until no more boundaries remain or no progress is made.

#### `align_loops_and_loft`

```python
def align_loops_and_loft(
    loop_a: np.ndarray,
    loop_b: np.ndarray,
    tolerance: float,
) -> pv.PolyData
```

Given two boundary loops, align their winding direction and starting vertex (by closest-segment matching), then loft a quad strip between them. Used to build wall faces between body and plane-offset patches.

#### `project_loop_to_plane`

```python
def project_loop_to_plane(
    loop: np.ndarray,
    plane_axis: int,
    plane_coord: float,
) -> np.ndarray
```

Snap all vertices of `loop` to the given coordinate plane along `plane_axis`.

#### `split_and_offset_plane_faces`

```python
def split_and_offset_plane_faces(
    mesh: pv.PolyData,
    plane_normal: tuple[float, float, float],
    plane_origin: tuple[float, float, float],
    offset_amount: float,
    tolerance: float,
) -> tuple[pv.PolyData, pv.PolyData]
```

Split a boundary cell mesh into two patches: the body (faces not on the cutting plane) and a translated copy of the cutting-plane faces (offset by `offset_amount` along `plane_normal`). Returns `(body_patch, offset_patch)`.

#### `point_distance_to_mesh_surface`

```python
def point_distance_to_mesh_surface(
    point: np.ndarray,
    surface: pv.PolyData,
) -> tuple[float, np.ndarray]
```

Compute the shortest distance from a 3D point to the nearest location on `surface` (may be interior to a triangle, not just a vertex). Uses a VTK cell locator. Returns `(distance, closest_point_on_surface)`.

#### `validate_polyline_surfaces`

```python
def validate_polyline_surfaces(
    polylines: list[np.ndarray],
    tolerance: float,
    snap_tolerance: float | None = None,
) -> list[tuple[int, int, bool, int]]
```

Check every neighbor pair for fan-surface overlaps. Builds fan surfaces from centroids, intersects them, and identifies intersection points not on shared edges. Returns `(idx_a, idx_b, has_overlap, non_shared_count)` for each checked pair.

#### `orient_normals_outward`

```python
def orient_normals_outward(mesh: pv.PolyData) -> pv.PolyData
```

Flip any inward-facing normals so all normals point outward from the mesh surface. Uses VTK's normal consistency with point normal computation.

#### `analyze_and_generate_surfaces`

```python
def analyze_and_generate_surfaces(
    polylines: list[np.ndarray],
    loft_bounds: tuple[float, ...],
    tolerance: float,
    discontinuity_angle_degrees: float = 170.0,
    extrusion_multiplier: float = -1.0,
    small_cell_extrusion_factor: float = 0.1,
    extrusion_scale_origin: np.ndarray | None = None,
    planar_scale_factors: tuple[float, float] = (0.5, 0.5),
    slice_plane_x: float = 0.0,
) -> SurfaceGenerationResult
```

Fit a plane, bounding box, and circumscribed circle to each polyline; classify cells by size (relative to the average ratio); compute extrusion direction and scaled circle center for each cell. Returns a `SurfaceGenerationResult` with all `CurveAnalysis` objects.

#### `default_snap_tolerance`

```python
def default_snap_tolerance(tolerance: float) -> float
```

Return the canonical snap tolerance derived from the base line tolerance. Used consistently throughout the pipeline for neighbor-detection grid snapping. (Currently returns `tolerance * 10`.)

---

## CLI — `materialize`

The CLI is installed as the `materialize` command. Run `uv run materialize --help` for full usage.

### `generate`

```
materialize generate [OPTIONS]
```

Run the full voronoi shell pipeline and export to STL.

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--config PATH` | `-c` | — | Load all parameters from a saved config JSON |
| `--surface PATH` | `-s` | `data/lofted_surface_inputs.json` | Surface geometry inputs file |
| `--points PATH` | `-p` | `data/voronoi_points_inputs.json` | Voronoi seed inputs file |
| `--extrusion FLOAT` | `-e` | `-0.2` | Extrusion multiplier (negative = inward) |
| `--scale-x FLOAT` | | `0.5` | X-axis scale factor for inner surface |
| `--scale-y FLOAT` | | `0.5` | Y-axis scale factor for inner surface |
| `--seed INT` | | — | Override random seed from JSON |
| `--seed-count INT` | | — | Override voronoi seed count from JSON |
| `--retry INT` | `-r` | `10` | Max auto-retry attempts with different seeds |
| `--export-dir PATH` | `-o` | `exports` | Directory for STL output |
| `--configs-dir PATH` | | `configs` | Directory for config snapshots |
| `--save-config / --no-save-config` | | save | Save config snapshot after run |
| `--viewer / --no-viewer` | | viewer | Open interactive 3D viewer after export |
| `--screenshot PATH` | | — | Save PNG screenshot to path |
| `--quiet` | `-q` | — | Suppress progress output |
| `--smoothing / --no-smoothing` | | smoothing | Apply radii/spacing smoothing before loft |

```bash
# Use default data files
uv run materialize generate

# Load a saved config
uv run materialize generate --config configs/20260407_134445.json

# Custom parameters, no viewer, save screenshot
uv run materialize generate \
  --extrusion -0.3 --scale-x 0.6 --scale-y 0.6 \
  --seed 999 --retry 5 --no-viewer \
  --screenshot exports/preview.png
```

---

### `run`

```
materialize run [OPTIONS]
```

Create a config from explicit parameter values, save it, and run the pipeline in one step. Same options as `generate` except uses `--radii`, `--z-increment`, `--seed-count`, `--random-seed` instead of loading from JSON files.

| Option | Default | Description |
|--------|---------|-------------|
| `--radii TEXT` | `"8.91,10.446,..."` | Comma-separated list of exactly 8 radii |
| `--z-increment FLOAT` | `13.38` | Z spacing between rings |
| `--seed-count INT` | `78` | Number of voronoi seeds |
| `--random-seed INT` | `12` | RNG seed |
| `--extrusion FLOAT` | `-0.2` | Extrusion multiplier |
| `--scale-x FLOAT` | `0.5` | X scale |
| `--scale-y FLOAT` | `0.5` | Y scale |
| `--retry INT` | `10` | Max retry attempts |
| `--export-dir PATH` | `exports` | STL output directory |
| `--configs-dir PATH` | `configs` | Config snapshot directory |
| `--viewer / --no-viewer` | viewer | Interactive viewer |
| `--screenshot PATH` | — | PNG screenshot path |
| `--quiet` | — | Suppress output |
| `--smoothing / --no-smoothing` | smoothing | Apply smoothing |

```bash
uv run materialize run \
  --radii "10,20,30,25,20,15,10,12" \
  --z-increment 14.0 \
  --seed-count 100 \
  --random-seed 42 \
  --extrusion -0.25 \
  --no-viewer
```

---

### `new-config`

```
materialize new-config [OPTIONS]
```

Create and save a new config snapshot without running the pipeline. Accepts the same geometry options as `run`.

```bash
uv run materialize new-config \
  --radii "15,20,25,30,25,20,15,18" \
  --z-increment 14.0 \
  --seed-count 90 \
  --random-seed 555
```

---

### `show-config`

```
materialize show-config PATH
```

Pretty-print the contents of a saved config JSON file.

```bash
uv run materialize show-config configs/20260407_134445.json
```

---

### `list-configs`

```
materialize list-configs [--configs-dir PATH]
```

List all saved config stems in the configs directory, newest first.

```bash
uv run materialize list-configs
uv run materialize list-configs --configs-dir my_configs/
```

---

### `view`

```
materialize view STL_PATH [--title TEXT]
```

Open an interactive 3D viewer for an existing STL file. Blocks until the window is closed.

```bash
uv run materialize view exports/voronoi_shell_20260407_134445.stl
uv run materialize view exports/my_shell.stl --title "My Print Preview"
```

---

## Config File Format

Config files are saved as JSON in the `configs/` directory with timestamped names (`YYYYMMDD_HHMMSS.json`). The `save_pipeline_config` function skips writing if the exact same config already exists on disk.

```json
{
  "radii": [15.108, 15.567, 31.134, 19.134, 15.409, 17.249, 22.514, 18.677],
  "z_increment": 21.428571428571427,
  "seed_count": 66,
  "random_seed": 4096,
  "extrusion_multiplier": -0.054,
  "scale_x": 0.5,
  "scale_y": 0.5,
  "circle_resolution": 120,
  "bbox_padding": 4.0,
  "line_tolerance": 0.001,
  "z_levels": [0.0, 18.704, 37.408, 56.112, 74.816, 93.52, 112.224, 130.928]
}
```

The `z_levels` key is optional. When present (after radii smoothing), it provides non-uniform Z ring positions and takes priority over `z_increment`.

---

## Constants Reference

| Constant | Module | Value | Role |
|----------|--------|-------|------|
| `MAX_MODEL_SPAN` | config, lofted_surface_voronoi | `150.0` | Max geometry size in any dimension |
| `MIN_RADIUS` | config, lofted_surface_voronoi | `5.0` | Minimum ring radius |
| `MAX_RADIUS` | config | `70.0` | Max radius for smoothing/CLI |
| `MAX_RADIUS` | lofted_surface_voronoi | `75.0` | Max radius enforced in `load_generation_config` |
| `SMALL_CELL_EXTRUSION_FACTOR` | config | `0.1` | Extrusion scale for small cells |
| `MAX_CONSECUTIVE_RATIO` | smoothing | `2.5` | Ratio trigger for smoothing |
| `TARGET_RATIO` | smoothing | `2.0` | Post-smoothing max ratio |
| `MIN_ANGLE_FROM_VERTICAL_DEG` | smoothing | `25.0` | Min loft surface angle from vertical |
| `MIN_ANGLE_BETWEEN_SEGMENTS_DEG` | smoothing | `50.0` | Min turn angle between loft segments |
| `HEIGHT_LIMIT_FACTOR` | smoothing | `2.0` | Max Z span multiple before nudging |
| `ELONGATED_WIDTH_RATIO_THRESHOLD` | pipeline | `0.20` | OBB width ratio below which cells are removed |
| `EXTREME_ASPECT_RATIO_THRESHOLD` | lofted_surface_voronoi | `2.0` | Aspect ratio threshold for extreme classification |
| `EXTREME_SCALE_FACTOR` | lofted_surface_voronoi | `0.3` | Scale used for extreme cells |
| `EXTREME_EXTRUSION_FACTOR` | lofted_surface_voronoi | `0.5` | Extrusion used for extreme cells |
| `EXTREME_PLANARITY_RATIO` | lofted_surface_voronoi | `0.03` | Planarity ratio below which a cell is extreme |
