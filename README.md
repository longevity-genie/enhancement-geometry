# Lofted Surface Voronoi Shell Generator

Generate **3D-printable voronoi shell geometry** from parametric inputs. Each run produces a unique shape driven by radius, spacing, seed, and extrusion parameters, exported as STL.

## Quick start

```bash
# Install dependencies
uv sync

# Run with default data files
uv run materialize generate

# Run without interactive viewer
uv run materialize generate --no-viewer
```

## Pipeline

1. **Lofted surface** — 8 circles at different radii stacked along Z, lofted into a surface
2. **Voronoi cells** — random seed points generate bounded voronoi cells
3. **Intersection curves** — voronoi cells intersect the lofted surface, producing polyline curves
4. **Per-cell solids** — each cell patch is scaled, its open edges lofted into walls, and closed into a watertight solid
5. **Plane face offset** — faces at the cutting plane (x=0) are offset by -2mm to prevent thin-wall artifacts
6. **Assembly** — all cell solids are combined, normals oriented outward, and exported as STL via trimesh
7. **Auto-retry** — if the mesh fails volume validation, the pipeline retries with different voronoi seeds (up to 10 attempts)

## CLI usage

The package installs a `materialize` command.

### Generate geometry

```bash
# From default data files with default parameters
uv run materialize generate

# From a saved config
uv run materialize generate --config configs/20260325_172401.json

# Override parameters
uv run materialize generate --extrusion -0.5 --scale-x 0.7 --scale-y 0.3 --seed 42

# Override seed count and export to a custom directory
uv run materialize generate --seed-count 150 --export-dir my_exports

# Save a screenshot alongside the STL
uv run materialize generate --screenshot preview.png

# Skip viewer and suppress output
uv run materialize generate --no-viewer --quiet
```

### Manage configs

```bash
# List all saved configs
uv run materialize list-configs

# Show a config file
uv run materialize show-config configs/20260325_172401.json

# Create a new config (saved into configs/ with timestamp)
uv run materialize new-config --radii "10,15,20,25,20,15,10,12" --seed-count 100 --random-seed 42

# Create a config AND run the pipeline in one step
uv run materialize run --radii "10,15,20,25,20,15,10,12" --seed-count 100 --random-seed 42 --extrusion -0.3 --scale-x 0.6 --scale-y 0.4
```

### View existing STL files

```bash
uv run materialize view exports/voronoi_shell_20260325_172143.stl
```

## Interactive notebook

The Jupyter notebook provides an interactive exploration workflow with widget controls:

```bash
uv run jupyter lab
# Then open lofted_surface_voronoi_generation.ipynb
```

The notebook offers slider-based parameter editing, step-by-step visualization of each pipeline stage, curve inspection, and per-cell solid debugging.

## Physical scale (units)

| Axis | Max size (units) | Real-world note |
|------|------------------|-----------------|
| **X / Y** | **150** | Treat as **150 mm** in the plane |
| **Z** | **150** | Height along Z (7 spacings) |

Circle radius is constrained to **5–75** units.

## Parameters

| Parameter | Range / rule |
|-----------|----------------|
| Radius | **5.00–75.00** (three decimals), one per circle |
| Spacing | **4.00–21.43** between circles |
| Points | **2–300** voronoi seed points |
| Seed | **0–9999** random seed |
| Extrusion | **-3.00 to 3.00** |
| Scale X/Y | **0.10–1.50** non-uniform XY scaling |

## Geometric constraints

The pipeline automatically detects and handles cells that would produce degenerate geometry. Each cell's intersection curve is analyzed with a best-fit plane and an oriented bounding box in the `(U, V, normal)` frame of that plane.

### Cell classification

Every cell is classified into one of three categories based on two metrics:

| Metric | Definition | Purpose |
|--------|------------|---------|
| **Volume/length ratio** | `bbox_volume / curve_length` | Separates large cells (ratio >= population mean) from small cells |
| **Bbox aspect ratio** | `max(u_span, v_span) / min(u_span, v_span)` | Detects elongated cells that would produce spiky geometry |

Classification rules:

| Category | Condition | Surface method |
|----------|-----------|----------------|
| **Large** | ratio >= mean AND aspect ratio <= 3.0 | Staged offset lofts (scale 0.5, then 0.45) |
| **Small** | ratio < mean AND aspect ratio <= 3.0 | Triangle fan from offset center |
| **Extreme** | aspect ratio > 3.0 (any ratio) | Extreme cell lofts (see below) |

### Extreme cell handling

When a cell's bounding box aspect ratio exceeds **3.0**, it is classified as extreme regardless of its volume/length ratio. Extreme cells get three corrections to prevent spiky, degenerate geometry:

| Correction | Value | Effect |
|------------|-------|--------|
| **Inner curve projection** | Projected onto fitted plane | The inner scaled copies are flattened onto the cell's best-fit plane, removing 3D undulation. The original intersection curve stays untouched in 3D. |
| **Scale factor** | **0.3** (vs 0.5 for normal large cells) | The inner curve is scaled closer to the center, producing thicker walls |
| **Extrusion reduction** | **0.5x** normal extrusion | The offset vector is halved, preventing extreme cells from extending too far outward |

The loft for extreme cells goes: original 3D curve (unchanged) -> flat inner edge (0.3 scale) -> flatter inner edge (0.27 scale). This creates a clean transition from the curved surface to a straight, thick inner wall.

### Constants

These values are defined in `lofted_surface_voronoi.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| `EXTREME_ASPECT_RATIO_THRESHOLD` | 3.0 | Bbox aspect ratio above which a cell is classified as extreme |
| `EXTREME_SCALE_FACTOR` | 0.3 | Inner curve scale factor for extreme cells (vs 0.5 default) |
| `EXTREME_EXTRUSION_FACTOR` | 0.5 | Multiplier applied to the extrusion offset for extreme cells |
| `MIN_RADIUS` | 5.0 | Minimum allowed circle radius |
| `MAX_RADIUS` | 75.0 | Maximum allowed circle radius |
| `MAX_MODEL_SPAN` | 150.0 | Maximum extent in any axis |

## Project structure

```
compass-web/
├── pyproject.toml                          # Package config, dependencies, CLI entry point
├── data/
│   ├── lofted_surface_inputs.json          # Default radii and spacing
│   └── voronoi_points_inputs.json          # Default seed count and random seed
├── configs/                                # Saved parameter configs (timestamped JSON)
├── exports/                                # Timestamped STL outputs (gitignored)
├── src/compass_web/
│   ├── __init__.py                         # Public API exports
│   ├── lofted_surface_voronoi.py           # Core geometry functions (loft, voronoi, mesh ops)
│   ├── config.py                           # PipelineConfig dataclass, JSON I/O, config management
│   ├── pipeline.py                         # End-to-end pipeline orchestration and export
│   ├── visualization.py                    # Camera, bounds, scene rendering, PyVista viewers
│   └── cli.py                              # Typer CLI (generate, view, config management)
├── lofted_surface_voronoi_generation.ipynb  # Interactive notebook with widget controls
└── voronoi_jewelry.ipynb                   # Separate demo: spherical voronoi filigree pendant
```

## Module overview

- **`lofted_surface_voronoi`** — All low-level geometry: circle sampling, loft construction, Voronoi cell building, surface intersection, mesh cleanup, naked edge handling, mesh repair, and STL export primitives.
- **`config`** — `PipelineConfig` dataclass unifying all parameters for a run, with JSON serialization and duplicate-aware saving.
- **`pipeline`** — `run_pipeline()` and `run_pipeline_with_retry()` orchestrate the full pipeline from config to trimesh result. `export_stl()` writes the output.
- **`visualization`** — Camera positioning, bounds helpers, static PNG rendering, and interactive VTK viewer windows.
- **`cli`** — Typer-based CLI with `generate`, `view`, `new-config`, `show-config`, and `list-configs` commands.

## Python API

```python
from compass_web import PipelineConfig, run_pipeline, export_stl

config = PipelineConfig(
    radii=(10, 15, 20, 25, 20, 15, 10, 12),
    z_increment=13.38,
    seed_count=100,
    random_seed=42,
    extrusion_multiplier=-0.2,
    scale_x=0.5,
    scale_y=0.5,
)

result = run_pipeline(config)
path = export_stl(result, "exports")
print(f"Exported to {path}, volume valid: {result.is_valid_volume}")
```

## License

MIT
