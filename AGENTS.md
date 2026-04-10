## Learned User Preferences

- Notebook cells should be separated by step; do not combine multiple stages in one cell or destroy previous results
- Always provide explanations and beginner-friendly comments in notebook code cells and markdown text cells
- Use native PyVista render windows for interactive 3D views (orbit/pan/zoom); do not rely on trame notebook backend
- Static screenshot cameras must zoom to fit the model tightly; use selection-aware framing, not full-scene bounds
- When modifying geometry logic, implement in both the Python module and the notebook; test with before-vs-after comparisons
- JSON config files must be saved inside the `configs/` folder, never outside it
- The CLI entrypoint command uses "materialize" naming, not "compass-web"
- Use `uv run jupyter lab` to launch the notebook; always use `uv run` for commands in this project
- Allow slider controls to have both drag and click-to-type input for all numeric parameters
- When adding new pipeline stages, keep existing results intact and add new code cells downstream
- Radii allow up to 3 decimal places; spacing loads from JSON, not hardcoded
- When searching geometry for matches, prefer targeted neighborhood-first search; expand to non-neighbors only for interior (non-boundary) cells; increase snap radius only for non-neighbour searches

## Learned Workspace Facts

- Parametric geometry generator: takes enhancement data from JSON, builds lofted-surface Voronoi 3D geometry
- Main module: `src/compass_web/lofted_surface_voronoi.py`; CLI: `src/compass_web/cli.py`; pipeline: `src/compass_web/pipeline.py`; config: `src/compass_web/config.py`
- Primary notebook: `lofted_surface_voronoi_generation.ipynb` — interactive development and visualization interface
- Geometry uses circles at different Z levels, lofted together, sliced by a YZ plane at x=0, with bounded Voronoi cells
- Non-uniform scaling applies to X and Y axes only (Z unchanged); pivot is center of initial full loft bounding box
- Extrusion: from original to scaled circle center; multiplier can be negative; small cells use factor 0.1; plane-face selection uses multi-pass vertex propagation (≥2 verts or ≥75% of face vertices near plane, bounded by distance limit); wall lofts use winding-direction check and closest-segment alignment; edge-cell walls trace boundary vertices to polyline origins
- Pipeline stages: intersect cells → filter isolated → filter extreme-elongated boundary cells (width_ratio threshold) → compact shapes (tail removal) → rebuild from discontinuities (incl. cross-polyline intersection points) → align neighbours (boundary vertices protected from displacement) → fix overlaps → close free vertices → extrude
- Shared helpers: `default_snap_tolerance(tolerance)` for canonical snap tolerance; `_splice_points_into_polyline()` for inserting vertices into polylines by segment
- Max geometry size: 150 units in XY (~150 mm), radius range 5–75; height = 7 × spacing
- Input configs: `data/lofted_surface_inputs.json` (radii, spacing), `data/voronoi_points_inputs.json` (seed, points); saved runs go to `configs/`
- Uses PyVista + VTK for 3D visualization and mesh ops; trame is unreliable in this environment; STL exports go to `exports/`
- Vertex-count validation: each cell's vertex count is checked against neighbours' shared vertices; missing vertices are added to close tiling gaps
