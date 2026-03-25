# Lofted Surface Voronoi Shell Generator

This project generates **3D-printable voronoi shell geometry** from parametric inputs. Each run produces a unique shape driven by radius, spacing, seed, and extrusion parameters.

## Pipeline

1. **Lofted surface** — 8 circles at different radii stacked along Z, lofted into a surface
2. **Voronoi cells** — random seed points generate bounded voronoi cells
3. **Intersection curves** — voronoi cells intersect the lofted surface, producing polyline curves
4. **Per-cell solids** — each cell patch is scaled, its open edges lofted into walls, and closed into a watertight solid
5. **Plane face offset** — faces at the cutting plane (x=0) are offset by -2mm to prevent thin-wall artifacts
6. **Assembly** — all cell solids are combined, normals oriented outward, and exported as STL via trimesh
7. **Auto-retry** — if the mesh fails volume validation, the pipeline retries with different voronoi seeds (up to 10 attempts)

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

## Config management

Slider values can be saved/loaded via the **Config** dropdown in the notebook. Configs are stored as JSON in the `configs/` directory with timestamps. Duplicate configs are not saved.

## Files

- `data/lofted_surface_inputs.json` — default radii and spacing
- `data/voronoi_points_inputs.json` — default seed count and random seed
- `src/compass_web/lofted_surface_voronoi.py` — all geometry functions
- `lofted_surface_voronoi_generation.ipynb` — interactive notebook
- `exports/` — timestamped STL outputs (gitignored)
- `configs/` — saved parameter configs (gitignored)
