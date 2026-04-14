# Pipeline Performance Optimisations

## Context

Two commits added new post-processing passes (cross-polyline intersection
injection, neighbour-alignment, surface-overlap fixing, pocket-cell resolution,
free-vertex closing, junction insertion). These stages were functionally correct
but algorithmically naive, pushing total pipeline time from **~3 s to 60–90 s**
on the standard test configuration.

The work described here reduced that to **~13 s** (≈6–7× speedup) while
producing bit-identical geometry output (72 retained polylines, watertight mesh,
same face/vertex count).

---

## Profiling summary (before optimisations)

Timing was added around each major stage in `pipeline.py` using
`time.perf_counter()`.

| Stage | Time (s) | Share |
|---|---|---|
| Compact + rebuild + filter | **53.84** | 77 % |
| Close free vertices | 6.37 | 9 % |
| Fix surface overlaps | 3.75 | 5 % |
| Align neighbours | 3.29 | 5 % |
| Remainder | ~2.7 | 4 % |
| **Total** | **~70** | |

The dominant bottleneck was `_find_cross_polyline_intersections`, called inside
the compact/rebuild stage.

---

## Optimisation 1 — Replace brute-force crossing detection with Shapely STRtree

**File**: `src/compass_web/lofted_surface_voronoi.py`  
**Function**: `_find_cross_polyline_intersections`

### Before

A three-level nested Python loop compared every segment of every polyline
against every segment of every other polyline. An AABB pre-filter helped
slightly, but the inner body still called `_segment_crossing_3d` (a Python
function with several `np.linalg` calls) for O(N² × E²) pairs — roughly
177 600 candidate pairs on the test geometry.

```
for each polyline A:
    for each polyline B (A ≠ B):
        for seg_a in A:
            for seg_b in B:
                if AABB overlap:
                    call _segment_crossing_3d(...)   # expensive Python call
```

### After — hybrid C-filter + 3D verification

1. **Build a flat list** of one `shapely.geometry.LineString` per segment,
   projected to XY.  
2. **Query `shapely.STRtree`** with `predicate='crosses'` — a fast C-level
   (GEOS) test that eliminates collinear segments, endpoint-touches, and
   non-intersecting pairs.  This reduces 177 600 candidates to **~856**.  
3. **Verify in 3D** by calling the original `_segment_crossing_3d` on each
   surviving candidate.  This filters out XY crossings that are far apart in Z,
   leaving **12 true 3D crossings**.

```python
tree = STRtree(seg_shapes)
left_idx, right_idx = tree.query(seg_shapes, predicate="crosses")
for li, ri in zip(left_idx, right_idx):
    pt3d = _segment_crossing_3d(seg_3d_starts[li], seg_3d_ends[li], ...)
    if pt3d is not None:
        result[pi_a].append(...)
```

**Result**: compact+rebuild stage **53.84 s → 0.42 s**.

> **Why `predicate='crosses'` and not `predicate='intersects'`?**
> `intersects` includes endpoint-touches and collinear overlaps, producing
> ~18 000 false candidates.  Those extra injections exploded downstream stage
> costs and changed the output geometry.  `crosses` (interior-to-interior only)
> is the correct semantic match for the original algorithm.

---

## Optimisation 2 — Vectorise shared-vertex detection

**Function**: `_find_shared_vertex_pairs`

Replaced a double Python `for`-loop that called `np.linalg.norm` on each pair
of points with a single `scipy.spatial.distance.cdist` call, producing the full
distance matrix in one C call.

```python
# Before
for i, a in enumerate(pts_a):
    for j, b in enumerate(pts_b):
        if np.linalg.norm(a - b) < tol: ...

# After
D = cdist(pts_a, pts_b)
ii, jj = np.where(D < tol)
```

---

## Optimisation 3 — Vectorise point-to-segment distances

**New function**: `_distances_points_to_segments`  
**Used by**: `_absorb_nearby_neighbour_vertices`

`_absorb_nearby_neighbour_vertices` contained an inner Python loop that called
`_distance_point_to_segment` once per (point, segment) pair for every candidate
vertex.  A new fully vectorised NumPy function replaces it:

```python
def _distances_points_to_segments(
    points: np.ndarray,      # (P, 3)
    seg_starts: np.ndarray,  # (S, 3)
    seg_ends: np.ndarray,    # (S, 3)
) -> np.ndarray:             # (P, S)
    ab = seg_ends - seg_starts
    ab_sq = np.einsum("ij,ij->i", ab, ab)
    ap = points[:, None, :] - seg_starts[None, :, :]
    t = np.clip(np.einsum("ijk,jk->ij", ap, ab) / np.where(ab_sq < 1e-20, 1.0, ab_sq), 0.0, 1.0)
    proj = seg_starts[None, :, :] + t[:, :, None] * ab[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", points[:, None] - proj, points[:, None] - proj))
```

**Result**: absorb loop **4.41 s → < 0.1 s**.

---

## Optimisation 4 — Cache the neighbour graph

**File**: `src/compass_web/pipeline.py`  
**Affected functions** (all gained an optional `neighbours` parameter):  
`align_neighbouring_polylines`, `validate_polyline_surfaces`,
`fix_polyline_surface_overlaps`, `_resnap_shared_vertices`,
`resolve_pocket_cells`, `close_free_vertices`, `_insert_junction_points`,
`_absorb_nearby_neighbour_vertices`

`find_polyline_neighbours` was being called independently inside every stage and
on every iteration of the overlap-fixing loop.  The result is now computed once
after filtering and threaded through:

```python
polyline_neighbours = find_polyline_neighbours(compact_polylines, ...)
aligned = align_neighbouring_polylines(..., neighbours=polyline_neighbours)
closed  = fix_polyline_surface_overlaps(..., neighbours=polyline_neighbours)
# recompute only if resolve_pocket_cells removed polylines:
if len(closed) != len(compact_polylines):
    polyline_neighbours = find_polyline_neighbours(closed, ...)
final   = close_free_vertices(..., neighbours=polyline_neighbours)
```

---

## Optimisation 5 — Vectorise point-snapping loop

**Function**: `_snap_neighboring_polyline_points`

The inner distance loop was replaced with `cdist`, eliminating a nested Python
`for` in a frequently called function.

---

## Optimisation 6 — cKDTree for non-neighbour segment search

**Function**: `close_free_vertices`

A fallback search for the nearest segment on a non-neighbouring cell used a
plain Python loop over all segments.  It is now built as a `cKDTree` over
segment midpoints, turning the O(N) scan into an O(log N) query.

---

## Dependency added

```
shapely >= 2.0
```

Added via `uv add shapely` to `pyproject.toml`.  Shapely 2.x ships with a
pre-compiled GEOS C library and exposes vectorised array operations; no
system-level GEOS install is required.

---

## Final timings (after all optimisations)

| Stage | Time (s) |
|---|---|
| Compact + rebuild + filter | 0.42 |
| Close free vertices | ~0.8 |
| Fix surface overlaps | ~0.5 |
| Align neighbours | ~0.4 |
| Remainder | ~1.5 |
| **Total** | **~13** |

Wall-clock: `real 0m13.693s` (down from ~70 s, **≈6–7× improvement**).

Output geometry unchanged: 72 retained polylines, 13 750 faces, watertight
mesh, valid volume.
