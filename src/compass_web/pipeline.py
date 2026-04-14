"""End-to-end voronoi shell generation pipeline.

This module consolidates the full pipeline that was previously split across
notebook cells: loft -> clip -> voronoi -> intersect -> analyze -> build
solids -> export.  It is designed to work both from the notebook and from CLI.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

logger = logging.getLogger(__name__)

from compass_web.config import PipelineConfig, SMALL_CELL_EXTRUSION_FACTOR
from compass_web.lofted_surface_voronoi import (
    _loft_between_polylines,
    _merge_meshes,
    align_loops_and_loft,
    align_neighbouring_polylines,
    analyze_and_generate_surfaces,
    build_analysis_output_meshes,
    build_bounded_voronoi_cells,
    build_lofted_surface,
    build_mesh_printability_report,
    build_polyline_mesh,
    clean_meshes_without_naked_edges,
    clip_surface_in_half,
    close_mesh_boundaries,
    compact_polyline_shapes,
    count_connected_regions,
    default_snap_tolerance,
    extract_naked_edge_loops,
    extract_surface_mesh,
    find_polyline_neighbours,
    close_free_vertices,
    fix_polyline_surface_overlaps,
    resolve_pocket_cells,
    intersect_cells_with_surface,
    orient_normals_outward,
    pad_bounds,
    random_points_in_bounds,
    rebuild_polylines_from_discontinuities,
    scale_points_in_xy,
    scale_polydata_in_xy,
    split_and_offset_plane_faces,
    unify_mesh_normals,
    validate_polyline_surfaces,
)


ELONGATED_WIDTH_RATIO_THRESHOLD = 0.20
BOUNDARY_VERTEX_PLANE_TOL_FACTOR = 50


def restore_boundary_vertices(
    polylines: list[np.ndarray],
    original_polylines: list[np.ndarray],
    plane_axis: int,
    plane_coord: float,
    tolerance: float,
) -> list[np.ndarray]:
    """Restore the plane-axis coordinate of vertices that were at the cutting plane.

    Only the coordinate along `plane_axis` is restored; the other two axes
    keep their aligned values so neighbour tiling isn't broken.
    """
    plane_tol = tolerance * BOUNDARY_VERTEX_PLANE_TOL_FACTOR
    result: list[np.ndarray] = []
    for pl, orig in zip(polylines, original_polylines):
        out = pl.copy()
        n = min(len(pl), len(orig))
        for vi in range(n):
            if abs(float(orig[vi, plane_axis]) - plane_coord) < plane_tol:
                out[vi, plane_axis] = orig[vi, plane_axis]
        result.append(out)
    return result


def polyline_width_ratio(polyline: np.ndarray, tolerance: float) -> float:
    """OBB width ratio (min_span / max_span) of a polyline. 0 = linear, 1 = square."""
    pts = polyline[:-1] if len(polyline) > 1 and np.allclose(polyline[0], polyline[-1], atol=tolerance) else polyline
    if len(pts) < 3:
        return 1.0
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = centered.T @ centered
    _, eigvecs = np.linalg.eigh(cov)
    u_proj = centered @ eigvecs[:, 2]
    v_proj = centered @ eigvecs[:, 1]
    u_span = float(u_proj.max() - u_proj.min())
    v_span = float(v_proj.max() - v_proj.min())
    min_span = min(u_span, v_span)
    max_span = max(u_span, v_span)
    return min_span / max_span if max_span > tolerance else 0.0


def filter_elongated_polylines(
    polylines: list[np.ndarray],
    tolerance: float,
    width_ratio_threshold: float = ELONGATED_WIDTH_RATIO_THRESHOLD,
) -> tuple[list[np.ndarray], int, list[str]]:
    """Remove polylines whose OBB width ratio is below threshold (too elongated)."""
    if not polylines:
        return [], 0, []
    kept: list[np.ndarray] = []
    messages: list[str] = []
    removed = 0
    for i, pl in enumerate(polylines):
        wr = polyline_width_ratio(pl, tolerance)
        if wr < width_ratio_threshold:
            messages.append(f"Polyline {i}: removed (width_ratio={wr:.3f} < {width_ratio_threshold})")
            removed += 1
        else:
            kept.append(pl)
    return kept, removed, messages


def polyline_point_keys(
    polyline: np.ndarray,
    tolerance: float,
) -> set[tuple[int, int, int]]:
    unique_points = polyline[:-1] if len(polyline) > 1 else polyline
    return {
        tuple(np.round(np.asarray(point, dtype=float) / tolerance).astype(int).tolist())
        for point in unique_points
    }


def filter_isolated_polylines(
    polylines: list[np.ndarray],
    tolerance: float,
) -> tuple[list[np.ndarray], list[int], list[int]]:
    """Remove polylines that share no boundary points with any other polyline."""
    if not polylines:
        return [], [], []
    point_key_sets = [
        polyline_point_keys(pl, tolerance=tolerance) for pl in polylines
    ]
    kept_indices: list[int] = []
    discarded_indices: list[int] = []
    for index, point_keys in enumerate(point_key_sets):
        has_neighbor = any(
            index != other_index and len(point_keys.intersection(other_point_keys)) > 0
            for other_index, other_point_keys in enumerate(point_key_sets)
        )
        if has_neighbor:
            kept_indices.append(index)
        else:
            discarded_indices.append(index)
    return [polylines[i] for i in kept_indices], kept_indices, discarded_indices


@dataclass
class PipelineResult:
    """Holds everything produced by a single pipeline run."""

    trimesh_result: trimesh.Trimesh
    cell_solids: list[pv.PolyData]
    generated_surface: pv.PolyData
    is_valid_volume: bool
    stats: dict


def build_export_trimesh(solids_list: list[pv.PolyData]) -> trimesh.Trimesh:
    """Convert a list of PyVista cell solids into a single trimesh, with
    outward normals and fixed winding, rotated for printing (sliced face
    on XY).

    Individual cells that fail ``is_volume`` after normal repair are dropped
    to prevent one broken cell from poisoning the combined mesh.
    """
    cell_tms: list[trimesh.Trimesh] = []
    dropped = 0
    for solid in solids_list:
        solid_o = orient_normals_outward(solid)
        pts = np.asarray(solid_o.points, dtype=float)
        fraw = np.asarray(solid_o.faces, dtype=int)
        face_verts: list[list[int]] = []
        cursor = 0
        while cursor < len(fraw):
            n = int(fraw[cursor])
            if n == 3:
                face_verts.append(
                    [int(fraw[cursor + 1]), int(fraw[cursor + 2]), int(fraw[cursor + 3])]
                )
            cursor += n + 1
        tm = trimesh.Trimesh(
            vertices=pts, faces=np.array(face_verts), process=True
        )
        trimesh.repair.fix_normals(tm)
        trimesh.repair.fix_winding(tm)
        if tm.is_volume:
            cell_tms.append(tm)
        else:
            dropped += 1

    if dropped > 0:
        logger.info("build_export_trimesh: dropped %d/%d cells with broken normals",
                     dropped, dropped + len(cell_tms))

    if not cell_tms:
        return trimesh.Trimesh()

    combined = trimesh.util.concatenate(cell_tms)
    trimesh.repair.fix_normals(combined, multibody=True)

    rot = trimesh.transformations.rotation_matrix(np.radians(-90), [0, 1, 0])
    combined.apply_transform(rot)
    min_z = float(combined.vertices[:, 2].min())
    combined.vertices[:, 2] -= min_z
    return combined


def _extract_plane_edge_loops(
    mesh: pv.PolyData,
    plane_axis: int,
    plane_coord: float,
    tolerance: float,
) -> list[np.ndarray]:
    """Extract boundary sub-loops whose vertices lie near a coordinate plane.

    Unlike the previous version which required the ENTIRE loop to lie at
    the plane, this extracts the contiguous run of vertices near the plane
    from each boundary loop.  If a loop has vertices both on and off the
    plane, only the on-plane portion is returned (as a closed sub-loop
    with the first and last on-plane vertex connected).

    Returns closed polyline arrays suitable for ``_build_cap_from_loop``.
    """
    from compass_web.lofted_surface_voronoi import (
        _extract_polylines, _unique_polyline_points, _close_polyline,
    )

    boundary = mesh.extract_feature_edges(
        boundary_edges=True, feature_edges=False,
        manifold_edges=False, non_manifold_edges=False,
    ).clean()
    if boundary.n_points < 2 or boundary.n_lines == 0:
        return []

    plane_tol = tolerance * 50
    result: list[np.ndarray] = []

    for loop in _extract_polylines(boundary, tolerance=tolerance):
        unique = _unique_polyline_points(loop, tolerance)
        if len(unique) < 2:
            continue

        on_plane = np.abs(unique[:, plane_axis] - plane_coord) < plane_tol
        if not np.any(on_plane):
            continue

        if np.all(on_plane) and len(unique) >= 3:
            result.append(_close_polyline(unique, tolerance))
            continue

        n = len(unique)
        in_run = False
        run_pts: list[np.ndarray] = []
        runs: list[list[np.ndarray]] = []

        for scan in range(2 * n):
            idx = scan % n
            if on_plane[idx]:
                if not in_run:
                    in_run = True
                    run_pts = []
                run_pts.append(unique[idx].copy())
            else:
                if in_run:
                    runs.append(run_pts)
                    in_run = False
        if in_run and runs and np.allclose(run_pts[-1], runs[0][0], atol=tolerance):
            runs[0] = run_pts + runs[0]
        elif in_run:
            runs.append(run_pts)

        for run in runs:
            if len(run) < 2:
                continue
            arr = np.array(run, dtype=float)
            snapped = arr.copy()
            snapped[:, plane_axis] = plane_coord
            result.append(_close_polyline(snapped, tolerance))

    return result


def _build_cap_from_loop(
    loop: np.ndarray,
    tolerance: float,
) -> pv.PolyData:
    """Build a triangulated fan-cap from a boundary loop."""
    from compass_web.lofted_surface_voronoi import _unique_polyline_points, _fan_surface_from_center
    unique = _unique_polyline_points(loop, tolerance)
    if len(unique) < 3:
        return pv.PolyData()
    center = unique.mean(axis=0)
    return _fan_surface_from_center(center, unique)


def _build_cell_solids(
    cell_patches: list[pv.PolyData],
    *,
    loft_bbox_center: np.ndarray,
    scale_x: float,
    scale_y: float,
    slice_normal: tuple[float, float, float],
    slice_origin: tuple[float, float, float],
    tolerance: float,
) -> list[pv.PolyData]:
    """Build watertight cell solids from mesh patches — the Step 5 logic."""
    plane_axis = int(np.argmax(np.abs(slice_normal)))
    plane_coord = float(slice_origin[plane_axis])
    offset_amount = -2.0

    cell_solids: list[pv.PolyData] = []
    for cell_mesh in cell_patches:
        surf = extract_surface_mesh(cell_mesh)

        scaled_surf = scale_polydata_in_xy(
            surf, center=loft_bbox_center, scale_x=scale_x, scale_y=scale_y
        )
        _, open_edge_loops = extract_naked_edge_loops(surf, tolerance=tolerance)
        scaled_loops = [
            scale_points_in_xy(loop, center=loft_bbox_center, scale_x=scale_x, scale_y=scale_y)
            for loop in open_edge_loops
        ]
        loft_bands = [
            _loft_between_polylines(src, tgt)
            for src, tgt in zip(open_edge_loops, scaled_loops)
            if len(src) >= 2 and len(tgt) >= 2
        ]
        raw_solid = _merge_meshes(
            [p for p in [surf, scaled_surf] + loft_bands if p.n_cells > 0]
        )

        body_patch, moved_patch = split_and_offset_plane_faces(
            raw_solid,
            plane_normal=slice_normal,
            plane_origin=slice_origin,
            offset_amount=offset_amount,
            tolerance=tolerance,
        )

        if moved_patch.n_cells == 0:
            plane_loops = _extract_plane_edge_loops(
                raw_solid, plane_axis, plane_coord, tolerance,
            )
            if plane_loops:
                cap_meshes_fb: list[pv.PolyData] = []
                offset_cap_meshes_fb: list[pv.PolyData] = []
                for pl_loop in plane_loops:
                    cap = _build_cap_from_loop(pl_loop, tolerance)
                    if cap.n_cells == 0:
                        continue
                    cap_meshes_fb.append(cap)
                    offset_pts = np.asarray(cap.points, dtype=float).copy()
                    offset_pts[:, plane_axis] += offset_amount
                    offset_cap = cap.copy()
                    offset_cap.points = offset_pts
                    offset_cap_meshes_fb.append(offset_cap)

                if cap_meshes_fb:
                    body_patch = raw_solid
                    moved_patch = _merge_meshes(
                        [c for c in offset_cap_meshes_fb if c.n_cells > 0]
                    )

        if moved_patch.n_cells > 0:
            _, body_loops = extract_naked_edge_loops(body_patch, tolerance=tolerance)
            _, moved_loops = extract_naked_edge_loops(moved_patch, tolerance=tolerance)
            wall_lofts: list[pv.PolyData] = []
            used: set[int] = set()
            for bl in body_loops:
                bl_len = float(np.sum(np.linalg.norm(np.diff(bl, axis=0), axis=1))) if len(bl) >= 2 else 0.0
                best_mi: int | None = None
                best_d = float("inf")
                for mi, ml in enumerate(moved_loops):
                    if mi in used:
                        continue
                    d = float(np.linalg.norm(bl.mean(axis=0) - ml.mean(axis=0)))
                    ml_len = float(np.sum(np.linalg.norm(np.diff(ml, axis=0), axis=1))) if len(ml) >= 2 else 0.0
                    max_len = max(bl_len, ml_len)
                    len_ratio = min(bl_len, ml_len) / max_len if max_len > 0 else 0.0
                    if len_ratio < 0.3:
                        continue
                    if d < best_d:
                        best_d = d
                        best_mi = mi
                if best_mi is not None and len(bl) >= 2:
                    used.add(best_mi)
                    lr = align_loops_and_loft(bl, moved_loops[best_mi], tolerance=tolerance)
                    if lr.n_cells > 0:
                        wall_lofts.append(lr)
            solid = _merge_meshes(
                [p for p in [body_patch, moved_patch] + wall_lofts if p.n_cells > 0]
            )
        else:
            solid = raw_solid

        solid = close_mesh_boundaries(solid, tolerance=tolerance)
        solid = unify_mesh_normals(solid)
        for k in list(solid.cell_data.keys()):
            del solid.cell_data[k]
        cell_solids.append(solid)

    return cell_solids


def _filter_disconnected_cells(cell_solids: list[pv.PolyData]) -> list[pv.PolyData]:
    """Keep only cells that belong to the largest connected region."""
    non_empty = [s for s in cell_solids if s.n_cells > 0]
    if not non_empty:
        return cell_solids
    test_assembly = _merge_meshes(non_empty)
    n_regions = count_connected_regions(test_assembly)
    if n_regions <= 1:
        return cell_solids

    conn = test_assembly.connectivity()
    region_ids = np.asarray(conn["RegionId"], dtype=int)
    region_sizes = Counter(region_ids)
    largest_rid = max(region_sizes, key=region_sizes.get)

    face_offset = 0
    keep_mask: list[bool] = []
    for solid in cell_solids:
        if solid.n_cells == 0:
            keep_mask.append(False)
            continue
        mid = face_offset + solid.n_cells // 2
        in_main = int(region_ids[mid]) == largest_rid if mid < len(region_ids) else False
        keep_mask.append(in_main)
        face_offset += solid.n_cells

    return [s for s, k in zip(cell_solids, keep_mask) if k]


def run_pipeline(
    config: PipelineConfig,
    *,
    verbose: bool = True,
    apply_smoothing: bool = True,
) -> PipelineResult:
    """Execute the full voronoi shell pipeline for a given config.

    Set ``apply_smoothing=False`` to use ``config`` as-is (including optional
    ``z_levels``) without running radii/spacing smoothing.

    Returns a PipelineResult containing the trimesh, cell solids, assembled
    surface, volume validity flag, and summary stats dict.
    """
    import vtk as _vtk
    _vtk.vtkObject.GlobalWarningDisplayOff()

    from compass_web.smoothing import apply_smoothing_to_config

    smoothing = None
    if apply_smoothing:
        config, smoothing = apply_smoothing_to_config(config)

    surface_config = config.to_surface_config()
    point_config = config.to_point_config()
    extrusion_multiplier = config.effective_extrusion
    scale_x = config.scale_x
    scale_y = config.scale_y
    tolerance = config.line_tolerance

    if verbose:
        if smoothing is not None and smoothing.was_adjusted:
            print("Radii/spacing smoothed:")
            for adj in smoothing.adjustments:
                print(f"  {adj}")
        print(f"Radii: {list(surface_config.radii)}")
        print(f"Z positions: {list(surface_config.z_levels)}")
        print(f"Voronoi seeds: {point_config.seed_count}, random seed: {point_config.random_seed}")
        print(f"Extrusion: {extrusion_multiplier:.2f}, Scale X: {scale_x:.2f}, Scale Y: {scale_y:.2f}")

    full_surface = build_lofted_surface(surface_config)
    full_loft_bounds = full_surface.bounds
    loft_bbox_center = np.array([
        0.5 * (full_loft_bounds[0] + full_loft_bounds[1]),
        0.5 * (full_loft_bounds[2] + full_loft_bounds[3]),
        0.5 * (full_loft_bounds[4] + full_loft_bounds[5]),
    ], dtype=float)

    half_surface = clip_surface_in_half(
        full_surface,
        normal=surface_config.slice_normal,
        origin=surface_config.slice_origin,
    )
    padded_bounds = pad_bounds(half_surface.bounds, surface_config.bbox_padding)

    if verbose:
        print(f"Full loft: {full_surface.n_points} pts / {full_surface.n_cells} cells")
        print(f"Half surface: {half_surface.n_points} pts / {half_surface.n_cells} cells")

    seed_points = random_points_in_bounds(
        bounds=padded_bounds,
        count=point_config.seed_count,
        seed=point_config.random_seed,
    )
    voronoi_cells = build_bounded_voronoi_cells(seed_points, padded_bounds)
    raw_polylines = intersect_cells_with_surface(
        surface=half_surface, cells=voronoi_cells, tolerance=tolerance,
    )
    closed_polylines, _, _ = filter_isolated_polylines(raw_polylines, tolerance=tolerance)

    closed_polylines, compact_removed, compact_msgs = compact_polyline_shapes(
        closed_polylines, tolerance=tolerance,
    )
    if verbose and compact_removed > 0:
        for msg in compact_msgs:
            print(f"  {msg}")

    polyline_snap_tolerance = default_snap_tolerance(tolerance)
    closed_polylines = rebuild_polylines_from_discontinuities(
        closed_polylines,
        tolerance=tolerance,
        discontinuity_angle_degrees=176.0,
        neighbor_snap_tolerance=polyline_snap_tolerance,
    )

    closed_polylines, elongated_removed, elongated_msgs = filter_elongated_polylines(
        closed_polylines, tolerance=tolerance,
    )
    if verbose and elongated_removed > 0:
        print(f"Removed {elongated_removed} extreme elongated polyline(s) (WR < {ELONGATED_WIDTH_RATIO_THRESHOLD})")

    pre_alignment_polylines = [p.copy() for p in closed_polylines]

    polyline_neighbours = find_polyline_neighbours(
        closed_polylines, polyline_snap_tolerance,
    )

    closed_polylines = align_neighbouring_polylines(
        closed_polylines, tolerance=tolerance,
        slice_plane_x=float(surface_config.slice_origin[0]),
        neighbours=polyline_neighbours,
    )

    max_shift = 0.0
    for pre, post in zip(pre_alignment_polylines, closed_polylines):
        n = min(len(pre), len(post))
        if n > 0:
            shift = float(np.max(np.linalg.norm(pre[:n] - post[:n], axis=1)))
            max_shift = max(max_shift, shift)
    alignment_changed = max_shift > tolerance

    if verbose:
        if alignment_changed:
            print(f"Neighbour edge alignment applied (max vertex shift: {max_shift:.6f})")
        else:
            print("Neighbour edge alignment: no adjustment needed")

    pre_overlap_count = len(closed_polylines)
    closed_polylines, overlap_relocated, overlap_messages = fix_polyline_surface_overlaps(
        closed_polylines, tolerance=tolerance, neighbours=polyline_neighbours,
    )

    if len(closed_polylines) != pre_overlap_count:
        polyline_neighbours = find_polyline_neighbours(
            closed_polylines, polyline_snap_tolerance,
        )

    closed_polylines, free_snapped, free_msgs = close_free_vertices(
        closed_polylines, half_surface, tolerance=tolerance,
        neighbours=polyline_neighbours,
    )

    if verbose:
        if overlap_relocated > 0 or overlap_messages:
            for msg in overlap_messages:
                print(f"  {msg}")
        for msg in free_msgs:
            print(f"  {msg}")
        if overlap_relocated == 0 and free_snapped == 0 and not overlap_messages:
            print("Surface overlap check: no adjustments needed")
        print(f"Voronoi cells: {len(voronoi_cells)}, retained polylines: {len(closed_polylines)}")

    if not closed_polylines:
        return PipelineResult(
            trimesh_result=trimesh.Trimesh(),
            cell_solids=[],
            generated_surface=half_surface,
            is_valid_volume=False,
            stats={"polyline_count": 0},
        )

    if verbose:
        print("Classifying cells (large/small/extreme) from final polyline geometry...")

    curve_result = analyze_and_generate_surfaces(
        closed_polylines,
        loft_bounds=full_surface.bounds,
        tolerance=tolerance,
        extrusion_multiplier=extrusion_multiplier,
        small_cell_extrusion_factor=SMALL_CELL_EXTRUSION_FACTOR,
        extrusion_scale_origin=loft_bbox_center,
        planar_scale_factors=(scale_x, scale_y),
        slice_plane_x=surface_config.slice_origin[0],
    )

    analysis_output = build_analysis_output_meshes(
        curve_result.analyses,
        average_ratio=curve_result.average_ratio,
        loft_bounds=full_surface.bounds,
        tolerance=tolerance,
        extrusion_multiplier=extrusion_multiplier,
        small_cell_extrusion_factor=SMALL_CELL_EXTRUSION_FACTOR,
        slice_plane_x=surface_config.slice_origin[0],
    )

    mesh_cleanup = clean_meshes_without_naked_edges(
        list(analysis_output.output_meshes), tolerance=tolerance,
    )
    cell_patches = list(mesh_cleanup.kept_meshes)

    if verbose:
        print(f"Analyzed curves: {len(curve_result.analyses)}, cell patches: {len(cell_patches)}")

    cell_solids = _build_cell_solids(
        cell_patches,
        loft_bbox_center=loft_bbox_center,
        scale_x=scale_x,
        scale_y=scale_y,
        slice_normal=surface_config.slice_normal,
        slice_origin=surface_config.slice_origin,
        tolerance=tolerance,
    )
    cell_solids = _filter_disconnected_cells(cell_solids)

    if not cell_solids:
        return PipelineResult(
            trimesh_result=trimesh.Trimesh(),
            cell_solids=[],
            generated_surface=half_surface,
            is_valid_volume=False,
            stats={"polyline_count": len(closed_polylines), "cell_solid_count": 0},
        )

    generated_surface = _merge_meshes([s for s in cell_solids if s.n_cells > 0])
    result_mesh = build_export_trimesh(cell_solids)

    stats = {
        "polyline_count": len(closed_polylines),
        "elongated_removed": elongated_removed,
        "alignment_applied": alignment_changed,
        "alignment_max_shift": max_shift,
        "overlap_points_relocated": overlap_relocated,
        "curve_count": len(curve_result.analyses),
        "cell_patch_count": len(cell_patches),
        "cell_solid_count": len(cell_solids),
        "face_count": len(result_mesh.faces),
        "is_watertight": result_mesh.is_watertight,
        "is_volume": result_mesh.is_volume,
    }

    if verbose:
        print(
            f"Cell solids: {len(cell_solids)}, faces: {len(result_mesh.faces)}, "
            f"watertight: {result_mesh.is_watertight}, volume: {result_mesh.is_volume}"
        )

    return PipelineResult(
        trimesh_result=result_mesh,
        cell_solids=cell_solids,
        generated_surface=generated_surface,
        is_valid_volume=result_mesh.is_volume,
        stats=stats,
    )


def run_pipeline_with_retry(
    config: PipelineConfig,
    *,
    max_attempts: int = 10,
    verbose: bool = True,
    apply_smoothing: bool = True,
) -> tuple[PipelineResult, PipelineConfig]:
    """Run the pipeline, retrying with different seeds if the result is not a valid volume.

    Returns a tuple of (result, config_used) so the caller knows which seed succeeded.
    """
    result = run_pipeline(config, verbose=verbose, apply_smoothing=apply_smoothing)
    if result.is_valid_volume:
        return result, config

    if verbose:
        print("Mesh is not a valid volume. Starting auto-retry...")

    base_seed = config.random_seed
    for attempt in range(1, max_attempts + 1):
        new_seed = base_seed + attempt * 7
        if verbose:
            print(f"  Attempt {attempt}/{max_attempts} with seed {new_seed}...")
        retry_config = config.with_seed(new_seed)
        result = run_pipeline(retry_config, verbose=False, apply_smoothing=apply_smoothing)
        if result.is_valid_volume:
            if verbose:
                print(f"  SUCCESS with seed {new_seed}: {len(result.trimesh_result.faces)} faces")
            return result, retry_config
        elif verbose:
            print(f"  Seed {new_seed}: not a valid volume, trying next...")

    if verbose:
        print("  All attempts failed. Try adjusting other parameters.")
    return result, config.with_seed(base_seed + max_attempts * 7)


def export_stl(
    result: PipelineResult,
    export_dir: str | Path,
    *,
    suffix: str = "",
) -> Path:
    """Write the trimesh result to an STL file with a timestamped name."""
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"voronoi_shell_{ts}{suffix}.stl"
    path = export_dir / name
    result.trimesh_result.export(str(path))
    return path
