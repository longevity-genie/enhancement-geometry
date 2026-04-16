from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path

import numpy as np
import pyvista as pv
import vtk
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.spatial.distance import cdist
from shapely import STRtree
from shapely.geometry import LineString as _ShapelyLine

MIN_RADIUS = 5.0
MAX_RADIUS = 75.0
MAX_MODEL_SPAN = 150.0
EXTREME_ASPECT_RATIO_THRESHOLD = 2.0
EXTREME_SCALE_FACTOR = 0.3
EXTREME_EXTRUSION_FACTOR = 0.5
EXTREME_PLANARITY_RATIO = 0.03


@dataclass(frozen=True)
class LoftedVoronoiConfig:
    radii: tuple[float, ...]
    z_levels: tuple[float, ...]
    z_increment: float
    circle_resolution: int
    slice_normal: tuple[float, float, float]
    slice_origin: tuple[float, float, float]
    bbox_padding: float
    line_tolerance: float


@dataclass(frozen=True)
class VoronoiPointConfig:
    seed_count: int
    random_seed: int


@dataclass(frozen=True)
class CurveAnalysis:
    original_polyline: np.ndarray
    followup_polyline: np.ndarray
    discontinuity_points: np.ndarray
    plane_origin: np.ndarray
    plane_u: np.ndarray
    plane_v: np.ndarray
    plane_normal: np.ndarray
    circle_center: np.ndarray
    circle_radius: float
    bbox_mesh: pv.PolyData
    bbox_center: np.ndarray
    bbox_volume: float
    curve_length: float
    ratio: float
    bbox_aspect_ratio: float
    planarity_ratio: float
    scaled_circle_center: np.ndarray
    extrusion_base_vector: np.ndarray
    offset_direction: np.ndarray


@dataclass(frozen=True)
class SurfaceGenerationResult:
    analyses: tuple[CurveAnalysis, ...]
    average_ratio: float
    followup_polylines: tuple[np.ndarray, ...]
    generated_surface: pv.PolyData
    larger_surface: pv.PolyData
    smaller_surface: pv.PolyData


@dataclass(frozen=True)
class MeshCleanupResult:
    kept_meshes: tuple[pv.PolyData, ...]
    kept_surface_meshes: tuple[pv.PolyData, ...]
    kept_indices: tuple[int, ...]
    removed_indices: tuple[int, ...]
    naked_edge_meshes: tuple[pv.PolyData, ...]
    naked_edge_loops_by_mesh: tuple[tuple[np.ndarray, ...], ...]


@dataclass(frozen=True)
class MeshPrintabilityReport:
    point_count: int
    face_count: int
    connected_region_count: int
    boundary_edge_count: int
    boundary_loop_count: int
    non_manifold_edge_count: int
    is_closed: bool
    is_printable: bool


@dataclass(frozen=True)
class MeshPreparationResult:
    mesh: pv.PolyData
    initial_report: MeshPrintabilityReport
    final_report: MeshPrintabilityReport
    repair_attempted: bool
    repair_method: str | None


@dataclass(frozen=True)
class AnalysisOutputMeshes:
    preview_meshes: tuple[pv.PolyData, ...]
    output_meshes: tuple[pv.PolyData, ...]
    output_modes: tuple[str, ...]
    removed_by_retained_volume_indices: tuple[int, ...]


def load_generation_config(path: str | Path) -> LoftedVoronoiConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    radii = tuple(float(value) for value in raw["radii"])
    if len(radii) != 8:
        raise ValueError(f"Expected 8 radii, received {len(radii)}.")
    for radius in radii:
        if radius < MIN_RADIUS or radius > MAX_RADIUS:
            raise ValueError(f"Radius {radius} is outside the [{MIN_RADIUS:.0f}, {MAX_RADIUS:.0f}] range.")
        if round(radius, 3) != radius:
            raise ValueError(f"Radius {radius} must use at most 3 decimal places.")
    max_width = 2.0 * max(radii)
    if max_width > MAX_MODEL_SPAN:
        raise ValueError(
            f"Loft width {max_width:.2f} exceeds the {MAX_MODEL_SPAN:.0f} unit limit."
        )

    z_increment = float(raw.get("z_increment", 12.0))
    if z_increment <= 0.0:
        raise ValueError("z_increment must be greater than 0.")
    z_levels = tuple(index * z_increment for index in range(len(radii)))

    circle_resolution = int(raw.get("circle_resolution", 96))
    if circle_resolution < 8:
        raise ValueError("circle_resolution must be at least 8.")

    bbox_padding = float(raw.get("bbox_padding", 2.0))
    line_tolerance = float(raw.get("line_tolerance", 1e-4))

    return LoftedVoronoiConfig(
        radii=radii,
        z_levels=z_levels,
        z_increment=z_increment,
        circle_resolution=circle_resolution,
        slice_normal=(1.0, 0.0, 0.0),
        slice_origin=(0.0, 0.0, 0.0),
        bbox_padding=bbox_padding,
        line_tolerance=line_tolerance,
    )


def load_voronoi_point_config(path: str | Path) -> VoronoiPointConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    seed_count = int(raw["seed_count"])
    if seed_count < 2:
        raise ValueError("seed_count must be at least 2.")

    return VoronoiPointConfig(
        seed_count=seed_count,
        random_seed=int(raw["random_seed"]),
    )


def build_lofted_surface(config: LoftedVoronoiConfig) -> pv.PolyData:
    rings = [
        _circle_points(radius=radius, z_level=z_level, resolution=config.circle_resolution)
        for radius, z_level in zip(config.radii, config.z_levels, strict=True)
    ]
    points = np.vstack(rings)
    ring_size = config.circle_resolution
    faces: list[int] = []

    for ring_index in range(len(rings) - 1):
        ring_offset = ring_index * ring_size
        next_ring_offset = (ring_index + 1) * ring_size
        for point_index in range(ring_size):
            next_point_index = (point_index + 1) % ring_size
            a = ring_offset + point_index
            b = ring_offset + next_point_index
            c = next_ring_offset + next_point_index
            d = next_ring_offset + point_index
            faces.extend([3, a, b, c, 3, a, c, d])

    return pv.PolyData(points, faces=np.array(faces, dtype=np.int64)).clean().triangulate()


def clip_surface_in_half(
    surface: pv.PolyData,
    normal: tuple[float, float, float],
    origin: tuple[float, float, float] | None = None,
) -> pv.PolyData:
    clip_origin = origin or (0.0, 0.0, 0.0)
    clipped = surface.clip(normal=normal, origin=clip_origin, invert=False)
    return clipped.clean().triangulate()


def pad_bounds(bounds: tuple[float, float, float, float, float, float], padding: float) -> tuple[float, float, float, float, float, float]:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return (
        xmin - padding,
        xmax + padding,
        ymin - padding,
        ymax + padding,
        zmin - padding,
        zmax + padding,
    )


def random_points_in_bounds(
    bounds: tuple[float, float, float, float, float, float],
    count: int,
    seed: int,
) -> np.ndarray:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    lower = np.array([xmin, ymin, zmin], dtype=float)
    upper = np.array([xmax, ymax, zmax], dtype=float)
    margin = np.maximum((upper - lower) * 1e-3, 1e-6)
    rng = np.random.default_rng(seed)
    return rng.uniform(lower + margin, upper - margin, size=(count, 3))


def build_bounded_voronoi_cells(
    seed_points: np.ndarray,
    bounds: tuple[float, float, float, float, float, float],
) -> list[pv.PolyData]:
    return [_build_single_cell(seed_points, index, bounds) for index in range(len(seed_points))]


def intersect_cells_with_surface(
    surface: pv.PolyData,
    cells: list[pv.PolyData],
    tolerance: float,
) -> list[np.ndarray]:
    unique_polylines: list[np.ndarray] = []
    seen_keys: set[tuple[tuple[int, int, int], ...]] = set()

    triangulated_surface = surface.triangulate().clean()
    vtk.vtkObject.GlobalWarningDisplayOff()

    cells_with_intersection: set[int] = set()

    for ci, cell in enumerate(cells):
        if not _bounds_overlap(triangulated_surface.bounds, cell.bounds):
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*not valid.*vtkOriginalCellIds.*")
            intersection, _, _ = triangulated_surface.intersection(
                cell.triangulate().clean(),
                split_first=False,
                split_second=False,
            )
            intersection = _strip_stale_cell_arrays(intersection)
        if intersection.n_points == 0 or intersection.n_lines == 0:
            continue

        cells_with_intersection.add(ci)
        for polyline in _extract_polylines(intersection, tolerance=tolerance):
            polyline_key = _canonical_polyline_key(polyline, tolerance=tolerance)
            if polyline_key in seen_keys:
                continue
            seen_keys.add(polyline_key)
            unique_polylines.append(polyline)

    uncovered = _find_uncovered_surface_patches(
        triangulated_surface, unique_polylines, tolerance,
    )
    for polyline in uncovered:
        polyline_key = _canonical_polyline_key(polyline, tolerance=tolerance)
        if polyline_key not in seen_keys:
            seen_keys.add(polyline_key)
            unique_polylines.append(polyline)

    return unique_polylines


def _find_uncovered_surface_patches(
    surface: pv.PolyData,
    existing_polylines: list[np.ndarray],
    tolerance: float,
) -> list[np.ndarray]:
    """Find surface regions not near any existing intersection polyline.

    A surface face is "covered" if all its vertices are within a distance
    threshold of at least one polyline segment.  Uncovered faces form
    patches whose boundaries become new polylines.
    """
    result: list[np.ndarray] = []
    if surface.n_cells == 0 or not existing_polylines:
        return result

    surf_pts = np.asarray(surface.points, dtype=float)
    n_verts = len(surf_pts)

    all_poly_pts: list[np.ndarray] = []
    for pl in existing_polylines:
        u = _unique_polyline_points(pl, tolerance)
        if len(u) > 0:
            all_poly_pts.append(u)
    if not all_poly_pts:
        return result
    poly_cloud = np.vstack(all_poly_pts)

    min_dist = np.full(n_verts, float("inf"), dtype=float)
    chunk = 500
    for start in range(0, len(poly_cloud), chunk):
        pc = poly_cloud[start : start + chunk]
        d = np.linalg.norm(surf_pts[:, None, :] - pc[None, :, :], axis=2)
        min_dist = np.minimum(min_dist, d.min(axis=1))

    coverage_radius = max(tolerance * 500, 2.0)
    uncovered_mask = min_dist > coverage_radius

    if not np.any(uncovered_mask):
        return result

    face_arr = np.asarray(surface.faces, dtype=int)
    cursor = 0
    uncovered_face_ids: list[int] = []
    for fi in range(surface.n_cells):
        nv = face_arr[cursor]
        vert_ids = face_arr[cursor + 1 : cursor + 1 + nv]
        if np.all(uncovered_mask[vert_ids]):
            uncovered_face_ids.append(fi)
        cursor += nv + 1

    if not uncovered_face_ids:
        return result

    uncovered_sub = surface.extract_cells(np.array(uncovered_face_ids, dtype=int))
    if uncovered_sub.n_cells == 0:
        return result
    uncovered_sub = _strip_stale_cell_arrays(uncovered_sub.clean())

    boundary = uncovered_sub.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    ).clean()

    if boundary.n_points < 3 or boundary.n_lines == 0:
        return result

    min_area = tolerance * 100
    for polyline in _extract_polylines(boundary, tolerance=tolerance):
        unique = _unique_polyline_points(polyline, tolerance)
        if len(unique) < 3:
            continue
        po, pu, pv_vec, _ = _fit_plane(unique)
        p2d = _project_to_plane(unique, po, pu, pv_vec)
        area = _polygon_area_2d(p2d)
        if area > min_area:
            result.append(polyline)

    return result


def intersect_mesh_with_plane(
    mesh: pv.PolyData,
    normal: tuple[float, float, float],
    origin: tuple[float, float, float] | None = None,
    tolerance: float = 1e-4,
) -> list[np.ndarray]:
    if mesh.n_points == 0 or mesh.n_cells == 0:
        return []
    clip_origin = origin or (0.0, 0.0, 0.0)
    sliced = mesh.slice(normal=normal, origin=clip_origin)
    if sliced.n_points == 0:
        return []
    return _extract_polylines(sliced, tolerance=tolerance)


def filter_segments_against_curves(
    segments: list[np.ndarray],
    reference_curves: list[np.ndarray],
    tolerance: float,
) -> list[np.ndarray]:
    if not segments or not reference_curves:
        return segments

    ref_keys: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for curve in reference_curves:
        for si in range(len(curve) - 1):
            p1 = tuple(np.round(curve[si] / tolerance).astype(int).tolist())
            p2 = tuple(np.round(curve[si + 1] / tolerance).astype(int).tolist())
            ref_keys.add((min(p1, p2), max(p1, p2)))

    kept: list[np.ndarray] = []
    for seg in segments:
        if len(seg) != 2:
            kept.append(seg)
            continue
        s1 = tuple(np.round(seg[0] / tolerance).astype(int).tolist())
        s2 = tuple(np.round(seg[1] / tolerance).astype(int).tolist())
        seg_key = (min(s1, s2), max(s1, s2))
        if seg_key not in ref_keys:
            kept.append(seg)

    return kept


def filter_naked_loops_against_base_polylines(
    naked_loops: list[np.ndarray],
    base_polylines: list[np.ndarray],
    tolerance: float,
    overlap_threshold: float = 0.3,
) -> list[np.ndarray]:
    if not naked_loops or not base_polylines:
        return naked_loops

    base_keys: set[tuple[int, ...]] = set()
    for polyline in base_polylines:
        for pt in polyline:
            base_keys.add(tuple(np.round(pt / tolerance).astype(int).tolist()))

    kept: list[np.ndarray] = []
    for loop in naked_loops:
        loop_pts = loop[:-1] if len(loop) > 1 and np.allclose(loop[0], loop[-1], atol=tolerance) else loop
        if len(loop_pts) == 0:
            continue
        matching = sum(
            1 for pt in loop_pts
            if tuple(np.round(pt / tolerance).astype(int).tolist()) in base_keys
        )
        overlap = matching / max(len(loop_pts), 1)
        if overlap < overlap_threshold:
            kept.append(loop)

    return kept


def weld_mesh_vertices(mesh: pv.PolyData, tolerance: float | None = None) -> pv.PolyData:
    if mesh.n_points == 0:
        return mesh
    cleaned = mesh.clean(tolerance=tolerance) if tolerance is not None else mesh.clean()
    return cleaned.triangulate()


def build_polyline_mesh(polylines: list[np.ndarray]) -> pv.PolyData:
    if not polylines:
        return pv.PolyData()

    points: list[np.ndarray] = []
    lines: list[int] = []
    point_offset = 0

    for polyline in polylines:
        polyline_points = polyline
        points.extend(polyline_points)
        lines.extend([len(polyline_points), *range(point_offset, point_offset + len(polyline_points))])
        point_offset += len(polyline_points)

    return pv.PolyData(np.array(points, dtype=float), lines=np.array(lines, dtype=np.int64))


def scale_points_in_xy(
    points: np.ndarray,
    center: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    scaled_points = np.asarray(points, dtype=float).copy()
    scaled_points[:, 0] = center[0] + scale_x * (scaled_points[:, 0] - center[0])
    scaled_points[:, 1] = center[1] + scale_y * (scaled_points[:, 1] - center[1])
    return scaled_points


def scale_polydata_in_xy(
    mesh: pv.PolyData,
    center: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> pv.PolyData:
    if mesh.n_points == 0:
        return pv.PolyData()
    scaled = mesh.copy(deep=True)
    scaled.points = scale_points_in_xy(
        mesh.points,
        center=center,
        scale_x=scale_x,
        scale_y=scale_y,
    )
    return scaled


def count_connected_regions(mesh: pv.PolyData) -> int:
    if mesh.n_cells == 0:
        return 0
    connected = mesh.connectivity()
    region_ids = np.asarray(connected["RegionId"], dtype=int)
    return int(np.unique(region_ids).size) if region_ids.size > 0 else 0


def extract_surface_mesh(mesh: pv.PolyData) -> pv.PolyData:
    if mesh.n_points == 0 or mesh.n_cells == 0:
        return pv.PolyData()
    surface = mesh.extract_surface(algorithm="dataset_surface").clean()
    if surface.n_cells > 0:
        surface = _strip_stale_cell_arrays(_strip_stale_cell_arrays(surface).triangulate())
    return surface


def unify_mesh_normals(mesh: pv.PolyData) -> pv.PolyData:
    if mesh.n_points == 0 or mesh.n_cells == 0:
        return pv.PolyData()
    return mesh.compute_normals(
        consistent_normals=True,
        auto_orient_normals=True,
        cell_normals=True,
        point_normals=True,
        flip_normals=False,
    )


def remove_closed_regions(mesh: pv.PolyData) -> pv.PolyData:
    if mesh.n_points == 0 or mesh.n_cells == 0:
        return pv.PolyData()

    n_regions = count_connected_regions(mesh)
    if n_regions <= 1:
        boundary = mesh.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=False,
        ).clean()
        if boundary.n_cells > 0:
            return mesh
        return pv.PolyData()

    connected = mesh.connectivity()
    region_ids = np.asarray(connected["RegionId"], dtype=int)
    keep_mask = np.zeros(connected.n_cells, dtype=bool)

    for rid in range(n_regions):
        region_cell_mask = region_ids == rid
        region = connected.extract_cells(np.flatnonzero(region_cell_mask))
        region_surf = extract_surface_mesh(region)
        if region_surf.n_cells == 0:
            continue
        boundary = region_surf.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=False,
        ).clean()
        if boundary.n_cells > 0:
            keep_mask[region_cell_mask] = True

    if keep_mask.all():
        return mesh

    kept = connected.extract_cells(np.flatnonzero(keep_mask))
    return kept.extract_surface().triangulate().clean() if kept.n_cells > 0 else pv.PolyData()


def resolve_non_manifold_faces(surface: pv.PolyData) -> pv.PolyData:
    if surface.n_points == 0 or surface.n_cells == 0:
        return pv.PolyData()

    surface = surface.triangulate().clean()
    points = np.asarray(surface.points, dtype=float)
    faces_raw = np.asarray(surface.faces, dtype=int)

    face_verts: list[tuple[int, int, int]] = []
    cursor = 0
    while cursor < len(faces_raw):
        n = int(faces_raw[cursor])
        if n == 3:
            face_verts.append((
                int(faces_raw[cursor + 1]),
                int(faces_raw[cursor + 2]),
                int(faces_raw[cursor + 3]),
            ))
        cursor += n + 1

    if not face_verts:
        return surface

    def _face_edges(fi: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        a, b, c = face_verts[fi]
        return (min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(a, c), max(a, c))

    face_key_groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for fi, (a, b, c) in enumerate(face_verts):
        face_key_groups[tuple(sorted((a, b, c)))].append(fi)

    remove_set: set[int] = set()
    for group in face_key_groups.values():
        if len(group) > 1:
            for fi in group[1:]:
                remove_set.add(fi)

    for _iteration in range(200):
        alive = [fi for fi in range(len(face_verts)) if fi not in remove_set]
        edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
        for fi in alive:
            for edge in _face_edges(fi):
                edge_to_faces[edge].append(fi)

        nm_edges = {e: fis for e, fis in edge_to_faces.items() if len(fis) > 2}
        if not nm_edges:
            break

        manifold_count: dict[int, int] = defaultdict(int)
        for fi in alive:
            for edge in _face_edges(fi):
                if len(edge_to_faces[edge]) == 2:
                    manifold_count[fi] += 1

        votes_to_remove: dict[int, int] = defaultdict(int)
        for edge, face_indices in nm_edges.items():
            ranked = sorted(face_indices, key=lambda fi: manifold_count.get(fi, 0), reverse=True)
            for fi in ranked[2:]:
                votes_to_remove[fi] += 1

        if not votes_to_remove:
            break

        removed_this_round: set[int] = set()
        for fi in sorted(votes_to_remove, key=lambda fi: votes_to_remove[fi], reverse=True):
            if fi in removed_this_round:
                continue
            removed_this_round.add(fi)

        remove_set.update(removed_this_round)

    if not remove_set:
        return surface

    kept_faces = [face_verts[fi] for fi in range(len(face_verts)) if fi not in remove_set]
    if not kept_faces:
        return pv.PolyData()

    new_faces: list[int] = []
    for a, b, c in kept_faces:
        new_faces.extend([3, a, b, c])

    return pv.PolyData(points.copy(), faces=np.array(new_faces, dtype=np.int64)).clean().triangulate()


def extract_naked_edge_loops(surface: pv.PolyData, tolerance: float) -> tuple[pv.PolyData, list[np.ndarray]]:
    if surface.n_points == 0:
        return pv.PolyData(), []
    boundary_edges = surface.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    ).clean()
    polylines = _extract_polylines(boundary_edges, tolerance=tolerance)
    return build_polyline_mesh(polylines), polylines


def clean_meshes_without_naked_edges(
    meshes: list[pv.PolyData],
    tolerance: float,
) -> MeshCleanupResult:
    kept_meshes: list[pv.PolyData] = []
    kept_surface_meshes: list[pv.PolyData] = []
    kept_indices: list[int] = []
    removed_indices: list[int] = []
    naked_edge_meshes: list[pv.PolyData] = []
    naked_edge_loops_by_mesh: list[tuple[np.ndarray, ...]] = []

    for index, mesh in enumerate(meshes):
        surface = extract_surface_mesh(mesh)
        if _should_remove_cell_surface(surface, tolerance=tolerance):
            removed_indices.append(index)
            continue
        naked_edge_mesh, naked_edge_loops = extract_naked_edge_loops(surface, tolerance=tolerance)
        if not naked_edge_loops:
            removed_indices.append(index)
            continue
        kept_meshes.append(mesh)
        kept_surface_meshes.append(surface)
        kept_indices.append(index)
        naked_edge_meshes.append(naked_edge_mesh)
        naked_edge_loops_by_mesh.append(tuple(naked_edge_loops))

    return MeshCleanupResult(
        kept_meshes=tuple(kept_meshes),
        kept_surface_meshes=tuple(kept_surface_meshes),
        kept_indices=tuple(kept_indices),
        removed_indices=tuple(removed_indices),
        naked_edge_meshes=tuple(naked_edge_meshes),
        naked_edge_loops_by_mesh=tuple(naked_edge_loops_by_mesh),
    )


def _should_remove_cell_surface(surface: pv.PolyData, tolerance: float) -> bool:
    if surface.n_points == 0 or surface.n_cells == 0:
        return True

    boundary_edges = surface.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    ).clean()
    boundary_edge_count = int(boundary_edges.n_cells)

    non_manifold_edges = surface.extract_feature_edges(
        boundary_edges=False,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=True,
    ).clean()
    non_manifold_edge_count = int(non_manifold_edges.n_cells)

    if boundary_edge_count == 0:
        return True

    if non_manifold_edge_count > 0 and boundary_edge_count == 0:
        return True

    if non_manifold_edge_count > 2 * boundary_edge_count:
        return True

    return False


def join_two_point_segments_into_polylines(
    segments: list[np.ndarray],
    tolerance: float,
) -> list[np.ndarray]:
    if not segments:
        return []

    point_index_by_key: dict[tuple[int, int, int], int] = {}
    canonical_points: list[np.ndarray] = []

    def get_or_create_index(point: np.ndarray) -> int:
        key = tuple(np.round(np.asarray(point, dtype=float) / tolerance).astype(int).tolist())
        existing_index = point_index_by_key.get(key)
        if existing_index is not None:
            return existing_index
        new_index = len(canonical_points)
        point_index_by_key[key] = new_index
        canonical_points.append(np.asarray(point, dtype=float))
        return new_index

    unique_edges: set[tuple[int, int]] = set()
    for segment in segments:
        if len(segment) != 2:
            continue
        start_index = get_or_create_index(segment[0])
        end_index = get_or_create_index(segment[1])
        if start_index == end_index:
            continue
        unique_edges.add(tuple(sorted((start_index, end_index))))

    adjacency: dict[int, set[int]] = defaultdict(set)
    for start_index, end_index in unique_edges:
        adjacency[start_index].add(end_index)
        adjacency[end_index].add(start_index)

    unvisited_edges = set(unique_edges)
    polylines: list[np.ndarray] = []

    def pop_path(start_node: int) -> list[int]:
        ordered_nodes = [start_node]
        previous_node: int | None = None
        current_node = start_node

        while True:
            next_candidates = sorted(
                neighbor
                for neighbor in adjacency[current_node]
                if tuple(sorted((current_node, neighbor))) in unvisited_edges and neighbor != previous_node
            )
            if not next_candidates:
                break
            next_node = next_candidates[0]
            unvisited_edges.remove(tuple(sorted((current_node, next_node))))
            ordered_nodes.append(next_node)
            previous_node, current_node = current_node, next_node
        return ordered_nodes

    endpoint_nodes = sorted(node for node, neighbors in adjacency.items() if len(neighbors) == 1)
    for endpoint_node in endpoint_nodes:
        has_unvisited_edge = any(
            tuple(sorted((endpoint_node, neighbor))) in unvisited_edges
            for neighbor in adjacency[endpoint_node]
        )
        if not has_unvisited_edge:
            continue
        ordered_nodes = pop_path(endpoint_node)
        polylines.append(np.array([canonical_points[node] for node in ordered_nodes], dtype=float))

    while unvisited_edges:
        start_node = min(min(edge) for edge in unvisited_edges)
        ordered_nodes = pop_path(start_node)
        if len(ordered_nodes) > 1 and ordered_nodes[0] != ordered_nodes[-1]:
            ordered_nodes.append(ordered_nodes[0])
        polylines.append(np.array([canonical_points[node] for node in ordered_nodes], dtype=float))

    return polylines


def build_analysis_output_meshes(
    analyses: tuple[CurveAnalysis, ...],
    average_ratio: float,
    loft_bounds: tuple[float, float, float, float, float, float],
    tolerance: float,
    extrusion_multiplier: float,
    small_cell_extrusion_factor: float,
    slice_plane_x: float | None = None,
) -> AnalysisOutputMeshes:
    preview_meshes: list[pv.PolyData] = []
    output_meshes: list[pv.PolyData] = []
    output_modes: list[str] = []
    removed_by_retained_volume_indices: list[int] = []

    for index, analysis in enumerate(analyses):
        is_extreme = (
            analysis.bbox_aspect_ratio > EXTREME_ASPECT_RATIO_THRESHOLD
            or analysis.planarity_ratio < EXTREME_PLANARITY_RATIO
        )

        if analysis.ratio >= average_ratio:
            offset_vector = extrusion_multiplier * analysis.extrusion_base_vector
            if is_extreme:
                offset_vector = EXTREME_EXTRUSION_FACTOR * offset_vector
                staged_mesh, _, _ = _build_extreme_cell_lofts(
                    analysis.followup_polyline,
                    center=analysis.circle_center,
                    plane_origin=analysis.plane_origin,
                    plane_u=analysis.plane_u,
                    plane_v=analysis.plane_v,
                    plane_normal=analysis.plane_normal,
                    offset_vector=offset_vector,
                )
                preview_meshes.append(staged_mesh)
                output_meshes.append(staged_mesh)
                output_modes.append("extreme")
            else:
                staged_mesh, _, _ = _build_staged_offset_lofts(
                    analysis.followup_polyline,
                    center=analysis.circle_center,
                    plane_u=analysis.plane_u,
                    plane_v=analysis.plane_v,
                    offset_vector=offset_vector,
                )
                preview_meshes.append(staged_mesh)
                output_meshes.append(staged_mesh)
                output_modes.append("large")
            continue

        offset_vector = small_cell_extrusion_factor * extrusion_multiplier * analysis.extrusion_base_vector
        if is_extreme:
            offset_vector = EXTREME_EXTRUSION_FACTOR * offset_vector

        moved_center = analysis.circle_center + offset_vector
        fan_mesh = _fan_surface_from_center(moved_center, analysis.discontinuity_points)
        if _small_mesh_exceeds_retained_volume(
            fan_mesh,
            loft_bounds=loft_bounds,
            slice_plane_x=slice_plane_x,
            tolerance=tolerance,
        ):
            removed_by_retained_volume_indices.append(index)
            empty_mesh = pv.PolyData()
            preview_meshes.append(empty_mesh)
            output_meshes.append(empty_mesh)
            output_modes.append("small")
            continue

        preview_meshes.append(fan_mesh)
        output_meshes.append(fan_mesh)
        output_modes.append("small")

    return AnalysisOutputMeshes(
        preview_meshes=tuple(preview_meshes),
        output_meshes=tuple(output_meshes),
        output_modes=tuple(output_modes),
        removed_by_retained_volume_indices=tuple(removed_by_retained_volume_indices),
    )


def build_mesh_printability_report(mesh: pv.PolyData, tolerance: float) -> MeshPrintabilityReport:
    surface = extract_surface_mesh(mesh)
    boundary_edges = surface.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    ).clean()
    _, boundary_loops = extract_naked_edge_loops(surface, tolerance=tolerance)
    non_manifold_edges = surface.extract_feature_edges(
        boundary_edges=False,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=True,
    ).clean()
    boundary_edge_count = int(boundary_edges.n_cells)
    non_manifold_edge_count = int(non_manifold_edges.n_cells)
    is_closed = boundary_edge_count == 0 and non_manifold_edge_count == 0
    return MeshPrintabilityReport(
        point_count=int(surface.n_points),
        face_count=int(surface.n_cells),
        connected_region_count=count_connected_regions(surface),
        boundary_edge_count=boundary_edge_count,
        boundary_loop_count=len(boundary_loops),
        non_manifold_edge_count=non_manifold_edge_count,
        is_closed=is_closed,
        is_printable=bool(surface.n_cells > 0 and is_closed),
    )


def prepare_mesh_for_export(
    mesh: pv.PolyData,
    tolerance: float,
    attempt_repair: bool = True,
) -> MeshPreparationResult:
    surface = extract_surface_mesh(mesh)
    initial_report = build_mesh_printability_report(surface, tolerance=tolerance)
    if initial_report.is_printable or not attempt_repair:
        return MeshPreparationResult(
            mesh=surface,
            initial_report=initial_report,
            final_report=initial_report,
            repair_attempted=False,
            repair_method=None,
        )

    closed = close_mesh_boundaries(surface, tolerance=tolerance)
    closed_report = build_mesh_printability_report(closed, tolerance=tolerance)
    if closed_report.is_printable:
        return MeshPreparationResult(
            mesh=closed,
            initial_report=initial_report,
            final_report=closed_report,
            repair_attempted=True,
            repair_method="close_mesh_boundaries",
        )

    if importlib.util.find_spec("pymeshfix") is not None:
        from pymeshfix import MeshFix

        pymeshfix_input = closed.triangulate().clean()
        if pymeshfix_input.n_cells > 0:
            faces_array = pymeshfix_input.faces.reshape((-1, 4))[:, 1:]
            meshfix = MeshFix(pymeshfix_input.points.copy(), faces_array.copy())
            meshfix.repair(joincomp=True, remove_smallest_components=False)
            repaired_faces = np.hstack(
                [
                    np.full((len(meshfix.faces), 1), 3, dtype=np.int64),
                    np.asarray(meshfix.faces, dtype=np.int64),
                ]
            ).ravel()
            repaired_mesh = pv.PolyData(
                np.asarray(meshfix.points, dtype=float), faces=repaired_faces,
            ).triangulate().clean()
            repaired_report = build_mesh_printability_report(repaired_mesh, tolerance=tolerance)
            if repaired_report.is_printable and _repair_preserves_shape(surface, repaired_mesh):
                return MeshPreparationResult(
                    mesh=repaired_mesh,
                    initial_report=initial_report,
                    final_report=repaired_report,
                    repair_attempted=True,
                    repair_method="pymeshfix",
                )

    best_mesh, best_report = closed, closed_report
    if _repair_quality_score(initial_report) > _repair_quality_score(closed_report):
        best_mesh, best_report = surface, initial_report

    return MeshPreparationResult(
        mesh=best_mesh,
        initial_report=initial_report,
        final_report=best_report,
        repair_attempted=True,
        repair_method="close_mesh_boundaries_partial",
    )


def export_mesh_to_stl(
    mesh: pv.PolyData,
    output_path: str | Path,
    tolerance: float,
    attempt_repair: bool = True,
) -> Path:
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise ValueError("Cannot export an empty mesh to STL.")
    preparation = prepare_mesh_for_export(
        mesh,
        tolerance=tolerance,
        attempt_repair=attempt_repair,
    )
    if not preparation.final_report.is_printable:
        raise ValueError(
            "Mesh is not closed after export preparation. "
            f"Boundary loops: {preparation.final_report.boundary_loop_count}, "
            f"boundary edges: {preparation.final_report.boundary_edge_count}, "
            f"non-manifold edges: {preparation.final_report.non_manifold_edge_count}."
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    preparation.mesh.save(str(output))
    return output


def make_bounding_box(bounds: tuple[float, float, float, float, float, float]) -> pv.PolyData:
    return pv.Box(bounds=bounds)


def _mesh_diagonal_length(bounds: tuple[float, float, float, float, float, float]) -> float:
    spans = np.array(
        [
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4],
        ],
        dtype=float,
    )
    return float(np.linalg.norm(spans))


def _repair_preserves_shape(original_mesh: pv.PolyData, repaired_mesh: pv.PolyData) -> bool:
    original_bounds = np.array(original_mesh.bounds, dtype=float)
    repaired_bounds = np.array(repaired_mesh.bounds, dtype=float)
    diagonal = max(_mesh_diagonal_length(tuple(original_bounds.tolist())), 1e-9)
    max_bound_shift = float(np.max(np.abs(original_bounds - repaired_bounds)))
    if max_bound_shift > 0.01 * diagonal:
        return False

    original_area = float(original_mesh.area)
    repaired_area = float(repaired_mesh.area)
    if original_area <= 1e-9:
        return True
    area_ratio = repaired_area / original_area
    return 0.9 <= area_ratio <= 1.1


def close_mesh_boundaries(
    mesh: pv.PolyData,
    tolerance: float,
    max_iterations: int = 6,
) -> pv.PolyData:
    mesh = resolve_non_manifold_faces(mesh)

    for _ in range(max_iterations):
        report = build_mesh_printability_report(mesh, tolerance=tolerance)
        if report.is_closed:
            return mesh

        surface = extract_surface_mesh(mesh)
        _, simple_loops = extract_naked_edge_loops(surface, tolerance=tolerance)
        if simple_loops:
            caps = [
                _fan_surface_from_center(
                    loop[:-1].mean(axis=0) if np.allclose(loop[0], loop[-1], atol=tolerance) else loop.mean(axis=0),
                    loop[:-1] if np.allclose(loop[0], loop[-1], atol=tolerance) else loop,
                )
                for loop in simple_loops
                if len(loop) >= 4
            ]
            if caps:
                mesh = _merge_meshes([mesh] + [c for c in caps if c.n_cells > 0])
                mesh = resolve_non_manifold_faces(mesh)

        report = build_mesh_printability_report(mesh, tolerance=tolerance)
        if report.is_closed:
            return mesh

        branch_loops = _split_branch_boundaries(mesh, tolerance=tolerance)
        if branch_loops:
            branch_caps = [
                _fan_surface_from_center(loop.mean(axis=0), loop)
                for loop in branch_loops
                if len(loop) >= 3
            ]
            if branch_caps:
                mesh = _merge_meshes([mesh] + [c for c in branch_caps if c.n_cells > 0])
                mesh = resolve_non_manifold_faces(mesh)

        if report.boundary_edge_count > 0:
            mesh = mesh.fill_holes(
                hole_size=_mesh_diagonal_length(mesh.bounds) * 2.0,
            ).triangulate().clean()
            mesh = resolve_non_manifold_faces(mesh)

    return mesh


def _split_branch_boundaries(
    mesh: pv.PolyData,
    tolerance: float,
) -> list[np.ndarray]:
    surface = extract_surface_mesh(mesh)
    if surface.n_points == 0:
        return []

    boundary = surface.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    ).clean()
    if boundary.n_cells == 0:
        return []

    surf_pts = np.asarray(surface.points, dtype=float)
    bnd_pts = np.asarray(boundary.points, dtype=float)

    bnd_to_surf: dict[int, int] = {}
    for bi in range(len(bnd_pts)):
        dists = np.linalg.norm(surf_pts - bnd_pts[bi], axis=1)
        bnd_to_surf[bi] = int(np.argmin(dists))

    adj: dict[int, set[int]] = defaultdict(set)
    raw_lines = np.asarray(boundary.lines, dtype=int)
    cursor = 0
    while cursor < len(raw_lines):
        n = int(raw_lines[cursor])
        ids = raw_lines[cursor + 1 : cursor + 1 + n]
        for a, b in zip(ids[:-1], ids[1:]):
            sa, sb = bnd_to_surf[int(a)], bnd_to_surf[int(b)]
            adj[sa].add(sb)
            adj[sb].add(sa)
        cursor += n + 1

    branch_verts = {v for v, nbrs in adj.items() if len(nbrs) == 4}
    if not branch_verts:
        return []

    next_id = max(adj.keys()) + 1
    split_adj: dict[int, set[int]] = defaultdict(set)
    for v, nbrs in adj.items():
        if v not in branch_verts:
            for n in nbrs:
                split_adj[v].add(n)

    split_positions: dict[int, np.ndarray] = {}
    for v in adj:
        split_positions[v] = surf_pts[v]

    for bv in branch_verts:
        nbrs = sorted(adj[bv])
        bv_pos = surf_pts[bv]
        dirs = [surf_pts[n] - bv_pos for n in nbrs]
        norms = [np.linalg.norm(d) for d in dirs]
        unit_dirs = [d / max(n, 1e-12) for d, n in zip(dirs, norms)]

        best_score = -np.inf
        best_pairing: tuple[tuple[int, int], tuple[int, int]] | None = None
        for i in range(4):
            for j in range(i + 1, 4):
                others = [k for k in range(4) if k != i and k != j]
                score = -(float(np.dot(unit_dirs[i], unit_dirs[j])) + float(np.dot(unit_dirs[others[0]], unit_dirs[others[1]])))
                if score > best_score:
                    best_score = score
                    best_pairing = ((nbrs[i], nbrs[j]), (nbrs[others[0]], nbrs[others[1]]))

        if best_pairing is None:
            continue

        (a1, a2), (b1, b2) = best_pairing
        id_a = bv
        id_b = next_id
        next_id += 1

        split_positions[id_a] = bv_pos.copy()
        split_positions[id_b] = bv_pos.copy()

        for n in [a1, a2]:
            split_adj[id_a].add(n)
            split_adj[n].discard(bv)
            split_adj[n].add(id_a)
        for n in [b1, b2]:
            split_adj[id_b].add(n)
            split_adj[n].discard(bv)
            split_adj[n].add(id_b)

    visited: set[int] = set()
    loops: list[np.ndarray] = []

    for start_v in sorted(split_adj.keys()):
        if start_v in visited:
            continue
        if len(split_adj[start_v]) != 2:
            continue

        path = [start_v]
        visited.add(start_v)
        current = start_v
        prev = -1

        for _ in range(len(split_adj) + 10):
            nbrs = sorted(split_adj[current])
            candidates = [n for n in nbrs if n != prev]
            if not candidates:
                break
            next_v = candidates[0]

            if next_v == start_v and len(path) >= 3:
                loop_pts = np.array([split_positions[v] for v in path], dtype=float)
                loops.append(loop_pts)
                break

            if next_v in visited:
                break

            visited.add(next_v)
            path.append(next_v)
            prev = current
            current = next_v

    return loops


def align_loops_and_loft(
    loop_a: np.ndarray,
    loop_b: np.ndarray,
    tolerance: float,
) -> pv.PolyData:
    a_closed = np.allclose(loop_a[0], loop_a[-1], atol=tolerance)
    b_closed = np.allclose(loop_b[0], loop_b[-1], atol=tolerance)
    a_unique = loop_a[:-1] if a_closed else loop_a
    b_unique = loop_b[:-1] if b_closed else loop_b

    if len(a_unique) != len(b_unique) or len(a_unique) < 2:
        return pv.PolyData()

    n = len(a_unique)

    def _total_dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(np.linalg.norm(a - b, axis=1)))

    def _best_roll(a: np.ndarray, b: np.ndarray) -> tuple[int, float]:
        best_off, best_d = 0, _total_dist(a, b)
        for off in range(1, n):
            d = _total_dist(a, np.roll(b, -off, axis=0))
            if d < best_d:
                best_d = d
                best_off = off
        return best_off, best_d

    off_fwd, d_fwd = _best_roll(a_unique, b_unique)
    b_rev = b_unique[::-1]
    off_rev, d_rev = _best_roll(a_unique, b_rev)

    if d_rev < d_fwd:
        b_unique = np.roll(b_rev, -off_rev, axis=0)
    elif off_fwd != 0:
        b_unique = np.roll(b_unique, -off_fwd, axis=0)

    a_loop = np.vstack([a_unique, a_unique[0:1]])
    b_loop = np.vstack([b_unique, b_unique[0:1]])
    return _loft_between_polylines(a_loop, b_loop)


def project_loop_to_plane(
    loop: np.ndarray,
    plane_axis: int,
    plane_coord: float,
) -> np.ndarray:
    projected = loop.copy()
    projected[:, plane_axis] = plane_coord
    return projected


def split_and_offset_plane_faces(
    mesh: pv.PolyData,
    plane_normal: tuple[float, float, float],
    plane_origin: tuple[float, float, float],
    offset_amount: float = -2.0,
    tolerance: float = 1e-4,
) -> tuple[pv.PolyData, pv.PolyData]:
    """Split a boundary cell mesh into body and plane-face patches.

    Two-run approach:
      Run 1 — centroid check: faces whose centroid lies on the cutting plane
              are moved as whole faces (the original method).
      Run 2 — vertex search among remaining faces: faces where ALL vertices
              are within a wider tolerance AND at least 2 are right on the
              plane are flattened onto the plane, then offset.
    """
    plane_axis = int(np.argmax(np.abs(plane_normal)))
    plane_coord = float(plane_origin[plane_axis])

    if mesh.n_points == 0 or mesh.n_cells == 0:
        return mesh, pv.PolyData()

    points = np.asarray(mesh.points, dtype=float)
    faces_raw = np.asarray(mesh.faces, dtype=int)
    face_verts: list[tuple[int, ...]] = []
    cursor = 0
    while cursor < len(faces_raw):
        n = int(faces_raw[cursor])
        verts = tuple(int(faces_raw[cursor + 1 + j]) for j in range(n))
        face_verts.append(verts)
        cursor += n + 1

    plane_tol = tolerance * 50
    vertex_dists = np.abs(points[:, plane_axis] - plane_coord)

    is_boundary = bool(np.any(vertex_dists < plane_tol))
    if not is_boundary:
        return mesh, pv.PolyData()

    near_plane_tol = plane_tol * 5

    # ------------------------------------------------------------------
    # Run 1: centroid check — move entire face if centroid is at x ≈ 0
    # ------------------------------------------------------------------
    run1_faces: set[int] = set()
    for fi, fv in enumerate(face_verts):
        centroid_on_axis = sum(float(points[vi, plane_axis]) for vi in fv) / len(fv)
        if abs(centroid_on_axis - plane_coord) < plane_tol:
            run1_faces.add(fi)

    # ------------------------------------------------------------------
    # Run 2: among remaining faces, find those that can become flat
    #   - ALL vertices within near_plane_tol of the plane
    #   - at least 2 vertices right on the plane (within plane_tol)
    # ------------------------------------------------------------------
    run2_faces: set[int] = set()
    for fi, fv in enumerate(face_verts):
        if fi in run1_faces:
            continue
        all_near = all(vertex_dists[vi] < near_plane_tol for vi in fv)
        if not all_near:
            continue
        on_plane_count = sum(1 for vi in fv if vertex_dists[vi] < plane_tol)
        if on_plane_count >= 2:
            run2_faces.add(fi)

    selected_faces = run1_faces | run2_faces
    if not selected_faces:
        return mesh, pv.PolyData()

    if len(selected_faces) == len(face_verts):
        selected_faces = run1_faces if run1_faces else set()
        if not selected_faces:
            return mesh, pv.PolyData()

    # ------------------------------------------------------------------
    # Build body mesh (unselected faces, original points)
    # ------------------------------------------------------------------
    body_face_data: list[int] = []
    for fi, fv in enumerate(face_verts):
        if fi not in selected_faces:
            body_face_data.extend([len(fv)] + [int(v) for v in fv])

    body_mesh = (
        pv.PolyData(points.copy(), faces=np.array(body_face_data, dtype=np.int64))
        if body_face_data
        else pv.PolyData()
    )

    # ------------------------------------------------------------------
    # Build moved mesh
    #   Run 1 faces: offset as-is (they are already flat on the plane)
    #   Run 2 faces: flatten to plane first, then offset
    # ------------------------------------------------------------------
    moved_points = points.copy()
    for fi in run2_faces:
        for vi in face_verts[fi]:
            moved_points[vi, plane_axis] = plane_coord
    moved_points[:, plane_axis] += offset_amount

    plane_face_data: list[int] = []
    for fi in sorted(selected_faces):
        fv = face_verts[fi]
        plane_face_data.extend([len(fv)] + [int(v) for v in fv])

    plane_mesh = pv.PolyData(
        moved_points, faces=np.array(plane_face_data, dtype=np.int64)
    )
    plane_mesh = plane_mesh.clean().triangulate()

    return body_mesh, plane_mesh


def filter_closed_meshes(
    meshes: list[pv.PolyData],
) -> tuple[list[pv.PolyData], list[int], list[int]]:
    kept: list[pv.PolyData] = []
    kept_indices: list[int] = []
    removed_indices: list[int] = []
    for index, mesh in enumerate(meshes):
        if mesh.n_points == 0 or mesh.n_cells == 0:
            removed_indices.append(index)
            continue
        surface = extract_surface_mesh(mesh)
        boundary = surface.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=False,
        ).clean()
        if int(boundary.n_cells) == 0:
            removed_indices.append(index)
            continue
        kept.append(mesh)
        kept_indices.append(index)
    return kept, kept_indices, removed_indices


def _pick_best_candidate(
    candidates: list[tuple[pv.PolyData, MeshPrintabilityReport]],
) -> tuple[pv.PolyData, MeshPrintabilityReport]:
    best_mesh, best_report = candidates[0]
    best_score = _repair_quality_score(best_report)
    for mesh, report in candidates[1:]:
        score = _repair_quality_score(report)
        if score > best_score:
            best_mesh, best_report = mesh, report
            best_score = score
    return best_mesh, best_report


def _repair_quality_score(report: MeshPrintabilityReport, original_face_count: int = 1) -> float:
    retention = report.face_count / max(original_face_count, 1)
    if retention < 0.5:
        return -1e9
    if report.is_printable:
        return 1e9 + report.face_count
    score = float(report.face_count)
    score -= 100.0 * report.non_manifold_edge_count
    score -= 50.0 * report.boundary_edge_count
    return score


def _signed_polygon_area_2d(pts_2d: np.ndarray) -> float:
    """Signed area of a 2D polygon via the shoelace formula."""
    n = len(pts_2d)
    if n < 3:
        return 0.0
    x = pts_2d[:, 0]
    y = pts_2d[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _polygon_area_2d(pts_2d: np.ndarray) -> float:
    return abs(_signed_polygon_area_2d(pts_2d))


def _polygon_centroid_2d(pts_2d: np.ndarray) -> np.ndarray:
    """Centroid of a simple 2D polygon."""
    n = len(pts_2d)
    if n < 3:
        return pts_2d.mean(axis=0) if n > 0 else np.zeros(2)
    sa = _signed_polygon_area_2d(pts_2d)
    if abs(sa) < 1e-20:
        return pts_2d.mean(axis=0)
    x = pts_2d[:, 0]
    y = pts_2d[:, 1]
    xn = np.roll(x, -1)
    yn = np.roll(y, -1)
    cross = x * yn - xn * y
    cx = float(np.sum((x + xn) * cross)) / (6.0 * sa)
    cy = float(np.sum((y + yn) * cross)) / (6.0 * sa)
    return np.array([cx, cy], dtype=float)


def compact_polyline_shapes(
    polylines: list[np.ndarray],
    tolerance: float,
    *,
    tail_distance_ratio: float = 2.2,
    min_area_fraction: float = 0.4,
    min_vertices: int = 4,
) -> tuple[list[np.ndarray], int, list[str]]:
    """Remove tail vertices that make polyline shapes non-compact.

    For each polyline the function:
    1. Fits a best-fit plane and projects to 2D.
    2. Computes the polygon centroid and per-vertex distance from it.
    3. Vertices farther than ``tail_distance_ratio * median_distance`` are
       candidates for removal.
    4. A candidate is removed only if dropping it keeps >= *min_vertices*
       vertices and the resulting area is >= *min_area_fraction* of the
       original.

    This trims elongated peninsulas / tails that barely contribute area
    but extend far from the bulk of the cell.

    Returns ``(compacted_polylines, total_removed, log_messages)``.
    """
    result: list[np.ndarray] = []
    total_removed = 0
    messages: list[str] = []

    for ci, polyline in enumerate(polylines):
        unique = _unique_polyline_points(polyline, tolerance)
        if len(unique) < min_vertices:
            result.append(polyline.copy())
            continue

        po, pu, pv, _pn = _fit_plane(unique)
        pts_2d = _project_to_plane(unique, po, pu, pv)
        original_area = _polygon_area_2d(pts_2d)

        if original_area < 1e-12:
            result.append(polyline.copy())
            continue

        centroid = _polygon_centroid_2d(pts_2d)
        dists = np.linalg.norm(pts_2d - centroid, axis=1)
        median_dist = float(np.median(dists))
        if median_dist < 1e-12:
            result.append(polyline.copy())
            continue

        threshold = tail_distance_ratio * median_dist
        tail_candidates = np.where(dists > threshold)[0]

        if len(tail_candidates) == 0:
            result.append(polyline.copy())
            continue

        sorted_candidates = sorted(tail_candidates, key=lambda i: -dists[i])

        keep_mask = np.ones(len(unique), dtype=bool)
        removed_this_cell = 0

        for vi in sorted_candidates:
            if int(keep_mask.sum()) <= min_vertices:
                break
            test_mask = keep_mask.copy()
            test_mask[vi] = False
            test_pts = pts_2d[test_mask]
            test_area = _polygon_area_2d(test_pts)

            if test_area < min_area_fraction * original_area:
                continue

            keep_mask[vi] = False
            removed_this_cell += 1

        if removed_this_cell > 0:
            kept_pts = unique[keep_mask]
            result.append(_close_polyline(kept_pts, tolerance))
            total_removed += removed_this_cell
            messages.append(
                f"Cell {ci}: removed {removed_this_cell} tail vertex(es) "
                f"(area: {original_area:.2f} -> {_polygon_area_2d(pts_2d[keep_mask]):.2f})"
            )
        else:
            result.append(polyline.copy())

    return result, total_removed, messages


def _find_cross_polyline_intersections(
    unique_point_lists: list[np.ndarray],
    tolerance: float,
) -> dict[int, list[tuple[int, np.ndarray]]]:
    """Find all pairwise segment-segment crossings between different polylines.

    Returns a mapping ``{polyline_index: [(segment_index, crossing_point), ...]}``.
    Each crossing is recorded for BOTH involved polylines so both can insert
    the crossing vertex during rebuild.

    Uses a Shapely STRtree with ``predicate='crosses'`` as a fast C-level
    pre-filter to reduce O(n^2 * e^2) Python iterations to a small candidate
    set, then confirms each candidate with the original ``_segment_crossing_3d``
    3D check to preserve exact semantics (including the 3D skewness tolerance).
    """
    cross_tol = tolerance * 50
    result: dict[int, list[tuple[int, np.ndarray]]] = {
        i: [] for i in range(len(unique_point_lists))
    }

    n_poly = len(unique_point_lists)
    if n_poly < 2:
        return result

    # Build flat segment list: one Shapely LineString per edge, projected to XY.
    seg_shapes: list[_ShapelyLine] = []
    seg_meta: list[tuple[int, int]] = []  # (polyline_idx, segment_idx)
    seg_3d_starts: list[np.ndarray] = []
    seg_3d_ends: list[np.ndarray] = []

    for pi, pts in enumerate(unique_point_lists):
        n = len(pts)
        if n < 2:
            continue
        for si in range(n):
            si_next = (si + 1) % n
            p0, p1 = pts[si], pts[si_next]
            dx, dy = float(p1[0] - p0[0]), float(p1[1] - p0[1])
            if dx * dx + dy * dy < 1e-20:
                continue
            seg_shapes.append(
                _ShapelyLine([(float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))])
            )
            seg_meta.append((pi, si))
            seg_3d_starts.append(p0)
            seg_3d_ends.append(p1)

    if len(seg_shapes) < 2:
        return result

    tree = STRtree(seg_shapes)

    # predicate='crosses' is a fast C-level pre-filter: keeps only segments
    # whose XY projections geometrically cross (interior-to-interior, no
    # endpoint touches). This reduces candidates from O(n^2) to O(n log n)
    # before the 3D verification step.
    left_idx, right_idx = tree.query(seg_shapes, predicate="crosses")

    processed: set[tuple[int, int]] = set()

    for li, ri in zip(left_idx, right_idx):
        if li >= ri:
            continue  # process each unordered pair once; skip self-hits
        pi_a, si_a = seg_meta[li]
        pi_b, si_b = seg_meta[ri]
        if pi_a == pi_b:
            continue  # same polyline — not a cross-polyline crossing

        pair = (li, ri)
        if pair in processed:
            continue
        processed.add(pair)

        # 3D confirmation: rejects XY crossings that are far apart in Z
        # (e.g., segments from different height levels of the lofted surface).
        pt3d = _segment_crossing_3d(
            seg_3d_starts[li], seg_3d_ends[li],
            seg_3d_starts[ri], seg_3d_ends[ri],
            cross_tol,
        )
        if pt3d is None:
            continue

        result[pi_a].append((si_a, pt3d))
        result[pi_b].append((si_b, pt3d))

    return result


def _inject_crossing_points(
    unique_points: np.ndarray,
    crossings: list[tuple[int, np.ndarray]],
    tolerance: float,
) -> tuple[np.ndarray, set[int]]:
    """Insert crossing points into a polyline's point array.

    Returns ``(augmented_points, forced_discontinuity_indices)`` — the set
    of indices in the new array that correspond to injected crossings and
    must be treated as discontinuities regardless of angle/curvature.
    """
    if not crossings:
        return unique_points, set()

    n = len(unique_points)
    insertions: list[tuple[int, np.ndarray]] = []
    for seg_idx, pt in crossings:
        seg_idx = min(seg_idx, n - 1)
        if n > 0 and float(np.min(np.linalg.norm(unique_points - pt, axis=1))) < tolerance:
            continue
        already = any(
            float(np.linalg.norm(pt - ip)) < tolerance for _, ip in insertions
        )
        if already:
            continue
        insertions.append((seg_idx, pt))

    if not insertions:
        return unique_points, set()

    spliced = _splice_points_into_polyline(unique_points, insertions, tolerance)

    forced: set[int] = set()
    for _, ipt in insertions:
        for k in range(len(spliced)):
            if float(np.linalg.norm(spliced[k] - ipt)) < tolerance:
                forced.add(k)
                break

    return spliced, forced


def rebuild_polylines_from_discontinuities(
    polylines: list[np.ndarray],
    tolerance: float,
    discontinuity_angle_degrees: float = 176.0,
    neighbor_snap_tolerance: float | None = None,
) -> list[np.ndarray]:
    unique_lists = [
        _unique_polyline_points(p, tolerance=tolerance) for p in polylines
    ]

    crossings_map = _find_cross_polyline_intersections(unique_lists, tolerance)

    rebuilt_polylines: list[np.ndarray] = []
    for pi, polyline in enumerate(polylines):
        unique_points = unique_lists[pi]
        if len(unique_points) < 3:
            continue

        augmented, forced_indices = _inject_crossing_points(
            unique_points, crossings_map.get(pi, []), tolerance,
        )

        straight_polyline, _ = _build_straight_polyline_from_discontinuities(
            augmented,
            tolerance=tolerance,
            discontinuity_angle_degrees=discontinuity_angle_degrees,
            forced_discontinuity_indices=forced_indices,
        )
        rebuilt_polyline = _sanitize_closed_polyline(straight_polyline, tolerance=tolerance)
        if len(_unique_polyline_points(rebuilt_polyline, tolerance=tolerance)) < 3:
            continue
        rebuilt_polylines.append(rebuilt_polyline)

    if not rebuilt_polylines:
        return []

    snap_tolerance = neighbor_snap_tolerance if neighbor_snap_tolerance is not None else default_snap_tolerance(tolerance)
    return _snap_neighboring_polyline_points(
        rebuilt_polylines,
        tolerance=tolerance,
        snap_tolerance=snap_tolerance,
    )


def find_polyline_neighbours(
    polylines: list[np.ndarray],
    tolerance: float,
) -> dict[int, list[int]]:
    """Return mapping of polyline index to sorted list of neighbor indices.

    Two polylines are neighbors if they share at least one vertex
    (within *tolerance*).  Uses a spatial grid for O(n) lookup.
    """
    point_to_polys: dict[tuple[int, ...], set[int]] = {}
    for idx, poly in enumerate(polylines):
        unique = _unique_polyline_points(poly, tolerance)
        for pt in unique:
            key = tuple(np.round(pt / tolerance).astype(int).tolist())
            if key not in point_to_polys:
                point_to_polys[key] = set()
            point_to_polys[key].add(idx)

    neighbours: dict[int, set[int]] = {i: set() for i in range(len(polylines))}
    for poly_set in point_to_polys.values():
        indices = list(poly_set)
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                neighbours[indices[i]].add(indices[j])
                neighbours[indices[j]].add(indices[i])

    return {k: sorted(v) for k, v in neighbours.items()}


def align_neighbouring_polylines(
    polylines: list[np.ndarray],
    tolerance: float,
    snap_tolerance: float | None = None,
    slice_plane_x: float | None = None,
    neighbours: dict[int, list[int]] | None = None,
) -> list[np.ndarray]:
    """Align shared-edge segments between neighboring polylines.

    For each pair of neighbors the function:

    1. Finds shared vertices (voronoi face boundary corners).
    2. Detects edge–edge crossings that are **not** at shared vertices
       (these indicate misalignment).
    3. Extracts the sub-path between each pair of consecutive shared
       vertices from both polylines.
    4. If point counts match the corresponding positions are averaged;
       otherwise the shorter sub-path is resampled via arc-length
       interpolation to match, then averaged.

    When *slice_plane_x* is given, vertices near the cutting plane are
    dampened (moved only 20 % toward the midpoint) so boundary cells
    keep their extension toward the plane edge.

    Returns a new list of polylines with aligned shared edges.
    """
    if snap_tolerance is None:
        snap_tolerance = default_snap_tolerance(tolerance)

    if len(polylines) < 2:
        return [p.copy() for p in polylines]

    if neighbours is None:
        neighbours = find_polyline_neighbours(polylines, snap_tolerance)

    seg_updates: dict[int, dict[tuple[int, int], np.ndarray]] = {
        i: {} for i in range(len(polylines))
    }

    processed: set[tuple[int, int]] = set()
    for idx_a, nbrs in sorted(neighbours.items()):
        for idx_b in nbrs:
            pair = (min(idx_a, idx_b), max(idx_a, idx_b))
            if pair in processed:
                continue
            processed.add(pair)
            _compute_edge_alignment(
                polylines, idx_a, idx_b, tolerance, snap_tolerance, seg_updates,
                slice_plane_x=slice_plane_x,
            )

    result: list[np.ndarray] = []
    for idx in range(len(polylines)):
        if not seg_updates[idx]:
            result.append(polylines[idx].copy())
        else:
            unique = _unique_polyline_points(polylines[idx], tolerance)
            new_unique = _rebuild_polyline_from_updates(unique, seg_updates[idx])
            result.append(_close_polyline(new_unique, tolerance))

    return result


def _find_shared_vertex_pairs(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    tolerance: float,
) -> list[tuple[int, int]]:
    """Return ``(index_in_a, index_in_b)`` pairs of shared vertices."""
    if len(pts_a) == 0 or len(pts_b) == 0:
        return []
    dists = cdist(pts_a, pts_b)
    pairs: list[tuple[int, int]] = []
    used_b = np.zeros(len(pts_b), dtype=bool)
    for ia in range(len(pts_a)):
        row = dists[ia].copy()
        row[used_b] = np.inf
        ib = int(row.argmin())
        if row[ib] <= tolerance:
            pairs.append((ia, ib))
            used_b[ib] = True
    return pairs


def _cyclic_interior_indices(start: int, end: int, n: int) -> list[int]:
    """Indices strictly between *start* and *end* in a cyclic array of size *n*."""
    if start == end:
        return []
    indices: list[int] = []
    i = (start + 1) % n
    safety = 0
    while i != end:
        indices.append(i)
        i = (i + 1) % n
        safety += 1
        if safety > n:
            break
    return indices


def _segment_crossing_3d(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
    tolerance: float,
) -> np.ndarray | None:
    """Approximate crossing point of two 3-D line segments, or ``None``."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3

    a = float(np.dot(d1, d1))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d2, d2))
    d_val = float(np.dot(d1, r))
    e = float(np.dot(d2, r))

    denom = a * c - b * b
    if abs(denom) < 1e-20:
        return None

    t = (b * e - c * d_val) / denom
    s = (a * e - b * d_val) / denom

    eps = 0.01
    if t < eps or t > 1.0 - eps or s < eps or s > 1.0 - eps:
        return None

    pt1 = p1 + t * d1
    pt2 = p3 + s * d2
    if float(np.linalg.norm(pt1 - pt2)) > tolerance:
        return None

    return (pt1 + pt2) / 2.0


def _find_edge_crossings(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    tolerance: float,
    shared_positions: list[np.ndarray],
) -> list[np.ndarray]:
    """Edge–edge intersections between two polylines, excluding shared vertices."""
    crossings: list[np.ndarray] = []
    na, nb = len(pts_a), len(pts_b)
    cross_tol = tolerance * 50
    shared_arr = np.array(shared_positions) if shared_positions else np.empty((0, 3))

    for ia in range(na):
        ia_next = (ia + 1) % na
        for ib in range(nb):
            ib_next = (ib + 1) % nb
            pt = _segment_crossing_3d(
                pts_a[ia], pts_a[ia_next],
                pts_b[ib], pts_b[ib_next],
                cross_tol,
            )
            if pt is None:
                continue
            if len(shared_arr) > 0 and float(np.min(np.linalg.norm(shared_arr - pt, axis=1))) < cross_tol:
                continue
            crossings.append(pt)

    return crossings


def _pick_matching_direction(
    sub_a: np.ndarray,
    sub_b_fwd: np.ndarray,
    sub_b_rev: np.ndarray,
) -> tuple[np.ndarray | None, str]:
    """Choose the B sub-path direction that best matches A's sub-path."""
    if len(sub_a) == 0:
        if len(sub_b_fwd) > 0:
            return sub_b_fwd, "forward"
        if len(sub_b_rev) > 0:
            return sub_b_rev, "reverse"
        return None, "forward"

    mid_a = sub_a.mean(axis=0)
    d_fwd = (
        float(np.linalg.norm(sub_b_fwd.mean(axis=0) - mid_a))
        if len(sub_b_fwd) > 0
        else float("inf")
    )
    d_rev = (
        float(np.linalg.norm(sub_b_rev.mean(axis=0) - mid_a))
        if len(sub_b_rev) > 0
        else float("inf")
    )

    if d_fwd <= d_rev and len(sub_b_fwd) > 0:
        return sub_b_fwd, "forward"
    if len(sub_b_rev) > 0:
        return sub_b_rev, "reverse"
    if len(sub_b_fwd) > 0:
        return sub_b_fwd, "forward"
    return None, "forward"


def _resample_polyline_segment(
    points: np.ndarray,
    target_count: int,
) -> np.ndarray:
    """Resample a polyline segment to *target_count* points via arc-length interpolation."""
    if len(points) == target_count:
        return points.copy()
    if target_count <= 0:
        return np.empty((0, points.shape[1]), dtype=float)
    if target_count == 1:
        return points[len(points) // 2 : len(points) // 2 + 1].copy()

    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(diffs)])
    total_len = cumlen[-1]

    if total_len < 1e-12:
        return np.tile(points[0], (target_count, 1))

    new_params = np.linspace(0.0, total_len, target_count)
    result = np.zeros((target_count, points.shape[1]), dtype=float)
    for dim in range(points.shape[1]):
        result[:, dim] = np.interp(new_params, cumlen, points[:, dim])
    return result


def _compute_edge_alignment(
    polylines: list[np.ndarray],
    idx_a: int,
    idx_b: int,
    tolerance: float,
    snap_tolerance: float,
    seg_updates: dict[int, dict[tuple[int, int], np.ndarray]],
    slice_plane_x: float | None = None,
) -> None:
    """Compute aligned interior points for every shared edge of one neighbor pair."""
    unique_a = _unique_polyline_points(polylines[idx_a], tolerance)
    unique_b = _unique_polyline_points(polylines[idx_b], tolerance)

    if len(unique_a) < 3 or len(unique_b) < 3:
        return

    shared = _find_shared_vertex_pairs(unique_a, unique_b, snap_tolerance)
    if len(shared) < 2:
        return

    shared_positions = [unique_a[ia] for ia, _ in shared]
    crossings = _find_edge_crossings(unique_a, unique_b, tolerance, shared_positions)

    shared.sort(key=lambda p: p[0])
    na, nb = len(unique_a), len(unique_b)

    for k in range(len(shared)):
        ia1, ib1 = shared[k]
        ia2, ib2 = shared[(k + 1) % len(shared)]

        int_a_idx = _cyclic_interior_indices(ia1, ia2, na)
        if len(int_a_idx) == 0:
            continue

        int_b_fwd_idx = _cyclic_interior_indices(ib1, ib2, nb)
        int_b_rev_idx = _cyclic_interior_indices(ib2, ib1, nb)

        sub_a = unique_a[int_a_idx]
        sub_b_fwd = unique_b[int_b_fwd_idx] if len(int_b_fwd_idx) > 0 else np.empty((0, 3))
        sub_b_rev = (
            unique_b[int_b_rev_idx][::-1] if len(int_b_rev_idx) > 0 else np.empty((0, 3))
        )

        sub_b, b_dir = _pick_matching_direction(sub_a, sub_b_fwd, sub_b_rev)
        if sub_b is None or len(sub_b) == 0:
            continue

        mid_dist = float(np.linalg.norm(sub_a.mean(axis=0) - sub_b.mean(axis=0)))
        shared_span = float(np.linalg.norm(unique_a[ia1] - unique_a[ia2]))
        proximity_limit = max(shared_span * 0.5, snap_tolerance * 5)
        if mid_dist > proximity_limit:
            continue

        max_sub = max(len(sub_a), len(sub_b))
        min_sub = max(min(len(sub_a), len(sub_b)), 1)
        if max_sub > min_sub * 3:
            continue

        needs_align = len(crossings) > 0
        if not needs_align and len(sub_a) == len(sub_b):
            max_diff = float(np.max(np.linalg.norm(sub_a - sub_b, axis=1)))
            needs_align = max_diff > tolerance
        elif not needs_align:
            needs_align = True

        if not needs_align:
            continue

        target = max(len(sub_a), len(sub_b))
        ra = _resample_polyline_segment(sub_a, target) if len(sub_a) != target else sub_a
        rb = _resample_polyline_segment(sub_b, target) if len(sub_b) != target else sub_b
        aligned = (ra + rb) / 2.0

        if slice_plane_x is not None:
            plane_tol = tolerance * 50
            boundary_damping = 0.2
            for vi in range(len(aligned)):
                if (abs(float(ra[vi, 0]) - slice_plane_x) < plane_tol
                        or abs(float(rb[vi, 0]) - slice_plane_x) < plane_tol):
                    aligned[vi] = ra[vi] + boundary_damping * (aligned[vi] - ra[vi])

        max_shift_a = float(np.max(np.linalg.norm(ra - aligned, axis=1)))
        max_shift_b = float(np.max(np.linalg.norm(rb - aligned, axis=1)))
        shift_limit = max(shared_span * 0.5, snap_tolerance * 10)
        if max(max_shift_a, max_shift_b) > shift_limit:
            continue

        seg_updates[idx_a][(ia1, ia2)] = aligned

        if b_dir == "forward":
            seg_updates[idx_b][(ib1, ib2)] = aligned.copy()
        else:
            seg_updates[idx_b][(ib2, ib1)] = aligned[::-1].copy()


def _rebuild_polyline_from_updates(
    unique_pts: np.ndarray,
    updates: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    """Rebuild unique-point array by splicing in updated interior segments."""
    n = len(unique_pts)
    if not updates:
        return unique_pts.copy()

    boundary_set: set[int] = set()
    for s, e in updates:
        boundary_set.add(s)
        boundary_set.add(e)

    sorted_boundaries = sorted(boundary_set)
    if len(sorted_boundaries) < 2:
        return unique_pts.copy()

    parts: list[np.ndarray] = []
    for k in range(len(sorted_boundaries)):
        start = sorted_boundaries[k]
        end = sorted_boundaries[(k + 1) % len(sorted_boundaries)]

        parts.append(unique_pts[start : start + 1])

        if (start, end) in updates:
            parts.append(updates[(start, end)])
        else:
            interior = _cyclic_interior_indices(start, end, n)
            if len(interior) > 0:
                parts.append(unique_pts[interior])

    if not parts:
        return unique_pts.copy()
    return np.vstack(parts)


def point_distance_to_mesh_surface(
    point: np.ndarray,
    surface: pv.PolyData,
) -> tuple[float, np.ndarray]:
    """Distance from a 3D point to the nearest location on a mesh surface.

    Uses a VTK cell-locator so the returned closest point can lie in the
    interior of a triangle face, not only at mesh vertices.

    Returns ``(distance, closest_point_on_surface)``.
    """
    if surface.n_cells == 0 or surface.n_points == 0:
        return float("inf"), np.asarray(point, dtype=float).copy()

    tri = surface.triangulate().clean()
    if tri.n_cells == 0:
        return float("inf"), np.asarray(point, dtype=float).copy()

    locator = vtk.vtkCellLocator()
    locator.SetDataSet(tri)
    locator.BuildLocator()

    pt = [float(point[0]), float(point[1]), float(point[2])]
    cp = [0.0, 0.0, 0.0]
    gc = vtk.vtkGenericCell()
    cid = vtk.reference(0)
    sid = vtk.reference(0)
    d2 = vtk.reference(0.0)
    locator.FindClosestPoint(pt, cp, gc, cid, sid, d2)
    return float(d2) ** 0.5, np.array(cp, dtype=float)


def validate_polyline_surfaces(
    polylines: list[np.ndarray],
    tolerance: float,
    snap_tolerance: float | None = None,
    neighbours: dict[int, list[int]] | None = None,
) -> list[tuple[int, int, bool, int]]:
    """Check every neighbour pair for fan-surface overlaps.

    For each pair the function builds a fan surface (centroid to polygon
    edges), intersects the two surfaces, and checks whether every
    intersection point lies on *both* polyline boundaries (shared edge)
    or only on one (overlap).

    Returns ``(idx_a, idx_b, has_overlap, non_shared_intersection_point_count)``
    for every neighbour pair that was checked.
    """
    if snap_tolerance is None:
        snap_tolerance = default_snap_tolerance(tolerance)

    if neighbours is None:
        neighbours = find_polyline_neighbours(polylines, snap_tolerance)
    results: list[tuple[int, int, bool, int]] = []
    processed: set[tuple[int, int]] = set()

    vtk.vtkObject.GlobalWarningDisplayOff()

    for idx_a, nbrs in sorted(neighbours.items()):
        for idx_b in nbrs:
            pair = (min(idx_a, idx_b), max(idx_a, idx_b))
            if pair in processed:
                continue
            processed.add(pair)

            unique_a = _unique_polyline_points(polylines[idx_a], tolerance)
            unique_b = _unique_polyline_points(polylines[idx_b], tolerance)
            if len(unique_a) < 3 or len(unique_b) < 3:
                continue

            fan_a = _fan_surface_from_center(unique_a.mean(axis=0), unique_a)
            fan_b = _fan_surface_from_center(unique_b.mean(axis=0), unique_b)
            if fan_a.n_cells == 0 or fan_b.n_cells == 0:
                continue

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*not valid.*vtkOriginalCellIds.*")
                    intersection, _, _ = fan_a.triangulate().clean().intersection(
                        fan_b.triangulate().clean(),
                        split_first=False,
                        split_second=False,
                    )
                    intersection = _strip_stale_cell_arrays(intersection)
            except Exception:
                continue

            if intersection.n_points == 0:
                results.append((pair[0], pair[1], False, 0))
                continue

            check_tol = snap_tolerance * 3
            int_pts = np.asarray(intersection.points, dtype=float)
            non_shared = 0
            for ipt in int_pts:
                da = _distance_point_to_polyline(ipt, polylines[pair[0]])
                db = _distance_point_to_polyline(ipt, polylines[pair[1]])
                if not (da < check_tol and db < check_tol):
                    non_shared += 1

            results.append((pair[0], pair[1], non_shared > 0, non_shared))

    return results


def fix_polyline_surface_overlaps(
    polylines: list[np.ndarray],
    tolerance: float,
    snap_tolerance: float | None = None,
    max_iterations: int = 5,
    neighbours: dict[int, list[int]] | None = None,
) -> tuple[list[np.ndarray], int, list[str]]:
    """Fix overlapping neighbour surfaces by relocating offending vertices.

    For each neighbour pair, every non-shared vertex of polyline A is
    checked against polyline B's polygon (projected onto B's best-fit
    plane).  If the vertex lies inside B's polygon it is relocated to the
    nearest shared-edge segment (straight line between consecutive shared
    vertices).  The same check runs symmetrically for B against A.

    Returns ``(fixed_polylines, total_relocated, log_messages)``.
    """
    if snap_tolerance is None:
        snap_tolerance = default_snap_tolerance(tolerance)

    result = [p.copy() for p in polylines]
    total_relocated = 0
    messages: list[str] = []

    if neighbours is None:
        neighbours = find_polyline_neighbours(result, snap_tolerance)

    for iteration in range(max_iterations):
        relocated_this_round = 0
        processed: set[tuple[int, int]] = set()

        for idx_a, nbrs in sorted(neighbours.items()):
            for idx_b in nbrs:
                pair = (min(idx_a, idx_b), max(idx_a, idx_b))
                if pair in processed:
                    continue
                processed.add(pair)
                relocated_this_round += _fix_pair_overlap(
                    result, pair[0], pair[1], tolerance, snap_tolerance,
                )

        total_relocated += relocated_this_round
        if relocated_this_round > 0:
            messages.append(
                f"Overlap fix pass {iteration + 1}: relocated {relocated_this_round} point(s)"
            )
        if relocated_this_round == 0:
            break

    result = _resnap_shared_vertices(result, tolerance, snap_tolerance, neighbours=neighbours)

    result, pocket_removed, pocket_msgs = resolve_pocket_cells(result, tolerance, snap_tolerance, neighbours=neighbours)
    messages.extend(pocket_msgs)

    restored = 0
    for i in range(len(result)):
        unique_fixed = _unique_polyline_points(result[i], tolerance)
        if len(unique_fixed) < 3:
            result[i] = polylines[i].copy()
            restored += 1
    if restored > 0:
        messages.append(f"Restored {restored} degenerate polyline(s) to pre-fix state")

    return result, total_relocated, messages


def _resnap_shared_vertices(
    polylines: list[np.ndarray],
    tolerance: float,
    snap_tolerance: float,
    neighbours: dict[int, list[int]] | None = None,
) -> list[np.ndarray]:
    """Re-snap shared vertices between neighbours to close gaps."""
    unique_lists = [_unique_polyline_points(p, tolerance).copy() for p in polylines]
    closed_for_query = [_close_polyline(u, tolerance) for u in unique_lists]
    if neighbours is None:
        neighbours = find_polyline_neighbours(closed_for_query, snap_tolerance)

    processed: set[tuple[int, int]] = set()
    for idx_a, nbrs in sorted(neighbours.items()):
        for idx_b in nbrs:
            pair = (min(idx_a, idx_b), max(idx_a, idx_b))
            if pair in processed:
                continue
            processed.add(pair)

            shared = _find_shared_vertex_pairs(
                unique_lists[pair[0]], unique_lists[pair[1]], snap_tolerance,
            )
            for ia, ib in shared:
                avg = (unique_lists[pair[0]][ia] + unique_lists[pair[1]][ib]) / 2.0
                unique_lists[pair[0]][ia] = avg
                unique_lists[pair[1]][ib] = avg

    return [_close_polyline(u, tolerance) for u in unique_lists]


def resolve_pocket_cells(
    polylines: list[np.ndarray],
    tolerance: float,
    snap_tolerance: float | None = None,
    neighbours: dict[int, list[int]] | None = None,
) -> tuple[list[np.ndarray], list[int], list[str]]:
    """Remove pocket cells that are fully enclosed by a single larger neighbour.

    A cell is a pocket of neighbour N only when **every** polyline segment
    that is not shared with N is geometrically inside N's polygon.  If even
    one non-shared segment reaches outside N (e.g. toward a different
    neighbour), the cell is considered independent and kept.

    The check works on segment midpoints projected onto N's best-fit plane.

    Returns ``(filtered_polylines, removed_indices, log_messages)``.
    """
    if snap_tolerance is None:
        snap_tolerance = default_snap_tolerance(tolerance)

    if neighbours is None:
        neighbours = find_polyline_neighbours(polylines, snap_tolerance)
    pocket_indices: set[int] = set()
    messages: list[str] = []

    for idx in range(len(polylines)):
        u = _unique_polyline_points(polylines[idx], tolerance)
        if len(u) < 3:
            continue
        n_u = len(u)

        for nbr_idx in neighbours.get(idx, []):
            u_nbr = _unique_polyline_points(polylines[nbr_idx], tolerance)
            if len(u_nbr) < 3:
                continue
            if len(u_nbr) <= len(u):
                continue

            shared = _find_shared_vertex_pairs(u, u_nbr, snap_tolerance)
            if len(shared) < 2:
                continue

            shared_set = {ia for ia, _ in shared}

            po_n, pu_n, pv_n, pn_n = _fit_plane(u_nbr)
            p2d_n = _project_to_plane(u_nbr, po_n, pu_n, pv_n)
            max_oop = float(np.max(np.abs(np.dot(u_nbr - po_n, pn_n))))
            oop_limit = max_oop + snap_tolerance * 5

            non_shared_segs = 0
            inside_segs = 0
            for si in range(n_u):
                vi_a = si
                vi_b = (si + 1) % n_u
                if vi_a in shared_set and vi_b in shared_set:
                    continue
                non_shared_segs += 1
                midpoint = (u[vi_a] + u[vi_b]) * 0.5
                if abs(float(np.dot(midpoint - po_n, pn_n))) > oop_limit:
                    continue
                mid_2d = _project_to_plane(midpoint[None, :], po_n, pu_n, pv_n)[0]
                if _point_in_polygon_2d(mid_2d, p2d_n):
                    inside_segs += 1

            if non_shared_segs == 0:
                pocket_indices.add(idx)
                messages.append(
                    f"Pocket cell {idx} removed (all {n_u} segments shared "
                    f"with larger cell {nbr_idx})"
                )
                break

            if inside_segs == non_shared_segs:
                pocket_indices.add(idx)
                messages.append(
                    f"Pocket cell {idx} removed ({inside_segs}/{non_shared_segs} "
                    f"non-shared segments inside larger cell {nbr_idx}, "
                    f"{len(shared)}/{n_u} shared vertices)"
                )
                break

    removed = sorted(pocket_indices)
    kept = [p for i, p in enumerate(polylines) if i not in pocket_indices]
    return kept, removed, messages


def _find_nearest_segment_on_cells(
    point: np.ndarray,
    candidate_indices: list[int],
    polylines: list[np.ndarray],
    tolerance: float,
) -> tuple[float, np.ndarray, int, int]:
    """Find the nearest polyline segment across a set of candidate cells.

    Returns ``(distance, projected_point, cell_index, segment_index)``.
    """
    best_dist = float("inf")
    best_pt = point.copy()
    best_cell = -1
    best_seg = -1

    for ni in candidate_indices:
        if ni < 0 or ni >= len(polylines):
            continue
        un = _unique_polyline_points(polylines[ni], tolerance)
        if len(un) < 3:
            continue
        closed_n = _close_polyline(un, tolerance)
        for si in range(len(closed_n) - 1):
            ab = closed_n[si + 1] - closed_n[si]
            ab_sq = float(np.dot(ab, ab))
            if ab_sq < 1e-20:
                proj = closed_n[si].copy()
            else:
                t = max(0.0, min(1.0, float(np.dot(point - closed_n[si], ab)) / ab_sq))
                proj = closed_n[si] + t * ab
            d = float(np.linalg.norm(point - proj))
            if d < best_dist:
                best_dist = d
                best_pt = proj
                best_cell = ni
                best_seg = si

    return best_dist, best_pt, best_cell, best_seg


def close_free_vertices(
    polylines: list[np.ndarray],
    surface: pv.PolyData,
    tolerance: float,
    snap_tolerance: float | None = None,
    neighbours: dict[int, list[int]] | None = None,
) -> tuple[list[np.ndarray], int, list[str]]:
    """Snap free vertices of interior cells to neighbour edges and insert
    the snapped point into the neighbour's polyline so both cells share it.

    A *free vertex* is one not shared with any neighbour.  For cells
    whose polyline lies entirely inside the lofted surface (not an edge
    cell), every vertex should be shared with at least one neighbour.
    Free vertices create gaps between cells.

    For each free vertex the function:
    1. Projects it onto the nearest segment of the nearest neighbour.
    2. Moves the free vertex to that projected position.
    3. **Inserts** that position into the neighbour's polyline between
       the two segment endpoints, creating a new shared vertex.

    Returns ``(updated_polylines, total_snapped, log_messages)``.
    """
    if snap_tolerance is None:
        snap_tolerance = default_snap_tolerance(tolerance)

    result = [p.copy() for p in polylines]
    if neighbours is None:
        neighbours = find_polyline_neighbours(result, snap_tolerance)
    boundary_pts = _surface_boundary_points(surface, snap_tolerance)
    messages: list[str] = []
    total_snapped = 0

    insertions: dict[int, list[tuple[int, np.ndarray]]] = {}

    from scipy.spatial import cKDTree
    centroids = np.array([
        _unique_polyline_points(result[i], tolerance).mean(axis=0)
        if len(_unique_polyline_points(result[i], tolerance)) >= 3
        else np.full(3, np.inf)
        for i in range(len(result))
    ])
    centroid_tree = cKDTree(centroids[np.isfinite(centroids[:, 0])])
    valid_centroid_map = np.where(np.isfinite(centroids[:, 0]))[0]

    for ci in range(len(result)):
        u = _unique_polyline_points(result[ci], tolerance)
        if len(u) < 3:
            continue

        shared_indices: set[int] = set()
        for ni in neighbours.get(ci, []):
            un = _unique_polyline_points(result[ni], tolerance)
            for ia, _ in _find_shared_vertex_pairs(u, un, snap_tolerance):
                shared_indices.add(ia)

        free_indices = [i for i in range(len(u)) if i not in shared_indices]
        if not free_indices:
            continue

        u_copy = u.copy()
        snapped_this_cell = 0
        nbr_snap_limit = snap_tolerance * 20

        dists_from_center = np.linalg.norm(u - u.mean(axis=0), axis=1)
        cell_radius = float(dists_from_center.max()) if len(dists_from_center) > 0 else 1.0
        extended_snap_limit = min(
            max(snap_tolerance * 100, 2.0),
            cell_radius * 0.5,
        )

        nbr_set = set(neighbours.get(ci, []))

        for fi in free_indices:
            if _is_on_surface_boundary(u_copy[fi], boundary_pts, snap_tolerance * 5):
                continue

            best_dist, best_pt, best_nbr, best_seg_idx = _find_nearest_segment_on_cells(
                u_copy[fi], neighbours.get(ci, []), result, tolerance,
            )

            if best_dist > nbr_snap_limit:
                search_radius = max(cell_radius * 2.0, extended_snap_limit * 2.0)
                nearby_tree_indices = centroid_tree.query_ball_point(u_copy[fi], search_radius)
                non_nbr_candidates = [
                    int(valid_centroid_map[ti]) for ti in nearby_tree_indices
                    if int(valid_centroid_map[ti]) != ci and int(valid_centroid_map[ti]) not in nbr_set
                ]
                if non_nbr_candidates:
                    ext_dist, ext_pt, ext_nbr, ext_seg = _find_nearest_segment_on_cells(
                        u_copy[fi], non_nbr_candidates, result, tolerance,
                    )
                    if ext_dist < best_dist:
                        best_dist, best_pt, best_nbr, best_seg_idx = ext_dist, ext_pt, ext_nbr, ext_seg

            snap_limit = extended_snap_limit if best_dist > nbr_snap_limit else nbr_snap_limit
            if best_dist < snap_limit and best_nbr >= 0:
                u_copy[fi] = best_pt
                snapped_this_cell += 1
                if best_nbr not in insertions:
                    insertions[best_nbr] = []
                insertions[best_nbr].append((best_seg_idx, best_pt.copy()))

        if snapped_this_cell > 0:
            result[ci] = _close_polyline(u_copy, tolerance)
            total_snapped += snapped_this_cell

    for ni, ins_list in insertions.items():
        u_nbr = _unique_polyline_points(result[ni], tolerance).copy()
        spliced = _splice_points_into_polyline(
            u_nbr, ins_list, tolerance, dedup_against_existing=True,
        )
        result[ni] = _close_polyline(spliced, tolerance)

    if total_snapped > 0:
        messages.append(f"Snapped {total_snapped} free vertices to neighbour edges")

    total_absorbed = 0
    absorb_neighbours = find_polyline_neighbours(result, snap_tolerance)
    for _absorb_pass in range(5):
        result, absorbed = _absorb_nearby_neighbour_vertices(
            result, tolerance, snap_tolerance, neighbours=absorb_neighbours,
        )
        total_absorbed += absorbed
        if absorbed == 0:
            break
    if total_absorbed > 0:
        messages.append(f"Absorbed {total_absorbed} neighbour vertices into cell polylines")

    for i in range(len(result)):
        result[i] = _deduplicate_polyline(result[i], tolerance)

    result, junctions_added = _insert_junction_points(
        result, tolerance, snap_tolerance, neighbours=absorb_neighbours,
    )
    if junctions_added > 0:
        messages.append(f"Inserted {junctions_added} junction points from neighbour intersections")
        for i in range(len(result)):
            result[i] = _deduplicate_polyline(result[i], tolerance)

    return result, total_snapped + total_absorbed + junctions_added, messages


def _insert_junction_points(
    polylines: list[np.ndarray],
    tolerance: float,
    snap_tolerance: float,
    neighbours: dict[int, list[int]] | None = None,
) -> tuple[list[np.ndarray], int]:
    """Insert junction points that neighbours share but a cell lacks.

    A *junction point* is a vertex shared by two or more of cell C's
    neighbours.  If that point is close to C's polyline but C doesn't
    have it, there's a tiling gap at the junction.  This function finds
    such points and inserts them into C's polyline.

    Returns ``(updated_polylines, total_inserted)``.
    """
    result = [p.copy() for p in polylines]
    if neighbours is None:
        neighbours = find_polyline_neighbours(result, snap_tolerance)
    junction_limit = max(snap_tolerance * 30, 0.6)
    total_inserted = 0

    for ci in range(len(result)):
        u_c = _unique_polyline_points(result[ci], tolerance)
        if len(u_c) < 3:
            continue
        closed_c = _close_polyline(u_c, tolerance)
        cell_nbrs = neighbours.get(ci, [])
        if len(cell_nbrs) < 2:
            continue

        nbr_verts: dict[tuple[int, ...], list[tuple[int, int, np.ndarray]]] = {}
        for ni in cell_nbrs:
            u_n = _unique_polyline_points(result[ni], tolerance)
            for vi in range(len(u_n)):
                key = tuple(np.round(u_n[vi] / snap_tolerance).astype(int).tolist())
                if key not in nbr_verts:
                    nbr_verts[key] = []
                nbr_verts[key].append((ni, vi, u_n[vi]))

        to_insert: list[tuple[int, np.ndarray]] = []
        for key, entries in nbr_verts.items():
            if len(entries) < 2:
                continue
            unique_cells = len(set(ni for ni, _, _ in entries))
            if unique_cells < 2:
                continue

            pt = entries[0][2]
            if len(u_c) > 0 and float(np.min(np.linalg.norm(u_c - pt, axis=1))) < snap_tolerance:
                continue

            d_seg = _distance_point_to_polyline(pt, closed_c)
            if d_seg > junction_limit:
                continue

            best_seg = -1
            best_sd = float("inf")
            for si in range(len(closed_c) - 1):
                sd = _distance_point_to_segment(pt, closed_c[si], closed_c[si + 1])
                if sd < best_sd:
                    best_sd = sd
                    best_seg = si

            already = any(
                float(np.linalg.norm(pt - ip)) < tolerance for _, ip in to_insert
            )
            if not already and best_seg >= 0:
                to_insert.append((best_seg, pt.copy()))

        if not to_insert:
            continue

        spliced = _splice_points_into_polyline(u_c, to_insert, tolerance)
        result[ci] = _close_polyline(spliced, tolerance)
        total_inserted += len(to_insert)

    return result, total_inserted


def _deduplicate_polyline(polyline: np.ndarray, tolerance: float) -> np.ndarray:
    """Remove consecutive duplicate vertices from a closed polyline."""
    u = _unique_polyline_points(polyline, tolerance)
    if len(u) < 3:
        return polyline
    kept = [u[0]]
    for i in range(1, len(u)):
        if float(np.linalg.norm(u[i] - kept[-1])) > tolerance:
            is_dup = any(float(np.linalg.norm(u[i] - k)) < tolerance for k in kept)
            if not is_dup:
                kept.append(u[i])
    if len(kept) < 3:
        return polyline
    return _close_polyline(np.array(kept, dtype=float), tolerance)


def _absorb_nearby_neighbour_vertices(
    polylines: list[np.ndarray],
    tolerance: float,
    snap_tolerance: float,
    neighbours: dict[int, list[int]] | None = None,
) -> tuple[list[np.ndarray], int]:
    """Insert neighbour vertices that lie on our polyline but aren't shared.

    For each cell C and each neighbour N, any vertex of N that is close
    to a segment of C but not already matched to a vertex of C is inserted
    into C's polyline.  This closes triangle-shaped holes caused by one
    cell having a vertex at a junction that the adjacent cell lacks.

    Returns ``(updated_polylines, total_absorbed)``.
    """
    result = [p.copy() for p in polylines]
    if neighbours is None:
        neighbours = find_polyline_neighbours(result, snap_tolerance)
    absorb_limit = snap_tolerance * 10
    total_absorbed = 0

    for ci in range(len(result)):
        u_c = _unique_polyline_points(result[ci], tolerance)
        if len(u_c) < 3:
            continue
        closed_c = _close_polyline(u_c, tolerance)

        to_insert: list[tuple[int, np.ndarray]] = []

        for ni in neighbours.get(ci, []):
            u_n = _unique_polyline_points(result[ni], tolerance)
            if len(u_n) < 3:
                continue

            shared_c = {ia for ia, _ in _find_shared_vertex_pairs(u_c, u_n, snap_tolerance)}

            if len(u_n) == 0 or len(u_c) == 0:
                continue
            n_to_c_dists = cdist(u_n, u_c)
            n_min_dists = n_to_c_dists.min(axis=1)

            candidate_mask = n_min_dists >= snap_tolerance
            candidate_indices = np.where(candidate_mask)[0]
            if len(candidate_indices) == 0:
                continue

            n_segs = len(closed_c) - 1
            if n_segs > 0:
                seg_dists = _distances_points_to_segments(
                    u_n[candidate_indices],
                    closed_c[:-1],
                    closed_c[1:],
                )
                best_segs = seg_dists.argmin(axis=1)
                best_ds = seg_dists[np.arange(len(candidate_indices)), best_segs]

                for k, vi in enumerate(candidate_indices):
                    if best_ds[k] >= absorb_limit:
                        continue
                    pt = u_n[vi]
                    already = any(
                        float(np.linalg.norm(pt - ip)) < tolerance
                        for _, ip in to_insert
                    )
                    if not already:
                        to_insert.append((int(best_segs[k]), pt.copy()))

        if not to_insert:
            continue

        spliced = _splice_points_into_polyline(u_c, to_insert, tolerance)
        result[ci] = _close_polyline(spliced, tolerance)
        total_absorbed += len(to_insert)

    return result, total_absorbed


def _surface_boundary_points(
    surface: pv.PolyData,
    snap_tolerance: float,
) -> np.ndarray:
    """Extract boundary vertex positions from the lofted surface."""
    boundary = surface.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    ).clean()
    if boundary.n_points == 0:
        return np.empty((0, 3), dtype=float)
    return np.asarray(boundary.points, dtype=float)


def _is_on_surface_boundary(
    point: np.ndarray,
    boundary_pts: np.ndarray,
    edge_tol: float,
) -> bool:
    """Check if a single point is near the surface boundary."""
    if len(boundary_pts) == 0:
        return False
    dists = np.linalg.norm(boundary_pts - point, axis=1)
    return float(dists.min()) < edge_tol


def _detect_edge_cells(
    polylines: list[np.ndarray],
    surface: pv.PolyData,
    tolerance: float,
    snap_tolerance: float,
) -> set[int]:
    """Return indices of cells that touch the lofted surface boundary."""
    bnd_pts = _surface_boundary_points(surface, snap_tolerance)
    if len(bnd_pts) == 0:
        return set()

    edge_tol = snap_tolerance * 5
    edge_cells: set[int] = set()

    for ci in range(len(polylines)):
        u = _unique_polyline_points(polylines[ci], tolerance)
        for pt in u:
            dists = np.linalg.norm(bnd_pts - pt, axis=1)
            if float(dists.min()) < edge_tol:
                edge_cells.add(ci)
                break

    return edge_cells


def _distance_point_to_polyline(
    point: np.ndarray,
    polyline: np.ndarray,
) -> float:
    """Minimum distance from *point* to any segment of *polyline*."""
    if len(polyline) < 2:
        if len(polyline) == 1:
            return float(np.linalg.norm(point - polyline[0]))
        return float("inf")
    best = float("inf")
    for i in range(len(polyline) - 1):
        d = _distance_point_to_segment(point, polyline[i], polyline[i + 1])
        if d < best:
            best = d
    return best


def _distance_point_to_segment(
    point: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
) -> float:
    """Distance from *point* to the finite line segment *seg_a*--*seg_b*."""
    ab = seg_b - seg_a
    ab_sq = float(np.dot(ab, ab))
    if ab_sq < 1e-20:
        return float(np.linalg.norm(point - seg_a))
    t = max(0.0, min(1.0, float(np.dot(point - seg_a, ab)) / ab_sq))
    proj = seg_a + t * ab
    return float(np.linalg.norm(point - proj))


def _distances_points_to_segments(
    points: np.ndarray,
    seg_starts: np.ndarray,
    seg_ends: np.ndarray,
) -> np.ndarray:
    """Vectorized distance from each point to each segment.

    Returns shape ``(n_points, n_segments)``.
    """
    ab = seg_ends - seg_starts  # (S, 3)
    ab_sq = np.einsum("ij,ij->i", ab, ab)  # (S,)

    ap = points[:, None, :] - seg_starts[None, :, :]  # (P, S, 3)
    dot_ap_ab = np.einsum("ijk,jk->ij", ap, ab)  # (P, S)

    safe_ab_sq = np.where(ab_sq < 1e-20, 1.0, ab_sq)
    t = dot_ap_ab / safe_ab_sq[None, :]  # (P, S)
    t = np.clip(t, 0.0, 1.0)

    proj = seg_starts[None, :, :] + t[:, :, None] * ab[None, :, :]  # (P, S, 3)
    diff = points[:, None, :] - proj  # (P, S, 3)
    dists = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))  # (P, S)

    degenerate = ab_sq < 1e-20
    if np.any(degenerate):
        deg_dists = np.linalg.norm(
            points[:, None, :] - seg_starts[None, degenerate, :], axis=2
        )
        dists[:, degenerate] = deg_dists

    return dists


def _project_point_to_nearest_segment(
    point: np.ndarray,
    segments: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Project *point* onto the nearest segment, return the projected point."""
    best_dist = float("inf")
    best_proj = point.copy()
    for seg_a, seg_b in segments:
        ab = seg_b - seg_a
        ab_sq = float(np.dot(ab, ab))
        if ab_sq < 1e-20:
            proj = seg_a.copy()
        else:
            t = max(0.0, min(1.0, float(np.dot(point - seg_a, ab)) / ab_sq))
            proj = seg_a + t * ab
        d = float(np.linalg.norm(point - proj))
        if d < best_dist:
            best_dist = d
            best_proj = proj
    return best_proj


def _fix_pair_overlap(
    polylines: list[np.ndarray],
    idx_a: int,
    idx_b: int,
    tolerance: float,
    snap_tolerance: float,
) -> int:
    """Fix surface overlap for one neighbour pair.  Returns relocated count.

    Detects when one cell *dominates* the other (a legitimate smaller cell
    whose non-shared vertices are mostly inside the larger cell) and
    preserves the smaller cell instead of collapsing it onto the shared edge.
    """
    unique_a = _unique_polyline_points(polylines[idx_a], tolerance).copy()
    unique_b = _unique_polyline_points(polylines[idx_b], tolerance).copy()

    if len(unique_a) < 3 or len(unique_b) < 3:
        return 0

    shared = _find_shared_vertex_pairs(unique_a, unique_b, snap_tolerance)
    if len(shared) < 2:
        return 0

    shared_a_set = {ia for ia, _ in shared}
    shared_b_set = {ib for _, ib in shared}

    shared_sorted_a = sorted(shared, key=lambda p: p[0])
    segs_a: list[tuple[np.ndarray, np.ndarray]] = [
        (unique_a[shared_sorted_a[k][0]].copy(),
         unique_a[shared_sorted_a[(k + 1) % len(shared_sorted_a)][0]].copy())
        for k in range(len(shared_sorted_a))
    ]

    shared_sorted_b = sorted(shared, key=lambda p: p[1])
    segs_b: list[tuple[np.ndarray, np.ndarray]] = [
        (unique_b[shared_sorted_b[k][1]].copy(),
         unique_b[shared_sorted_b[(k + 1) % len(shared_sorted_b)][1]].copy())
        for k in range(len(shared_sorted_b))
    ]

    edge_proximity = tolerance * 10
    a_non_shared = [i for i in range(len(unique_a)) if i not in shared_a_set]
    b_non_shared = [i for i in range(len(unique_b)) if i not in shared_b_set]

    po_a, pu_a, pv_a, pn_a = _fit_plane(unique_a)
    po_b, pu_b, pv_b, pn_b = _fit_plane(unique_b)
    poly2d_a = _project_to_plane(unique_a, po_a, pu_a, pv_a)
    poly2d_b = _project_to_plane(unique_b, po_b, pu_b, pv_b)
    max_oop_a = float(np.max(np.abs(np.dot(unique_a - po_a, pn_a))))
    max_oop_b = float(np.max(np.abs(np.dot(unique_b - po_b, pn_b))))

    def _inside(pt: np.ndarray, po: np.ndarray, pu: np.ndarray, pv: np.ndarray,
                pn: np.ndarray, poly2d: np.ndarray, max_oop: float) -> bool:
        if abs(float(np.dot(pt - po, pn))) > max_oop + snap_tolerance * 5:
            return False
        return _point_in_polygon_2d(
            _project_to_plane(pt[None, :], po, pu, pv)[0], poly2d,
        )

    a_inside_b = [
        i for i in a_non_shared
        if _inside(unique_a[i], po_b, pu_b, pv_b, pn_b, poly2d_b, max_oop_b)
    ]
    b_inside_a = [
        i for i in b_non_shared
        if _inside(unique_b[i], po_a, pu_a, pv_a, pn_a, poly2d_a, max_oop_a)
    ]

    a_dominated = len(a_non_shared) > 0 and len(a_inside_b) > len(a_non_shared) * 0.85
    b_dominated = len(b_non_shared) > 0 and len(b_inside_a) > len(b_non_shared) * 0.85

    relocated = 0

    if not a_dominated:
        for i in a_inside_b:
            if segs_a and min(
                _distance_point_to_segment(unique_a[i], s1, s2) for s1, s2 in segs_a
            ) < edge_proximity:
                continue
            unique_a[i] = _project_point_to_nearest_segment(unique_a[i], segs_a)
            relocated += 1

    if not b_dominated:
        for i in b_inside_a:
            if segs_b and min(
                _distance_point_to_segment(unique_b[i], s1, s2) for s1, s2 in segs_b
            ) < edge_proximity:
                continue
            unique_b[i] = _project_point_to_nearest_segment(unique_b[i], segs_b)
            relocated += 1

    if relocated > 0:
        polylines[idx_a] = _close_polyline(unique_a, tolerance)
        polylines[idx_b] = _close_polyline(unique_b, tolerance)

    return relocated


def analyze_and_generate_surfaces(
    polylines: list[np.ndarray],
    loft_bounds: tuple[float, float, float, float, float, float],
    tolerance: float,
    discontinuity_angle_degrees: float = 176.0,
    extrusion_multiplier: float = 0.5,
    small_cell_extrusion_factor: float = 0.1,
    extrusion_scale_origin: np.ndarray | tuple[float, float, float] | None = None,
    planar_scale_factors: tuple[float, float] = (1.0, 1.0),
    slice_plane_x: float | None = None,
) -> SurfaceGenerationResult:
    loft_bbox_center = np.array(
        [
            0.5 * (loft_bounds[0] + loft_bounds[1]),
            0.5 * (loft_bounds[2] + loft_bounds[3]),
            0.5 * (loft_bounds[4] + loft_bounds[5]),
        ],
        dtype=float,
    )
    scale_origin = np.asarray(extrusion_scale_origin if extrusion_scale_origin is not None else loft_bbox_center, dtype=float)
    scale_x, scale_y = planar_scale_factors

    analyses: list[CurveAnalysis] = []
    for polyline in polylines:
        unique_points = _unique_polyline_points(polyline, tolerance=tolerance)
        if len(unique_points) < 3:
            continue

        discontinuity_points = unique_points.copy()
        followup_polyline = _sanitize_closed_polyline(polyline, tolerance=tolerance)

        plane_origin, plane_u, plane_v, plane_normal = _fit_plane(unique_points)
        circle_center, circle_radius = _fit_circle_on_plane(unique_points, plane_origin, plane_u, plane_v)
        circle_center = _ensure_center_inside_polygon(
            circle_center, unique_points, plane_origin, plane_u, plane_v,
        )
        bbox_mesh, bbox_center, bbox_volume, bbox_extents = _build_plane_aligned_bounding_box(
            unique_points,
            plane_origin,
            plane_u,
            plane_v,
            plane_normal,
            tolerance=tolerance,
        )
        u_span, v_span, n_span = bbox_extents
        min_span = min(u_span, v_span)
        bbox_aspect_ratio = max(u_span, v_span) / min_span if min_span > tolerance else 1.0
        max_planar_span = max(u_span, v_span)
        planarity_ratio = n_span / max_planar_span if max_planar_span > tolerance else 0.0
        curve_length = _polyline_length(followup_polyline)
        if curve_length <= tolerance:
            continue

        scaled_circle_center = circle_center.copy()
        scaled_circle_center[0] = scale_origin[0] + scale_x * (scaled_circle_center[0] - scale_origin[0])
        scaled_circle_center[1] = scale_origin[1] + scale_y * (scaled_circle_center[1] - scale_origin[1])
        direction_vector = scaled_circle_center - circle_center
        direction_length = float(np.linalg.norm(direction_vector))
        if direction_length <= tolerance:
            offset_direction = np.zeros(3, dtype=float)
        else:
            offset_direction = direction_vector / direction_length

        analyses.append(
            CurveAnalysis(
                original_polyline=followup_polyline,
                followup_polyline=followup_polyline,
                discontinuity_points=discontinuity_points,
                plane_origin=plane_origin,
                plane_u=plane_u,
                plane_v=plane_v,
                plane_normal=plane_normal,
                circle_center=circle_center,
                circle_radius=circle_radius,
                bbox_mesh=bbox_mesh,
                bbox_center=bbox_center,
                bbox_volume=bbox_volume,
                curve_length=curve_length,
                ratio=bbox_volume / curve_length,
                bbox_aspect_ratio=bbox_aspect_ratio,
                planarity_ratio=planarity_ratio,
                scaled_circle_center=scaled_circle_center,
                extrusion_base_vector=direction_vector,
                offset_direction=offset_direction,
            )
        )

    if not analyses:
        empty = pv.PolyData()
        return SurfaceGenerationResult(
            analyses=tuple(),
            average_ratio=0.0,
            followup_polylines=tuple(),
            generated_surface=empty,
            larger_surface=empty,
            smaller_surface=empty,
        )

    average_ratio = float(np.mean([analysis.ratio for analysis in analyses]))
    larger_meshes: list[pv.PolyData] = []
    smaller_meshes: list[pv.PolyData] = []

    for analysis in analyses:
        is_extreme = (
            analysis.bbox_aspect_ratio > EXTREME_ASPECT_RATIO_THRESHOLD
            or analysis.planarity_ratio < EXTREME_PLANARITY_RATIO
        )

        if analysis.ratio >= average_ratio:
            offset_vector = extrusion_multiplier * analysis.extrusion_base_vector
            if is_extreme:
                offset_vector = EXTREME_EXTRUSION_FACTOR * offset_vector
                staged_loft_mesh, _, _ = _build_extreme_cell_lofts(
                    analysis.followup_polyline,
                    center=analysis.circle_center,
                    plane_origin=analysis.plane_origin,
                    plane_u=analysis.plane_u,
                    plane_v=analysis.plane_v,
                    plane_normal=analysis.plane_normal,
                    offset_vector=offset_vector,
                )
            else:
                staged_loft_mesh, _, _ = _build_staged_offset_lofts(
                    analysis.followup_polyline,
                    center=analysis.circle_center,
                    plane_u=analysis.plane_u,
                    plane_v=analysis.plane_v,
                    offset_vector=offset_vector,
                )
            larger_meshes.append(staged_loft_mesh)
        else:
            offset_vector = small_cell_extrusion_factor * extrusion_multiplier * analysis.extrusion_base_vector
            if is_extreme:
                offset_vector = EXTREME_EXTRUSION_FACTOR * offset_vector
            moved_center = analysis.circle_center + offset_vector
            small_mesh = _fan_surface_from_center(moved_center, analysis.discontinuity_points)
            if _small_mesh_exceeds_retained_volume(
                small_mesh,
                loft_bounds=loft_bounds,
                slice_plane_x=slice_plane_x,
                tolerance=tolerance,
            ):
                continue
            smaller_meshes.append(small_mesh)

    larger_surface = _merge_meshes(larger_meshes)
    smaller_surface = _merge_meshes(smaller_meshes)
    generated_surface = _merge_meshes([mesh for mesh in [larger_surface, smaller_surface] if mesh.n_cells > 0])

    return SurfaceGenerationResult(
        analyses=tuple(analyses),
        average_ratio=average_ratio,
        followup_polylines=tuple(analysis.followup_polyline for analysis in analyses),
        generated_surface=generated_surface,
        larger_surface=larger_surface,
        smaller_surface=smaller_surface,
    )


def _circle_points(radius: float, z_level: float, resolution: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False)
    x_coords = radius * np.cos(angles)
    y_coords = radius * np.sin(angles)
    z_coords = np.full(resolution, z_level, dtype=float)
    return np.column_stack((x_coords, y_coords, z_coords))


def _build_single_cell(
    seed_points: np.ndarray,
    index: int,
    bounds: tuple[float, float, float, float, float, float],
) -> pv.PolyData:
    interior_point = seed_points[index]
    halfspaces = [_bounding_box_halfspaces(bounds)]

    for other_index, other_point in enumerate(seed_points):
        if other_index == index:
            continue
        normal = 2.0 * (other_point - interior_point)
        offset = float(np.dot(interior_point, interior_point) - np.dot(other_point, other_point))
        halfspaces.append(np.array([[normal[0], normal[1], normal[2], offset]], dtype=float))

    bounded_halfspaces = np.vstack(halfspaces)
    intersections = HalfspaceIntersection(bounded_halfspaces, interior_point=interior_point)
    vertices = intersections.intersections
    hull = ConvexHull(vertices)

    faces: list[int] = []
    for triangle in hull.simplices:
        faces.extend([3, *triangle.tolist()])

    return pv.PolyData(vertices, faces=np.array(faces, dtype=np.int64)).clean().triangulate()


def _bounding_box_halfspaces(bounds: tuple[float, float, float, float, float, float]) -> np.ndarray:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return np.array(
        [
            [-1.0, 0.0, 0.0, xmin],
            [1.0, 0.0, 0.0, -xmax],
            [0.0, -1.0, 0.0, ymin],
            [0.0, 1.0, 0.0, -ymax],
            [0.0, 0.0, -1.0, zmin],
            [0.0, 0.0, 1.0, -zmax],
        ],
        dtype=float,
    )


def _extract_polylines(line_mesh: pv.PolyData, tolerance: float) -> list[np.ndarray]:
    segments = _extract_segments(line_mesh)
    if not segments:
        return []

    canonical_points, edges = _merge_points_and_edges(line_mesh.points, segments, tolerance=tolerance)
    adjacency: dict[int, set[int]] = defaultdict(set)
    for start, end in edges:
        adjacency[start].add(end)
        adjacency[end].add(start)

    visited_nodes: set[int] = set()
    polylines: list[np.ndarray] = []
    for node in adjacency:
        if node in visited_nodes:
            continue
        component_nodes = _connected_component(node, adjacency)
        visited_nodes.update(component_nodes)
        degrees = {component_node: len(adjacency[component_node]) for component_node in component_nodes}
        endpoints = [component_node for component_node, degree in degrees.items() if degree == 1]

        if len(endpoints) == 0 and all(degree == 2 for degree in degrees.values()):
            ordered_nodes = _ordered_cycle(component_nodes, adjacency)
        elif len(endpoints) == 2 and all(degree in {1, 2} for degree in degrees.values()):
            ordered_nodes = _ordered_path(start_node=min(endpoints), adjacency=adjacency)
        else:
            continue
        polyline = np.array([canonical_points[index] for index in ordered_nodes], dtype=float)
        if not np.allclose(polyline[0], polyline[-1], atol=tolerance):
            polyline = np.vstack([polyline, polyline[0]])
        polylines.append(polyline)

    return polylines


def _extract_segments(line_mesh: pv.PolyData) -> list[tuple[int, int]]:
    raw_lines = line_mesh.lines
    segments: list[tuple[int, int]] = []
    cursor = 0

    while cursor < len(raw_lines):
        point_count = int(raw_lines[cursor])
        point_ids = raw_lines[cursor + 1 : cursor + 1 + point_count]
        for start, end in zip(point_ids[:-1], point_ids[1:], strict=True):
            if int(start) != int(end):
                segments.append((int(start), int(end)))
        cursor += point_count + 1

    return segments


def _merge_points_and_edges(
    points: np.ndarray,
    segments: list[tuple[int, int]],
    tolerance: float,
) -> tuple[list[np.ndarray], set[tuple[int, int]]]:
    point_index_by_key: dict[tuple[int, int, int], int] = {}
    canonical_points: list[np.ndarray] = []

    def get_or_create_index(point: np.ndarray) -> int:
        key = tuple(np.round(point / tolerance).astype(int).tolist())
        existing_index = point_index_by_key.get(key)
        if existing_index is not None:
            return existing_index
        new_index = len(canonical_points)
        point_index_by_key[key] = new_index
        canonical_points.append(point)
        return new_index

    edges: set[tuple[int, int]] = set()
    for start, end in segments:
        start_index = get_or_create_index(points[start])
        end_index = get_or_create_index(points[end])
        if start_index == end_index:
            continue
        edges.add(tuple(sorted((start_index, end_index))))

    return canonical_points, edges


def _connected_component(start_node: int, adjacency: dict[int, set[int]]) -> set[int]:
    stack = [start_node]
    component: set[int] = set()

    while stack:
        node = stack.pop()
        if node in component:
            continue
        component.add(node)
        stack.extend(adjacency[node] - component)

    return component


def _ordered_cycle(component_nodes: set[int], adjacency: dict[int, set[int]]) -> list[int]:
    start_node = min(component_nodes)
    ordered = [start_node]
    previous_node = -1
    current_node = start_node

    while True:
        neighbors = sorted(adjacency[current_node])
        next_candidates = [neighbor for neighbor in neighbors if neighbor != previous_node]
        if not next_candidates:
            break
        next_node = next_candidates[0]
        if next_node == start_node:
            ordered.append(start_node)
            break
        ordered.append(next_node)
        previous_node, current_node = current_node, next_node

    return ordered


def _ordered_path(start_node: int, adjacency: dict[int, set[int]]) -> list[int]:
    ordered = [start_node]
    previous_node = -1
    current_node = start_node

    while True:
        neighbors = sorted(adjacency[current_node])
        next_candidates = [neighbor for neighbor in neighbors if neighbor != previous_node]
        if not next_candidates:
            break
        next_node = next_candidates[0]
        ordered.append(next_node)
        previous_node, current_node = current_node, next_node

    return ordered


def _canonical_polyline_key(polyline: np.ndarray, tolerance: float) -> tuple[tuple[int, int, int], ...]:
    points = polyline[:-1] if np.allclose(polyline[0], polyline[-1], atol=tolerance) else polyline
    rounded_points = [tuple(np.round(point / tolerance).astype(int).tolist()) for point in points]

    forward_rotations = [tuple(rounded_points[index:] + rounded_points[:index]) for index in range(len(rounded_points))]
    reversed_points = list(reversed(rounded_points))
    reverse_rotations = [tuple(reversed_points[index:] + reversed_points[:index]) for index in range(len(reversed_points))]
    return min(forward_rotations + reverse_rotations)


def _unique_polyline_points(polyline: np.ndarray, tolerance: float) -> np.ndarray:
    if len(polyline) == 0:
        return polyline
    if np.allclose(polyline[0], polyline[-1], atol=tolerance):
        return polyline[:-1]
    return polyline


def _build_straight_polyline_from_discontinuities(
    points: np.ndarray,
    tolerance: float,
    discontinuity_angle_degrees: float,
    forced_discontinuity_indices: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    discontinuity_indices = _detect_discontinuity_indices(points, discontinuity_angle_degrees)
    if forced_discontinuity_indices:
        discontinuity_indices = sorted(set(discontinuity_indices) | forced_discontinuity_indices)
    clustered_indices = _cluster_cyclic_indices(discontinuity_indices, len(points), max_gap=0)
    if len(clustered_indices) < 3:
        clustered_indices = _fallback_sample_indices(len(points), minimum_count=3)

    straight_points = points[clustered_indices]
    if not np.allclose(straight_points[0], straight_points[-1], atol=tolerance):
        straight_points = np.vstack([straight_points, straight_points[0]])

    return straight_points, points[clustered_indices]


def _snap_neighboring_polyline_points(
    polylines: list[np.ndarray],
    tolerance: float,
    snap_tolerance: float,
) -> list[np.ndarray]:
    if len(polylines) < 2 or snap_tolerance <= tolerance:
        return [_sanitize_closed_polyline(polyline, tolerance=tolerance) for polyline in polylines]

    polyline_points = [_unique_polyline_points(polyline, tolerance=tolerance).copy() for polyline in polylines]
    point_refs: list[tuple[int, int]] = []
    flat_points: list[np.ndarray] = []
    for polyline_index, points in enumerate(polyline_points):
        for point_index, point in enumerate(points):
            point_refs.append((polyline_index, point_index))
            flat_points.append(point)

    if len(flat_points) < 2:
        return [_sanitize_closed_polyline(polyline, tolerance=tolerance) for polyline in polylines]

    points_array = np.array(flat_points, dtype=float)
    parents = list(range(len(points_array)))
    ranks = [0] * len(points_array)

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        root_first = find(first)
        root_second = find(second)
        if root_first == root_second:
            return
        if ranks[root_first] < ranks[root_second]:
            parents[root_first] = root_second
        elif ranks[root_first] > ranks[root_second]:
            parents[root_second] = root_first
        else:
            parents[root_second] = root_first
            ranks[root_first] += 1

    poly_indices = np.array([pr[0] for pr in point_refs], dtype=int)
    dist_matrix = cdist(points_array, points_array)
    close_pairs = np.argwhere(
        (dist_matrix <= snap_tolerance) & (np.arange(len(points_array))[:, None] < np.arange(len(points_array))[None, :])
    )
    for first_index, second_index in close_pairs:
        if poly_indices[first_index] != poly_indices[second_index]:
            union(first_index, second_index)

    cluster_members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(points_array)):
        cluster_members[find(index)].append(index)

    snapped_points = points_array.copy()
    for member_indices in cluster_members.values():
        if len(member_indices) < 2:
            continue
        snapped_location = points_array[member_indices].mean(axis=0)
        snapped_points[member_indices] = snapped_location

    for flat_index, (polyline_index, point_index) in enumerate(point_refs):
        polyline_points[polyline_index][point_index] = snapped_points[flat_index]

    return [
        _sanitize_closed_polyline(_close_polyline(points, tolerance=tolerance), tolerance=tolerance)
        for points in polyline_points
        if len(points) >= 3
    ]


def _detect_discontinuity_indices(points: np.ndarray, discontinuity_angle_degrees: float) -> list[int]:
    if len(points) < 3:
        return list(range(len(points)))

    edge_vectors = np.roll(points, -1, axis=0) - points
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    unit_tangents = np.zeros_like(edge_vectors)
    nonzero_mask = edge_lengths > 1e-12
    unit_tangents[nonzero_mask] = edge_vectors[nonzero_mask] / edge_lengths[nonzero_mask, None]

    curvatures = _discrete_curvature_magnitudes(points)
    curvature_gradient = _discrete_curvature_gradient(curvatures)
    nonzero_curvatures = curvatures[curvatures > 1e-12]
    nonzero_gradient = curvature_gradient[curvature_gradient > 1e-12]
    baseline_curvature = float(np.median(nonzero_curvatures)) if len(nonzero_curvatures) else 0.0
    baseline_gradient = float(np.median(nonzero_gradient)) if len(nonzero_gradient) else 0.0
    kink_turn_angle_threshold = max(180.0 - discontinuity_angle_degrees, 0.0)
    curvature_change_ratio = 1.35
    curvature_floor = max(0.08 * baseline_curvature, 0.04 * baseline_gradient, 5e-4)

    turning_angles = np.zeros(len(points), dtype=float)
    degenerate_indices: list[int] = []
    for index in range(len(points)):
        previous_tangent = unit_tangents[index - 1]
        next_tangent = unit_tangents[index]
        if np.linalg.norm(previous_tangent) <= 1e-12 or np.linalg.norm(next_tangent) <= 1e-12:
            degenerate_indices.append(index)
            continue

        tangent_cosine = float(np.clip(np.dot(previous_tangent, next_tangent), -1.0, 1.0))
        turning_angles[index] = float(np.degrees(np.arccos(tangent_cosine)))

    indices: list[int] = [0, *degenerate_indices]
    tangent_active = turning_angles >= kink_turn_angle_threshold
    for run_indices in _cyclic_true_runs(tangent_active):
        indices.extend(_run_boundary_and_peak_indices(run_indices, turning_angles))

    curvature_threshold = max(curvature_floor, curvature_change_ratio * baseline_gradient)
    curvature_active = curvature_gradient >= curvature_threshold
    for run_indices in _cyclic_true_runs(curvature_active):
        indices.extend(_run_boundary_and_peak_indices(run_indices, curvature_gradient))

    return sorted(set(indices))


def _discrete_curvature_magnitudes(points: np.ndarray) -> np.ndarray:
    point_count = len(points)
    if point_count < 3:
        return np.zeros(point_count, dtype=float)

    curvatures = np.zeros(point_count, dtype=float)
    for index in range(point_count):
        previous_point = points[index - 1]
        current_point = points[index]
        next_point = points[(index + 1) % point_count]

        first_edge = current_point - previous_point
        second_edge = next_point - current_point
        chord = next_point - previous_point

        first_length = float(np.linalg.norm(first_edge))
        second_length = float(np.linalg.norm(second_edge))
        chord_length = float(np.linalg.norm(chord))
        if first_length <= 1e-12 or second_length <= 1e-12 or chord_length <= 1e-12:
            continue

        doubled_triangle_area = float(np.linalg.norm(np.cross(first_edge, second_edge)))
        curvatures[index] = 2.0 * doubled_triangle_area / (first_length * second_length * chord_length)

    return curvatures


def _discrete_curvature_gradient(curvatures: np.ndarray) -> np.ndarray:
    if len(curvatures) == 0:
        return np.zeros(0, dtype=float)

    previous_curvatures = np.roll(curvatures, 1)
    next_curvatures = np.roll(curvatures, -1)
    return np.maximum(
        np.abs(curvatures - previous_curvatures),
        np.abs(next_curvatures - curvatures),
    )


def _cyclic_true_runs(mask: np.ndarray) -> list[list[int]]:
    if len(mask) == 0 or not np.any(mask):
        return []
    if np.all(mask):
        return [list(range(len(mask)))]

    false_indices = np.flatnonzero(~mask)
    start_index = int((false_indices[0] + 1) % len(mask))
    ordered_indices = list(range(start_index, len(mask))) + list(range(start_index))

    runs: list[list[int]] = []
    current_run: list[int] = []
    for index in ordered_indices:
        if bool(mask[index]):
            current_run.append(index)
        elif current_run:
            runs.append(current_run)
            current_run = []
    if current_run:
        runs.append(current_run)

    return runs


def _run_boundary_and_peak_indices(run_indices: list[int], signal: np.ndarray) -> list[int]:
    if not run_indices:
        return []
    if len(run_indices) == 1:
        return [run_indices[0]]

    peak_index = max(run_indices, key=lambda index: (float(signal[index]), -index))
    return sorted(set([run_indices[0], peak_index, run_indices[-1]]))


def _cluster_cyclic_indices(indices: list[int], point_count: int, max_gap: int = 0) -> list[int]:
    if not indices:
        return []
    if len(indices) == 1:
        return indices

    sorted_indices = sorted(set(indices))
    groups: list[list[int]] = [[sorted_indices[0]]]
    for index in sorted_indices[1:]:
        if index - groups[-1][-1] <= max_gap:
            groups[-1].append(index)
        else:
            groups.append([index])

    wrap_gap = (sorted_indices[0] + point_count) - sorted_indices[-1]
    if len(groups) > 1 and wrap_gap <= max_gap:
        merged_group = groups[-1] + [value + point_count for value in groups[0]]
        groups = [merged_group] + groups[1:-1]

    representatives: list[int] = []
    for group in groups:
        representative = int(round(sum(group) / len(group))) % point_count
        representatives.append(representative)

    return sorted(set(representatives))


def _fallback_sample_indices(point_count: int, minimum_count: int) -> list[int]:
    sampled = np.linspace(0, point_count - 1, minimum_count, dtype=int)
    return sorted(set(int(index) for index in sampled))


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    plane_origin = points.mean(axis=0)
    centered = points - plane_origin
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    plane_u = vh[0] / np.linalg.norm(vh[0])
    plane_v = vh[1] / np.linalg.norm(vh[1])
    plane_normal = np.cross(plane_u, plane_v)
    plane_normal /= np.linalg.norm(plane_normal)
    return plane_origin, plane_u, plane_v, plane_normal


def _ensure_center_inside_polygon(
    center_3d: np.ndarray,
    polygon_points: np.ndarray,
    plane_origin: np.ndarray,
    plane_u: np.ndarray,
    plane_v: np.ndarray,
) -> np.ndarray:
    poly_2d = _project_to_plane(polygon_points, plane_origin, plane_u, plane_v)
    center_2d = _project_to_plane(center_3d[None, :], plane_origin, plane_u, plane_v)[0]

    if _point_in_polygon_2d(center_2d, poly_2d):
        return center_3d

    fallback_2d = _ear_clip_centroid(poly_2d)
    return plane_origin + fallback_2d[0] * plane_u + fallback_2d[1] * plane_v


def _point_in_polygon_2d(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = float(point[0]), float(point[1])
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(polygon[i, 0]), float(polygon[i, 1])
        xj, yj = float(polygon[j, 0]), float(polygon[j, 1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _ear_clip_centroid(polygon_2d: np.ndarray) -> np.ndarray:
    pts = list(polygon_2d)
    if len(pts) < 3:
        return polygon_2d.mean(axis=0)

    best_centroid = polygon_2d.mean(axis=0)
    indices = list(range(len(pts)))

    for _ in range(len(pts)):
        if len(indices) < 3:
            break
        for i in range(len(indices)):
            prev_i = indices[(i - 1) % len(indices)]
            curr_i = indices[i]
            next_i = indices[(i + 1) % len(indices)]

            a = polygon_2d[prev_i]
            b = polygon_2d[curr_i]
            c = polygon_2d[next_i]

            cross = float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
            if cross <= 0:
                continue

            centroid = (a + b + c) / 3.0
            if _point_in_polygon_2d(centroid, polygon_2d):
                return centroid

            indices.pop(i)
            best_centroid = centroid
            break
        else:
            break

    return best_centroid


def _fit_circle_on_plane(
    points: np.ndarray,
    plane_origin: np.ndarray,
    plane_u: np.ndarray,
    plane_v: np.ndarray,
) -> tuple[np.ndarray, float]:
    local_2d = _project_to_plane(points, plane_origin, plane_u, plane_v)
    x_coords = local_2d[:, 0]
    y_coords = local_2d[:, 1]
    system = np.column_stack([x_coords, y_coords, np.ones(len(local_2d))])
    rhs = -(x_coords ** 2 + y_coords ** 2)
    coefficients, *_ = np.linalg.lstsq(system, rhs, rcond=None)
    a_coef, b_coef, c_coef = coefficients
    center_2d = np.array([-0.5 * a_coef, -0.5 * b_coef], dtype=float)
    radius = float(np.sqrt(max(center_2d.dot(center_2d) - c_coef, 0.0)))
    center_3d = plane_origin + center_2d[0] * plane_u + center_2d[1] * plane_v
    return center_3d, radius


def _project_to_plane(
    points: np.ndarray,
    plane_origin: np.ndarray,
    plane_u: np.ndarray,
    plane_v: np.ndarray,
) -> np.ndarray:
    centered = points - plane_origin
    return np.column_stack(
        [
            centered @ plane_u,
            centered @ plane_v,
        ]
    )


def _build_plane_aligned_bounding_box(
    points: np.ndarray,
    plane_origin: np.ndarray,
    plane_u: np.ndarray,
    plane_v: np.ndarray,
    plane_normal: np.ndarray,
    tolerance: float,
) -> tuple[pv.PolyData, np.ndarray, float, tuple[float, float, float]]:
    centered = points - plane_origin
    local_coordinates = np.column_stack(
        [
            centered @ plane_u,
            centered @ plane_v,
            centered @ plane_normal,
        ]
    )
    mins = local_coordinates.min(axis=0)
    maxs = local_coordinates.max(axis=0)
    if maxs[2] - mins[2] < tolerance:
        mins[2] -= 0.5 * tolerance
        maxs[2] += 0.5 * tolerance

    corners_local = np.array(
        [
            [mins[0], mins[1], mins[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], maxs[1], mins[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], maxs[2]],
            [mins[0], maxs[1], maxs[2]],
        ],
        dtype=float,
    )
    transform = np.column_stack([plane_u, plane_v, plane_normal])
    corners_world = plane_origin + corners_local @ transform.T
    faces = np.array(
        [
            4, 0, 1, 2, 3,
            4, 4, 5, 6, 7,
            4, 0, 1, 5, 4,
            4, 1, 2, 6, 5,
            4, 2, 3, 7, 6,
            4, 3, 0, 4, 7,
        ],
        dtype=np.int64,
    )
    bbox_mesh = pv.PolyData(corners_world, faces=faces).triangulate().clean()
    bbox_center_local = 0.5 * (mins + maxs)
    bbox_center = plane_origin + bbox_center_local @ transform.T
    bbox_volume = float(np.prod(maxs - mins))
    extents = (float(maxs[0] - mins[0]), float(maxs[1] - mins[1]), float(maxs[2] - mins[2]))
    return bbox_mesh, bbox_center, bbox_volume, extents


def _polyline_length(polyline: np.ndarray) -> float:
    return float(np.linalg.norm(polyline[1:] - polyline[:-1], axis=1).sum())


def _close_polyline(points: np.ndarray, tolerance: float) -> np.ndarray:
    if len(points) == 0:
        return points
    if np.allclose(points[0], points[-1], atol=tolerance):
        return points
    return np.vstack([points, points[0]])


def _splice_points_into_polyline(
    unique_pts: np.ndarray,
    insertions: list[tuple[int, np.ndarray]],
    tolerance: float,
    *,
    dedup_against_existing: bool = False,
) -> np.ndarray:
    """Insert points into a closed polyline, ordered by parametric t along each segment.

    *insertions* is a list of ``(segment_index, point)`` pairs.  Points
    targeting the same segment are sorted by their projection parameter
    along that segment so they appear in the correct order.

    When *dedup_against_existing* is True, points that are within
    *tolerance* of an existing vertex or an already-queued insertion for
    the same segment are silently dropped.

    Returns a new unique-point array (NOT closed) ready for
    ``_close_polyline``.
    """
    n = len(unique_pts)
    if not insertions or n == 0:
        return unique_pts.copy()

    ins_by_seg: dict[int, list[np.ndarray]] = {}
    for seg_idx, pt in insertions:
        seg_idx = min(seg_idx, n - 1)
        if seg_idx not in ins_by_seg:
            ins_by_seg[seg_idx] = []
        if dedup_against_existing:
            already = any(float(np.linalg.norm(pt - ex)) < tolerance for ex in ins_by_seg[seg_idx])
            near_existing = any(float(np.linalg.norm(pt - v)) < tolerance for v in unique_pts)
            if already or near_existing:
                continue
        ins_by_seg[seg_idx].append(pt)

    if not ins_by_seg:
        return unique_pts.copy()

    new_pts: list[np.ndarray] = []
    for vi in range(n):
        new_pts.append(unique_pts[vi])
        if vi in ins_by_seg:
            seg_start = unique_pts[vi]
            seg_end = unique_pts[(vi + 1) % n]
            seg_dir = seg_end - seg_start
            seg_len_sq = max(float(np.dot(seg_dir, seg_dir)), 1e-20)
            pts_t = [
                (float(np.dot(pt - seg_start, seg_dir)) / seg_len_sq, pt)
                for pt in ins_by_seg[vi]
            ]
            pts_t.sort(key=lambda x: x[0])
            for _, pt in pts_t:
                new_pts.append(pt)

    return np.array(new_pts, dtype=float)


def _sanitize_closed_polyline(polyline: np.ndarray, tolerance: float) -> np.ndarray:
    unique_points = _unique_polyline_points(polyline, tolerance=tolerance)
    if len(unique_points) == 0:
        return np.zeros((0, 3), dtype=float)

    cleaned_points = [unique_points[0]]
    for point in unique_points[1:]:
        if float(np.linalg.norm(point - cleaned_points[-1])) > tolerance:
            cleaned_points.append(point)

    cleaned_array = np.array(cleaned_points, dtype=float)
    if len(cleaned_array) > 1 and float(np.linalg.norm(cleaned_array[0] - cleaned_array[-1])) <= tolerance:
        cleaned_array = cleaned_array[:-1]

    if len(cleaned_array) == 0:
        return np.zeros((0, 3), dtype=float)
    return np.vstack([cleaned_array, cleaned_array[0]])


def _scale_and_offset_polyline(
    polyline: np.ndarray,
    center: np.ndarray,
    plane_u: np.ndarray,
    plane_v: np.ndarray,
    scale_factor: float,
    offset_vector: np.ndarray,
) -> np.ndarray:
    unique_points = polyline[:-1]
    centered = unique_points - center
    u_coords = centered @ plane_u
    v_coords = centered @ plane_v
    scaled_points = center + scale_factor * u_coords[:, None] * plane_u + scale_factor * v_coords[:, None] * plane_v
    moved_points = scaled_points + offset_vector
    return np.vstack([moved_points, moved_points[0]])


def _build_staged_offset_lofts(
    polyline: np.ndarray,
    center: np.ndarray,
    plane_u: np.ndarray,
    plane_v: np.ndarray,
    offset_vector: np.ndarray,
    primary_scale_factor: float = 0.5,
    secondary_scale_ratio: float = 0.9,
) -> tuple[pv.PolyData, np.ndarray, np.ndarray]:
    first_scaled_polyline = _scale_and_offset_polyline(
        polyline,
        center=center,
        plane_u=plane_u,
        plane_v=plane_v,
        scale_factor=primary_scale_factor,
        offset_vector=offset_vector,
    )
    second_scaled_polyline = _scale_and_offset_polyline(
        polyline,
        center=center,
        plane_u=plane_u,
        plane_v=plane_v,
        scale_factor=primary_scale_factor * secondary_scale_ratio,
        offset_vector=offset_vector,
    )
    staged_mesh = _merge_meshes(
        [
            _loft_between_polylines(polyline, first_scaled_polyline),
            _loft_between_polylines(first_scaled_polyline, second_scaled_polyline),
        ]
    )
    return staged_mesh, first_scaled_polyline, second_scaled_polyline


def _project_polyline_to_plane(
    polyline: np.ndarray,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray:
    """Project each point of a polyline onto a plane, flattening any 3D undulation."""
    vecs = polyline - plane_origin
    dists = (vecs @ plane_normal)[:, None]
    return polyline - dists * plane_normal


def _build_extreme_cell_lofts(
    polyline: np.ndarray,
    center: np.ndarray,
    plane_origin: np.ndarray,
    plane_u: np.ndarray,
    plane_v: np.ndarray,
    plane_normal: np.ndarray,
    offset_vector: np.ndarray,
    primary_scale_factor: float = EXTREME_SCALE_FACTOR,
    secondary_scale_ratio: float = 0.9,
) -> tuple[pv.PolyData, np.ndarray, np.ndarray]:
    """Variant of _build_staged_offset_lofts for extreme-aspect-ratio cells.

    The original polyline is kept as-is (3D surface intersection); only the
    inner scaled copies are projected onto the fitted plane to flatten them,
    and a more aggressive scale factor is used for thicker walls.
    """
    first = _scale_and_offset_polyline(
        polyline, center, plane_u, plane_v, primary_scale_factor, offset_vector,
    )
    first = _project_polyline_to_plane(first, plane_origin, plane_normal)
    second = _scale_and_offset_polyline(
        polyline, center, plane_u, plane_v,
        primary_scale_factor * secondary_scale_ratio, offset_vector,
    )
    second = _project_polyline_to_plane(second, plane_origin, plane_normal)
    staged_mesh = _merge_meshes([
        _loft_between_polylines(polyline, first),
        _loft_between_polylines(first, second),
    ])
    return staged_mesh, first, second


def _small_mesh_exceeds_retained_volume(
    mesh: pv.PolyData,
    loft_bounds: tuple[float, float, float, float, float, float],
    tolerance: float,
    slice_plane_x: float | None = None,
) -> bool:
    if mesh.n_points == 0:
        return False

    xmin, _, _, _, zmin, zmax = mesh.bounds
    loft_zmin = loft_bounds[4]
    loft_zmax = loft_bounds[5]

    crosses_slice_plane = slice_plane_x is not None and xmin < slice_plane_x - tolerance
    crosses_bottom_cap = zmin < loft_zmin - tolerance
    crosses_top_cap = zmax > loft_zmax + tolerance
    return bool(crosses_slice_plane or crosses_bottom_cap or crosses_top_cap)


def _loft_between_polylines(first: np.ndarray, second: np.ndarray) -> pv.PolyData:
    first_unique = first[:-1]
    second_unique = second[:-1]
    if len(first_unique) != len(second_unique):
        raise ValueError("Lofted polylines must have the same number of vertices.")

    points = np.vstack([first_unique, second_unique])
    point_count = len(first_unique)
    faces: list[int] = []
    for index in range(point_count):
        next_index = (index + 1) % point_count
        a = index
        b = next_index
        c = point_count + next_index
        d = point_count + index
        faces.extend([3, a, b, c, 3, a, c, d])

    return pv.PolyData(points, faces=np.array(faces, dtype=np.int64)).clean()


def _fan_surface_from_center(center: np.ndarray, polygon_points: np.ndarray) -> pv.PolyData:
    unique_points = polygon_points
    if len(unique_points) < 3:
        return pv.PolyData()

    points = np.vstack([center[None, :], unique_points])
    faces: list[int] = []
    for index in range(len(unique_points)):
        next_index = (index + 1) % len(unique_points)
        faces.extend([3, 0, index + 1, next_index + 1])

    return pv.PolyData(points, faces=np.array(faces, dtype=np.int64)).clean()


def orient_normals_outward(mesh: pv.PolyData) -> pv.PolyData:
    if mesh.n_cells < 2:
        return mesh

    points = np.asarray(mesh.points, dtype=float)
    faces_raw = np.asarray(mesh.faces, dtype=int)
    face_verts: list[list[int]] = []
    cursor = 0
    while cursor < len(faces_raw):
        n = int(faces_raw[cursor])
        if n == 3:
            face_verts.append([int(faces_raw[cursor + 1]), int(faces_raw[cursor + 2]), int(faces_raw[cursor + 3])])
        cursor += n + 1

    if not face_verts:
        return mesh

    center = points.mean(axis=0)
    changed = False
    for fi, (a, b, c) in enumerate(face_verts):
        face_center = (points[a] + points[b] + points[c]) / 3.0
        normal = np.cross(points[b] - points[a], points[c] - points[a])
        if float(np.dot(normal, face_center - center)) < 0:
            face_verts[fi] = [a, c, b]
            changed = True

    if not changed:
        return mesh

    new_faces: list[int] = []
    for a, b, c in face_verts:
        new_faces.extend([3, a, b, c])
    return pv.PolyData(points.copy(), faces=np.array(new_faces, dtype=np.int64))


def _fix_mesh_winding(mesh: pv.PolyData) -> pv.PolyData:
    if mesh.n_cells < 2:
        return mesh

    points = np.asarray(mesh.points, dtype=float)
    faces_raw = np.asarray(mesh.faces, dtype=int)
    face_verts: list[list[int]] = []
    cursor = 0
    while cursor < len(faces_raw):
        n = int(faces_raw[cursor])
        if n == 3:
            face_verts.append([int(faces_raw[cursor + 1]), int(faces_raw[cursor + 2]), int(faces_raw[cursor + 3])])
        cursor += n + 1

    if not face_verts:
        return mesh

    normals: list[np.ndarray] = []
    for a, b, c in face_verts:
        e1 = points[b] - points[a]
        e2 = points[c] - points[a]
        n = np.cross(e1, e2)
        length = float(np.linalg.norm(n))
        normals.append(n / length if length > 1e-12 else np.zeros(3))

    avg_normal = np.mean(normals, axis=0)
    avg_len = float(np.linalg.norm(avg_normal))
    if avg_len < 1e-12:
        return mesh
    avg_normal /= avg_len

    changed = False
    for fi in range(len(face_verts)):
        if float(np.dot(normals[fi], avg_normal)) < 0:
            face_verts[fi][1], face_verts[fi][2] = face_verts[fi][2], face_verts[fi][1]
            changed = True

    if not changed:
        return mesh

    new_faces: list[int] = []
    for a, b, c in face_verts:
        new_faces.extend([3, a, b, c])
    return pv.PolyData(points.copy(), faces=np.array(new_faces, dtype=np.int64))


def _sort_polygon_by_angle(center: np.ndarray, polygon_points: np.ndarray) -> np.ndarray:
    if len(polygon_points) < 3:
        return polygon_points

    _, plane_u, plane_v, _ = _fit_plane(polygon_points)
    centered = polygon_points - center
    u_coords = centered @ plane_u
    v_coords = centered @ plane_v
    angles = np.arctan2(v_coords, u_coords)
    return polygon_points[np.argsort(angles)]



def _strip_stale_cell_arrays(mesh: pv.PolyData) -> pv.PolyData:
    """Remove cell-data arrays whose length doesn't match n_cells."""
    for name in list(mesh.cell_data.keys()):
        if len(mesh.cell_data[name]) != mesh.n_cells:
            del mesh.cell_data[name]
    return mesh


def _merge_meshes(meshes: list[pv.PolyData]) -> pv.PolyData:
    non_empty = [_strip_stale_cell_arrays(mesh) for mesh in meshes if mesh.n_cells > 0]
    if not non_empty:
        return pv.PolyData()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*PolyData.*not valid.*vtkOriginalCellIds.*")
        merged = non_empty[0].copy()
        for mesh in non_empty[1:]:
            merged = merged.merge(mesh)
    return _strip_stale_cell_arrays(merged).clean()


def _bounds_overlap(
    first: tuple[float, float, float, float, float, float],
    second: tuple[float, float, float, float, float, float],
) -> bool:
    return not (
        first[1] < second[0]
        or second[1] < first[0]
        or first[3] < second[2]
        or second[3] < first[2]
        or first[5] < second[4]
        or second[5] < first[4]
    )


def default_snap_tolerance(tolerance: float) -> float:
    """Canonical snap tolerance derived from the base line tolerance."""
    return max(20.0 * tolerance, 0.02)
