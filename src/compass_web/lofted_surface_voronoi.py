from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path

import numpy as np
import pyvista as pv
import vtk
from scipy.spatial import ConvexHull, HalfspaceIntersection

MIN_RADIUS = 5.0
MAX_RADIUS = 75.0
MAX_MODEL_SPAN = 150.0


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
    for cell in cells:
        if not _bounds_overlap(triangulated_surface.bounds, cell.bounds):
            continue
        intersection, _, _ = triangulated_surface.intersection(
            cell.triangulate().clean(),
            split_first=False,
            split_second=False,
        )
        if intersection.n_points == 0 or intersection.n_lines == 0:
            continue

        for polyline in _extract_polylines(intersection, tolerance=tolerance):
            polyline_key = _canonical_polyline_key(polyline, tolerance=tolerance)
            if polyline_key in seen_keys:
                continue
            seen_keys.add(polyline_key)
            unique_polylines.append(polyline)

    return unique_polylines


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
    return mesh.extract_surface(algorithm="dataset_surface").triangulate().clean()


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
        if analysis.ratio >= average_ratio:
            offset_vector = extrusion_multiplier * analysis.extrusion_base_vector
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


def rebuild_polylines_from_discontinuities(
    polylines: list[np.ndarray],
    tolerance: float,
    discontinuity_angle_degrees: float = 176.0,
    neighbor_snap_tolerance: float | None = None,
) -> list[np.ndarray]:
    rebuilt_polylines: list[np.ndarray] = []
    for polyline in polylines:
        unique_points = _unique_polyline_points(polyline, tolerance=tolerance)
        if len(unique_points) < 3:
            continue
        straight_polyline, _ = _build_straight_polyline_from_discontinuities(
            unique_points,
            tolerance=tolerance,
            discontinuity_angle_degrees=discontinuity_angle_degrees,
        )
        rebuilt_polyline = _sanitize_closed_polyline(straight_polyline, tolerance=tolerance)
        if len(_unique_polyline_points(rebuilt_polyline, tolerance=tolerance)) < 3:
            continue
        rebuilt_polylines.append(rebuilt_polyline)

    if not rebuilt_polylines:
        return []

    snap_tolerance = neighbor_snap_tolerance if neighbor_snap_tolerance is not None else max(20.0 * tolerance, 0.02)
    return _snap_neighboring_polyline_points(
        rebuilt_polylines,
        tolerance=tolerance,
        snap_tolerance=snap_tolerance,
    )


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
        bbox_mesh, bbox_center, bbox_volume = _build_plane_aligned_bounding_box(
            unique_points,
            plane_origin,
            plane_u,
            plane_v,
            plane_normal,
            tolerance=tolerance,
        )
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
        if analysis.ratio >= average_ratio:
            offset_vector = extrusion_multiplier * analysis.extrusion_base_vector
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
) -> tuple[np.ndarray, np.ndarray]:
    discontinuity_indices = _detect_discontinuity_indices(points, discontinuity_angle_degrees)
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

    for first_index, first_point in enumerate(points_array[:-1]):
        first_polyline_index, _ = point_refs[first_index]
        for second_index in range(first_index + 1, len(points_array)):
            second_polyline_index, _ = point_refs[second_index]
            if first_polyline_index == second_polyline_index:
                continue
            if float(np.linalg.norm(first_point - points_array[second_index])) <= snap_tolerance:
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
) -> tuple[pv.PolyData, np.ndarray, float]:
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
    return bbox_mesh, bbox_center, bbox_volume


def _polyline_length(polyline: np.ndarray) -> float:
    return float(np.linalg.norm(polyline[1:] - polyline[:-1], axis=1).sum())


def _close_polyline(points: np.ndarray, tolerance: float) -> np.ndarray:
    if len(points) == 0:
        return points
    if np.allclose(points[0], points[-1], atol=tolerance):
        return points
    return np.vstack([points, points[0]])


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

    return pv.PolyData(points, faces=np.array(faces, dtype=np.int64)).triangulate().clean()


def _fan_surface_from_center(center: np.ndarray, polygon_points: np.ndarray) -> pv.PolyData:
    unique_points = polygon_points
    if len(unique_points) < 3:
        return pv.PolyData()

    points = np.vstack([center[None, :], unique_points])
    faces: list[int] = []
    for index in range(len(unique_points)):
        next_index = (index + 1) % len(unique_points)
        faces.extend([3, 0, index + 1, next_index + 1])

    return pv.PolyData(points, faces=np.array(faces, dtype=np.int64)).triangulate().clean()


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



def _merge_meshes(meshes: list[pv.PolyData]) -> pv.PolyData:
    non_empty = [mesh for mesh in meshes if mesh.n_cells > 0]
    if not non_empty:
        return pv.PolyData()

    merged = non_empty[0].copy()
    for mesh in non_empty[1:]:
        merged = merged.merge(mesh)
    return merged.clean().triangulate()


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
