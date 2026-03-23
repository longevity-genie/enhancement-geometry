from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pyvista as pv
import vtk
from scipy.spatial import ConvexHull, HalfspaceIntersection

MIN_RADIUS = 5.0
MAX_RADIUS = 70.0
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
    offset_direction: np.ndarray


@dataclass(frozen=True)
class SurfaceGenerationResult:
    analyses: tuple[CurveAnalysis, ...]
    average_ratio: float
    followup_polylines: tuple[np.ndarray, ...]
    generated_surface: pv.PolyData
    larger_surface: pv.PolyData
    smaller_surface: pv.PolyData


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


def make_bounding_box(bounds: tuple[float, float, float, float, float, float]) -> pv.PolyData:
    return pv.Box(bounds=bounds)


def analyze_and_generate_surfaces(
    polylines: list[np.ndarray],
    loft_bounds: tuple[float, float, float, float, float, float],
    tolerance: float,
    discontinuity_angle_degrees: float = 176.0,
    extrusion_multiplier: float = 0.5,
) -> SurfaceGenerationResult:
    if extrusion_multiplier < 0.0:
        raise ValueError("extrusion_multiplier must be non-negative.")

    loft_bbox_center = np.array(
        [
            0.5 * (loft_bounds[0] + loft_bounds[1]),
            0.5 * (loft_bounds[2] + loft_bounds[3]),
            0.5 * (loft_bounds[4] + loft_bounds[5]),
        ],
        dtype=float,
    )

    analyses: list[CurveAnalysis] = []
    for polyline in polylines:
        unique_points = _unique_polyline_points(polyline, tolerance=tolerance)
        if len(unique_points) < 3:
            continue

        discontinuity_indices = _detect_discontinuity_indices(unique_points, discontinuity_angle_degrees)
        clustered_indices = _cluster_cyclic_indices(discontinuity_indices, len(unique_points), max_gap=0)
        if len(clustered_indices) < 3:
            clustered_indices = _fallback_sample_indices(len(unique_points), minimum_count=3)

        discontinuity_points = unique_points[clustered_indices]
        followup_polyline = np.vstack([unique_points, unique_points[0]])

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

        direction_vector = circle_center - loft_bbox_center
        direction_length = float(np.linalg.norm(direction_vector))
        if direction_length <= tolerance:
            offset_direction = plane_normal
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
        ratio_factor = analysis.ratio / average_ratio if average_ratio > tolerance else 1.0
        offset_distance = extrusion_multiplier * ratio_factor
        offset_vector = analysis.offset_direction * offset_distance
        if analysis.ratio >= average_ratio:
            moved_scaled_polyline = _scale_and_offset_polyline(
                analysis.followup_polyline,
                center=analysis.circle_center,
                plane_u=analysis.plane_u,
                plane_v=analysis.plane_v,
                scale_factor=0.5,
                offset_vector=offset_vector,
            )
            larger_meshes.append(_loft_between_polylines(analysis.followup_polyline, moved_scaled_polyline))
        else:
            moved_center = analysis.circle_center + offset_vector
            smaller_meshes.append(_fan_surface_from_center(moved_center, analysis.discontinuity_points))

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
