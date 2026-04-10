"""compass-web: 3D-printable voronoi shell geometry generator."""

from compass_web.config import (
    PipelineConfig,
    load_pipeline_config,
    load_pipeline_config_from_saved,
    save_pipeline_config,
    list_saved_configs,
    validate_geometry_limits,
)
from compass_web.lofted_surface_voronoi import (
    align_neighbouring_polylines,
    compact_polyline_shapes,
    default_snap_tolerance,
    find_polyline_neighbours,
    fix_polyline_surface_overlaps,
    point_distance_to_mesh_surface,
    validate_polyline_surfaces,
)
from compass_web.pipeline import (
    PipelineResult,
    build_export_trimesh,
    export_stl,
    filter_isolated_polylines,
    run_pipeline,
    run_pipeline_with_retry,
)
from compass_web.smoothing import (
    SmoothingResult,
    smooth_radii_and_spacing,
    apply_smoothing_to_config,
)

__all__ = [
    "PipelineConfig",
    "PipelineResult",
    "SmoothingResult",
    "align_neighbouring_polylines",
    "apply_smoothing_to_config",
    "build_export_trimesh",
    "export_stl",
    "filter_isolated_polylines",
    "find_polyline_neighbours",
    "fix_polyline_surface_overlaps",
    "list_saved_configs",
    "point_distance_to_mesh_surface",
    "load_pipeline_config",
    "load_pipeline_config_from_saved",
    "run_pipeline",
    "run_pipeline_with_retry",
    "save_pipeline_config",
    "smooth_radii_and_spacing",
    "default_snap_tolerance",
    "validate_geometry_limits",
    "validate_polyline_surfaces",
]
