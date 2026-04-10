"""Pipeline configuration: dataclass, JSON I/O, and config management."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

from compass_web.lofted_surface_voronoi import (
    LoftedVoronoiConfig,
    VoronoiPointConfig,
    load_generation_config,
    load_voronoi_point_config,
)

MAX_MODEL_SPAN = 150.0
MIN_RADIUS = 5.0
MAX_RADIUS = 70.0
MAX_Z_INCREMENT = MAX_MODEL_SPAN / 7.0
SMALL_CELL_EXTRUSION_FACTOR = 0.1


@dataclass(frozen=True)
class PipelineConfig:
    """Unified config holding all parameters needed for a single pipeline run."""

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
    # When set (e.g. after radii smoothing), non-uniform Z ring positions; else derived from z_increment.
    z_levels: tuple[float, ...] | None = None

    @property
    def effective_extrusion(self) -> float:
        return 5.0 * self.extrusion_multiplier

    def to_surface_config(
        self,
        z_levels_override: tuple[float, ...] | None = None,
    ) -> LoftedVoronoiConfig:
        z_levels = (
            z_levels_override
            or self.z_levels
            or tuple(i * self.z_increment for i in range(len(self.radii)))
        )
        return LoftedVoronoiConfig(
            radii=self.radii,
            z_levels=z_levels,
            z_increment=self.z_increment,
            circle_resolution=self.circle_resolution,
            slice_normal=(1.0, 0.0, 0.0),
            slice_origin=(0.0, 0.0, 0.0),
            bbox_padding=self.bbox_padding,
            line_tolerance=self.line_tolerance,
        )

    def to_point_config(self) -> VoronoiPointConfig:
        return VoronoiPointConfig(
            seed_count=self.seed_count,
            random_seed=self.random_seed,
        )

    def with_seed(self, new_seed: int) -> PipelineConfig:
        return replace(self, random_seed=new_seed)

    def to_dict(self) -> dict:
        d = {
            "radii": list(self.radii),
            "z_increment": self.z_increment,
            "seed_count": self.seed_count,
            "random_seed": self.random_seed,
            "extrusion_multiplier": self.extrusion_multiplier,
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "circle_resolution": self.circle_resolution,
            "bbox_padding": self.bbox_padding,
            "line_tolerance": self.line_tolerance,
        }
        if self.z_levels is not None:
            d["z_levels"] = list(self.z_levels)
        return d


def validate_geometry_limits(
    radii: tuple[float, ...],
    z_increment: float,
    *,
    z_levels: tuple[float, ...] | None = None,
) -> tuple[float, float]:
    max_width = 2.0 * max(radii)
    if z_levels is not None and len(z_levels) >= 2:
        max_height = float(max(z_levels) - min(z_levels))
    else:
        max_height = z_increment * (len(radii) - 1)
    if max_width > MAX_MODEL_SPAN + 1e-9:
        raise ValueError(
            f"The widest circle would produce {max_width:.2f} units in width, "
            f"which is above the {MAX_MODEL_SPAN:.0f} unit limit."
        )
    if max_height > MAX_MODEL_SPAN + 1e-9:
        raise ValueError(
            f"The stacked circles would span {max_height:.2f} units in Z, "
            f"which is above the {MAX_MODEL_SPAN:.0f} unit limit."
        )
    return max_width, max_height


def load_pipeline_config(
    surface_path: str | Path,
    point_path: str | Path,
    *,
    extrusion_multiplier: float = -0.2,
    scale_x: float = 0.5,
    scale_y: float = 0.5,
) -> PipelineConfig:
    """Build a PipelineConfig from the two standard JSON input files."""
    sc = load_generation_config(surface_path)
    pc = load_voronoi_point_config(point_path)
    return PipelineConfig(
        radii=sc.radii,
        z_increment=sc.z_increment,
        seed_count=pc.seed_count,
        random_seed=pc.random_seed,
        extrusion_multiplier=extrusion_multiplier,
        scale_x=scale_x,
        scale_y=scale_y,
        circle_resolution=sc.circle_resolution,
        bbox_padding=sc.bbox_padding,
        line_tolerance=sc.line_tolerance,
    )


def load_pipeline_config_from_saved(path: str | Path) -> PipelineConfig:
    """Load a PipelineConfig from a previously-saved config JSON file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    z_levels_raw = raw.get("z_levels")
    z_levels: tuple[float, ...] | None
    if z_levels_raw is not None:
        z_levels = tuple(float(v) for v in z_levels_raw)
    else:
        z_levels = None
    return PipelineConfig(
        radii=tuple(float(v) for v in raw["radii"]),
        z_increment=float(raw["z_increment"]),
        seed_count=int(raw["seed_count"]),
        random_seed=int(raw["random_seed"]),
        extrusion_multiplier=float(raw["extrusion_multiplier"]),
        scale_x=float(raw["scale_x"]),
        scale_y=float(raw["scale_y"]),
        circle_resolution=int(raw.get("circle_resolution", 120)),
        bbox_padding=float(raw.get("bbox_padding", 4.0)),
        line_tolerance=float(raw.get("line_tolerance", 0.001)),
        z_levels=z_levels,
    )


def save_pipeline_config(
    config: PipelineConfig,
    configs_dir: str | Path,
    *,
    allow_duplicates: bool = False,
) -> Path | None:
    """Save config JSON to configs_dir with a timestamp name.

    Returns the path written, or None if a duplicate already exists (and
    allow_duplicates is False).
    """
    configs_dir = Path(configs_dir)
    configs_dir.mkdir(parents=True, exist_ok=True)
    cfg_data = config.to_dict()

    if not allow_duplicates:
        for existing_path in sorted(configs_dir.glob("*.json"), reverse=True):
            existing_data = json.loads(existing_path.read_text(encoding="utf-8"))
            if existing_data == cfg_data:
                return None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = configs_dir / f"{ts}.json"
    path.write_text(json.dumps(cfg_data, indent=2), encoding="utf-8")
    return path


def list_saved_configs(configs_dir: str | Path) -> list[str]:
    """Return sorted list of saved config stems (newest first)."""
    return sorted(
        [f.stem for f in Path(configs_dir).glob("*.json")],
        reverse=True,
    )
