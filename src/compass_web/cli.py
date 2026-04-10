"""CLI for the compass-web voronoi shell generator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(
    name="materialize",
    help="Generate 3D-printable voronoi shell geometry from parametric inputs.",
    add_completion=False,
)


@app.command()
def generate(
    config: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to a saved pipeline config JSON file."),
    ] = None,
    surface: Annotated[
        Path,
        typer.Option("--surface", "-s", help="Path to lofted_surface_inputs.json."),
    ] = Path("data/lofted_surface_inputs.json"),
    points: Annotated[
        Path,
        typer.Option("--points", "-p", help="Path to voronoi_points_inputs.json."),
    ] = Path("data/voronoi_points_inputs.json"),
    extrusion: Annotated[
        float,
        typer.Option("--extrusion", "-e", help="Extrusion multiplier (-3.0 to 3.0)."),
    ] = -0.2,
    scale_x: Annotated[
        float,
        typer.Option("--scale-x", help="XY scale factor for X (0.1–1.5)."),
    ] = 0.5,
    scale_y: Annotated[
        float,
        typer.Option("--scale-y", help="XY scale factor for Y (0.1–1.5)."),
    ] = 0.5,
    seed: Annotated[
        Optional[int],
        typer.Option("--seed", help="Override the random seed from JSON."),
    ] = None,
    seed_count: Annotated[
        Optional[int],
        typer.Option("--seed-count", help="Override the voronoi seed point count from JSON."),
    ] = None,
    retry: Annotated[
        int,
        typer.Option("--retry", "-r", help="Max auto-retry attempts with different seeds."),
    ] = 10,
    export_dir: Annotated[
        Path,
        typer.Option("--export-dir", "-o", help="Directory for exported STL files."),
    ] = Path("exports"),
    configs_dir: Annotated[
        Path,
        typer.Option("--configs-dir", help="Directory for saving config snapshots."),
    ] = Path("configs"),
    save_config: Annotated[
        bool,
        typer.Option("--save-config/--no-save-config", help="Save the config used for this run."),
    ] = True,
    viewer: Annotated[
        bool,
        typer.Option("--viewer/--no-viewer", help="Open interactive 3D viewer after generation."),
    ] = True,
    screenshot: Annotated[
        Optional[Path],
        typer.Option("--screenshot", help="Save a PNG screenshot to this path."),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress progress output."),
    ] = False,
    apply_smoothing: Annotated[
        bool,
        typer.Option(
            "--smoothing/--no-smoothing",
            help="Apply radii/spacing smoothing before loft.",
        ),
    ] = True,
) -> None:
    """Run the full voronoi shell pipeline and export to STL."""
    from compass_web.config import (
        PipelineConfig,
        load_pipeline_config,
        load_pipeline_config_from_saved,
        save_pipeline_config,
        validate_geometry_limits,
    )
    from compass_web.pipeline import export_stl, run_pipeline_with_retry

    if config is not None:
        if not config.exists():
            typer.echo(f"Config file not found: {config}", err=True)
            raise typer.Exit(1)
        pipeline_config = load_pipeline_config_from_saved(config)
        if not quiet:
            typer.echo(f"Loaded config from {config}")
    else:
        if not surface.exists():
            typer.echo(f"Surface inputs file not found: {surface}", err=True)
            raise typer.Exit(1)
        if not points.exists():
            typer.echo(f"Point inputs file not found: {points}", err=True)
            raise typer.Exit(1)
        pipeline_config = load_pipeline_config(
            surface, points,
            extrusion_multiplier=extrusion,
            scale_x=scale_x,
            scale_y=scale_y,
        )

    if seed is not None:
        pipeline_config = pipeline_config.with_seed(seed)
    if seed_count is not None:
        from dataclasses import replace
        pipeline_config = replace(pipeline_config, seed_count=seed_count)

    validate_geometry_limits(
        pipeline_config.radii,
        pipeline_config.z_increment,
        z_levels=pipeline_config.z_levels,
    )

    if not quiet:
        typer.echo("Running voronoi shell pipeline...")

    result, used_config = run_pipeline_with_retry(
        pipeline_config,
        max_attempts=retry,
        verbose=not quiet,
        apply_smoothing=apply_smoothing,
    )

    if not result.is_valid_volume:
        typer.echo("WARNING: Final mesh is not a valid volume.", err=True)

    stl_path = export_stl(result, export_dir)
    typer.echo(f"STL exported: {stl_path} ({stl_path.stat().st_size / 1024:.1f} KB)")

    if save_config:
        saved = save_pipeline_config(used_config, configs_dir)
        if saved is not None:
            typer.echo(f"Config saved: {saved}")
        elif not quiet:
            typer.echo("Config identical to existing, not saved.")

    if screenshot is not None:
        from compass_web.visualization import save_screenshot
        save_screenshot(result.generated_surface, screenshot)
        typer.echo(f"Screenshot saved: {screenshot}")

    if viewer:
        from compass_web.visualization import show_mesh_interactive
        typer.echo("Opening interactive viewer (close window to continue)...")
        show_mesh_interactive(result.generated_surface, title="Voronoi Shell — CLI")


@app.command(name="show-config")
def show_config(
    path: Annotated[
        Path,
        typer.Argument(help="Path to a config JSON file to display."),
    ],
) -> None:
    """Display the contents of a saved config file."""
    if not path.exists():
        typer.echo(f"File not found: {path}", err=True)
        raise typer.Exit(1)
    data = json.loads(path.read_text(encoding="utf-8"))
    typer.echo(json.dumps(data, indent=2))


@app.command(name="list-configs")
def list_configs(
    configs_dir: Annotated[
        Path,
        typer.Option("--configs-dir", help="Directory containing config files."),
    ] = Path("configs"),
) -> None:
    """List all saved config snapshots."""
    from compass_web.config import list_saved_configs

    if not configs_dir.exists():
        typer.echo(f"Configs directory not found: {configs_dir}", err=True)
        raise typer.Exit(1)

    names = list_saved_configs(configs_dir)
    if not names:
        typer.echo("No saved configs found.")
        return

    typer.echo(f"Found {len(names)} saved config(s) in {configs_dir}/:")
    for name in names:
        typer.echo(f"  {name}.json")


def _build_config_from_options(
    radii: str,
    z_increment: float,
    seed_count: int,
    random_seed: int,
    extrusion: float,
    scale_x: float,
    scale_y: float,
) -> "PipelineConfig":
    from compass_web.config import PipelineConfig, validate_geometry_limits

    radii_values = tuple(float(v.strip()) for v in radii.split(","))
    if len(radii_values) != 8:
        typer.echo(f"Expected 8 radii values, got {len(radii_values)}.", err=True)
        raise typer.Exit(1)

    validate_geometry_limits(radii_values, z_increment)

    return PipelineConfig(
        radii=radii_values,
        z_increment=z_increment,
        seed_count=seed_count,
        random_seed=random_seed,
        extrusion_multiplier=extrusion,
        scale_x=scale_x,
        scale_y=scale_y,
    )


@app.command(name="new-config")
def new_config(
    radii: Annotated[
        str,
        typer.Option("--radii", help="Comma-separated list of 8 radii values."),
    ] = "8.91,10.446,10.46,17.66,11.26,19.299,11.26,14.46",
    z_increment: Annotated[
        float,
        typer.Option("--z-increment", help="Z spacing between circles."),
    ] = 13.38,
    seed_count: Annotated[
        int,
        typer.Option("--seed-count", help="Number of voronoi seed points."),
    ] = 78,
    random_seed: Annotated[
        int,
        typer.Option("--random-seed", help="Random seed for voronoi generation."),
    ] = 12,
    extrusion: Annotated[
        float,
        typer.Option("--extrusion", help="Extrusion multiplier."),
    ] = -0.2,
    scale_x: Annotated[
        float,
        typer.Option("--scale-x", help="Scale factor for X."),
    ] = 0.5,
    scale_y: Annotated[
        float,
        typer.Option("--scale-y", help="Scale factor for Y."),
    ] = 0.5,
    configs_dir: Annotated[
        Path,
        typer.Option("--configs-dir", help="Directory to save the config into."),
    ] = Path("configs"),
) -> None:
    """Create a new config and save it into configs/ with a timestamp."""
    from compass_web.config import save_pipeline_config

    cfg = _build_config_from_options(
        radii, z_increment, seed_count, random_seed, extrusion, scale_x, scale_y,
    )
    saved = save_pipeline_config(cfg, configs_dir)
    if saved is not None:
        typer.echo(f"Config saved: {saved}")
    else:
        typer.echo("Config identical to existing, not saved.")


@app.command()
def run(
    radii: Annotated[
        str,
        typer.Option("--radii", help="Comma-separated list of 8 radii values."),
    ] = "8.91,10.446,10.46,17.66,11.26,19.299,11.26,14.46",
    z_increment: Annotated[
        float,
        typer.Option("--z-increment", help="Z spacing between circles."),
    ] = 13.38,
    seed_count: Annotated[
        int,
        typer.Option("--seed-count", help="Number of voronoi seed points."),
    ] = 78,
    random_seed: Annotated[
        int,
        typer.Option("--random-seed", help="Random seed for voronoi generation."),
    ] = 12,
    extrusion: Annotated[
        float,
        typer.Option("--extrusion", help="Extrusion multiplier."),
    ] = -0.2,
    scale_x: Annotated[
        float,
        typer.Option("--scale-x", help="Scale factor for X."),
    ] = 0.5,
    scale_y: Annotated[
        float,
        typer.Option("--scale-y", help="Scale factor for Y."),
    ] = 0.5,
    retry: Annotated[
        int,
        typer.Option("--retry", "-r", help="Max auto-retry attempts with different seeds."),
    ] = 10,
    export_dir: Annotated[
        Path,
        typer.Option("--export-dir", "-o", help="Directory for exported STL files."),
    ] = Path("exports"),
    configs_dir: Annotated[
        Path,
        typer.Option("--configs-dir", help="Directory to save the config into."),
    ] = Path("configs"),
    viewer: Annotated[
        bool,
        typer.Option("--viewer/--no-viewer", help="Open interactive 3D viewer after generation."),
    ] = True,
    screenshot: Annotated[
        Optional[Path],
        typer.Option("--screenshot", help="Save a PNG screenshot to this path."),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress progress output."),
    ] = False,
    apply_smoothing: Annotated[
        bool,
        typer.Option(
            "--smoothing/--no-smoothing",
            help="Apply radii/spacing smoothing before loft.",
        ),
    ] = True,
) -> None:
    """Create a config from parameters, save it, and run the pipeline in one step."""
    from compass_web.config import save_pipeline_config
    from compass_web.pipeline import export_stl, run_pipeline_with_retry

    cfg = _build_config_from_options(
        radii, z_increment, seed_count, random_seed, extrusion, scale_x, scale_y,
    )

    saved = save_pipeline_config(cfg, configs_dir)
    if saved is not None:
        typer.echo(f"Config saved: {saved}")
    else:
        if not quiet:
            typer.echo("Config identical to existing, not saved.")

    if not quiet:
        typer.echo("Running voronoi shell pipeline...")

    result, used_config = run_pipeline_with_retry(
        cfg, max_attempts=retry, verbose=not quiet, apply_smoothing=apply_smoothing,
    )

    if not result.is_valid_volume:
        typer.echo("WARNING: Final mesh is not a valid volume.", err=True)

    stl_path = export_stl(result, export_dir)
    typer.echo(f"STL exported: {stl_path} ({stl_path.stat().st_size / 1024:.1f} KB)")

    if saved is None and used_config is not cfg:
        retry_saved = save_pipeline_config(used_config, configs_dir)
        if retry_saved is not None:
            typer.echo(f"Retry config saved: {retry_saved}")

    if screenshot is not None:
        from compass_web.visualization import save_screenshot
        save_screenshot(result.generated_surface, screenshot)
        typer.echo(f"Screenshot saved: {screenshot}")

    if viewer:
        from compass_web.visualization import show_mesh_interactive
        typer.echo("Opening interactive viewer (close window to continue)...")
        show_mesh_interactive(result.generated_surface, title="Voronoi Shell")


@app.command()
def view(
    stl_path: Annotated[
        Path,
        typer.Argument(help="Path to an STL file to view."),
    ],
    title: Annotated[
        str,
        typer.Option("--title", help="Window title."),
    ] = "Voronoi Shell Viewer",
) -> None:
    """Open an interactive 3D viewer for an existing STL file."""
    if not stl_path.exists():
        typer.echo(f"File not found: {stl_path}", err=True)
        raise typer.Exit(1)

    import pyvista as pv
    mesh = pv.read(str(stl_path))
    from compass_web.visualization import show_mesh_interactive
    typer.echo(f"Opening viewer for {stl_path} ({mesh.n_points} points, {mesh.n_cells} faces)...")
    show_mesh_interactive(mesh, title=title)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
