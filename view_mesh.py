import argparse
import os

import gradio as gr
import numpy as np
import trimesh


def _load_trimesh(path: str) -> trimesh.Trimesh:
    """Load a mesh file and return a single Trimesh (merging scenes if needed)."""
    loaded = trimesh.load(path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No triangle geometry found in {path}")
        return trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    raise ValueError(f"Unsupported mesh type: {type(loaded)}")


def _axis_marker_mesh(length: float, thickness: float) -> trimesh.Trimesh:
    """Create RGB axis bars centered at origin (+X red, +Y green, +Z blue)."""
    x_bar = trimesh.creation.box(extents=[length, thickness, thickness])
    x_bar.apply_translation([length / 2.0, 0.0, 0.0])
    x_bar.visual.vertex_colors = [255, 64, 64, 255]

    y_bar = trimesh.creation.box(extents=[thickness, length, thickness])
    y_bar.apply_translation([0.0, length / 2.0, 0.0])
    y_bar.visual.vertex_colors = [64, 255, 64, 255]

    z_bar = trimesh.creation.box(extents=[thickness, thickness, length])
    z_bar.apply_translation([0.0, 0.0, length / 2.0])
    z_bar.visual.vertex_colors = [64, 64, 255, 255]

    origin = trimesh.creation.icosphere(subdivisions=2, radius=thickness * 1.5)
    origin.visual.vertex_colors = [250, 250, 250, 255]

    return trimesh.util.concatenate([x_bar, y_bar, z_bar, origin])


def _scale_grid_mesh(
    half_extent: float,
    line_thickness: float,
    step: float,
) -> trimesh.Trimesh:
    """
    Create a simple X/Y ground grid at Z=0 using thin box lines.
    Major lines (every 5 steps) are brighter to improve readability.
    """
    lines: list[trimesh.Trimesh] = []
    z = -line_thickness * 0.6  # Slightly below origin to avoid z-fighting with axes.
    max_index = int(np.floor(half_extent / step))

    for i in range(-max_index, max_index + 1):
        coord = i * step
        is_major = (i % 5) == 0
        alpha = 120 if is_major else 70
        color = [220, 220, 220, alpha] if is_major else [180, 180, 180, alpha]

        # Line parallel to X (varying Y).
        x_line = trimesh.creation.box(
            extents=[2.0 * half_extent, line_thickness, line_thickness]
        )
        x_line.apply_translation([0.0, coord, z])
        x_line.visual.vertex_colors = color
        lines.append(x_line)

        # Line parallel to Y (varying X).
        y_line = trimesh.creation.box(
            extents=[line_thickness, 2.0 * half_extent, line_thickness]
        )
        y_line.apply_translation([coord, 0.0, z])
        y_line.visual.vertex_colors = color
        lines.append(y_line)

    if not lines:
        raise ValueError("Unable to generate scale grid lines.")
    return trimesh.util.concatenate(lines)


def _mesh_with_axes(mesh_path: str, out_path: str) -> str:
    """Write a copy of mesh_path with coordinate axes and a scale grid."""
    mesh = _load_trimesh(mesh_path).copy()
    max_extent = float(np.max(mesh.extents)) if len(mesh.extents) else 1.0
    axis_len = max(max_extent * 0.6, 0.2)
    axis_thickness = max(axis_len * 0.03, 0.005)
    axes = _axis_marker_mesh(axis_len, axis_thickness)
    grid_half_extent = max(max_extent * 0.75, 0.3)
    grid_step = max(grid_half_extent / 10.0, 0.02)
    grid = _scale_grid_mesh(
        half_extent=grid_half_extent,
        line_thickness=max(axis_thickness * 0.35, 0.0015),
        step=grid_step,
    )
    combined = trimesh.util.concatenate([mesh, axes, grid])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    combined.export(out_path)
    return out_path


def build_viewer(
    mesh_path: str,
    source_mesh_path: str | None = None,
    render_image_path: str | None = None,
    output_dir: str | None = None,
) -> gr.Blocks:
    mesh_name = os.path.basename(mesh_path)
    compare_mode = source_mesh_path is not None
    source_name = os.path.basename(source_mesh_path) if source_mesh_path else ""
    output_dir = output_dir or os.path.dirname(mesh_path) or "."
    vis_dir = os.path.join(output_dir, "viewer_axes")
    recon_axes_path = _mesh_with_axes(
        mesh_path,
        os.path.join(vis_dir, "reconstruction_with_axes.glb"),
    )
    source_axes_path = None
    if compare_mode:
        source_axes_path = _mesh_with_axes(
            source_mesh_path,
            os.path.join(vis_dir, "source_with_axes.glb"),
        )

    with gr.Blocks(title=f"Mesh Viewer - {mesh_name}") as app:
        gr.Markdown(
            "# Mesh Viewer\n"
            f"Viewing `{mesh_name}` with coordinate axes "
            "(+X red, +Y green, +Z blue, origin white) and a ground scale grid."
        )
        with gr.Row(equal_height=True):
            if compare_mode and source_axes_path:
                gr.Model3D(
                    value=source_axes_path,
                    clear_color=[1.0, 1.0, 1.0, 1.0],
                    label=f"Original source mesh + axes ({source_name})",
                )
            gr.Model3D(
                value=recon_axes_path,
                clear_color=[1.0, 1.0, 1.0, 1.0],
                label=(
                    "Reconstructed mesh + axes (drag to rotate, scroll to zoom)"
                    if compare_mode
                    else "Mesh + axes (drag to rotate, scroll to zoom)"
                ),
            )
        with gr.Row():
            if render_image_path and os.path.exists(render_image_path):
                gr.Image(
                    value=render_image_path,
                    label="Rendered input image",
                    interactive=False,
                )
            gr.File(value=mesh_path, label="Download reconstructed mesh")

    return app


def launch_viewer(
    mesh_path: str,
    source_mesh_path: str | None,
    render_image_path: str | None,
    output_dir: str,
    port: int = 7861,
    listen: bool = False,
    share: bool = True,
) -> None:
    app = build_viewer(
        mesh_path=mesh_path,
        source_mesh_path=source_mesh_path,
        render_image_path=render_image_path,
        output_dir=output_dir,
    )
    kwargs = {
        "server_name": "0.0.0.0" if listen else "localhost",
        "server_port": port,
    }
    allowed_paths = [mesh_path, output_dir]
    if source_mesh_path:
        allowed_paths.append(source_mesh_path)
    if render_image_path:
        allowed_paths.append(render_image_path)
    try:
        app.launch(allowed_paths=allowed_paths, share=share, **kwargs)
    except TypeError:
        app.launch(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open an OBJ/GLB mesh in a local browser-based 3D viewer."
    )
    parser.add_argument("mesh", help="Path to the mesh file to display.")
    parser.add_argument(
        "--source-mesh",
        default=None,
        help="Optional source mesh to compare against.",
    )
    parser.add_argument(
        "--render-image",
        default=None,
        help="Optional rendered input image shown alongside the mesh.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write temporary viewer meshes with axes.",
    )
    parser.add_argument("--port", type=int, default=7861, help="Port for the local viewer.")
    parser.add_argument(
        "--listen",
        action="store_true",
        help="Bind to 0.0.0.0 so the viewer is reachable from other machines.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Ask Gradio to create a temporary public share link.",
    )
    args = parser.parse_args()

    mesh_path = os.path.abspath(os.path.expanduser(args.mesh))
    source_mesh_path = os.path.abspath(os.path.expanduser(args.source_mesh)) if args.source_mesh else None
    render_image_path = (
        os.path.abspath(os.path.expanduser(args.render_image)) if args.render_image else None
    )
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir)) if args.output_dir else os.path.dirname(mesh_path)

    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    if source_mesh_path and not os.path.exists(source_mesh_path):
        raise FileNotFoundError(f"Source mesh file not found: {source_mesh_path}")
    if render_image_path and not os.path.exists(render_image_path):
        raise FileNotFoundError(f"Render image not found: {render_image_path}")

    app = build_viewer(
        mesh_path=mesh_path,
        source_mesh_path=source_mesh_path,
        render_image_path=render_image_path,
        output_dir=output_dir,
    )
    launch_kwargs = {
        "share": args.share,
        "server_name": "0.0.0.0" if args.listen else None,
        "server_port": args.port,
    }

    # Older Gradio releases do not support allowed_paths.
    try:
        app.launch(
            allowed_paths=[
                mesh_path,
                output_dir,
                *([source_mesh_path] if source_mesh_path else []),
                *([render_image_path] if render_image_path else []),
            ],
            **launch_kwargs,
        )
    except TypeError:
        app.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
