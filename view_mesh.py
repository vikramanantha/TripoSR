import argparse
import json
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


def _rotation_tripo_recon_to_pyrender_world(T_camera_to_world: np.ndarray) -> np.ndarray:
    """
    Map TripoSR recon coords (+X toward camera, +Y right, +Z up) to pyrender world
    using only the camera-to-world rotation columns (see render_to_triposr camera_extrinsics.json).
    """
    return np.stack(
        [T_camera_to_world[:3, 2], T_camera_to_world[:3, 0], T_camera_to_world[:3, 1]],
        axis=1,
    )


def _load_recon_rotation_from_extrinsics(path: str) -> np.ndarray | None:
    """Return 3×3 R with p_world = R @ p_recon, or None if file missing/invalid."""
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[overlay] Could not read camera extrinsics {path}: {e}")
        return None
    if "R_tripo_recon_to_pyrender_world_3x3" in data:
        R = np.asarray(data["R_tripo_recon_to_pyrender_world_3x3"], dtype=np.float64)
        if R.shape == (3, 3):
            print(f"[overlay] Using R from {path} (Rᵀ applied to unit source in overlay)")
            return R
    if "T_camera_to_world_4x4" in data:
        T = np.asarray(data["T_camera_to_world_4x4"], dtype=np.float64)
        if T.shape == (4, 4):
            R = _rotation_tripo_recon_to_pyrender_world(T)
            print(f"[overlay] Derived R from T_camera_to_world in {path}")
            return R
    print(f"[overlay] No usable R in {path}")
    return None


def _apply_rotation_3x3(mesh: trimesh.Trimesh, R: np.ndarray) -> trimesh.Trimesh:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    m = mesh.copy()
    m.apply_transform(T)
    return m


def _normalize_mesh_to_unit_cube(mesh: trimesh.Trimesh, label: str) -> trimesh.Trimesh:
    """
    Center this mesh's axis-aligned bounding box at the origin and scale uniformly
    so its longest AABB edge has length 1 (fits in [-0.5, 0.5] along that axis).
    Independent of any other mesh.
    """
    m = mesh.copy()
    center = (m.bounds[0] + m.bounds[1]) / 2.0
    m.apply_translation(-center)
    longest = float(np.max(m.extents))
    if longest <= 0 or not np.isfinite(longest):
        longest = 1.0
    s = 1.0 / longest
    m.apply_scale(s)
    print(f"[overlay] {label}: per-mesh unit cube (AABB center, scale 1/max_extent = {s:.6f})")
    return m


def _copy_mesh_solid_color(mesh: trimesh.Trimesh, rgba: list[int]) -> trimesh.Trimesh:
    """Copy mesh with a uniform PBR material (robust in GLB exports)."""
    m = mesh.copy()
    base_color = [c / 255.0 for c in rgba]
    try:
        material = trimesh.visual.material.PBRMaterial(
            name="overlay",
            baseColorFactor=base_color,
            metallicFactor=0.0,
            roughnessFactor=0.8,
            alphaMode="BLEND" if rgba[3] < 255 else "OPAQUE",
            doubleSided=True,
        )
        uv = np.zeros((len(m.vertices), 2), dtype=np.float32)
        m.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    except Exception:
        vc = np.tile(np.asarray(rgba, dtype=np.uint8), (len(m.vertices), 1))
        m.visual = trimesh.visual.ColorVisuals(vertex_colors=vc)
    return m


def _combined_overlay_with_axes(
    source_mesh_path: str,
    recon_mesh_path: str,
    out_path: str,
    *,
    show_source: bool = True,
    show_recon: bool = True,
    recon_R_world_from_recon: np.ndarray | None = None,
) -> str:
    """
    Single scene: optional source + optional reconstruction.

    **Source (blue):** unit AABB (bbox center at origin, longest edge = 1) in pyrender
    world. If ``recon_R_world_from_recon`` (``R`` from ``camera_extrinsics.json``) is
    set, then ``p' = Rᵀ @ p`` is applied so the source sits in TripoSR’s recon-like
    frame while the recon mesh is left unrotated.

    **Recon (orange):** TripoSR export only — no rotation, no rescaling, no translation
    beyond what the file contains.
    """
    source = _load_trimesh(source_mesh_path)
    recon = _load_trimesh(recon_mesh_path)

    source_extent = float(np.max(source.extents)) if source.extents.size else 0.0
    recon_extent = float(np.max(recon.extents)) if recon.extents.size else 0.0
    print(
        "[overlay] raw source bounds min/max: "
        f"{np.round(source.bounds[0], 4).tolist()} / {np.round(source.bounds[1], 4).tolist()} "
        f"(max extent {source_extent:.4f})"
    )
    print(
        "[overlay] raw recon bounds (TripoSR export) min/max: "
        f"{np.round(recon.bounds[0], 4).tolist()} / {np.round(recon.bounds[1], 4).tolist()} "
        f"(max extent {recon_extent:.4f})"
    )

    source = _normalize_mesh_to_unit_cube(source, "source")
    if recon_R_world_from_recon is not None:
        R_inv = recon_R_world_from_recon.T
        source = _apply_rotation_3x3(source, R_inv)
        print(
            "[overlay] source after unit cube + Rᵀ (pyrender world → recon frame) min/max: "
            f"{np.round(source.bounds[0], 4).tolist()} / {np.round(source.bounds[1], 4).tolist()}"
        )
    else:
        print("[overlay] source: unit cube only (no Rᵀ — missing camera_extrinsics.json)")
    print("[overlay] reconstruction: unchanged from TripoSR export (no rotation)")

    bounds = np.stack([source.bounds, recon.bounds], axis=0)
    combined_min = bounds[:, 0, :].min(axis=0)
    combined_max = bounds[:, 1, :].max(axis=0)
    extents_union = combined_max - combined_min
    max_extent = float(np.max(extents_union)) if extents_union.size else 1.0

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

    scene = trimesh.Scene()
    if show_source:
        scene.add_geometry(
            _copy_mesh_solid_color(source, [70, 130, 255, 200]),
            geom_name="source",
        )
    if show_recon:
        scene.add_geometry(
            _copy_mesh_solid_color(recon, [255, 120, 60, 220]),
            geom_name="reconstruction",
        )
    scene.add_geometry(axes, geom_name="axes")
    scene.add_geometry(grid, geom_name="grid")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    scene.export(out_path)
    return out_path


def build_viewer(
    mesh_path: str,
    source_mesh_path: str | None = None,
    render_image_path: str | None = None,
    output_dir: str | None = None,
    camera_extrinsics_path: str | None = None,
) -> gr.Blocks:
    mesh_name = os.path.basename(mesh_path)
    compare_mode = source_mesh_path is not None
    source_name = os.path.basename(source_mesh_path) if source_mesh_path else ""
    output_dir = output_dir or os.path.dirname(mesh_path) or "."

    if camera_extrinsics_path is None:
        _default_ext = os.path.join(output_dir, "camera_extrinsics.json")
        if os.path.isfile(_default_ext):
            camera_extrinsics_path = _default_ext
    recon_R = _load_recon_rotation_from_extrinsics(camera_extrinsics_path) if camera_extrinsics_path else None

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
        overlay_path = _combined_overlay_with_axes(
            source_mesh_path,
            mesh_path,
            os.path.join(vis_dir, "overlay_both_meshes.glb"),
            recon_R_world_from_recon=recon_R,
        )
    else:
        overlay_path = None

    with gr.Blocks(title=f"Mesh Viewer - {mesh_name}") as app:
        intro = (
            "# Mesh Viewer\n"
            f"Viewing `{mesh_name}` with coordinate axes "
            "(+X red, +Y green, +Z blue, origin white) and a ground scale grid."
        )
        if compare_mode:
            _rot_note = (
                " Blue source gets **Rᵀ** from `camera_extrinsics.json` after unit-cube normalize "
                "(same capture as TripoSR input); orange recon is **not** rotated."
                if recon_R is not None
                else " Place `camera_extrinsics.json` next to the render (from `render_to_triposr.py`) "
                "to apply **Rᵀ** to the source in the overlay."
            )
            intro += (
                "\n\n**Overlay (bottom right):** blue = source (unit AABB), orange = recon"
                + _rot_note
                + " Use the checkboxes to show or hide each mesh."
            )
        gr.Markdown(intro)
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
            elif compare_mode and overlay_path:
                gr.Markdown("")
            if compare_mode and overlay_path:
                overlay_glb_path = os.path.join(vis_dir, "overlay_both_meshes.glb")

                def _refresh_overlay(show_src: bool, show_recon: bool):
                    _combined_overlay_with_axes(
                        source_mesh_path,
                        mesh_path,
                        overlay_glb_path,
                        show_source=show_src,
                        show_recon=show_recon,
                        recon_R_world_from_recon=recon_R,
                    )
                    return gr.update(value=overlay_glb_path)

                with gr.Column():
                    with gr.Row():
                        show_source_cb = gr.Checkbox(
                            value=True,
                            label="Show source (blue)",
                        )
                        show_recon_cb = gr.Checkbox(
                            value=True,
                            label="Show reconstruction (orange)",
                        )
                    overlay_model = gr.Model3D(
                        value=overlay_path,
                        clear_color=[1.0, 1.0, 1.0, 1.0],
                        label="Overlay: same coordinates (toggle meshes above)",
                    )
                    show_source_cb.change(
                        _refresh_overlay,
                        inputs=[show_source_cb, show_recon_cb],
                        outputs=[overlay_model],
                    )
                    show_recon_cb.change(
                        _refresh_overlay,
                        inputs=[show_source_cb, show_recon_cb],
                        outputs=[overlay_model],
                    )
            elif not compare_mode:
                gr.File(value=mesh_path, label="Download reconstructed mesh")
        if compare_mode:
            with gr.Row():
                gr.File(value=mesh_path, label="Download reconstructed mesh")
                gr.File(value=source_mesh_path, label="Download source mesh")

    return app


def launch_viewer(
    mesh_path: str,
    source_mesh_path: str | None,
    render_image_path: str | None,
    output_dir: str,
    port: int = 7861,
    listen: bool = False,
    share: bool = True,
    camera_extrinsics_path: str | None = None,
) -> None:
    app = build_viewer(
        mesh_path=mesh_path,
        source_mesh_path=source_mesh_path,
        render_image_path=render_image_path,
        output_dir=output_dir,
        camera_extrinsics_path=camera_extrinsics_path,
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
    if camera_extrinsics_path:
        allowed_paths.append(camera_extrinsics_path)
    elif os.path.isfile(os.path.join(output_dir, "camera_extrinsics.json")):
        allowed_paths.append(os.path.join(output_dir, "camera_extrinsics.json"))
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
    parser.add_argument(
        "--camera-extrinsics",
        default=None,
        help="JSON from render_to_triposr (camera_extrinsics.json) to rotate recon in overlay.",
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
    camera_extrinsics_path = (
        os.path.abspath(os.path.expanduser(args.camera_extrinsics)) if args.camera_extrinsics else None
    )

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
        camera_extrinsics_path=camera_extrinsics_path,
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
                *([camera_extrinsics_path] if camera_extrinsics_path else []),
            ],
            **launch_kwargs,
        )
    except TypeError:
        app.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
