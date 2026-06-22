"""
infer_sdf_mesh.py  —  SDF MLP inference on a new mesh.

Given a saved SDF MLP checkpoint (from train_sdf_head.py) and a mesh source
(Objaverse UID, UID index, or local file), this script:

  1. Normalises and renders the mesh to an input PNG.
  2. Runs frozen TripoSR to extract the triplane (scene codes).
  3. Reconstructs a mesh via the trained SDF MLP (marching cubes on SDF field).
  4. Reconstructs a mesh via TripoSR's original NeRF density decoder (baseline).
  5. Opens a Gradio viewer showing all three meshes + the input render.

Usage
-----
    python infer_sdf_mesh.py \\
        --checkpoint sdf_checkpoints/sdf_head_epoch0500.pt \\
        --uid 1b5e7a72f9cc42a0bd76d1d64db6d3c5

    python infer_sdf_mesh.py \\
        --checkpoint sdf_checkpoints/sdf_head_epoch0500.pt \\
        --uid-index 0 --azimuth 45 --mc-resolution 128

    python infer_sdf_mesh.py \\
        --checkpoint sdf_checkpoints/sdf_head_epoch0500.pt \\
        --mesh path/to/object.obj --azimuth 90

Optional flags
--------------
    --model           TripoSR pretrained model ID          (default stabilityai/TripoSR)
    --output-dir      Directory for renders + meshes       (default infer_output/)
    --azimuth         Camera azimuth in degrees            (default 45)
    --elevation       Camera elevation in degrees          (default 30)
    --fov             Vertical field of view in degrees    (default 40)
    --size            Render image size in pixels          (default 256)
    --mc-resolution   Marching-cubes grid resolution       (default 128)
    --nerf-threshold  Density threshold for NeRF MC        (default 25.0)
    --port            Gradio viewer port                   (default 7861)
    --listen          Bind Gradio to 0.0.0.0
    --share           Create a public Gradio share link
"""

import argparse
import gc
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYOPENGL_USE_ACCELERATE", "0")

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import trimesh
import gradio as gr
from PIL import Image

# Reuse core utilities from train_sdf_head (no model loading side-effects on import)
from train_sdf_head import (
    SDFMLP,
    load_and_normalize_mesh,
    load_R_world_from_recon_json,
    reconstruct_mesh_from_triplane,
    reconstruct_mesh_nerf_decoder,
    render_mesh_to_image,
)

# Reuse viewer helpers from view_mesh
from view_mesh import (
    _axis_marker_mesh,
    _copy_mesh_solid_color,
    _mesh_with_axes,
    _scale_grid_mesh,
)


# ─── Objaverse helpers ────────────────────────────────────────────────────────

def uid_from_index(index: int) -> str:
    import objaverse
    print("[objaverse] Loading UID list...")
    uids_list = list(objaverse.load_uids())
    if not (0 <= index < len(uids_list)):
        raise IndexError(f"--uid-index {index} out of range ({len(uids_list)} objects)")
    uid = uids_list[index]
    print(f"[objaverse] UID at index {index}: {uid}")
    return uid


def fetch_objaverse_glb(uid: str, cache_dir: str) -> str:
    import objaverse
    cached = os.path.join(cache_dir, uid)
    for ext in (".glb", ".obj", ".stl"):
        candidate = os.path.join(cached, f"{uid}{ext}")
        if os.path.exists(candidate):
            print(f"[objaverse] Cache hit: {candidate}")
            return candidate
    print(f"[objaverse] Downloading UID: {uid}")
    objects = objaverse.load_objects(uids=[uid], download_processes=1)
    if uid not in objects:
        raise RuntimeError(f"Objaverse returned nothing for UID: {uid}")
    src = objects[uid]
    os.makedirs(cached, exist_ok=True)
    dst = os.path.join(cached, os.path.basename(src))
    if src != dst:
        shutil.copy2(src, dst)
    print(f"[objaverse] Saved: {dst}")
    return dst


# ─── Checkpoint loading ───────────────────────────────────────────────────────

def load_sdf_mlp_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[SDFMLP, dict, argparse.Namespace]:
    """Rebuild SDFMLP architecture from a checkpoint saved by train_sdf_head.py.

    Returns (sdf_mlp, meta, saved_args).  The model is already on ``device``
    with weights loaded and set to eval mode.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    saved_args = argparse.Namespace(**ckpt["args"])

    feat_dim: int = meta["feat_dim"]
    n_freqs: int = getattr(saved_args, "n_freqs", 0)
    pe_dim: int = (3 + 6 * n_freqs) if n_freqs > 0 else 0

    use_triplane: bool = getattr(saved_args, "use_triplane_features", True)
    hidden_dim: int = getattr(saved_args, "hidden_dim", 256)
    if use_triplane:
        mlp_feat_dim = feat_dim
        mlp_hidden_dim = hidden_dim
    else:
        mlp_feat_dim = 0
        mlp_hidden_dim = getattr(saved_args, "hidden_dim_no_triplane", hidden_dim)

    mlp_in_dim = mlp_feat_dim + pe_dim

    sdf_mlp = SDFMLP(
        in_dim=mlp_in_dim,
        hidden_dim=mlp_hidden_dim,
        n_hidden=getattr(saved_args, "n_hidden", 3),
        use_tanh_output=getattr(saved_args, "use_tanh_output", False),
        feat_dim=mlp_feat_dim,
        pe_dim=pe_dim,
    ).to(device)
    sdf_mlp.load_state_dict(ckpt["model"])
    sdf_mlp.eval()

    epoch = ckpt.get("epoch", "?")
    n_hidden_val = getattr(saved_args, "n_hidden", 3)
    print(
        f"[checkpoint] Loaded SDF MLP (epoch {epoch}): "
        f"in_dim={mlp_in_dim} (feat={mlp_feat_dim} + pe={pe_dim}), "
        f"hidden_dim={mlp_hidden_dim}, n_hidden={n_hidden_val}"
    )
    return sdf_mlp, meta, saved_args


# ─── Three-mesh overlay ───────────────────────────────────────────────────────

def _combined_three_way_overlay(
    source_mesh: trimesh.Trimesh,
    nerf_mesh: trimesh.Trimesh | None,
    sdf_mesh: trimesh.Trimesh | None,
    out_path: str,
) -> str:
    """Write a GLB combining all three meshes (each coloured differently) with axes.

    All meshes must already be in the same world coordinate frame
    (i.e. the TripoSR recon → world rotation has already been applied).

    Colours:  blue = GT source,  green = NeRF decoder,  orange = SDF MLP.
    """
    extents = [
        float(np.max(source_mesh.extents)) if source_mesh.extents.size else 1.0,
    ]
    if nerf_mesh is not None and nerf_mesh.extents.size:
        extents.append(float(np.max(nerf_mesh.extents)))
    if sdf_mesh is not None and sdf_mesh.extents.size:
        extents.append(float(np.max(sdf_mesh.extents)))
    max_extent = max(extents)

    axis_len = max(max_extent * 0.6, 0.2)
    axis_thickness = max(axis_len * 0.03, 0.005)
    axes = _axis_marker_mesh(axis_len, axis_thickness)
    grid_half = max(max_extent * 0.75, 0.3)
    grid_step = max(grid_half / 10.0, 0.02)
    grid = _scale_grid_mesh(
        half_extent=grid_half,
        line_thickness=max(axis_thickness * 0.35, 0.0015),
        step=grid_step,
    )

    scene = trimesh.Scene()
    scene.add_geometry(
        _copy_mesh_solid_color(source_mesh, [70, 130, 255, 200]),
        geom_name="source_gt",
    )
    if nerf_mesh is not None:
        scene.add_geometry(
            _copy_mesh_solid_color(nerf_mesh, [60, 200, 80, 220]),
            geom_name="nerf_decoder",
        )
    if sdf_mesh is not None:
        scene.add_geometry(
            _copy_mesh_solid_color(sdf_mesh, [255, 100, 50, 220]),
            geom_name="sdf_mlp",
        )
    scene.add_geometry(axes, geom_name="axes")
    scene.add_geometry(grid, geom_name="grid")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    scene.export(out_path)
    return out_path


# ─── Gradio viewer ────────────────────────────────────────────────────────────

def _format_timing_table(timing: dict) -> str:
    """Format a timing dict as a Gradio-friendly markdown table."""
    model_keys = {"SDF MLP checkpoint load", "TripoSR model load"}
    pipeline_total = sum(v for k, v in timing.items() if k not in model_keys)
    grand_total = sum(timing.values())

    rows = ["| Stage | Time |", "|:------|-----:|"]
    for label, seconds in timing.items():
        rows.append(f"| {label} | {seconds:.2f} s |")
    rows.append("|  |  |")
    rows.append(f"| **Pipeline total** (excl. model loads) | **{pipeline_total:.2f} s** |")
    rows.append(f"| **Grand total** | **{grand_total:.2f} s** |")
    return "### Inference Timing\n\n" + "\n".join(rows)


def launch_three_way_viewer(
    source_mesh_path: str,
    nerf_mesh_path: str | None,
    sdf_mesh_path: str,
    render_image_path: str | None,
    output_dir: str,
    timing: dict | None = None,
    port: int = 7861,
    listen: bool = False,
    share: bool = True,
) -> None:
    """Gradio viewer: GT source  |  NeRF decoder  |  SDF MLP  +  three-way overlay."""
    vis_dir = os.path.join(output_dir, "viewer_axes")

    # Per-mesh views with coordinate axes
    source_axes = _mesh_with_axes(
        source_mesh_path, os.path.join(vis_dir, "source_with_axes.glb")
    )
    nerf_axes = (
        _mesh_with_axes(nerf_mesh_path, os.path.join(vis_dir, "nerf_with_axes.glb"))
        if nerf_mesh_path
        else None
    )
    sdf_axes = _mesh_with_axes(
        sdf_mesh_path, os.path.join(vis_dir, "sdf_mlp_with_axes.glb")
    )

    # Three-way overlay (all in world / normalised-mesh frame)
    source_tm = trimesh.load(source_mesh_path, force="mesh")
    nerf_tm = trimesh.load(nerf_mesh_path, force="mesh") if nerf_mesh_path else None
    sdf_tm = trimesh.load(sdf_mesh_path, force="mesh")
    overlay_path = os.path.join(vis_dir, "overlay_three_way.glb")
    _combined_three_way_overlay(source_tm, nerf_tm, sdf_tm, overlay_path)

    with gr.Blocks(title="SDF MLP Inference Viewer") as app:
        gr.Markdown(
            "# SDF MLP Inference — Mesh Comparison\n"
            "**Blue** = source GT mesh  |  "
            "**Green** = TripoSR NeRF decoder  |  "
            "**Orange** = SDF MLP prediction  \n"
            "Drag to rotate · scroll to zoom · axes: +X red, +Y green, +Z blue"
        )
        if timing:
            gr.Markdown(_format_timing_table(timing))

        # Top row: individual meshes
        with gr.Row(equal_height=True):
            gr.Model3D(
                value=source_axes,
                clear_color=[1.0, 1.0, 1.0, 1.0],
                label="Source GT mesh",
            )
            if nerf_axes:
                gr.Model3D(
                    value=nerf_axes,
                    clear_color=[1.0, 1.0, 1.0, 1.0],
                    label="TripoSR NeRF decoder baseline",
                )
            gr.Model3D(
                value=sdf_axes,
                clear_color=[1.0, 1.0, 1.0, 1.0],
                label="SDF MLP prediction",
            )

        # Bottom row: input image + overlay
        with gr.Row(equal_height=True):
            if render_image_path and os.path.exists(render_image_path):
                gr.Image(
                    value=render_image_path,
                    label="TripoSR input image",
                    interactive=False,
                )
            gr.Model3D(
                value=overlay_path,
                clear_color=[1.0, 1.0, 1.0, 1.0],
                label="Overlay — blue: GT  ·  green: NeRF  ·  orange: SDF MLP",
            )

        # Downloads
        with gr.Row():
            gr.File(value=sdf_mesh_path, label="Download SDF MLP mesh")
            if nerf_mesh_path:
                gr.File(value=nerf_mesh_path, label="Download NeRF decoder mesh")
            gr.File(value=source_mesh_path, label="Download source GT mesh")

    allowed = [output_dir, source_mesh_path, sdf_mesh_path]
    if nerf_mesh_path:
        allowed.append(nerf_mesh_path)
    if render_image_path:
        allowed.append(render_image_path)

    launch_kwargs = dict(
        server_name="0.0.0.0" if listen else "localhost",
        server_port=port,
        share=share,
    )
    try:
        app.launch(allowed_paths=allowed, **launch_kwargs)
    except TypeError:
        app.launch(**launch_kwargs)


# ─── Main inference pipeline ──────────────────────────────────────────────────

def run_inference(args: argparse.Namespace) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Inference on {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timing: dict = {}

    # ── 1. Load SDF MLP checkpoint ────────────────────────────────────────────
    print(f"\n[1/5] Loading SDF MLP checkpoint: {args.checkpoint}")
    _t = time.perf_counter()
    sdf_mlp, meta, saved_args = load_sdf_mlp_from_checkpoint(args.checkpoint, device)
    timing["SDF MLP checkpoint load"] = time.perf_counter() - _t
    radius: float = meta["radius"]
    feature_reduction: str = meta["feature_reduction"]
    n_freqs: int = getattr(saved_args, "n_freqs", 0)
    use_triplane: bool = getattr(saved_args, "use_triplane_features", True)

    # ── 2. Load frozen TripoSR ────────────────────────────────────────────────
    print(f"\n[2/5] Loading TripoSR ({args.model})...")
    from tsr.system import TSR

    _t = time.perf_counter()
    triposr = TSR.from_pretrained(args.model, config_name="config.yaml", weight_name="model.ckpt")
    triposr.renderer.set_chunk_size(8192)
    triposr.to(device).eval()
    for p in triposr.parameters():
        p.requires_grad_(False)
    timing["TripoSR model load"] = time.perf_counter() - _t

    triposr_decoder = triposr.decoder
    _density_activation: str = triposr.renderer.cfg.density_activation
    _density_bias: float = float(triposr.renderer.cfg.density_bias)

    # ── 3. Resolve mesh path and normalise ────────────────────────────────────
    print("\n[3/5] Loading and normalising mesh...")
    cache_dir = str(output_dir / "objaverse_cache")
    if args.uid is not None:
        mesh_path = fetch_objaverse_glb(args.uid, cache_dir)
    elif args.uid_index is not None:
        uid = uid_from_index(args.uid_index)
        mesh_path = fetch_objaverse_glb(uid, cache_dir)
    else:
        mesh_path = os.path.abspath(os.path.expanduser(args.mesh))
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    _t = time.perf_counter()
    mesh = load_and_normalize_mesh(mesh_path, radius)
    source_obj_path = str(output_dir / "source_mesh.obj")
    mesh.export(source_obj_path)
    timing["Mesh load & normalize"] = time.perf_counter() - _t
    print(f"[mesh] Normalised source ({len(mesh.vertices):,}v, {len(mesh.faces):,}f) → {source_obj_path}")

    # ── 4. Render + TripoSR triplane extraction ───────────────────────────────
    print(f"\n[4/5] Rendering (az={args.azimuth}°, el={args.elevation}°, fov={args.fov}°) "
          f"and running TripoSR...")
    extrinsics_path = output_dir / "camera_extrinsics.json"
    render_path = str(output_dir / "input_render.png")

    _t = time.perf_counter()
    pil_image = render_mesh_to_image(
        mesh,
        elevation=args.elevation,
        fov=args.fov,
        size=args.size,
        azimuth=args.azimuth,
        extrinsics_json_path=str(extrinsics_path),
    )
    pil_image.save(render_path)
    timing["Render to image"] = time.perf_counter() - _t
    print(f"[render] Input image → {render_path}")

    # Re-load the saved PNG as a NumPy array — byte-identical to precompute.
    image_np = np.array(Image.open(render_path).convert("RGB"))

    _t = time.perf_counter()
    with torch.no_grad():
        scene_codes = triposr([image_np], device=device)
    triplane = scene_codes[0].float()
    timing["TripoSR triplane extraction"] = time.perf_counter() - _t
    print(f"[triposr] Triplane shape: {tuple(triplane.shape)}")

    # R maps TripoSR recon frame → normalised mesh world frame.
    # load_R_world_from_recon_json expects the directory containing camera_extrinsics.json.
    R_np = load_R_world_from_recon_json(output_dir)
    if R_np is None:
        print("[warning] Could not load camera extrinsics — meshes will be in TripoSR recon space.")

    # ── 5. Reconstruct meshes via marching cubes ──────────────────────────────
    print(f"\n[5/5] Marching cubes (resolution={args.mc_resolution})...")

    print("[reconstruct] SDF MLP ...")
    _t = time.perf_counter()
    sdf_mesh = reconstruct_mesh_from_triplane(
        sdf_mlp,
        triplane,
        radius,
        feature_reduction,
        resolution=args.mc_resolution,
        device=device,
        n_freqs=n_freqs,
        R_world_from_trip=R_np,
        use_triplane_features=use_triplane,
    )
    timing["SDF MLP marching cubes"] = time.perf_counter() - _t
    if sdf_mesh is None:
        print("[warning] SDF MLP marching cubes: no zero crossing found.")
        sdf_obj_path = None
    else:
        sdf_obj_path = str(output_dir / "sdf_mlp_mesh.obj")
        sdf_mesh.export(sdf_obj_path)
        print(f"[sdf_mlp] → {sdf_obj_path}  ({len(sdf_mesh.vertices):,}v, {len(sdf_mesh.faces):,}f)")

    print("[reconstruct] TripoSR NeRF decoder ...")
    _t = time.perf_counter()
    nerf_mesh = reconstruct_mesh_nerf_decoder(
        triposr_decoder,
        triplane,
        radius,
        feature_reduction,
        _density_activation,
        _density_bias,
        resolution=args.mc_resolution,
        threshold=args.nerf_threshold,
        device=device,
        R_world_from_trip=R_np,
    )
    timing["NeRF decoder marching cubes"] = time.perf_counter() - _t
    if nerf_mesh is None:
        print("[warning] NeRF decoder marching cubes: no surface found.")
        nerf_obj_path = None
    else:
        nerf_obj_path = str(output_dir / "nerf_decoder_mesh.obj")
        nerf_mesh.export(nerf_obj_path)
        print(f"[nerf] → {nerf_obj_path}  ({len(nerf_mesh.vertices):,}v, {len(nerf_mesh.faces):,}f)")

    # Free GPU memory before viewer
    del triposr, triplane
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if sdf_obj_path is None:
        print("\n[error] SDF MLP produced no mesh — cannot open viewer.")
        return

    print("\nLaunching Gradio viewer...")
    launch_three_way_viewer(
        source_mesh_path=source_obj_path,
        nerf_mesh_path=nerf_obj_path,
        sdf_mesh_path=sdf_obj_path,
        render_image_path=render_path,
        output_dir=str(output_dir),
        timing=timing,
        port=args.port,
        listen=args.listen,
        share=True,
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SDF MLP inference: render mesh → TripoSR triplane → SDF MLP + NeRF → Gradio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--uid", default=None,
        help="Objaverse UID to download and render.",
    )
    source.add_argument(
        "--uid-index", type=int, default=None, metavar="N",
        help="Index into the Objaverse UID list (0-based).",
    )
    source.add_argument(
        "--mesh", default=None,
        help="Path to a local mesh file (OBJ / GLB / STL).",
    )

    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the saved SDF MLP checkpoint (.pt) from train_sdf_head.py.",
    )
    parser.add_argument(
        "--model", default="stabilityai/TripoSR",
        help="TripoSR pretrained model ID or local path.",
    )
    parser.add_argument(
        "--output-dir", default="infer_output",
        help="Directory for renders and output meshes.",
    )
    parser.add_argument(
        "--azimuth", type=float, default=45.0,
        help="Camera azimuth in degrees.",
    )
    parser.add_argument(
        "--elevation", type=float, default=30.0,
        help="Camera elevation in degrees.",
    )
    parser.add_argument(
        "--fov", type=float, default=40.0,
        help="Vertical field of view in degrees.",
    )
    parser.add_argument(
        "--size", type=int, default=256,
        help="Render image size in pixels (matched to precompute IMAGE_SIZE).",
    )
    parser.add_argument(
        "--mc-resolution", type=int, default=128,
        help="Marching-cubes grid resolution.",
    )
    parser.add_argument(
        "--nerf-threshold", type=float, default=25.0,
        help="Density threshold for the NeRF decoder marching cubes.",
    )
    parser.add_argument(
        "--port", type=int, default=7861,
        help="Gradio viewer port.",
    )
    parser.add_argument(
        "--listen", action="store_true",
        help="Bind Gradio to 0.0.0.0 (accessible from other machines).",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Ask Gradio to create a temporary public share link.",
    )

    args = parser.parse_args()
    args.output_dir = os.path.abspath(args.output_dir)
    args.checkpoint = os.path.abspath(os.path.expanduser(args.checkpoint))

    run_inference(args)


if __name__ == "__main__":
    main()
