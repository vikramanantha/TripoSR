"""
visualize_sdf_diagnostics.py  —  Cross-section, error-heatmap, and metric
diagnostics for a trained SDF MLP, across multiple objects at once.

Given a saved SDF MLP checkpoint (from train_sdf_head.py) and one or more
mesh sources (Objaverse UIDs, UID indices, or local files), this script
reconstructs each object and opens a Gradio grid — one ROW per object, one
COLUMN per visualization:

  1. Overview      — GT vs SDF-MLP mesh render + the input image fed to TripoSR
                      (reuses train_sdf_head.create_mesh_comparison_visualization)
  2. SDF slice     — 2D cross-section through the volume, GT | predicted side
                      by side, colored by sign, zero-contour overlaid. The
                      slice axis + offset are SHARED sliders above the grid
                      that update every row's slice at once.
  3. Sign map      — the same slice, colored red/green by sign(pred) ==
                      sign(GT). Separates topology mistakes (what SIGN_BCE
                      supervises) from pure magnitude error, which a single
                      MSE number conflates.
  4. Surface error — the predicted mesh, vertices colored by nearest-neighbor
                      distance to the GT surface. Shows WHERE on the object
                      it's wrong instead of one aggregate number.
  5. Metrics       — pointwise SDF MSE (raw + TSDF-clamped, two-pass — see
                      infer_sdf_mesh.compute_pointwise_sdf_mse), Chamfer-L2,
                      and F-score, per object.

Only axis-aligned cross-sections (X / Y / Z) are implemented, not arbitrary
plane rotation — a deliberately simple slider ("sweep the cut position along
one of three axes") rather than full 3D plane-orientation control.

Usage
-----
    # Multiple Objaverse UIDs, using a trained checkpoint
    python visualize_sdf_diagnostics.py \\
        --checkpoint sdf_checkpoints/sdf_head_v0.61_100_epoch0100.pt \\
        --uids 137f0eb28c524ab0aac86c5105ba33bb 1b5e7a72f9cc42a0bd76d1d64db6d3c5

    # By index into the Objaverse UID list
    python visualize_sdf_diagnostics.py \\
        --checkpoint sdf_checkpoints/sdf_head_v0.61_100_epoch0100.pt \\
        --uid-indices 0 1 2 5

    # Local mesh files (OBJ / GLB / STL)
    python visualize_sdf_diagnostics.py \\
        --checkpoint sdf_checkpoints/sdf_head_v0.61_100_epoch0100.pt \\
        --meshes path/to/a.obj path/to/b.glb

    # No arguments at all: every value falls back to the CONFIGURATION
    # globals below (same pattern as train_sdf_head.py).
    python visualize_sdf_diagnostics.py

Optional flags
--------------
    --model                   TripoSR pretrained model ID    (default stabilityai/TripoSR)
    --output-dir              Renders + meshes directory      (default sdf_diagnostics_output/)
    --azimuth / --elevation / --fov / --size   Input-render camera params
    --mc-resolution           Marching-cubes grid resolution  (default 128)
    --sdf-clamp               TSDF clamp δ for clamped MSE    (default 0.1)
    --mse-points              Pass-1 uniform points for MSE   (default 32768)
    --near-surface-threshold / --near-surface-ratio / --near-surface-sigma
                              Pass-2 near-surface refinement knobs (see infer_sdf_mesh.py)
    --fscore-tau              F-score distance threshold      (default 0.01)
    --mesh-metric-samples     Surface samples for Chamfer/F-score (default 50000)
    --slice-axis              Initial cross-section axis (x/y/z) (default z)
    --slice-offset            Initial cross-section offset    (default 0.0)
    --slice-resolution        Slice heatmap grid resolution   (default 200)
    --port / --listen / --share   Gradio viewer options
"""

import argparse
import gc
import itertools
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYOPENGL_USE_ACCELERATE", "0")

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import gradio as gr
from PIL import Image

from train_sdf_head import (
    compute_sdf,
    create_mesh_comparison_visualization,
    fourier_encode,
    load_and_normalize_mesh,
    load_R_world_from_recon_json,
    mesh_surface_metrics,
    query_triplane_features,
    reconstruct_mesh_from_triplane,
    render_mesh_to_image,
    _camera_pose,
)
from infer_sdf_mesh import (
    apply_finetuned_lora,
    compute_pointwise_sdf_mse,
    fetch_objaverse_glb,
    load_sdf_mlp_from_checkpoint,
    uid_from_index,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — used for any value not passed on the command line
# ═══════════════════════════════════════════════════════════════════════════════

CHECKPOINT = "/home/markiv/TripoSR/sdf_checkpoints/sdf_head_v0.63_100_epoch0175.pt"
MODEL      = "stabilityai/TripoSR"

# Mesh source — the first non-empty of these three wins.
UIDS: list[str]        = ["b55517c209d74762b371bc6ce0d1e56f", "be4a1e32e9784960abe7f4ec21e7b6dc", "b3c309beb89d4188b68aad324f60b338", "add3ff3c6ece459fb0ff7e66b9fe14f8"]
UID_INDICES: list[int] = [0, 1, 2]
MESHES: list[str]      = []

OUTPUT_DIR    = "sdf_diagnostics_output"
AZIMUTH       = 45.0
ELEVATION     = 30.0
FOV           = 40.0
IMAGE_SIZE    = 256
MC_RESOLUTION = 128

SDF_CLAMP              = 0.1
MSE_POINTS             = 32768
NEAR_SURFACE_THRESHOLD = 0.05
NEAR_SURFACE_RATIO     = 1.0
NEAR_SURFACE_SIGMA     = 0.02

FSCORE_TAU          = 0.01
MESH_METRIC_SAMPLES = 50000

SLICE_AXIS       = "z"   # 'x' | 'y' | 'z' — initial cross-section plane
SLICE_OFFSET     = 0.0   # position along that axis, mesh units
SLICE_RESOLUTION = 200   # heatmap grid resolution (pixels per side)

PORT    = 7862
LISTEN  = False
SHARE   = True

# ═══════════════════════════════════════════════════════════════════════════════

_unique_id = itertools.count()  # cache-busting suffix for slider-updated image files


# ─── Slice computation ─────────────────────────────────────────────────────────

def _slice_axis_labels(axis: str) -> tuple[str, str]:
    return {"z": ("X", "Y"), "y": ("X", "Z"), "x": ("Y", "Z")}[axis]


def compute_sdf_slice(
    sdf_mlp,
    triplane: torch.Tensor,
    gt_mesh: trimesh.Trimesh,
    radius: float,
    feature_reduction: str,
    R_np: np.ndarray | None,
    n_freqs: int,
    use_triplane: bool,
    device: torch.device,
    axis: str = "z",
    offset: float = 0.0,
    resolution: int = 200,
    batch_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate GT and predicted SDF on a 2D axis-aligned slice through the volume.

    Returns ``(coords, gt_grid, pred_grid)``: ``coords`` is the shared 1D axis
    (for imshow extent / contour), ``gt_grid``/``pred_grid`` are
    ``(resolution, resolution)`` SDF value arrays in mesh/world frame.
    """
    coords = np.linspace(-radius, radius, resolution).astype(np.float32)
    a, b = np.meshgrid(coords, coords, indexing="xy")
    if axis == "z":
        pts = np.stack([a, b, np.full_like(a, offset, dtype=np.float32)], axis=-1)
    elif axis == "y":
        pts = np.stack([a, np.full_like(a, offset, dtype=np.float32), b], axis=-1)
    elif axis == "x":
        pts = np.stack([np.full_like(a, offset, dtype=np.float32), a, b], axis=-1)
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z' — got {axis!r}")
    pts_flat = pts.reshape(-1, 3)

    gt_grid = compute_sdf(gt_mesh, pts_flat.astype(np.float64)).reshape(resolution, resolution)

    pts_mesh = torch.from_numpy(pts_flat).to(device)
    if R_np is not None:
        R = torch.from_numpy(np.asarray(R_np, dtype=np.float32)).to(device)
        pts_trip_all = pts_mesh @ R.T
    else:
        pts_trip_all = pts_mesh

    triplane_dev = triplane.to(device)
    preds: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, pts_trip_all.shape[0], batch_size):
            batch = pts_trip_all[i : i + batch_size]
            if use_triplane:
                feats = query_triplane_features(batch, triplane_dev, radius, feature_reduction)
                if n_freqs > 0:
                    feats = torch.cat([feats, fourier_encode(batch, n_freqs)], dim=-1)
            else:
                feats = fourier_encode(batch, n_freqs) if n_freqs > 0 else batch
            preds.append(sdf_mlp(feats))
    pred_grid = torch.cat(preds).cpu().numpy().reshape(resolution, resolution)

    return coords, gt_grid, pred_grid


def render_slice_heatmap(
    coords: np.ndarray, gt_grid: np.ndarray, pred_grid: np.ndarray,
    axis: str, offset: float, radius: float, save_path: str,
) -> str:
    """GT | predicted SDF slice, diverging sign colormap, zero-contour overlaid."""
    xlabel, ylabel = _slice_axis_labels(axis)
    vmax = float(max(np.abs(gt_grid).max(), np.abs(pred_grid).max(), 1e-6))

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6))
    im = None
    for ax_, grid, title in zip(axes, (gt_grid, pred_grid), ("GT", "Predicted")):
        im = ax_.imshow(grid, extent=[-radius, radius, -radius, radius], origin="lower",
                         cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax_.contour(coords, coords, grid, levels=[0.0], colors="black", linewidths=1.1)
        ax_.set_title(title, fontsize=10)
        ax_.set_xlabel(xlabel, fontsize=8)
        ax_.set_ylabel(ylabel, fontsize=8)
        ax_.set_xticks([]); ax_.set_yticks([])
    fig.suptitle(f"SDF slice — {axis}={offset:.3f}", fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.85, label="SDF", pad=0.02)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return save_path


def render_sign_map(
    coords: np.ndarray, gt_grid: np.ndarray, pred_grid: np.ndarray,
    axis: str, offset: float, radius: float, save_path: str,
) -> str:
    """Slice colored red/green by sign(pred) == sign(GT); GT zero-contour overlaid."""
    xlabel, ylabel = _slice_axis_labels(axis)
    match = (np.sign(gt_grid) == np.sign(pred_grid)).astype(np.float32)
    acc = float(match.mean())

    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    ax.imshow(match, extent=[-radius, radius, -radius, radius], origin="lower",
              cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.contour(coords, coords, gt_grid, levels=[0.0], colors="black", linewidths=1.0)
    ax.set_title(f"Sign match — {axis}={offset:.3f}\n{acc * 100:.1f}% correct", fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ─── Surface error heatmap ──────────────────────────────────────────────────────

def render_surface_error_heatmap(
    pred_mesh: trimesh.Trimesh,
    gt_mesh: trimesh.Trimesh,
    save_path: str,
    image_size: int = 320,
    azimuth: float = 45.0,
    elevation: float = 30.0,
    fov: float = 40.0,
    cmap_name: str = "hot",
) -> tuple[str, float, float]:
    """Render ``pred_mesh`` with vertices colored by nearest-neighbor distance to
    ``gt_mesh``'s surface. Both meshes must already be in the same (world) frame.
    Returns ``(save_path, mean_dist, max_dist)``."""
    import pyrender

    prox = trimesh.proximity.ProximityQuery(gt_mesh)
    _, dist, _ = prox.on_surface(pred_mesh.vertices)
    vmax = max(float(np.percentile(dist, 95)) if dist.size else 1e-6, 1e-6)
    norm = np.clip(dist / vmax, 0.0, 1.0)
    rgba = (matplotlib.colormaps[cmap_name](norm) * 255).astype(np.uint8)

    colored = pred_mesh.copy()
    colored.visual.vertex_colors = rgba

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.4, 0.4, 0.4])
    # No explicit material -> pyrender derives it from the mesh's own vertex colors.
    pr_mesh = pyrender.Mesh.from_trimesh(colored, smooth=False)
    scene.add(pr_mesh)
    fov_rad = np.radians(fov)
    distance = 0.7 / np.tan(fov_rad / 2.0)
    T_cam = _camera_pose(azimuth, elevation, distance)
    scene.add(pyrender.PerspectiveCamera(yfov=fov_rad, aspectRatio=1.0), pose=T_cam)
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0), pose=T_cam)

    r = pyrender.OffscreenRenderer(image_size, image_size)
    color, _ = r.render(scene)
    r.delete()
    scene.clear()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    Image.fromarray(color).save(save_path)
    return save_path, float(dist.mean()), float(dist.max())


# ─── Per-object row ─────────────────────────────────────────────────────────────

def build_row(
    label: str,
    mesh_path: str,
    sdf_mlp,
    triposr_model,
    meta: dict,
    saved_args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    """Render, reconstruct, and diagnose ONE object. Returns a dict consumed by
    the Gradio grid (image paths, metrics markdown, and the live tensors/meshes
    the slice sliders need to recompute on demand)."""
    radius: float = meta["radius"]
    feature_reduction: str = meta["feature_reduction"]
    n_freqs: int = getattr(saved_args, "n_freqs", 0)
    use_triplane: bool = getattr(saved_args, "use_triplane_features", True)

    row_dir = output_dir / label
    row_dir.mkdir(parents=True, exist_ok=True)

    gt_mesh = load_and_normalize_mesh(mesh_path, radius)
    source_obj_path = str(row_dir / "source_mesh.obj")
    gt_mesh.export(source_obj_path)

    extrinsics_path = row_dir / "camera_extrinsics.json"
    render_path = str(row_dir / "input_render.png")
    pil_image = render_mesh_to_image(
        gt_mesh, elevation=args.elevation, fov=args.fov, size=args.size,
        azimuth=args.azimuth, extrinsics_json_path=str(extrinsics_path),
    )
    pil_image.save(render_path)
    image_np = np.array(Image.open(render_path).convert("RGB"))

    with torch.no_grad():
        scene_codes = triposr_model([image_np], device=device)
    triplane = scene_codes[0].float()

    R_np = load_R_world_from_recon_json(row_dir)
    if R_np is None:
        print(f"[warning] {label}: no camera extrinsics — not rotation-aligned.")

    pred_mesh = reconstruct_mesh_from_triplane(
        sdf_mlp, triplane, radius, feature_reduction,
        resolution=args.mc_resolution, device=device, n_freqs=n_freqs,
        R_world_from_trip=R_np, use_triplane_features=use_triplane,
    )
    if pred_mesh is None:
        print(f"[warning] {label}: marching cubes found no surface — skipping row.")
        return None
    pred_obj_path = str(row_dir / "sdf_mlp_mesh.obj")
    pred_mesh.export(pred_obj_path)

    # ── Metrics: two-pass pointwise MSE + Chamfer/F-score ──────────────────────
    mse_metrics = compute_pointwise_sdf_mse(
        sdf_mlp, triplane, gt_mesh, radius, feature_reduction,
        R_np, n_freqs, use_triplane, device,
        n_points=args.mse_points, sdf_clamp=args.sdf_clamp,
        near_surface_threshold=args.near_surface_threshold,
        near_surface_ratio=args.near_surface_ratio,
        near_surface_sigma=args.near_surface_sigma,
    )
    surf_metrics = mesh_surface_metrics(
        gt_mesh, pred_mesh, n_samples=args.mesh_metric_samples, fscore_tau=args.fscore_tau)

    metrics_md = (
        f"**{label[:16]}**\n\n"
        f"MSE (raw): `{mse_metrics['mse']:.6f}`  \n"
        f"MSE (clamped ±{mse_metrics['sdf_clamp']}): `{mse_metrics['mse_clamped']:.6f}`  \n"
        f"Chamfer-L2: `{surf_metrics['chamfer']:.6f}`  \n"
        f"F-score@{args.fscore_tau}: `{surf_metrics['fscore']:.3f}`  \n"
        f"[{mse_metrics['n_points_uniform']:,} + {mse_metrics['n_points_near_surface']:,} pts]"
        + ("" if mse_metrics["rotation_aligned"] else "  \n⚠️ not rotation-aligned")
    )

    # ── Overview: GT vs SDF-MLP render + input image ────────────────────────────
    overview_path = create_mesh_comparison_visualization(
        gt_mesh, pred_mesh, title=label[:24], save_path=row_dir / "overview.png",
        phi_values=(45, 225), input_image=image_np,
    )

    # ── Surface error heatmap ───────────────────────────────────────────────────
    surf_err_path, err_mean, err_max = render_surface_error_heatmap(
        pred_mesh, gt_mesh, str(row_dir / "surface_error.png"),
        azimuth=args.azimuth, elevation=args.elevation, fov=args.fov,
    )
    metrics_md += f"  \nSurface err: mean `{err_mean:.4f}` / max `{err_max:.4f}`"

    # ── Initial slice + sign map ────────────────────────────────────────────────
    coords, gt_grid, pred_grid = compute_sdf_slice(
        sdf_mlp, triplane, gt_mesh, radius, feature_reduction, R_np, n_freqs,
        use_triplane, device, axis=args.slice_axis, offset=args.slice_offset,
        resolution=args.slice_resolution,
    )
    slice_path = render_slice_heatmap(
        coords, gt_grid, pred_grid, args.slice_axis, args.slice_offset, radius,
        str(row_dir / f"slice_{next(_unique_id)}.png"))
    sign_path = render_sign_map(
        coords, gt_grid, pred_grid, args.slice_axis, args.slice_offset, radius,
        str(row_dir / f"sign_{next(_unique_id)}.png"))

    return {
        "label": label,
        "row_dir": row_dir,
        "radius": radius,
        "feature_reduction": feature_reduction,
        "n_freqs": n_freqs,
        "use_triplane": use_triplane,
        "triplane": triplane,
        "gt_mesh": gt_mesh,
        "R_np": R_np,
        "overview_path": overview_path,
        "surface_error_path": surf_err_path,
        "slice_path": slice_path,
        "sign_path": sign_path,
        "metrics_md": metrics_md,
    }


# ─── Gradio grid viewer ─────────────────────────────────────────────────────────

def launch_grid_viewer(
    rows: list[dict],
    sdf_mlp,
    device: torch.device,
    output_dir: Path,
    default_axis: str,
    default_offset: float,
    resolution: int,
    port: int,
    listen: bool,
    share: bool,
) -> None:
    def _recompute_row(row: dict, axis: str, offset: float) -> tuple[str, str]:
        coords, gt_grid, pred_grid = compute_sdf_slice(
            sdf_mlp, row["triplane"], row["gt_mesh"], row["radius"],
            row["feature_reduction"], row["R_np"], row["n_freqs"],
            row["use_triplane"], device, axis=axis, offset=offset, resolution=resolution,
        )
        slice_path = render_slice_heatmap(
            coords, gt_grid, pred_grid, axis, offset, row["radius"],
            str(row["row_dir"] / f"slice_{next(_unique_id)}.png"))
        sign_path = render_sign_map(
            coords, gt_grid, pred_grid, axis, offset, row["radius"],
            str(row["row_dir"] / f"sign_{next(_unique_id)}.png"))
        return slice_path, sign_path

    def _on_slider_change(axis: str, offset: float):
        outs: list[str] = []
        for row in rows:
            s_path, sg_path = _recompute_row(row, axis, offset)
            outs.extend([s_path, sg_path])
        return outs

    radius_max = max(r["radius"] for r in rows)

    with gr.Blocks(title="SDF Model Diagnostics") as app:
        gr.Markdown(
            "# SDF Model Diagnostics\n"
            "One row per object, one column per visualization. The slice axis "
            "and offset are shared — moving them updates every row at once."
        )
        with gr.Row():
            axis_radio = gr.Radio(choices=["x", "y", "z"], value=default_axis, label="Slice axis")
            offset_slider = gr.Slider(minimum=-radius_max, maximum=radius_max,
                                      value=default_offset, step=radius_max / 100,
                                      label="Slice offset")
            update_btn = gr.Button("Update slices")

        with gr.Row():
            gr.Markdown("**Object**")
            gr.Markdown("**Overview**")
            gr.Markdown("**SDF slice (GT | Pred)**")
            gr.Markdown("**Sign match**")
            gr.Markdown("**Surface error**")
            gr.Markdown("**Metrics**")

        slice_imgs: list[gr.Image] = []
        sign_imgs: list[gr.Image] = []
        for row in rows:
            with gr.Row(equal_height=True):
                gr.Markdown(f"`{row['label'][:18]}`")
                gr.Image(value=row["overview_path"], show_label=False, height=200)
                s_img = gr.Image(value=row["slice_path"], show_label=False, height=200)
                sg_img = gr.Image(value=row["sign_path"], show_label=False, height=200)
                gr.Image(value=row["surface_error_path"], show_label=False, height=200)
                gr.Markdown(row["metrics_md"])
            slice_imgs.append(s_img)
            sign_imgs.append(sg_img)

        all_outputs = [c for pair in zip(slice_imgs, sign_imgs) for c in pair]
        axis_radio.change(_on_slider_change, inputs=[axis_radio, offset_slider], outputs=all_outputs)
        offset_slider.release(_on_slider_change, inputs=[axis_radio, offset_slider], outputs=all_outputs)
        update_btn.click(_on_slider_change, inputs=[axis_radio, offset_slider], outputs=all_outputs)

    allowed = [str(output_dir)]
    launch_kwargs = dict(
        server_name="0.0.0.0" if listen else "localhost",
        server_port=port,
        share=share,
    )
    try:
        app.launch(allowed_paths=allowed, **launch_kwargs)
    except TypeError:
        app.launch(**launch_kwargs)


# ─── Main pipeline ───────────────────────────────────────────────────────────

def run_diagnostics(args: argparse.Namespace) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Diagnostics on {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = str(output_dir / "objaverse_cache")

    print(f"\n[1/3] Loading SDF MLP checkpoint: {args.checkpoint}")
    sdf_mlp, meta, saved_args = load_sdf_mlp_from_checkpoint(args.checkpoint, device)

    print(f"\n[2/3] Loading TripoSR ({args.model})...")
    from tsr.system import TSR
    triposr_model = TSR.from_pretrained(args.model, config_name="config.yaml", weight_name="model.ckpt")
    triposr_model.renderer.set_chunk_size(8192)
    triposr_model.to(device).eval()
    for p in triposr_model.parameters():
        p.requires_grad_(False)
    # REQUIRED: the SDF MLP was trained on triplanes from the LoRA fine-tuned
    # TripoSR, not the stock one. Without this every diagnostic below (slices,
    # sign maps, surface error, MSE) is computed on out-of-distribution features
    # and looks far worse than train_sdf_head's own wandb visualizations, which
    # reuse the live fine-tuned model in memory.
    apply_finetuned_lora(triposr_model, args.checkpoint, device)

    # ── Resolve every mesh source to (label, path) pairs ────────────────────────
    sources: list[tuple[str, str]] = []
    if args.uids:
        for uid in args.uids:
            sources.append((uid, fetch_objaverse_glb(uid, cache_dir)))
    elif args.uid_indices:
        for idx in args.uid_indices:
            uid = uid_from_index(idx)
            sources.append((uid, fetch_objaverse_glb(uid, cache_dir)))
    elif args.meshes:
        for m in args.meshes:
            mesh_path = os.path.abspath(os.path.expanduser(m))
            if not os.path.exists(mesh_path):
                raise FileNotFoundError(f"Mesh not found: {mesh_path}")
            sources.append((Path(mesh_path).stem, mesh_path))
    else:
        raise ValueError("No mesh sources resolved — pass --uids / --uid-indices / --meshes, "
                          "or set UIDS / UID_INDICES / MESHES globals.")

    print(f"\n[3/3] Building {len(sources)} row(s): "
          f"{', '.join(label[:12] for label, _ in sources)}")
    rows: list[dict] = []
    for label, mesh_path in sources:
        print(f"\n[row] {label} ({mesh_path})")
        _t = time.perf_counter()
        row = build_row(label, mesh_path, sdf_mlp, triposr_model, meta, saved_args,
                        device, output_dir, args)
        if row is not None:
            rows.append(row)
            print(f"[row] {label} done in {time.perf_counter() - _t:.1f}s — {row['metrics_md']!s}".replace("\n", "  "))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        print("\n[error] No rows produced a valid reconstruction — nothing to show.")
        return

    print("\nLaunching Gradio grid viewer...")
    launch_grid_viewer(
        rows, sdf_mlp, device, output_dir,
        default_axis=args.slice_axis, default_offset=args.slice_offset,
        resolution=args.slice_resolution,
        port=args.port, listen=args.listen, share=args.share,
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SDF model diagnostics: cross-section slices, sign maps, surface "
                     "error heatmaps, and MSE/Chamfer/F-score for one checkpoint across "
                     "multiple objects (Gradio grid: rows = objects, columns = visualizations).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--uids", nargs="+", default=None,
                        help="Objaverse UIDs to download and diagnose (one row each).")
    source.add_argument("--uid-indices", nargs="+", type=int, default=None, metavar="N",
                        help="Indices into the Objaverse UID list (0-based).")
    source.add_argument("--meshes", nargs="+", default=None,
                        help="Local mesh files (OBJ / GLB / STL).")

    parser.add_argument("--checkpoint", default=None,
                        help="Path to the saved SDF MLP checkpoint (.pt). "
                             f"Default: CHECKPOINT global ({CHECKPOINT}).")
    parser.add_argument("--model", default=None, help="TripoSR pretrained model ID or local path.")
    parser.add_argument("--output-dir", default=None, help="Directory for renders and output meshes.")
    parser.add_argument("--azimuth", type=float, default=None, help="Camera azimuth in degrees.")
    parser.add_argument("--elevation", type=float, default=None, help="Camera elevation in degrees.")
    parser.add_argument("--fov", type=float, default=None, help="Vertical field of view in degrees.")
    parser.add_argument("--size", type=int, default=None, help="Render image size in pixels.")
    parser.add_argument("--mc-resolution", type=int, default=None, help="Marching-cubes grid resolution.")
    parser.add_argument("--sdf-clamp", type=float, default=None,
                        help="TSDF clamp delta for the reported clamped MSE. Set 0 to skip.")
    parser.add_argument("--mse-points", type=int, default=None,
                        help="Pass-1 uniform points for the pointwise SDF MSE.")
    parser.add_argument("--near-surface-threshold", type=float, default=None,
                        help="Pass 2: |pred SDF| below this reseeds a refinement point.")
    parser.add_argument("--near-surface-ratio", type=float, default=None,
                        help="Pass 2: refinement points requested = ratio * --mse-points.")
    parser.add_argument("--near-surface-sigma", type=float, default=None,
                        help="Pass 2: Gaussian noise std for reseeded points.")
    parser.add_argument("--fscore-tau", type=float, default=None, help="F-score distance threshold.")
    parser.add_argument("--mesh-metric-samples", type=int, default=None,
                        help="Surface samples for Chamfer / F-score.")
    parser.add_argument("--slice-axis", choices=("x", "y", "z"), default=None,
                        help="Initial cross-section axis.")
    parser.add_argument("--slice-offset", type=float, default=None,
                        help="Initial cross-section offset along that axis.")
    parser.add_argument("--slice-resolution", type=int, default=None,
                        help="Slice heatmap grid resolution (pixels per side).")
    parser.add_argument("--port", type=int, default=None, help="Gradio viewer port.")
    parser.add_argument("--listen", action="store_true", help="Bind Gradio to 0.0.0.0.")
    parser.add_argument("--share", action="store_true", help="Force a public Gradio share link.")

    args = parser.parse_args()

    def _dflt(v, fallback):
        return fallback if v is None else v

    # Mesh source: if NONE of --uids/--uid-indices/--meshes were passed at all,
    # fall back entirely to the globals (first non-empty of UIDS/UID_INDICES/MESHES).
    if args.uids is None and args.uid_indices is None and args.meshes is None:
        if UIDS:
            args.uids = UIDS
        elif UID_INDICES:
            args.uid_indices = UID_INDICES
        elif MESHES:
            args.meshes = MESHES

    args.checkpoint    = os.path.abspath(os.path.expanduser(_dflt(args.checkpoint, CHECKPOINT)))
    args.model         = _dflt(args.model, MODEL)
    args.output_dir    = os.path.abspath(_dflt(args.output_dir, OUTPUT_DIR))
    args.azimuth       = _dflt(args.azimuth, AZIMUTH)
    args.elevation     = _dflt(args.elevation, ELEVATION)
    args.fov           = _dflt(args.fov, FOV)
    args.size          = _dflt(args.size, IMAGE_SIZE)
    args.mc_resolution = _dflt(args.mc_resolution, MC_RESOLUTION)
    args.sdf_clamp     = _dflt(args.sdf_clamp, SDF_CLAMP)
    args.mse_points    = _dflt(args.mse_points, MSE_POINTS)
    args.near_surface_threshold = _dflt(args.near_surface_threshold, NEAR_SURFACE_THRESHOLD)
    args.near_surface_ratio     = _dflt(args.near_surface_ratio, NEAR_SURFACE_RATIO)
    args.near_surface_sigma     = _dflt(args.near_surface_sigma, NEAR_SURFACE_SIGMA)
    args.fscore_tau             = _dflt(args.fscore_tau, FSCORE_TAU)
    args.mesh_metric_samples    = _dflt(args.mesh_metric_samples, MESH_METRIC_SAMPLES)
    args.slice_axis             = _dflt(args.slice_axis, SLICE_AXIS)
    args.slice_offset           = _dflt(args.slice_offset, SLICE_OFFSET)
    args.slice_resolution       = _dflt(args.slice_resolution, SLICE_RESOLUTION)
    args.port    = _dflt(args.port, PORT)
    args.listen  = args.listen or LISTEN
    args.share   = args.share or SHARE

    if not (args.uids or args.uid_indices or args.meshes):
        raise SystemExit(
            "No mesh sources: pass --uids / --uid-indices / --meshes, or set "
            "UIDS / UID_INDICES / MESHES in the CONFIGURATION block at the top of this file."
        )

    run_diagnostics(args)


if __name__ == "__main__":
    main()
