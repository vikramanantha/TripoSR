"""
infer_old.py  —  SDF MLP inference on a new mesh (BATCH pipeline).

Snapshot of infer_sdf_mesh.py BEFORE the interactive UI-first viewer, the
live X/Y/Z preview sliders, the repaired/watertight-mesh matching, and the
greyscale (material-stripping) option were added. Computes everything first,
then opens the three-way Gradio viewer, as it originally did.

Kept from the later work (these were bug fixes / explicitly requested, not
part of the reverted features):
  * apply_finetuned_lora() -- loads ckpt["lora_model"]. Without it the SDF MLP
    is fed STOCK triplanes it was never trained on and the output is garbage.
  * two-pass pointwise SDF MSE (raw + TSDF-clamped).
  * CONFIGURATION globals so it runs with no CLI arguments.
  * NeRF baseline decodes the STOCK triplane (unmodified TripoSR end to end).

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

    # No arguments at all: every value falls back to the CONFIGURATION
    # globals below (same pattern as train_sdf_head.py / visualize_sdf_diagnostics.py).
    python infer_sdf_mesh.py

Optional flags
--------------
    --checkpoint      SDF MLP checkpoint (.pt)              (default CHECKPOINT global)
    --model           TripoSR pretrained model ID          (default stabilityai/TripoSR)
    --output-dir      Directory for renders + meshes       (default infer_output/)
    --azimuth         Camera azimuth in degrees            (default 45)
    --elevation       Camera elevation in degrees          (default 30)
    --fov             Vertical field of view in degrees    (default 40)
    --size            Render image size in pixels          (default 256)
    --mc-resolution   Marching-cubes grid resolution       (default 128)
    --nerf-threshold  Density threshold for NeRF MC        (default 25.0)
    --sdf-clamp / --mse-points / --near-surface-*   Two-pass pointwise-MSE knobs
    --port            Gradio viewer port                   (default 7861)
    --listen          Bind Gradio to 0.0.0.0
    --share           Create a public Gradio share link

If no CLI arguments are given at all, every value falls back to the GLOBAL
variables defined below (same pattern as train_sdf_head.py's CONFIGURATION
block and visualize_sdf_diagnostics.py).
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
import torch.nn.functional as F
import trimesh
import gradio as gr
from PIL import Image

# Reuse core utilities from train_sdf_head (no model loading side-effects on import)
from train_sdf_head import (
    REPAIR_VOXEL_METHOD,
    REPAIR_VOXEL_RES,
    SDFMLP,
    _load_trimesh,
    _normalize_mesh_copy,
    compute_sdf,
    fourier_encode,
    load_and_normalize_mesh,
    repair_mesh_watertight,
    load_R_world_from_recon_json,
    mesh_surface_metrics,
    query_triplane_features,
    reconstruct_mesh_from_triplane,
    reconstruct_mesh_nerf_decoder,
    render_mesh_to_image,
    stock_triposr,
)

# Reuse viewer helpers from view_mesh
from view_mesh import (
    _axis_marker_mesh,
    _copy_mesh_solid_color,
    _mesh_with_axes,
    _scale_grid_mesh,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — used for any value not passed on the command line
# ═══════════════════════════════════════════════════════════════════════════════

CHECKPOINT = "/home/markiv/TripoSR/sdf_checkpoints/sdf_head_v0.61_100_epoch0175.pt"
MODEL      = "stabilityai/TripoSR"

# Mesh source — the first non-None of these three wins.
# 91ceac9fa0d14f6a9813765ea2e21b1b -- rectangle
# 97a038cd7a304bce81890c118fadd793 -- frog
UID: str | None       = "91ceac9fa0d14f6a9813765ea2e21b1b" # "b55517c209d74762b371bc6ce0d1e56f" # 85739db9b70e47c28be0340e2d1907b7
UID_INDEX: int | None = 1
MESH: str | None      = None

OUTPUT_DIR     = "infer_output"
AZIMUTH        = 0.0
ELEVATION      = 30.0
FOV            = 40.0
IMAGE_SIZE     = 256
MC_RESOLUTION  = 128
NERF_THRESHOLD = 25.0

SDF_CLAMP              = 0.1
MSE_POINTS             = 32768
NEAR_SURFACE_THRESHOLD = 0.05
NEAR_SURFACE_RATIO     = 1.0
NEAR_SURFACE_SIGMA     = 0.02

PORT    = 7860
LISTEN  = False
SHARE   = True

# ═══════════════════════════════════════════════════════════════════════════════


# ─── Objaverse helpers ────────────────────────────────────────────────────────

def uid_from_index(index: int) -> str:
    import objaverse
    from objaverse_paths import configure_objaverse
    configure_objaverse()
    print("[objaverse] Loading UID list...")
    uids_list = list(objaverse.load_uids())
    if not (0 <= index < len(uids_list)):
        raise IndexError(f"--uid-index {index} out of range ({len(uids_list)} objects)")
    uid = uids_list[index]
    print(f"[objaverse] UID at index {index}: {uid}")
    return uid


def fetch_objaverse_glb(uid: str, cache_dir: str) -> str:
    import objaverse
    from objaverse_paths import configure_objaverse
    configure_objaverse()
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


def apply_finetuned_lora(triposr, checkpoint_path: str, device: torch.device) -> bool:
    """Convert a stock TripoSR **in place** into the LoRA-adapted model that
    actually produced the triplanes the SDF MLP was trained on.

    This is REQUIRED, not optional. train_sdf_head fine-tunes TripoSR jointly
    with the SDF head (LoRA on backbone blocks + a fully unfrozen
    post_processor) and saves those weights as ckpt["lora_model"]. Decoding a
    STOCK TripoSR triplane with that SDF MLP feeds it a feature distribution it
    never saw in training, and the marching-cubes output is garbage — which is
    exactly what this script produced before, while train_sdf_head's own wandb
    visualizations looked fine (they reuse the live fine-tuned model in memory).

    Ordering matters: apply_lora_to_triposr must run BEFORE load_state_dict so
    the injected module names (base.weight / lora_A / lora_B) exist to match the
    saved keys. Returns True if fine-tuned weights were applied.
    """
    # apply_lora_to_triposr snapshots the pristine post_processor internally,
    # before any fine-tuned weights land, so stock_triposr() works afterwards.
    from train_sdf_head import apply_lora_to_triposr

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "lora_model" not in ckpt:
        print("[lora] checkpoint has no 'lora_model' — using stock TripoSR "
              "(correct only for checkpoints trained against a frozen TripoSR).")
        return False

    ck_args = ckpt["args"]
    apply_lora_to_triposr(
        triposr,
        ck_args["lora_block_start"], ck_args["lora_block_end"],
        ck_args["lora_rank"], ck_args["lora_alpha"],
    )
    triposr.to(device)
    missing, unexpected = triposr.load_state_dict(ckpt["lora_model"], strict=False)
    # strict=False is needed (buffers like image_mean/std are non-persistent),
    # but a LoRA/post_processor key going missing means a silent mismatch, so
    # surface those specifically rather than trusting the load blindly.
    sus = [k for k in missing if "lora_" in k or k.startswith("post_processor")]
    if sus:
        print(f"[lora] WARNING: {len(sus)} fine-tuned keys did not load "
              f"(e.g. {sus[:3]}) — reconstruction may be wrong.")
    n_lora = sum(1 for k in ckpt["lora_model"] if ".lora_A" in k or ".lora_B" in k)
    print(f"[lora] Applied fine-tuned TripoSR from checkpoint "
          f"(blocks {ck_args['lora_block_start']}-{ck_args['lora_block_end'] - 1}, "
          f"rank={ck_args['lora_rank']}, {n_lora} LoRA tensors + post_processor).")
    for p in triposr.parameters():
        p.requires_grad_(False)
    triposr.eval()
    return True


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


# ─── Pointwise SDF MSE (raw + TSDF-clamped) ──────────────────────────────────

def compute_pointwise_sdf_mse(
    sdf_mlp: SDFMLP,
    triplane: torch.Tensor,
    mesh: trimesh.Trimesh,
    radius: float,
    feature_reduction: str,
    R_np: np.ndarray | None,
    n_freqs: int,
    use_triplane: bool,
    device: torch.device,
    n_points: int = 32768,
    sdf_clamp: float = 0.1,
    batch_size: int = 32768,
    near_surface_threshold: float = 0.05,
    near_surface_ratio: float = 1.0,
    near_surface_sigma: float = 0.02,
) -> dict:
    """Two-pass pointwise SDF MSE of the trained MLP vs the GT source mesh.

    Pass 1 (uniform): sample ``n_points`` uniformly in [-radius, radius]^3
    (mesh/world frame, same convention as training), compute GT SDF directly
    on the mesh, and query the MLP through the SAME recon-frame rotation used
    at training time (``p_trip = p_mesh @ R.T``).

    Pass 2 (near-surface): flag Pass-1 points where |pred| <
    ``near_surface_threshold``, resample them (with replacement, up to
    ``near_surface_ratio`` × n_points) with Gaussian jitter
    (``near_surface_sigma``), recompute exact GT SDF on the mesh at the
    jittered locations, and query the MLP there too. This mirrors the
    model's own former "train how you test" near-surface refinement, now run
    once at eval time rather than online every training step (training now
    uses a fixed precomputed point mix — see NEAR_SURFACE_FRACTION in
    train_sdf_head.py — but a single inference call has no per-step DDP time
    budget to worry about, so the full two-pass probe-and-refine is cheap
    here and gives the surface-focused number this is actually judged by).

    Final MSE / clamped MSE are computed over the COMBINED uniform +
    near-surface pool, so the reported number isn't diluted by far-field
    points the model was never expected to get exactly right.

    ``sdf_clamp`` is applied post-hoc here regardless of what SDF_CLAMP (if
    any) the checkpoint was actually trained with — so the clamped number is
    directly comparable across checkpoints trained with different (or no)
    clamping, unlike the checkpoint's own logged training-time metrics.
    """
    R = (torch.from_numpy(np.asarray(R_np, dtype=np.float32)).to(device)
         if R_np is not None else None)
    triplane_dev = triplane.to(device)

    def _predict(pts_mesh: torch.Tensor) -> torch.Tensor:
        pts_trip = pts_mesh @ R.T if R is not None else pts_mesh
        preds: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, pts_trip.shape[0], batch_size):
                batch = pts_trip[i : i + batch_size]
                if use_triplane:
                    feats = query_triplane_features(batch, triplane_dev, radius, feature_reduction)
                    if n_freqs > 0:
                        feats = torch.cat([feats, fourier_encode(batch, n_freqs)], dim=-1)
                else:
                    feats = fourier_encode(batch, n_freqs) if n_freqs > 0 else batch
                preds.append(sdf_mlp(feats))
        return torch.cat(preds)

    # ── Pass 1: uniform probe ──────────────────────────────────────────────
    pts_uniform_np = np.random.uniform(-radius, radius, (n_points, 3)).astype(np.float32)
    sdf_gt_uniform = torch.from_numpy(compute_sdf(mesh, pts_uniform_np)).to(device)
    pts_uniform = torch.from_numpy(pts_uniform_np).to(device)
    sdf_pred_uniform = _predict(pts_uniform)

    # ── Pass 2: resample near the surface, recompute exact GT there ───────
    near_mask = sdf_pred_uniform.abs() < near_surface_threshold
    n_near_found = int(near_mask.sum().item())
    n_near_used = 0

    if n_near_found > 0:
        target = max(1, int(n_points * near_surface_ratio))
        idx = torch.randint(0, n_near_found, (target,), device=device)
        seeds = pts_uniform[near_mask][idx]
        pts_near = (seeds + torch.randn_like(seeds) * near_surface_sigma
                    ).clamp(-radius, radius)
        sdf_gt_near = torch.from_numpy(
            compute_sdf(mesh, pts_near.detach().cpu().numpy().astype(np.float64))
        ).to(device)
        sdf_pred_near = _predict(pts_near)
        sdf_pred = torch.cat([sdf_pred_uniform, sdf_pred_near])
        sdf_gt = torch.cat([sdf_gt_uniform, sdf_gt_near])
        n_near_used = pts_near.shape[0]
    else:
        sdf_pred, sdf_gt = sdf_pred_uniform, sdf_gt_uniform

    mse = float(F.mse_loss(sdf_pred, sdf_gt).item())
    c = float(sdf_clamp)
    mse_clamped = (float(F.mse_loss(sdf_pred.clamp(-c, c), sdf_gt.clamp(-c, c)).item())
                   if c > 0 else mse)

    return {
        "mse": mse,
        "mse_clamped": mse_clamped,
        "sdf_clamp": c,
        "n_points": int(sdf_pred.shape[0]),
        "n_points_uniform": n_points,
        "n_points_near_surface": n_near_used,
        "radius": radius,
        "rotation_aligned": R_np is not None,
    }


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
    metrics_md: str | None = None,
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
        if metrics_md:
            gr.Markdown(metrics_md)

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

    # Adapt TripoSR to the checkpoint's fine-tuned weights, then extract TWO
    # triplanes from the same instance:
    #   • fine-tuned -> the SDF MLP (it was trained only on these)
    #   • stock      -> the NeRF decoder, giving a genuine
    #                   "original TripoSR vs. my model" before/after
    # stock_triposr() disables the LoRA deltas and swaps the pretrained
    # post_processor back in, so no second 427M-param model is needed.
    _t = time.perf_counter()
    is_finetuned = apply_finetuned_lora(triposr, args.checkpoint, device)
    timing["Fine-tuned weight load"] = time.perf_counter() - _t

    from train_sdf_head import stock_triposr
    _t = time.perf_counter()
    with torch.no_grad():
        triplane = triposr([image_np], device=device)[0].float()
        with stock_triposr(triposr):
            triplane_stock = triposr([image_np], device=device)[0].float()
    timing["TripoSR triplane extraction"] = time.perf_counter() - _t
    print(f"[triposr] Triplane shape: {tuple(triplane.shape)} "
          f"({'FINE-TUNED' if is_finetuned else 'STOCK'} weights)")
    if is_finetuned:
        d = (triplane - triplane_stock).abs().mean().item()
        s = triplane_stock.abs().mean().item()
        print(f"[triposr] fine-tuned vs stock triplane: mean|Δ|={d:.4f} "
              f"(stock mean|x|={s:.4f}, relative {d / max(s, 1e-9):.1%})")

    # R maps TripoSR recon frame → normalised mesh world frame.
    # load_R_world_from_recon_json expects the directory containing camera_extrinsics.json.
    R_np = load_R_world_from_recon_json(output_dir)
    if R_np is None:
        print("[warning] Could not load camera extrinsics — meshes will be in TripoSR recon space.")

    print(f"\n[metric] Pointwise SDF MSE — two-pass (δ={args.sdf_clamp}, "
          f"{args.mse_points:,} uniform + up to "
          f"{int(args.mse_points * args.near_surface_ratio):,} near-surface)...")
    _t = time.perf_counter()
    mse_metrics = compute_pointwise_sdf_mse(
        sdf_mlp, triplane, mesh, radius, feature_reduction,
        R_np, n_freqs, use_triplane, device,
        n_points=args.mse_points, sdf_clamp=args.sdf_clamp,
        near_surface_threshold=args.near_surface_threshold,
        near_surface_ratio=args.near_surface_ratio,
        near_surface_sigma=args.near_surface_sigma,
    )
    timing["Pointwise SDF MSE"] = time.perf_counter() - _t
    print(f"[metric] MSE (raw) = {mse_metrics['mse']:.6f}   "
          f"MSE (clamped ±{mse_metrics['sdf_clamp']}) = {mse_metrics['mse_clamped']:.6f}   "
          f"[{mse_metrics['n_points_uniform']:,} uniform + "
          f"{mse_metrics['n_points_near_surface']:,} near-surface]"
          + ("" if mse_metrics["rotation_aligned"]
             else "  ⚠️ no camera_extrinsics.json — NOT rotation-aligned, values inflated"))
    metrics_md = (
        "### Pointwise SDF MSE vs ground truth (two-pass)\n\n"
        "| Metric | Value |\n|:---|---:|\n"
        f"| MSE (raw) | `{mse_metrics['mse']:.6f}` |\n"
        f"| MSE (clamped ±{mse_metrics['sdf_clamp']}) | `{mse_metrics['mse_clamped']:.6f}` |\n\n"
        f"Pass 1: {mse_metrics['n_points_uniform']:,} points uniform in "
        f"[−{radius:.3f}, {radius:.3f}]³. Pass 2: {mse_metrics['n_points_near_surface']:,} "
        f"points reseeded near where Pass 1 predicted |SDF| < "
        f"{args.near_surface_threshold} and rechecked against exact GT. Both passes "
        f"pooled for the MSE above, so the number reflects near-surface accuracy "
        f"rather than being diluted by far-field points. The clamped value is applied "
        f"post-hoc with this script's `--sdf-clamp`, independent of whatever SDF_CLAMP "
        f"(if any) this checkpoint was trained with — so it is directly comparable "
        f"across checkpoints. "
        + ("Rotation-aligned via `camera_extrinsics.json`."
           if mse_metrics["rotation_aligned"] else
           "⚠️ No `camera_extrinsics.json` — **not** rotation-aligned, values inflated.")
    )

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
        # STOCK triplane + the original (frozen) density head = unmodified
        # TripoSR end to end, so this pane is a true baseline to compare the
        # fine-tuned SDF model against. Matches train_sdf_head's NeRF viz row.
        triplane_stock,
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
        return mse_metrics

    print("\nLaunching Gradio viewer...")
    launch_three_way_viewer(
        source_mesh_path=source_obj_path,
        nerf_mesh_path=nerf_obj_path,
        sdf_mesh_path=sdf_obj_path,
        render_image_path=render_path,
        output_dir=str(output_dir),
        timing=timing,
        metrics_md=metrics_md,
        port=args.port,
        listen=args.listen,
        share=args.share,
    )
    return mse_metrics


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SDF MLP inference: render mesh → TripoSR triplane → SDF MLP + NeRF → Gradio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group()
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
        "--checkpoint", default=None,
        help=f"Path to the saved SDF MLP checkpoint (.pt). Default: CHECKPOINT global ({CHECKPOINT}).",
    )
    parser.add_argument(
        "--model", default=None,
        help="TripoSR pretrained model ID or local path.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for renders and output meshes.",
    )
    parser.add_argument(
        "--azimuth", type=float, default=None,
        help="Camera azimuth in degrees.",
    )
    parser.add_argument(
        "--elevation", type=float, default=None,
        help="Camera elevation in degrees.",
    )
    parser.add_argument(
        "--fov", type=float, default=None,
        help="Vertical field of view in degrees.",
    )
    parser.add_argument(
        "--size", type=int, default=None,
        help="Render image size in pixels (matched to precompute IMAGE_SIZE).",
    )
    parser.add_argument(
        "--mc-resolution", type=int, default=None,
        help="Marching-cubes grid resolution.",
    )
    parser.add_argument(
        "--nerf-threshold", type=float, default=None,
        help="Density threshold for the NeRF decoder marching cubes.",
    )
    parser.add_argument(
        "--sdf-clamp", type=float, default=None,
        help="TSDF clamp delta for the reported clamped MSE (pred & GT SDF clamped to "
             "+/-delta before computing MSE). Applied post-hoc regardless of what "
             "SDF_CLAMP the checkpoint was trained with, so this number is comparable "
             "across checkpoints. Set 0 to skip (clamped MSE falls back to the raw MSE).",
    )
    parser.add_argument(
        "--mse-points", type=int, default=None,
        help="Number of Pass-1 uniform points (in [-radius, radius]^3) used for the "
             "pointwise SDF MSE. Pass 2 adds near-surface refinement points on top.",
    )
    parser.add_argument(
        "--near-surface-threshold", type=float, default=None,
        help="Pass 2: Pass-1 points with |pred SDF| below this are treated as "
             "near-surface and reseeded for refinement. Set 0 to disable Pass 2 "
             "(MSE falls back to Pass-1-only, uniform points).",
    )
    parser.add_argument(
        "--near-surface-ratio", type=float, default=None,
        help="Pass 2: number of refinement points requested = ratio * --mse-points "
             "(1.0 = as many near-surface points as uniform ones).",
    )
    parser.add_argument(
        "--near-surface-sigma", type=float, default=None,
        help="Pass 2: Gaussian noise std used to jitter reseeded near-surface points.",
    )
    parser.add_argument(
        "--port", type=int, default=None,
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

    def _dflt(v, fallback):
        return fallback if v is None else v

    # Mesh source: if NONE of --uid/--uid-index/--mesh were passed at all,
    # fall back entirely to the globals (first non-None of UID/UID_INDEX/MESH).
    if args.uid is None and args.uid_index is None and args.mesh is None:
        if UID is not None:
            args.uid = UID
        elif UID_INDEX is not None:
            args.uid_index = UID_INDEX
        elif MESH is not None:
            args.mesh = MESH

    args.checkpoint    = os.path.abspath(os.path.expanduser(_dflt(args.checkpoint, CHECKPOINT)))
    args.model         = _dflt(args.model, MODEL)
    args.output_dir    = os.path.abspath(_dflt(args.output_dir, OUTPUT_DIR))
    args.azimuth       = _dflt(args.azimuth, AZIMUTH)
    args.elevation     = _dflt(args.elevation, ELEVATION)
    args.fov           = _dflt(args.fov, FOV)
    args.size          = _dflt(args.size, IMAGE_SIZE)
    args.mc_resolution = _dflt(args.mc_resolution, MC_RESOLUTION)
    args.nerf_threshold = _dflt(args.nerf_threshold, NERF_THRESHOLD)
    args.sdf_clamp      = _dflt(args.sdf_clamp, SDF_CLAMP)
    args.mse_points     = _dflt(args.mse_points, MSE_POINTS)
    args.near_surface_threshold = _dflt(args.near_surface_threshold, NEAR_SURFACE_THRESHOLD)
    args.near_surface_ratio     = _dflt(args.near_surface_ratio, NEAR_SURFACE_RATIO)
    args.near_surface_sigma     = _dflt(args.near_surface_sigma, NEAR_SURFACE_SIGMA)
    args.port    = _dflt(args.port, PORT)
    args.listen  = args.listen or LISTEN
    args.share   = args.share or SHARE

    if args.uid is None and args.uid_index is None and args.mesh is None:
        raise SystemExit(
            "No mesh source: pass --uid / --uid-index / --mesh, or set "
            "UID / UID_INDEX / MESH in the CONFIGURATION block at the top of this file."
        )

    run_inference(args)


if __name__ == "__main__":
    main()
