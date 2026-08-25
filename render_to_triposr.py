"""
Render a 3D mesh with pyrender (OpenGL-quality), feed the image to TripoSR,
and open the reconstructed mesh in a Gradio viewer.

Usage
-----
    # By index into the Objaverse UID list (0 = first object)
    python render_to_triposr.py --uid-index 0

    # By explicit Objaverse UID
    python render_to_triposr.py --uid 1b5e7a72f9cc42a0bd76d1d64db6d3c5

    # Local mesh file (OBJ / GLB / STL)
    python render_to_triposr.py --mesh path/to/object.obj
    
    # most common command:
    python render_to_triposr.py --uid 137f0eb28c524ab0aac86c5105ba33bb --finetuned-ckpt sdf_checkpoints/sdf_head_v0.61_100_epoch0100.pt --azimuth 225 --elevation 30

Optional flags
--------------
    --azimuth     Camera azimuth  in degrees (default  45)
    --elevation   Camera elevation in degrees (default  30)
    --fov         Vertical field of view      (default  40)
    --size        Output image width/height   (default 512)
    --output-dir  Where to save results       (default render_output/)
    --port        Gradio viewer port          (default 7861)
    --listen      Bind viewer to 0.0.0.0

Setup
-----
    pip install pyrender trimesh[all]

    Pyrender needs a headless OpenGL backend.  Default is EGL:
        python render_to_triposr.py ...
    If EGL is unavailable and your OpenGL stack supports it:
        PYOPENGL_PLATFORM=osmesa python render_to_triposr.py ...
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

# Must be set before pyrender / OpenGL are imported.
# In this environment EGL works reliably after upgrading PyOpenGL.
# Override with: PYOPENGL_PLATFORM=osmesa python render_to_triposr.py ...
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
# Work around OpenGL_accelerate wrapper incompatibilities on some Python 3.12
# + driver stacks (e.g. glGenTextures ctypes CArgObject handler errors).
os.environ.setdefault("PYOPENGL_USE_ACCELERATE", "0")

import numpy as np
import trimesh
from PIL import Image

import pyrender
from view_mesh import launch_viewer


def _patch_pyrender_osmesa() -> None:
    """Teach pyrender to use legacy OSMesa APIs if that backend is requested."""
    if os.environ.get("PYOPENGL_PLATFORM") != "osmesa":
        return

    from OpenGL import arrays
    from OpenGL import GL as gl
    from OpenGL.raw.osmesa.mesa import (
        OSMESA_RGBA,
        OSMesaCreateContextExt,
        OSMesaDestroyContext,
        OSMesaMakeCurrent,
    )
    from pyrender.platforms.osmesa import OSMesaPlatform

    def init_context(self):
        self._context = OSMesaCreateContextExt(OSMESA_RGBA, 24, 0, 0, None)
        self._buffer = arrays.GLubyteArray.zeros(
            (self.viewport_height, self.viewport_width, 4)
        )

    def make_current(self):
        assert OSMesaMakeCurrent(
            self._context,
            self._buffer,
            gl.GL_UNSIGNED_BYTE,
            self.viewport_width,
            self.viewport_height,
        )

    def delete_context(self):
        OSMesaDestroyContext(self._context)
        self._context = None
        self._buffer = None

    OSMesaPlatform.init_context = init_context
    OSMesaPlatform.make_current = make_current
    OSMesaPlatform.delete_context = delete_context


_patch_pyrender_osmesa()


# ---------------------------------------------------------------------------
# Objaverse helpers
# ---------------------------------------------------------------------------

def uid_from_index(index: int) -> str:
    import objaverse
    from objaverse_paths import configure_objaverse
    configure_objaverse()
    print("[objaverse] Loading UID list…")
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
    # Return any previously downloaded file
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


# ---------------------------------------------------------------------------
# Mesh loading + scene building
# ---------------------------------------------------------------------------

def _load_trimesh(path: str) -> trimesh.Trimesh:
    """Load any mesh file and return a single Trimesh (merges scenes)."""
    loaded = trimesh.load(path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values()
                  if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No triangle geometry found in {path}")
        mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise ValueError(f"Unsupported mesh type: {type(loaded)}")
    return mesh


def _camera_pose(azimuth_deg: float, elevation_deg: float, distance: float) -> np.ndarray:
    """
    4×4 **camera-to-world** transform T_cam→world (pyrender / OpenGL).

    **World frame** for the render pass: right-handed, mesh centered at the origin
    after `mesh.apply_translation(-centroid)` and scaled so its longest edge is 1.

    Columns of the upper-left 3×3 are camera basis vectors expressed in world coords:
      - Column 0: camera +X (points **right** in the image).
      - Column 1: camera +Y (points **up** in the image).
      - Column 2: camera +Z (points from the scene toward the camera, i.e. “out of”
        the monitor in camera space; in world this is **from origin toward cam_pos**).

    The camera **position** is ``T[:3, 3] = cam_pos``. The camera **looks** from
    ``cam_pos`` toward the origin (OpenGL look direction is **-camera +Z** in world,
    i.e. from camera into the scene).

    Spherical placement: azimuth in XY from +X, elevation from XY plane; distance
    is ``||cam_pos||`` (camera looks at origin).
    """
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)

    cam_pos = np.array([
        distance * np.cos(el) * np.cos(az),
        distance * np.cos(el) * np.sin(az),
        distance * np.sin(el),
    ])

    forward = -cam_pos / np.linalg.norm(cam_pos)           # points at origin
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:                       # near-vertical edge case
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward   # OpenGL: camera looks along -Z
    pose[:3, 3] = cam_pos
    return pose


def _tripo_recon_rotation_to_pyrender_world(T_camera_to_world: np.ndarray) -> np.ndarray:
    """
    TripoSR recon meshes are observed to align with the **capturing camera**:
    +X toward the camera, +Y to the right in the image, +Z up.

    In pyrender world, those directions are camera +Z, +X, +Y respectively, i.e.
    columns 2, 0, 1 of T_cam→world. Returns R (3×3) with
    ``p_world = R @ p_recon`` (same translation as before; rotation only).
    """
    R = np.stack(
        [T_camera_to_world[:3, 2], T_camera_to_world[:3, 0], T_camera_to_world[:3, 1]],
        axis=1,
    )
    return R


def _write_camera_extrinsics_json(
    path: str,
    T_camera_to_world: np.ndarray,
    *,
    azimuth_deg: float,
    elevation_deg: float,
    distance: float,
    fov_deg: float,
) -> None:
    """Save pose + R (recon→pyrender world); ``view_mesh`` overlay applies Rᵀ to the source."""
    R = _tripo_recon_rotation_to_pyrender_world(T_camera_to_world)
    cam_pos = T_camera_to_world[:3, 3]
    forward_into_scene = -T_camera_to_world[:3, 2]  # OpenGL -Z_cam in world
    data = {
        "schema": "tripo_sr_render_to_triposr_v1",
        "world_frame": (
            "Mesh at origin after centroid centering; longest edge scaled to 1. "
            "Right-handed. Camera pose is OpenGL camera-to-world."
        ),
        "capture": {
            "azimuth_deg": azimuth_deg,
            "elevation_deg": elevation_deg,
            "camera_distance": float(distance),
            "fov_deg": fov_deg,
        },
        "T_camera_to_world_4x4": T_camera_to_world.tolist(),
        "cam_position_world": cam_pos.tolist(),
        "camera_looks_from_cam_along_minus_Z_in_world": forward_into_scene.tolist(),
        "camera_right_in_world": T_camera_to_world[:3, 0].tolist(),
        "camera_up_in_world": T_camera_to_world[:3, 1].tolist(),
        "from_origin_toward_camera_unit": T_camera_to_world[:3, 2].tolist(),
        "R_tripo_recon_to_pyrender_world_3x3": R.tolist(),
        "notes": (
            "TripoSR recon frame: +X toward camera, +Y image-right, +Z image-up. "
            "R maps recon→pyrender world. Combined overlay applies Rᵀ to the **unit** "
            "source mesh only; recon vertices are unchanged."
        ),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[render] Camera extrinsics → {path}")


def _directional_light_pose(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Point a directional light at the origin from the given direction."""
    return _camera_pose(azimuth_deg, elevation_deg, 1.0)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_mesh(
    mesh_path: str,
    out_image_path: str,
    azimuth: float = 45.0,
    elevation: float = 30.0,
    fov: float = 40.0,
    size: int = 512,
) -> None:
    """Render *mesh_path* with pyrender and save a gray-background PNG."""

    mesh = _load_trimesh(mesh_path)

    # Centre at origin, scale longest axis to 1
    mesh.apply_translation(-mesh.centroid)
    longest = max(mesh.extents) if max(mesh.extents) > 0 else 1.0
    mesh.apply_scale(1.0 / longest)

    # Build scene  ── gray background, soft ambient + two directional lights
    scene = pyrender.Scene(
        bg_color=[0.5, 0.5, 0.5, 1.0],
        ambient_light=[0.25, 0.25, 0.25],
    )

    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene.add(pr_mesh)

    # Key light (front-upper-left)
    key = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
    scene.add(key, pose=_directional_light_pose(azimuth + 20, elevation + 20))

    # Fill light (opposite side, softer)
    fill = pyrender.DirectionalLight(color=[0.7, 0.8, 1.0], intensity=1.5)
    scene.add(fill, pose=_directional_light_pose(azimuth + 180, elevation - 10))

    # Camera – distance chosen so the unit-scaled mesh fills ~70 % of frame
    fov_rad = np.radians(fov)
    distance = (0.7 / np.tan(fov_rad / 2))
    camera = pyrender.PerspectiveCamera(yfov=fov_rad, aspectRatio=1.0)
    T_cam_to_world = _camera_pose(azimuth, elevation, distance)
    scene.add(camera, pose=T_cam_to_world)

    extrinsics_path = os.path.join(os.path.dirname(out_image_path) or ".", "camera_extrinsics.json")
    _write_camera_extrinsics_json(
        extrinsics_path,
        T_cam_to_world,
        azimuth_deg=azimuth,
        elevation_deg=elevation,
        distance=float(distance),
        fov_deg=fov,
    )

    # Render at 2× then downsample for free anti-aliasing
    render_w = render_h = size * 2
    try:
        r = pyrender.OffscreenRenderer(render_w, render_h)
        color, _ = r.render(scene)
        r.delete()
    except Exception as exc:
        raise RuntimeError(
            f"pyrender OffscreenRenderer failed ({exc}).\n"
            "Try running with an explicit backend:\n"
            "  PYOPENGL_PLATFORM=egl python render_to_triposr.py ...\n"
            "or, if your system supports it:\n"
            "  PYOPENGL_PLATFORM=osmesa python render_to_triposr.py ..."
        ) from exc

    img = Image.fromarray(color).resize((size, size), Image.LANCZOS)
    os.makedirs(os.path.dirname(out_image_path) or ".", exist_ok=True)
    img.save(out_image_path)
    print(f"[render] Saved → {out_image_path}")


# ---------------------------------------------------------------------------
# TripoSR inference
# ---------------------------------------------------------------------------

def run_triposr(image_path: str, output_dir: str) -> str:
    run_dir = os.path.join(output_dir, "0")
    os.makedirs(run_dir, exist_ok=True)

    saved_input = os.path.join(run_dir, "render.png")
    shutil.copy2(image_path, saved_input)
    print(f"[triposr] Input image copied → {saved_input}")

    script = os.path.join(os.path.dirname(__file__), "run.py")
    cmd = [
        sys.executable, script,
        image_path,
        "--output-dir", output_dir,
        "--no-remove-bg",
        "--model-save-format", "obj",
        "--roundtrip-scene-codes",
    ]
    print(f"[triposr] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    mesh_path = os.path.join(output_dir, "0", "mesh.obj")
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"TripoSR did not produce a mesh at {mesh_path}")
    print(f"[triposr] Mesh saved → {mesh_path}")
    return mesh_path


# ---------------------------------------------------------------------------
# Fine-tuned reconstruction: LoRA-adapted TripoSR transformer + SDF head
# ---------------------------------------------------------------------------

def run_finetuned_sdf(
    image_path: str,
    ckpt_path: str,
    output_dir: str,
    resolution: int = 128,
) -> str | None:
    """Reconstruct a mesh from the fine-tuned model: run the rendered image
    through the LoRA-adapted TripoSR transformer to get a triplane, then decode
    it with the trained SDF MLP head via marching cubes.

    This is distinct from the NeRF comparison (run_triposr), which uses the
    stock TripoSR transformer + original NeRF density head. Here both the
    transformer (LoRA) and the decoder (SDF head) are the fine-tuned ones.
    """
    import torch

    from tsr.system import TSR
    from train_sdf_head import (
        SDFMLP,
        apply_lora_to_triposr,
        reconstruct_mesh_from_triplane,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ck_args = ckpt["args"]
    meta = ckpt["meta"]
    radius = float(meta["radius"])
    feature_reduction = meta["feature_reduction"]
    feat_dim = int(meta["feat_dim"])

    # ── Rebuild the fine-tuned TripoSR ────────────────────────────────────────
    # LoRA modules must be injected (same block range / rank / alpha as training)
    # BEFORE load_state_dict, so the state-dict keys (base.weight, lora_A/B) match.
    triposr = TSR.from_pretrained(
        ck_args.get("model", "stabilityai/TripoSR"),
        config_name="config.yaml", weight_name="model.ckpt",
    )
    apply_lora_to_triposr(
        triposr,
        ck_args["lora_block_start"], ck_args["lora_block_end"],
        ck_args["lora_rank"], ck_args["lora_alpha"],
    )
    triposr.to(device)
    triposr.load_state_dict(ckpt["lora_model"], strict=False)
    triposr.eval()

    # ── Rebuild the SDF head (same dims as training) ──────────────────────────
    n_freqs = int(ck_args["n_freqs"])
    pe_dim = (3 + 6 * n_freqs) if n_freqs > 0 else 0
    use_trip = bool(ck_args["use_triplane_features"])
    mlp_feat_dim = feat_dim if use_trip else 0
    mlp_hidden = ck_args["hidden_dim"] if use_trip else ck_args["hidden_dim_no_triplane"]
    sdf_mlp = SDFMLP(
        in_dim=mlp_feat_dim + pe_dim,
        hidden_dim=mlp_hidden,
        n_hidden=ck_args["n_hidden"],
        use_tanh_output=ck_args["use_tanh_output"],
        feat_dim=mlp_feat_dim,
        pe_dim=pe_dim,
    ).to(device)
    sdf_mlp.load_state_dict(ckpt["model"])
    sdf_mlp.eval()

    # ── Image → triplane → SDF mesh ───────────────────────────────────────────
    image_np = np.array(Image.open(image_path).convert("RGB"))
    with torch.no_grad():
        scene_codes = triposr([image_np], device=device)
    triplane = scene_codes[0]  # (3, Cp, Hp, Wp)

    # No R rotation: leave the mesh in TripoSR's native triplane frame, the SAME
    # frame the NeRF comparison mesh (run_triposr) is in, so the two recons line
    # up. (The training visualiser applies R only because it overlays against the
    # GT source mesh in world frame — not relevant here.)
    mesh = reconstruct_mesh_from_triplane(
        sdf_mlp, triplane, radius, feature_reduction,
        resolution=resolution, device=device, n_freqs=n_freqs,
        R_world_from_trip=None, use_triplane_features=use_trip,
    )
    if mesh is None:
        print("[sdf] marching cubes produced no surface (empty SDF grid)")
        return None

    out_path = os.path.join(output_dir, "sdf_mesh.obj")
    mesh.export(out_path)
    print(f"[sdf] Fine-tuned SDF-head mesh saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# SDF MSE metric (mesh vs mesh) — same metric as train_sdf_head epoch_mse and
# TRELLIS/trellis_from_mesh._sdf_mse, so the numbers are directly comparable.
# ---------------------------------------------------------------------------

_MSE_N_POINTS = 32768
_MSE_RADIUS = 0.87
_FSCORE_TAU = 0.01        # F-score threshold (fraction of unit-normalized mesh scale)
_SURFACE_SAMPLES = 50000  # surface samples for Chamfer / F-score (dense enough that the
                          # NN-distance floor sits well below _FSCORE_TAU)


def _normalize_unit(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Centroid to origin, longest edge scaled to 1 (same as training/TRELLIS)."""
    m = mesh.copy()
    m.apply_translation(-m.centroid)
    longest = float(np.max(m.extents)) if m.extents.size and np.max(m.extents) > 0 else 1.0
    m.apply_scale(1.0 / longest)
    return m


def _mesh_sdf(mesh: trimesh.Trimesh, points: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    """Signed distance (positive outside) — identical to compute_sdf in training."""
    import trimesh.proximity
    points = np.asarray(points, dtype=np.float64)
    prox = trimesh.proximity.ProximityQuery(mesh)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    sdf = np.empty(len(points), dtype=np.float32)
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        closest, distances, tri_ids = prox.on_surface(batch)
        direction = batch - closest
        dot = np.einsum("ij,ij->i", direction, face_normals[tri_ids])
        sign = np.where(dot >= 0, 1.0, -1.0)
        sdf[i : i + batch_size] = (sign * distances).astype(np.float32)
    del prox
    return sdf


def _surface_metrics(
    gt_mesh: trimesh.Trimesh,
    rec_mesh: trimesh.Trimesh,
    n_samples: int = _SURFACE_SAMPLES,
    fscore_tau: float = _FSCORE_TAU,
) -> dict[str, float]:
    """Chamfer-L2 + F-score between surface samples of two ALIGNED meshes.

    Standard protocol (MeshLRM / TripoSG / Dora-bench). Complements SDF MSE,
    which is dominated by far-field error that marching cubes never sees.
    """
    from scipy.spatial import cKDTree
    gt_pts  = np.asarray(gt_mesh.sample(n_samples),  dtype=np.float64)
    rec_pts = np.asarray(rec_mesh.sample(n_samples), dtype=np.float64)
    d_rec_to_gt = cKDTree(gt_pts).query(rec_pts, workers=-1)[0]
    d_gt_to_rec = cKDTree(rec_pts).query(gt_pts, workers=-1)[0]
    chamfer   = float((d_rec_to_gt ** 2).mean() + (d_gt_to_rec ** 2).mean())
    precision = float((d_rec_to_gt < fscore_tau).mean())
    recall    = float((d_gt_to_rec < fscore_tau).mean())
    fscore    = (2 * precision * recall / (precision + recall)
                 if precision + recall > 0 else 0.0)
    return {"chamfer": chamfer, "fscore": fscore,
            "precision": precision, "recall": recall}


def sdf_mse_report(
    source_mesh_path: str,
    recons: list[tuple[str, str | None, np.ndarray | None]],
    n_points: int = _MSE_N_POINTS,
    radius: float = _MSE_RADIUS,
) -> dict[str, dict[str, float]]:
    """SDF MSE + Chamfer/F-score of each reconstruction vs the source.

    ``recons`` is a list of ``(name, mesh_path, R_world_from_recon)``. The GT SDF
    is computed ONCE on a single point set and reused for every recon (both
    faster and a fairer comparison than sampling separately). Each recon's R
    rotates it from TripoSR's camera-centric frame into the source world frame.
    Same 32768-pt / radius-0.87 protocol as train_sdf_head epoch_mse and TRELLIS;
    Chamfer / F-score use surface samples of the unit-normalized meshes.
    Per-stage timing is printed so a slow source mesh is easy to spot.

    Returns ``{name: {"mse", "chamfer", "fscore", "precision", "recall"}}``
    (surface-metric keys absent if that computation failed).
    """
    import time

    t0 = time.perf_counter()
    src = _normalize_unit(_load_trimesh(source_mesh_path))
    pts = np.random.uniform(-radius, radius, (n_points, 3)).astype(np.float32)
    pts = np.clip(pts, -radius, radius)
    sdf_gt = _mesh_sdf(src, pts)
    print(f"[metric]   ground-truth SDF ({src.faces.shape[0]:,} faces): "
          f"{time.perf_counter() - t0:.1f}s")

    results: dict[str, dict[str, float]] = {}
    for name, path, R in recons:
        if path is None:
            continue
        t1 = time.perf_counter()
        try:
            rec = _load_trimesh(path)
            if R is not None:
                rec = rec.copy()
                rec.vertices = rec.vertices @ R  # recon → world
            rec = _normalize_unit(rec)
            sdf_rec = _mesh_sdf(rec, pts)
            m: dict[str, float] = {"mse": float(np.mean((sdf_rec - sdf_gt) ** 2))}
            try:
                m.update(_surface_metrics(src, rec))
            except Exception as e:
                print(f"[metric]   {name}: surface metrics failed ({e})")
            results[name] = m
            extra = (f", chamfer = {m['chamfer']:.6f}, F@{_FSCORE_TAU} = {m['fscore']:.3f}"
                     if "chamfer" in m else "")
            print(f"[metric]   {name}: MSE = {m['mse']:.6f}{extra}  "
                  f"({rec.faces.shape[0]:,} faces, {time.perf_counter() - t1:.1f}s)")
        except Exception as e:
            print(f"[metric]   {name}: failed ({e})")
    return results


# ---------------------------------------------------------------------------
# Pipeline cache
# ---------------------------------------------------------------------------

def _pipeline_cache_key(
    mesh_path: str,
    azimuth: float,
    elevation: float,
    fov: float,
    size: int,
    finetuned_ckpt: str | None,
    sdf_resolution: int,
) -> str:
    """Stable hash of everything that determines the outputs. A file's identity
    includes its mtime + size, so replacing the source mesh or the checkpoint
    (e.g. a newer training epoch) invalidates the cache automatically."""
    import hashlib

    def _sig(p: str | None) -> str:
        if not p:
            return "none"
        try:
            st = os.stat(p)
            return f"{os.path.abspath(p)}|{st.st_mtime_ns}|{st.st_size}"
        except OSError:
            return str(p)

    parts = [
        _sig(mesh_path),
        f"az={azimuth}", f"el={elevation}", f"fov={fov}", f"size={size}",
        f"sdfres={sdf_resolution}",
        _sig(finetuned_ckpt),
    ]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mesh renderer → TripoSR → Gradio viewer pipeline."
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--uid",       default=None, help="Objaverse UID to download and render.")
    source.add_argument("--uid-index", type=int, default=None, metavar="N",
                        help="Index into the Objaverse UID list (0-based).")
    source.add_argument("--mesh",      default=None,
                        help="Path to a local mesh file (OBJ / GLB / STL).")

    parser.add_argument("--output-dir", default="render_output",
                        help="Directory for renders and mesh files. Default: render_output/")
    parser.add_argument("--azimuth",   type=float, default=45.0,
                        help="Camera azimuth in degrees  (default 45)")
    parser.add_argument("--elevation", type=float, default=30.0,
                        help="Camera elevation in degrees (default 30)")
    parser.add_argument("--fov",       type=float, default=40.0,
                        help="Vertical field of view      (default 40)")
    parser.add_argument("--size",      type=int,   default=512,
                        help="Output image size in pixels (default 512)")
    parser.add_argument("--port",      type=int,   default=7861,
                        help="Gradio viewer port          (default 7861)")
    parser.add_argument("--listen",    action="store_true",
                        help="Bind Gradio to 0.0.0.0 (accessible from other machines)")
    parser.add_argument("--finetuned-ckpt", default=None,
                        help="Path to a fine-tuned checkpoint (.pt). If given, also "
                             "reconstruct with the LoRA TripoSR transformer + SDF head "
                             "alongside the stock-NeRF comparison.")
    parser.add_argument("--sdf-resolution", type=int, default=128,
                        help="Marching-cubes grid resolution for the SDF-head mesh (default 128)")
    parser.add_argument("--refresh", action="store_true",
                        help="Ignore the cache and recompute render/recon/MSE from scratch.")
    args = parser.parse_args()

    output_dir  = os.path.abspath(args.output_dir)
    cache_dir   = os.path.join(output_dir, "objaverse_cache")
    render_path = os.path.join(output_dir, "render.png")

    # Resolve mesh path
    if args.uid is not None:
        mesh_path = fetch_objaverse_glb(args.uid, cache_dir)
    elif args.uid_index is not None:
        uid = uid_from_index(args.uid_index)
        mesh_path = fetch_objaverse_glb(uid, cache_dir)
    elif args.mesh is not None:
        mesh_path = os.path.abspath(os.path.expanduser(args.mesh))
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    else:
        parser.error("Provide one of --uid, --uid-index, or --mesh.")

    # ── Cache: reuse render/recon/MSE when the inputs are unchanged ───────────
    key = _pipeline_cache_key(
        mesh_path, args.azimuth, args.elevation, args.fov, args.size,
        args.finetuned_ckpt, args.sdf_resolution,
    )
    cdir = os.path.join(output_dir, "cache", key)
    c_render, c_ext = os.path.join(cdir, "render.png"), os.path.join(cdir, "camera_extrinsics.json")
    c_nerf, c_sdf   = os.path.join(cdir, "mesh.obj"),   os.path.join(cdir, "sdf_mesh.obj")
    c_metrics       = os.path.join(cdir, "metrics.json")
    want_sdf = args.finetuned_ckpt is not None

    cache_hit = (
        not args.refresh
        and os.path.isfile(c_render) and os.path.isfile(c_nerf) and os.path.isfile(c_metrics)
        and (not want_sdf or os.path.isfile(c_sdf))
    )

    if cache_hit:
        print(f"[cache] HIT {cdir}\n[cache] reusing meshes + MSE (pass --refresh to recompute)")
        render_path = c_render
        out_mesh    = c_nerf
        sdf_mesh    = c_sdf if (want_sdf and os.path.isfile(c_sdf)) else None
        extrinsics_path = c_ext if os.path.isfile(c_ext) else None
        with open(c_metrics) as f:
            _m = json.load(f)
        nerf_mse, sdf_mse, aligned = _m.get("nerf_mse"), _m.get("sdf_mse"), bool(_m.get("aligned", False))
        # Surface metrics may be absent in caches written before they existed.
        nerf_cd, nerf_f = _m.get("nerf_chamfer"), _m.get("nerf_fscore")
        sdf_cd,  sdf_f  = _m.get("sdf_chamfer"),  _m.get("sdf_fscore")
    else:
        # Step 1 – render
        render_mesh(
            mesh_path, render_path,
            azimuth=args.azimuth, elevation=args.elevation, fov=args.fov, size=args.size,
        )

        # Step 2 – NeRF comparison: stock TripoSR transformer + original NeRF head
        out_mesh = run_triposr(render_path, output_dir)

        # Step 2b – Fine-tuned reconstruction: LoRA TripoSR transformer + SDF head
        sdf_mesh = None
        if args.finetuned_ckpt is not None:
            sdf_mesh = run_finetuned_sdf(
                render_path, args.finetuned_ckpt, output_dir, resolution=args.sdf_resolution,
            )

        # Step 2c – SDF MSE vs the ground-truth source mesh (same metric as
        # train_sdf_head epoch_mse / TRELLIS). Rotate recons into the source frame
        # via R from camera_extrinsics.json so the comparison is shape-aligned.
        extrinsics_path = os.path.join(output_dir, "camera_extrinsics.json")
        recon_R = None
        if os.path.isfile(extrinsics_path):
            from pathlib import Path
            from train_sdf_head import load_R_world_from_recon_json
            recon_R = load_R_world_from_recon_json(Path(output_dir))
        aligned = recon_R is not None
        print(f"[metric] SDF MSE vs GT ({_MSE_N_POINTS:,} pts, radius {_MSE_RADIUS}, "
              f"{'rotation-aligned' if aligned else 'NO R — rotation NOT aligned'}):")
        _recons = [("NeRF (stock TripoSR)", out_mesh, recon_R)]
        if sdf_mesh is not None:
            _recons.append(("SDF+LoRA (fine-tuned)", sdf_mesh, recon_R))
        _metrics = sdf_mse_report(mesh_path, _recons)
        _nerf_m = _metrics.get("NeRF (stock TripoSR)") or {}
        _sdf_m  = _metrics.get("SDF+LoRA (fine-tuned)") or {}
        nerf_mse, nerf_cd, nerf_f = _nerf_m.get("mse"), _nerf_m.get("chamfer"), _nerf_m.get("fscore")
        sdf_mse,  sdf_cd,  sdf_f  = _sdf_m.get("mse"),  _sdf_m.get("chamfer"),  _sdf_m.get("fscore")

        # ── Persist artifacts to the cache for next time ──────────────────────
        try:
            os.makedirs(cdir, exist_ok=True)
            shutil.copy2(render_path, c_render)
            if os.path.isfile(extrinsics_path):
                shutil.copy2(extrinsics_path, c_ext)
            shutil.copy2(out_mesh, c_nerf)
            if sdf_mesh and os.path.isfile(sdf_mesh):
                shutil.copy2(sdf_mesh, c_sdf)
            with open(c_metrics, "w") as f:
                json.dump({
                    "nerf_mse": nerf_mse, "sdf_mse": sdf_mse, "aligned": aligned,
                    "nerf_chamfer": nerf_cd, "nerf_fscore": nerf_f,
                    "sdf_chamfer": sdf_cd, "sdf_fscore": sdf_f,
                    "fscore_tau": _FSCORE_TAU,
                    "azimuth": args.azimuth, "elevation": args.elevation,
                    "fov": args.fov, "size": args.size,
                    "sdf_resolution": args.sdf_resolution,
                    "finetuned_ckpt": args.finetuned_ckpt,
                }, f, indent=2)
            print(f"[cache] saved → {cdir}")
        except Exception as e:
            print(f"[cache] save failed (continuing): {e}")

    # Markdown summary shown in the Gradio window.
    def _fmt(v, nd=6):
        return f"`{v:.{nd}f}`" if v is not None else "—"

    _rows = [f"| Reconstruction | SDF MSE vs GT | Chamfer-L2 | F-score@{_FSCORE_TAU} |",
             "|:---|---:|---:|---:|"]
    if nerf_mse is not None:
        _rows.append(f"| NeRF (stock TripoSR) | {_fmt(nerf_mse)} | {_fmt(nerf_cd)} | {_fmt(nerf_f, 3)} |")
    if sdf_mse is not None:
        _rows.append(f"| **SDF+LoRA (fine-tuned)** | **{_fmt(sdf_mse)}** | **{_fmt(sdf_cd)}** | **{_fmt(sdf_f, 3)}** |")
    metrics_md = (
        "### Reconstruction metrics vs ground truth\n\n"
        + "\n".join(_rows)
        + f"\n\nSDF MSE: `mean((sdf_recon − sdf_gt)²)` over {_MSE_N_POINTS:,} points uniform in "
        f"[−{_MSE_RADIUS}, {_MSE_RADIUS}]³ on unit-normalised meshes — same metric as "
        f"`train_sdf_head.py` `epoch_mse` and TRELLIS. Chamfer-L2 / F-score: bidirectional "
        f"nearest-neighbour over {_SURFACE_SAMPLES:,} surface samples (surface accuracy; "
        f"SDF MSE is dominated by far-field error marching cubes never sees). "
        + ("Recon rotation-aligned to GT via `camera_extrinsics.json`."
           if aligned else
           "⚠️ No `camera_extrinsics.json` — rotation **not** aligned, metrics inflated.")
    )

    # Step 3 – view (overlay uses camera_extrinsics.json next to render for recon rotation)
    viewer_kwargs = dict(
        port=args.port,
        listen=args.listen,
        camera_extrinsics_path=(extrinsics_path
                                if extrinsics_path and os.path.isfile(extrinsics_path)
                                else None),
        metrics_md=metrics_md,
    )
    # Pass the SDF mesh to the viewer if it exposes a slot for a second recon;
    # otherwise it is saved to disk and its path is printed above.
    if sdf_mesh is not None:
        import inspect
        vparams = inspect.signature(launch_viewer).parameters
        for name in ("sdf_mesh", "sdf_mesh_path", "extra_mesh", "second_mesh"):
            if name in vparams:
                viewer_kwargs[name] = sdf_mesh
                break
        else:
            print(f"[viewer] launch_viewer has no slot for a second mesh; "
                  f"the SDF mesh is saved at {sdf_mesh}")
    launch_viewer(
        out_mesh,
        mesh_path,
        render_path,
        output_dir,
        **viewer_kwargs,
    )


if __name__ == "__main__":
    main()
