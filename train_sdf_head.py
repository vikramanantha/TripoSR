"""
train_sdf_head.py  —  Two-phase SDF MLP training on frozen TripoSR features.

Commands
────────
  precompute   Download Objaverse meshes, render input views, run frozen TripoSR,
               compute GT SDF, save (triplane, query_pts, sdf_gt) + camera artifacts.

  train        Load precomputed dataset, train a small SDF MLP with L1 + eikonal loss.

Typical workflow
────────────────
  ./train_sdf.sh --precompute       # precompute only
  ./train_sdf.sh --train            # train only (dataset must exist)
  python train_sdf_head.py          # uses COMMAND variable below
"""

import contextlib
import gc
import json
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.modules["torchmcubes"] = MagicMock()
sys.modules["rembg"] = MagicMock()

sys.path.insert(0, str(Path(__file__).parent))

import argparse
import random
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from einops import rearrange
from PIL import Image
from skimage.measure import marching_cubes
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tsr.utils import scale_tensor

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

COMMAND = "both"

# ── Shared ──────────────────────────────────────────────────────────────────
DATASET_DIR     = "/home/markiv/TripoSR/sdf_dataset"

# ── Precompute ───────────────────────────────────────────────────────────────
MODEL           = "stabilityai/TripoSR"
N_OBJECTS       = 10
AZIMUTHS_PER_MESH = 10
N_POINTS        = 262144
IMAGE_SIZE      = 256
ELEVATION       = 30.0
FOV             = 40.0
MAX_MESH_MB     = 0.0
VERBOSE         = False

# ── Train ────────────────────────────────────────────────────────────────────
OUTPUT_DIR      = "/home/markiv/TripoSR/sdf_checkpoints"
EPOCHS          = 500
SAVE_EVERY      = 10
HIDDEN_DIM      = 64
N_HIDDEN        = 10
N_FREQS         = 6
LR              = 1e-5
EIKONAL_WEIGHT  = 0.0
SDF_CLAMP       = 0.0
NUM_WORKERS     = 4
RUN_NAME        = "v0.3"
TEST_FRACTION   = 0.2        # fraction of meshes (UIDs) held out as unseen
TEST_VIEW_FRACTION = 0.2     # fraction of azimuth views held out per mesh (0 = use all views)
VIS_EVERY       = 5
VIS_SEEN        = 3
VIS_UNSEEN      = 3
VIS_RESOLUTION  = 64
BATCH_SIZE      = 4096
RESUME          = None
WEIGHT_DECAY    = 1e-4
USE_TANH_OUTPUT = True
TARGET_SCALE_QUANTILE = 0.95
MIN_TARGET_SCALE = 1e-3

# ═══════════════════════════════════════════════════════════════════════════════


_SKIP_WANDB_MODULE_KEYS = frozenset({
    "__name__", "__doc__", "__file__", "__package__", "__loader__",
    "__spec__", "__builtins__", "__cached__", "__annotations__",
})


def _wandb_jsonable(obj):
    """Convert a value to something wandb JSON / config can store."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, argparse.Namespace):
        return {k: _wandb_jsonable(v) for k, v in vars(obj).items()}
    if isinstance(obj, (set, frozenset)):
        try:
            return sorted(obj)
        except TypeError:
            return [repr(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _wandb_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        if len(obj) > 128:
            return f"{type(obj).__name__}(len={len(obj)})"
        return [_wandb_jsonable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return {"__tensor__": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, np.ndarray):
        if obj.size <= 64 and obj.ndim <= 2:
            return obj.tolist()
        return {"__ndarray__": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, types.ModuleType):
        return f"<module {getattr(obj, '__name__', repr(obj))}>"
    if isinstance(obj, type):
        return f"<class {obj.__module__}.{obj.__qualname__}>"
    if callable(obj):
        return f"<callable {getattr(obj, '__qualname__', getattr(obj, '__name__', repr(obj)))}>"
    return repr(obj)[:2000]


def wandb_collect_module_globals() -> dict:
    """Serializable script-level globals (skips imported modules, classes, and functions)."""
    mod = sys.modules[__name__]
    out: dict = {}
    for name, val in vars(mod).items():
        if name.startswith("_") or name in _SKIP_WANDB_MODULE_KEYS:
            continue
        if isinstance(val, types.ModuleType):
            continue
        if isinstance(val, type):
            continue
        if callable(val):
            continue
        try:
            out[name] = _wandb_jsonable(val)
        except Exception as e:
            out[name] = f"<serialize_error {type(val).__name__}: {e}>"
    return out


def wandb_model_parameter_config(model: nn.Module) -> dict:
    """Flat config entries for every parameter tensor (weights, biases, norms)."""
    cfg: dict = {}
    total = trainable = 0
    with torch.no_grad():
        for pname, p in model.named_parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n
            pf = p.detach().float().cpu()
            prefix = f"model/params/{pname.replace('.', '/')}"
            cfg[f"{prefix}/numel"] = int(n)
            cfg[f"{prefix}/shape"] = list(p.shape)
            cfg[f"{prefix}/dtype"] = str(p.dtype)
            cfg[f"{prefix}/mean"] = float(pf.mean())
            cfg[f"{prefix}/std"] = float(pf.std())
            cfg[f"{prefix}/min"] = float(pf.min())
            cfg[f"{prefix}/max"] = float(pf.max())
            cfg[f"{prefix}/requires_grad"] = bool(p.requires_grad)
    cfg["model/total_params"] = int(total)
    cfg["model/trainable_params"] = int(trainable)
    cfg["model/repr"] = repr(model)
    return cfg


def wandb_log_model_parameter_table(model: nn.Module) -> None:
    """Single wandb Table with one row per parameter (for the UI)."""
    rows = []
    with torch.no_grad():
        for pname, p in model.named_parameters():
            pf = p.detach().float().cpu()
            rows.append([
                pname,
                str(list(p.shape)),
                int(p.numel()),
                str(p.dtype),
                bool(p.requires_grad),
                float(pf.mean()),
                float(pf.std()),
                float(pf.min()),
                float(pf.max()),
            ])
    table = wandb.Table(
        columns=[
            "name", "shape", "numel", "dtype", "requires_grad",
            "mean", "std", "min", "max",
        ],
        data=rows,
    )
    wandb.log({"model/parameter_table": table})


# ─── SDF MLP ─────────────────────────────────────────────────────────────────

class SDFMLP(nn.Module):
    """Triplane features + optional Fourier PE -> scalar signed distance."""

    def __init__(
        self,
        in_dim: int = 120,
        hidden_dim: int = 256,
        n_hidden: int = 3,
        use_tanh_output: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
        ]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        self.use_tanh_output = use_tanh_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x).squeeze(-1)
        if self.use_tanh_output:
            out = torch.tanh(out)
        return out


# ─── Triplane sampling ───────────────────────────────────────────────────────

def query_triplane_features(
    positions: torch.Tensor,
    triplane: torch.Tensor,
    radius: float,
    feature_reduction: str = "concat",
) -> torch.Tensor:
    """Bilinear-sample a triplane at 3-D positions -> raw feature vectors."""
    input_shape = positions.shape[:-1]
    flat = positions.reshape(-1, 3)
    norm = scale_tensor(flat, (-radius, radius), (-1, 1))

    idx2d = torch.stack(
        (norm[..., [0, 1]], norm[..., [0, 2]], norm[..., [1, 2]]),
        dim=-3,
    )

    out = F.grid_sample(
        rearrange(triplane, "Np Cp Hp Wp -> Np Cp Hp Wp", Np=3),
        rearrange(idx2d, "Np N Nd -> Np () N Nd", Np=3),
        align_corners=False,
        mode="bilinear",
    )

    if feature_reduction == "concat":
        out = rearrange(out, "Np Cp () N -> N (Np Cp)", Np=3)
    else:
        from einops import reduce as ered
        out = ered(out, "Np Cp () N -> N Cp", Np=3, reduction="mean")

    return out.reshape(*input_shape, -1)


# ─── Fourier positional encoding ─────────────────────────────────────────────

def fourier_encode(pts: torch.Tensor, n_freqs: int = 6) -> torch.Tensor:
    """pts: (..., 3) -> (..., 3 + 6*n_freqs)"""
    if n_freqs == 0:
        return pts
    freqs = 2.0 ** torch.arange(n_freqs, dtype=pts.dtype, device=pts.device)
    x = pts[..., :, None] * freqs
    return torch.cat([pts, torch.sin(x).flatten(-2), torch.cos(x).flatten(-2)], dim=-1)


# ─── Mesh helpers ─────────────────────────────────────────────────────────────

def _load_trimesh(path: str):
    import trimesh
    loaded = trimesh.load(path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No triangle geometry in {path}")
        return trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    raise ValueError(f"Unsupported geometry type: {type(loaded)}")


def _normalize_mesh_copy(raw, radius: float):
    """Copy ``raw``, apply centroid removal + ``(2*radius)/longest`` scale (TripoSR volume)."""
    if len(raw.faces) == 0:
        raise ValueError("Mesh has no faces")
    centroid = np.asarray(raw.centroid, dtype=np.float64)
    longest = float(max(raw.extents) if max(raw.extents) > 0 else 1.0)
    mesh = raw.copy()
    mesh.apply_translation(-raw.centroid)
    mesh.apply_scale((2.0 * radius) / longest)
    return mesh, centroid, longest


def load_and_normalize_mesh(path: str, radius: float):
    raw = _load_trimesh(path)
    mesh, _, _ = _normalize_mesh_copy(raw, radius)
    return mesh


def apply_mesh_normalization_transform(
    pts: np.ndarray,
    centroid: np.ndarray,
    longest: float,
    radius: float,
) -> np.ndarray:
    """``p_norm = (p_raw - centroid) * (2*radius / longest)`` — same as ``load_and_normalize_mesh``."""
    s = (2.0 * radius) / float(longest)
    c = np.asarray(centroid, dtype=np.float64)
    return (np.asarray(pts, dtype=np.float64) - c) * s


def rotate_mesh_z(mesh, angle_deg: float):
    import trimesh
    R = trimesh.transformations.rotation_matrix(np.radians(angle_deg), [0, 0, 1])
    rotated = mesh.copy()
    rotated.apply_transform(R)
    return rotated


def _tripo_recon_rotation_to_pyrender_world(T_camera_to_world: np.ndarray) -> np.ndarray:
    """Same convention as ``render_to_triposr.py``: p_world_row = p_recon_row @ R."""
    return np.stack(
        [T_camera_to_world[:3, 2], T_camera_to_world[:3, 0], T_camera_to_world[:3, 1]],
        axis=1,
    )


def _write_camera_extrinsics_json(
    path: str | Path,
    T_camera_to_world: np.ndarray,
    *,
    azimuth_deg: float,
    elevation_deg: float,
    distance: float,
    fov_deg: float,
) -> None:
    """Same fields as ``render_to_triposr.py`` (schema trip_sr_render_to_triposr_v1)."""
    path = Path(path)
    R = _tripo_recon_rotation_to_pyrender_world(T_camera_to_world)
    cam_pos = T_camera_to_world[:3, 3]
    forward_into_scene = -T_camera_to_world[:3, 2]
    data = {
        "schema": "tripo_sr_render_to_triposr_v1",
        "world_frame": (
            "train_sdf_head precompute: mesh centroid at origin, longest edge scaled to 2*radius "
            "(TripoSR volume). Camera is OpenGL camera-to-world."
        ),
        "capture": {
            "azimuth_deg": float(azimuth_deg),
            "elevation_deg": float(elevation_deg),
            "camera_distance": float(distance),
            "fov_deg": float(fov_deg),
        },
        "T_camera_to_world_4x4": T_camera_to_world.tolist(),
        "cam_position_world": cam_pos.tolist(),
        "camera_looks_from_cam_along_minus_Z_in_world": forward_into_scene.tolist(),
        "camera_right_in_world": T_camera_to_world[:3, 0].tolist(),
        "camera_up_in_world": T_camera_to_world[:3, 1].tolist(),
        "from_origin_toward_camera_unit": T_camera_to_world[:3, 2].tolist(),
        "R_tripo_recon_to_pyrender_world_3x3": R.tolist(),
        "notes": (
            "Precompute moves the camera with azimuth (mesh stays normalized-only). "
            "Training uses p_trip = p_mesh @ R.T for triplane sampling; sdf_gt stays in mesh frame."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_R_world_from_recon_json(sample_dir: Path) -> np.ndarray | None:
    """Return R (3,3) with p_world_row = p_recon_row @ R, or None if missing."""
    p = sample_dir / "camera_extrinsics.json"
    if not p.is_file():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if "R_tripo_recon_to_pyrender_world_3x3" in data:
        R = np.asarray(data["R_tripo_recon_to_pyrender_world_3x3"], dtype=np.float64)
        if R.shape == (3, 3):
            return R
    if "T_camera_to_world_4x4" in data:
        T = np.asarray(data["T_camera_to_world_4x4"], dtype=np.float64)
        if T.shape == (4, 4):
            return _tripo_recon_rotation_to_pyrender_world(T)
    return None


def load_R_world_from_recon_json_strict(sample_dir: Path) -> np.ndarray:
    """Require valid ``camera_extrinsics.json`` (no legacy / no silent identity)."""
    R = load_R_world_from_recon_json(sample_dir)
    if R is None:
        raise RuntimeError(
            f"Missing or invalid camera_extrinsics.json under {sample_dir}. "
            "Re-run precompute — datasets without per-sample extrinsics are not supported."
        )
    return R


def compute_sdf(mesh, points: np.ndarray) -> np.ndarray:
    """Signed distance: negative inside, positive outside."""
    import trimesh.proximity
    _, distances, _ = trimesh.proximity.closest_point(mesh, points)
    inside = mesh.contains(points)
    return np.where(inside, -distances, distances).astype(np.float32)


def sample_query_points(
    raw_mesh,
    n_points: int,
    radius: float,
    centroid: np.ndarray,
    longest: float,
    near_surface_fraction: float = 0.5,
    near_surface_std: float | None = None,
) -> np.ndarray:
    """Sample in **raw** mesh coordinates, then apply the same translate+scale as
    ``load_and_normalize_mesh``. Optional ``near_surface_std`` is Gaussian noise **in raw units**;
    if ``None``, uses the same blur as the old ``radius * 0.04`` in normalized space.
    """
    s = (2.0 * radius) / float(longest)
    if near_surface_std is None:
        near_surface_std = float((radius * 0.04) / s)
    n_near = int(n_points * near_surface_fraction)
    n_uniform = n_points - n_near
    lo, hi = raw_mesh.bounds[0], raw_mesh.bounds[1]
    uniform = np.stack(
        [np.random.uniform(lo[i], hi[i], n_uniform) for i in range(3)],
        axis=1,
    )
    if len(raw_mesh.vertices) > 0 and n_near > 0:
        idx = np.random.choice(len(raw_mesh.vertices), n_near, replace=True)
        verts = np.asarray(raw_mesh.vertices[idx], dtype=np.float64)
        noise = np.random.normal(0.0, near_surface_std, (n_near, 3))
        near_raw = verts + noise
    else:
        near_raw = np.stack(
            [np.random.uniform(lo[i], hi[i], n_near) for i in range(3)],
            axis=1,
        )
    raw_pts = np.concatenate([uniform, near_raw], axis=0)
    out = apply_mesh_normalization_transform(raw_pts, centroid, longest, radius)
    return np.clip(out.astype(np.float32), -radius, radius)


# ─── Rendering ────────────────────────────────────────────────────────────────

def _camera_pose(azimuth_deg: float, elevation_deg: float, distance: float) -> np.ndarray:
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    cam_pos = np.array([
        distance * np.cos(el) * np.cos(az),
        distance * np.cos(el) * np.sin(az),
        distance * np.sin(el),
    ])
    forward = -cam_pos / np.linalg.norm(cam_pos)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = cam_pos
    return pose


def render_mesh_to_image(
    mesh,
    elevation: float = 30.0,
    fov: float = 40.0,
    size: int = 256,
    azimuth: float = 0.0,
    extrinsics_json_path: str | Path | None = None,
) -> Image.Image:
    """Render normalized mesh; main camera uses ``azimuth`` (deg) like precompute / TripoSR input."""
    import pyrender
    scene = pyrender.Scene(bg_color=[0.5, 0.5, 0.5, 1.0], ambient_light=[0.25, 0.25, 0.25])
    try:
        pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    except Exception:
        import trimesh as tr
        pr_mesh = pyrender.Mesh.from_trimesh(tr.Trimesh(vertices=mesh.vertices, faces=mesh.faces))
    scene.add(pr_mesh)
    fov_rad = np.radians(fov)
    distance = (max(mesh.extents) / 2.0 / 0.7) / np.tan(fov_rad / 2.0)
    T_cam = _camera_pose(azimuth, elevation, distance)
    scene.add(pyrender.PerspectiveCamera(yfov=fov_rad, aspectRatio=1.0), pose=T_cam)
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0),
              pose=_camera_pose(20, elevation + 20, 1.0))
    scene.add(pyrender.DirectionalLight(color=[0.7, 0.8, 1.0], intensity=1.5),
              pose=_camera_pose(180, elevation - 10, 1.0))
    if extrinsics_json_path is not None:
        _write_camera_extrinsics_json(
            extrinsics_json_path,
            T_cam,
            azimuth_deg=float(azimuth),
            elevation_deg=float(elevation),
            distance=float(distance),
            fov_deg=float(fov),
        )
    r = pyrender.OffscreenRenderer(size, size)
    color, _ = r.render(scene)
    r.delete()
    scene.clear()
    del scene
    return Image.fromarray(color)


# ─── Objaverse helpers ────────────────────────────────────────────────────────

def get_objaverse_uid_pool(seed: int = 42) -> list[str]:
    import objaverse
    print("[objaverse] Loading UID list...")
    all_uids = list(objaverse.load_uids())
    rng = random.Random(seed)
    rng.shuffle(all_uids)
    print(f"[objaverse] {len(all_uids)} UIDs available")
    return all_uids


def download_mesh(uid: str, cache_dir: str) -> str:
    import objaverse
    os.makedirs(cache_dir, exist_ok=True)
    cached_dir = os.path.join(cache_dir, uid)
    for ext in (".glb", ".obj", ".stl"):
        p = os.path.join(cached_dir, f"{uid}{ext}")
        if os.path.exists(p):
            return p
    with open(os.devnull, "w") as devnull, \
         contextlib.redirect_stdout(devnull), \
         contextlib.redirect_stderr(devnull):
        objects = objaverse.load_objects(uids=[uid], download_processes=1)
    if uid not in objects:
        raise RuntimeError(f"Objaverse returned nothing for UID: {uid}")
    src = objects[uid]
    os.makedirs(cached_dir, exist_ok=True)
    dst = os.path.join(cached_dir, os.path.basename(src))
    if src != dst:
        shutil.copy2(src, dst)
    return dst


# ─── PRECOMPUTE phase ─────────────────────────────────────────────────────────

def run_precompute(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir)
    samples_dir = dataset_dir / "samples"
    if samples_dir.exists() and any(samples_dir.iterdir()):
        existing = sum(1 for _ in samples_dir.iterdir() if not _.name.startswith("_tmp"))
        answer = input(
            f"Found {existing} existing samples in {samples_dir}.\n"
            f"Delete all and restart? [y/N] "
        ).strip().lower()
        if answer in ("y", "yes"):
            shutil.rmtree(samples_dir)
            print("Deleted existing samples.")
        else:
            print("Keeping existing samples (will skip already-done azimuths).")

    from tsr.system import TSR

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Precompute on {device}")

    print("Loading TripoSR...")
    model = TSR.from_pretrained(args.model, config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(8192)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    radius: float = float(model.renderer.cfg.radius)
    feature_reduction: str = model.renderer.cfg.feature_reduction
    feat_dim: int = model.decoder.cfg.in_channels

    dataset_dir = Path(args.dataset_dir)
    samples_dir = dataset_dir / "samples"
    cache_dir = str(dataset_dir / "mesh_cache")
    samples_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "radius": radius,
        "feature_reduction": feature_reduction,
        "feat_dim": feat_dim,
        "n_points": args.n_points,
    }
    with open(dataset_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata -> {dataset_dir / 'metadata.json'}")

    uids = get_objaverse_uid_pool()
    target_objects = args.n_objects
    azimuths = np.linspace(0, 360, args.azimuths_per_mesh, endpoint=False)

    pbar = tqdm(total=target_objects, unit="obj", dynamic_ncols=True)
    obj_saved = obj_skipped = 0

    for uid in uids:
        if obj_saved >= target_objects:
            break

        _done = [
            all((samples_dir / f"{uid}_az{int(az):03d}" / fn).exists()
                for fn in (
                    "triplane.pt",
                    "query_pts.pt",
                    "sdf_gt.pt",
                    "camera_extrinsics.json",
                ))
            for az in azimuths
        ]
        if all(_done):
            obj_saved += 1
            pbar.update(1)
            pbar.set_postfix(objects=obj_saved, skipped=obj_skipped)
            continue

        try:
            mesh_path = download_mesh(uid, cache_dir)
            if args.max_mesh_mb > 0:
                size_mb = os.path.getsize(mesh_path) / (1024 * 1024)
                if size_mb > args.max_mesh_mb:
                    tqdm.write(f"[skip mesh] {uid}: {size_mb:.1f} MB > limit {args.max_mesh_mb} MB")
                    obj_skipped += 1
                    pbar.set_postfix(objects=obj_saved, skipped=obj_skipped)
                    continue
            raw = _load_trimesh(mesh_path)
            mesh, centroid, longest = _normalize_mesh_copy(raw, radius)
            if not mesh.is_watertight:
                obj_skipped += 1
                pbar.set_postfix(objects=obj_saved, skipped=obj_skipped)
                continue
        except Exception as e:
            obj_skipped += 1
            pbar.set_postfix(objects=obj_saved, skipped=obj_skipped)
            if args.verbose:
                tqdm.write(f"[skip mesh] {uid}: {e}")
            continue

        for az, already_done in zip(azimuths, _done):
            sample_id = f"{uid}_az{int(az):03d}"
            sample_dir = samples_dir / sample_id

            if already_done:
                continue

            try:
                tmp_dir = sample_dir.parent / f"_tmp_{sample_id}"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                image = render_mesh_to_image(
                    mesh,
                    elevation=args.elevation,
                    fov=args.fov,
                    size=args.image_size,
                    azimuth=float(az),
                    extrinsics_json_path=tmp_dir / "camera_extrinsics.json",
                )

                with torch.no_grad():
                    scene_codes = model([image], device=device)
                triplane = scene_codes[0].half().cpu()

                query_pts_np = sample_query_points(
                    raw, args.n_points, radius, centroid, longest
                )
                sdf_gt_np = compute_sdf(mesh, query_pts_np)

                torch.save(triplane, tmp_dir / "triplane.pt")
                torch.save(torch.from_numpy(query_pts_np), tmp_dir / "query_pts.pt")
                torch.save(torch.from_numpy(sdf_gt_np), tmp_dir / "sdf_gt.pt")
                image.save(tmp_dir / "input_view.png")
                tmp_dir.rename(sample_dir)

                pbar.set_postfix(objects=obj_saved, skipped=obj_skipped, uid=uid[:8])

                del image, scene_codes, triplane
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                if args.verbose:
                    tqdm.write(f"[skip sample] {sample_id}: {e}")

        obj_saved += 1
        pbar.update(1)
        pbar.set_postfix(objects=obj_saved, skipped=obj_skipped)

        del mesh, raw
        gc.collect()

    pbar.close()
    total_samples = obj_saved * args.azimuths_per_mesh
    print(f"\nPrecompute done — {obj_saved} objects ({total_samples} samples), "
          f"{obj_skipped} skipped -> {dataset_dir}")


# ─── Dataset ──────────────────────────────────────────────────────────────────

class SDFPointDataset(Dataset):
    """Loads all precomputed samples, queries triplane features once, and
    stores a flat (feats, pts, sdf, sample_id) tensor dataset in memory.

    Every sample **must** have ``camera_extrinsics.json``. Triplane features use
    ``p_trip = p_mesh @ R.T``; ``sdf_gt`` is ``SDF(mesh, p_mesh)``. Stored
    ``query_pts`` are in the normalized mesh frame (same centroid + scale as
    ``load_and_normalize_mesh``); they are clipped to ``[-radius, radius]^3`` at
    precompute and checked on load.
    """

    def __init__(self, dataset_dir: str, uid_whitelist: set | None = None,
                 sample_whitelist: set | None = None):
        root = Path(dataset_dir)
        all_samples = sorted((root / "samples").glob("*/triplane.pt"))
        if uid_whitelist is not None:
            all_samples = [p for p in all_samples
                           if p.parent.name.split("_az")[0] in uid_whitelist]
        if sample_whitelist is not None:
            all_samples = [p for p in all_samples
                           if p.parent.name in sample_whitelist]
        if not all_samples:
            raise RuntimeError(f"No precomputed samples found under {root}/samples/")

        with open(root / "metadata.json") as f:
            self.meta: dict = json.load(f)

        radius = self.meta["radius"]
        feature_reduction = self.meta["feature_reduction"]

        ordered_dirs: list[Path] = []
        seen_dir: set[str] = set()
        for sample_path in all_samples:
            d = sample_path.parent
            key = str(d)
            if key not in seen_dir:
                seen_dir.add(key)
                ordered_dirs.append(d)
        path_to_si = {pp: i for i, pp in enumerate(ordered_dirs)}
        R_rows = [load_R_world_from_recon_json_strict(d) for d in ordered_dirs]
        self.R_stack = torch.from_numpy(np.stack(R_rows, axis=0)).float()

        pts_list, sdf_list, feats_list, sid_list = [], [], [], []
        print(f"Loading {len(all_samples)} samples and computing triplane features...")
        for sample_path in all_samples:
            p = sample_path.parent
            si = path_to_si[p]
            triplane = torch.load(p / "triplane.pt", map_location="cpu", weights_only=False).float()
            pts = torch.load(p / "query_pts.pt", map_location="cpu", weights_only=False)
            sdf = torch.load(p / "sdf_gt.pt", map_location="cpu", weights_only=False)
            if (pts.abs() > float(radius) + 1e-4).any():
                raise RuntimeError(
                    f"query_pts outside [-radius, radius] in {p} (radius={radius}). "
                    "Re-run precompute."
                )
            pts = pts.clamp(-float(radius), float(radius))
            R = self.R_stack[si]
            with torch.no_grad():
                pts_trip = pts @ R.T
                feats = query_triplane_features(pts_trip, triplane, radius, feature_reduction)
            pts_list.append(pts)
            sdf_list.append(sdf)
            feats_list.append(feats)
            sid_list.append(torch.full((pts.shape[0],), si, dtype=torch.long))

        self.all_pts = torch.cat(pts_list, dim=0)
        self.all_sdf = torch.cat(sdf_list, dim=0)
        self.all_feats = torch.cat(feats_list, dim=0)
        self.point_sample_id = torch.cat(sid_list, dim=0)
        self.sample_dirs = [p.parent for p in all_samples]
        abs_sdf = self.all_sdf.abs()
        self.sdf_scale = float(torch.quantile(abs_sdf, TARGET_SCALE_QUANTILE).item()) \
            if abs_sdf.numel() > 0 else 1.0
        self.sdf_scale = max(self.sdf_scale, MIN_TARGET_SCALE)
        print(f"Dataset ready: {self.all_pts.shape[0]} points, feats shape {self.all_feats.shape}")
        print(f"SDF normalization scale (q={TARGET_SCALE_QUANTILE:.2f}): {self.sdf_scale:.6f}")

    def __len__(self) -> int:
        return self.all_pts.shape[0]

    def __getitem__(self, idx: int):
        return self.all_feats[idx], self.all_pts[idx], self.all_sdf[idx], self.point_sample_id[idx]


# ─── Visualization helpers ────────────────────────────────────────────────────

def render_mesh_views(
    mesh,
    phi_values: tuple = (45, 135, 225, 315),
    theta_deg: float = 50.0,
    image_size: int = 256,
    base_color: tuple = (0.7, 0.7, 0.85),
) -> list:
    import pyrender
    import trimesh as tr

    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    max_dim = float(np.max(bounds[1] - bounds[0]))

    yfov = np.radians(40.0)
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[*base_color, 1.0], metallicFactor=0.1, roughnessFactor=0.6
    )
    try:
        pr_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    except Exception:
        pr_mesh = pyrender.Mesh.from_trimesh(
            tr.Trimesh(vertices=mesh.vertices, faces=mesh.faces), material=material
        )

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0])
    scene.add(pr_mesh)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.5))
    fill = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5)
    fill_pose = np.eye(4)
    fill_pose[:3, :3] = tr.transformations.rotation_matrix(np.radians(120), [0, 1, 0])[:3, :3]
    scene.add(fill, pose=fill_pose)

    r = pyrender.OffscreenRenderer(image_size, image_size)
    theta_rad = np.radians(theta_deg)
    r_dist = (max_dim * 1.2) / (2.0 * np.tan(yfov / 2.0))

    renders = []
    for phi_deg in phi_values:
        phi_rad = np.radians(phi_deg)
        cx = center[0] + r_dist * np.sin(theta_rad) * np.cos(phi_rad)
        cy = center[1] + r_dist * np.sin(theta_rad) * np.sin(phi_rad)
        cz = center[2] + r_dist * np.cos(theta_rad)
        cam_pos = np.array([cx, cy, cz])
        forward = (center - cam_pos)
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            world_up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward
        pose[:3, 3] = cam_pos
        cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
        cam_node = scene.add(cam, pose=pose)
        color, _ = r.render(scene)
        renders.append(color)
        scene.remove_node(cam_node)

    r.delete()
    scene.clear()
    return renders


def reconstruct_mesh_from_triplane(
    sdf_mlp: nn.Module,
    triplane: torch.Tensor,
    radius: float,
    feature_reduction: str,
    resolution: int = 64,
    batch_size: int = 32768,
    device: torch.device = None,
    n_freqs: int = 0,
    R_world_from_trip: np.ndarray | None = None,
):
    """Run marching cubes on a dense SDF grid in TripoSR coords; optional ``R`` maps verts to mesh world."""
    import trimesh as tr

    if device is None:
        device = next(sdf_mlp.parameters()).device

    coords = torch.linspace(-radius, radius, resolution)
    xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing="ij")
    grid_pts = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

    triplane_dev = triplane.to(device)
    all_sdfs: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(grid_pts), batch_size):
            batch = grid_pts[i : i + batch_size].to(device)
            feats = query_triplane_features(batch, triplane_dev, radius, feature_reduction)
            if n_freqs > 0:
                feats = torch.cat([feats, fourier_encode(batch, n_freqs)], dim=-1)
            all_sdfs.append(sdf_mlp(feats).cpu())

    sdf_vol = torch.cat(all_sdfs).numpy().reshape(resolution, resolution, resolution)

    try:
        verts, faces, normals, _ = marching_cubes(sdf_vol, level=0.0)
    except ValueError:
        return None

    voxel_size = (2.0 * radius) / (resolution - 1)
    verts = verts * voxel_size - radius
    if R_world_from_trip is not None:
        R = np.asarray(R_world_from_trip, dtype=np.float64)
        verts = verts @ R
    return tr.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)


def create_mesh_comparison_visualization(
    gt_mesh,
    pred_mesh,
    title: str,
    save_path: Path,
    phi_values: tuple = (45, 135, 225, 315),
    theta_deg: float = 50.0,
    input_image=None,
) -> Path:
    from matplotlib.gridspec import GridSpec

    gt_renders = render_mesh_views(gt_mesh, phi_values=phi_values, theta_deg=theta_deg,
                                   base_color=(0.6, 0.7, 0.85))
    pred_renders = render_mesh_views(pred_mesh, phi_values=phi_values, theta_deg=theta_deg,
                                     base_color=(0.85, 0.6, 0.6))

    n = len(phi_values)
    n_rows = 3 if input_image is not None else 2
    fig_h = 4 * n_rows
    fig = plt.figure(figsize=(4 * n, fig_h))
    gs = GridSpec(n_rows, n, figure=fig, hspace=0.35, wspace=0.05)

    if input_image is not None:
        span = min(2, n)
        start = (n - span) // 2
        ax_in = fig.add_subplot(gs[0, start : start + span])
        ax_in.imshow(input_image)
        ax_in.set_title("TripoSR input", fontsize=11, fontweight="bold")
        ax_in.axis("off")
        for col in range(n):
            if not (start <= col < start + span):
                fig.add_subplot(gs[0, col]).axis("off")
        gt_row, pred_row = 1, 2
    else:
        gt_row, pred_row = 0, 1

    for i, img in enumerate(gt_renders):
        ax = fig.add_subplot(gs[gt_row, i])
        ax.imshow(img)
        ax.set_title(f"GT  phi={phi_values[i]}", fontsize=10)
        ax.axis("off")
    for i, img in enumerate(pred_renders):
        ax = fig.add_subplot(gs[pred_row, i])
        ax.imshow(img)
        ax.set_title(f"Pred  phi={phi_values[i]}", fontsize=10)
        ax.axis("off")

    stats = (f"GT:   {len(gt_mesh.vertices):,}v  {len(gt_mesh.faces):,}f\n"
             f"Pred: {len(pred_mesh.vertices):,}v  {len(pred_mesh.faces):,}f")
    fig.text(0.01, 0.01, stats, fontsize=9, family="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path


def visualize_reconstructions(
    sdf_mlp: nn.Module,
    seen_dirs: list,
    unseen_dirs: list,
    radius: float,
    feature_reduction: str,
    cache_dir: str,
    epoch: int,
    output_dir: Path,
    wandb_enabled: bool,
    device,
    resolution: int = 64,
    n_freqs: int = 0,
    elevation: float = 30.0,
    fov: float = 40.0,
    image_size: int = 256,
) -> None:
    sdf_mlp.eval()

    for label, sample_dirs in (("seen", seen_dirs), ("unseen", unseen_dirs)):
        for sample_dir in sample_dirs:
            uid = sample_dir.name.split("_az")[0]
            try:
                triplane = torch.load(
                    sample_dir / "triplane.pt", map_location="cpu", weights_only=False
                ).float()

                R_np = load_R_world_from_recon_json_strict(sample_dir)
                pred_mesh = reconstruct_mesh_from_triplane(
                    sdf_mlp,
                    triplane,
                    radius,
                    feature_reduction,
                    resolution=resolution,
                    device=device,
                    n_freqs=n_freqs,
                    R_world_from_trip=R_np,
                )
                if pred_mesh is None:
                    tqdm.write(f"[vis] marching cubes failed for {uid}")
                    continue

                uid_cache = Path(cache_dir) / uid
                mesh_files = [f for f in uid_cache.glob("*") if f.is_file()]
                if not mesh_files:
                    tqdm.write(f"[vis] no cached mesh for {uid}")
                    continue
                gt_mesh = load_and_normalize_mesh(str(mesh_files[0]), radius)

                az_str = sample_dir.name.split("_az")[-1]
                az_deg = float(az_str) if az_str.isdigit() else 0.0
                input_pil = render_mesh_to_image(
                    gt_mesh,
                    elevation=elevation,
                    fov=fov,
                    size=image_size,
                    azimuth=az_deg,
                    extrinsics_json_path=None,
                )
                input_image = np.array(input_pil)

                save_path = output_dir / label / f"{uid}_epoch{epoch:04d}.png"
                create_mesh_comparison_visualization(
                    gt_mesh, pred_mesh,
                    title=f"{label} - {uid[:12]} - epoch {epoch}",
                    save_path=save_path,
                    input_image=input_image,
                )

                if wandb_enabled:
                    try:
                        wandb.log({
                            f"mesh_reconstruction/{label}/{uid[:12]}": wandb.Image(str(save_path)),
                            "mesh_reconstruction/epoch": epoch,
                        })
                    except Exception:
                        pass

                del triplane, pred_mesh, gt_mesh
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                tqdm.write(f"[vis] failed for {uid}: {e}")

    sdf_mlp.train()


# ─── TRAIN phase ──────────────────────────────────────────────────────────────

def run_train(args: argparse.Namespace) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # ── wandb ─────────────────────────────────────────────────────────────────
    wandb_enabled = False

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_dir = Path(args.dataset_dir)
    cache_dir = str(dataset_dir / "mesh_cache")
    all_sample_paths = sorted((dataset_dir / "samples").glob("*/triplane.pt"))
    all_uids = sorted({p.parent.name.split("_az")[0] for p in all_sample_paths})

    rng_split = random.Random(42)
    shuffled_uids = list(all_uids)
    rng_split.shuffle(shuffled_uids)
    n_test = int(len(shuffled_uids) * args.test_fraction)
    test_uids  = set(shuffled_uids[:n_test])
    train_uids = set(shuffled_uids[n_test:])

    # Split views within train UIDs
    all_sample_names = sorted({p.parent.name for p in all_sample_paths})
    train_sample_names = [s for s in all_sample_names if s.split("_az")[0] in train_uids] \
                         if n_test > 0 else list(all_sample_names)
    test_view_names: set = set()
    if args.test_view_fraction > 0 and len(train_sample_names) > 1:
        rng_view = random.Random(43)
        rng_view.shuffle(train_sample_names)
        n_test_views = max(1, int(len(train_sample_names) * args.test_view_fraction))
        test_view_names = set(train_sample_names[:n_test_views])
        train_view_names = set(train_sample_names[n_test_views:])
    else:
        train_view_names = set(train_sample_names)

    dataset = SDFPointDataset(args.dataset_dir, sample_whitelist=train_view_names)
    meta = dataset.meta
    radius: float = meta["radius"]
    feature_reduction: str = meta["feature_reduction"]
    feat_dim: int = meta["feat_dim"]
    sdf_scale: float = dataset.sdf_scale
    R_stack_device = dataset.R_stack.to(device)

    print(f"Dataset: {len(dataset)} points | "
          f"{n_test} unseen UIDs, {len(test_view_names)} unseen views | "
          f"radius={radius:.4f} | feat_dim={feat_dim} | sdf_scale={sdf_scale:.6f}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # ── Visualization samples ─────────────────────────────────────────────────
    def _pick_vis_from_uids(uid_set: set, n: int) -> list:
        if not uid_set:
            return []
        chosen = random.Random(0).sample(sorted(uid_set), min(n, len(uid_set)))
        dirs = []
        for uid in chosen:
            az0 = dataset_dir / "samples" / f"{uid}_az000"
            if az0.exists():
                dirs.append(az0)
            else:
                candidates = sorted((dataset_dir / "samples").glob(f"{uid}_az*"))
                if candidates:
                    dirs.append(candidates[0].parent)
        return dirs

    def _pick_vis_from_names(name_set: set, n: int) -> list:
        if not name_set:
            return []
        chosen = random.Random(0).sample(sorted(name_set), min(n, len(name_set)))
        return [dataset_dir / "samples" / name for name in chosen]

    vis_seen_dirs   = _pick_vis_from_uids(train_uids, args.vis_seen)
    vis_unseen_dirs = (_pick_vis_from_uids(test_uids, args.vis_unseen)
                       + _pick_vis_from_names(test_view_names, args.vis_unseen))
    vis_output_dir  = Path(args.output_dir) / "vis"

    # ── Model ─────────────────────────────────────────────────────────────────
    n_freqs: int = args.n_freqs
    pe_dim: int = (3 + 6 * n_freqs) if n_freqs > 0 else 0
    mlp_in_dim: int = feat_dim + pe_dim
    print(f"MLP input: {feat_dim}-dim triplane feats + {pe_dim}-dim PE = {mlp_in_dim}")

    global MLP_IN_DIM
    MLP_IN_DIM = mlp_in_dim

    sdf_mlp = SDFMLP(
        in_dim=mlp_in_dim,
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
        use_tanh_output=args.use_tanh_output,
    ).to(device)
    optimizer = torch.optim.AdamW(sdf_mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        sdf_mlp.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    try:
        derived_meta = {
            "radius": radius,
            "feature_reduction": feature_reduction,
            "feat_dim": feat_dim,
            "mlp_in_dim": mlp_in_dim,
            "pe_dim": pe_dim,
            "sdf_scale": sdf_scale,
            "total_dataset_points": len(dataset),
            "loader_batches_per_epoch": len(loader),
            "n_train_uids": len(train_uids),
            "n_test_uids": len(test_uids),
            "n_train_views": len(train_view_names),
            "n_test_views": len(test_view_names),
            "start_epoch": start_epoch,
        }
        wb_config: dict = {
            "globals": wandb_collect_module_globals(),
            "args": {k: _wandb_jsonable(v) for k, v in vars(args).items()},
            "derived": {k: _wandb_jsonable(v) for k, v in derived_meta.items()},
        }
        wb_config.update(wandb_model_parameter_config(sdf_mlp))
        wandb.init(
            project="simple-sdf",
            name=args.run_name,
            config=wb_config,
            settings=wandb.Settings(_disable_stats=True, console="off"),
        )
        wandb_log_model_parameter_table(sdf_mlp)
        if len(loader) > 0:
            wandb.watch(
                sdf_mlp,
                log="all",
                log_freq=max(1, len(loader)),
            )
        wandb_enabled = True
    except Exception as e:
        print(f"Warning: wandb init failed: {e}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────

    total_steps = args.epochs * len(loader)
    pbar = tqdm(total=total_steps, initial=start_epoch * len(loader),
                desc=f"Epoch 1/{args.epochs}", dynamic_ncols=True, unit="step")

    for epoch in range(start_epoch, args.epochs):
        sdf_mlp.train()
        epoch_loss = epoch_sdf = epoch_eik = 0.0
        diag_steps = 0
        diag_grad_mean_sum = 0.0
        diag_grad_min = float("inf")
        diag_grad_max = float("-inf")
        diag_pred_min = float("inf")
        diag_pred_max = float("-inf")
        diag_tgt_min = float("inf")
        diag_tgt_max = float("-inf")
        pbar.set_description(f"Epoch {epoch + 1}/{args.epochs}")

        for base_feats, query_pts, sdf_gt, sid in loader:
            base_feats = base_feats.to(device)
            query_pts = query_pts.to(device).requires_grad_(True)
            sdf_gt = sdf_gt.to(device)
            sid = sid.to(device)

            if n_freqs > 0:
                trip = query_pts.clone()
                Rs = R_stack_device
                for u in sid.unique():
                    m = sid == u
                    Ru = Rs[u.long()]
                    trip[m] = query_pts[m] @ Ru.T
                model_feats = torch.cat([base_feats, fourier_encode(trip, n_freqs)], dim=-1)
            else:
                model_feats = base_feats

            sdf_pred = sdf_mlp(model_feats)
            sdf_gt_norm = (sdf_gt / sdf_scale).clamp(-1.0, 1.0)
            sdf_loss = F.mse_loss(sdf_pred, sdf_gt_norm)

            gradients = torch.autograd.grad(
                outputs=sdf_pred,
                inputs=query_pts,
                grad_outputs=torch.ones_like(sdf_pred),
                create_graph=True,
                retain_graph=True,
            )[0]
            grad_norm = gradients.norm(dim=-1)
            eikonal_loss = ((grad_norm - 1.0) ** 2).mean()

            loss = sdf_loss + args.eikonal_weight * eikonal_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sdf_mlp.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_sdf += sdf_loss.item()
            epoch_eik += eikonal_loss.item()
            diag_steps += 1
            grad_mean_val = float(grad_norm.mean().detach().item())
            grad_min_val = float(grad_norm.min().detach().item())
            grad_max_val = float(grad_norm.max().detach().item())
            pred_min_val = float(sdf_pred.min().detach().item())
            pred_max_val = float(sdf_pred.max().detach().item())
            tgt_min_val = float(sdf_gt_norm.min().detach().item())
            tgt_max_val = float(sdf_gt_norm.max().detach().item())
            diag_grad_mean_sum += grad_mean_val
            diag_grad_min = min(diag_grad_min, grad_min_val)
            diag_grad_max = max(diag_grad_max, grad_max_val)
            diag_pred_min = min(diag_pred_min, pred_min_val)
            diag_pred_max = max(diag_pred_max, pred_max_val)
            diag_tgt_min = min(diag_tgt_min, tgt_min_val)
            diag_tgt_max = max(diag_tgt_max, tgt_max_val)

            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.5f}",
                             sdf=f"{sdf_loss.item():.5f}",
                             eik=f"{eikonal_loss.item():.5f}")

            if wandb_enabled:
                try:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/sdf_loss": sdf_loss.item(),
                        "train/eikonal_loss": eikonal_loss.item(),
                    })
                except Exception:
                    pass

        # ── End of epoch ──────────────────────────────────────────────────────
        n = len(loader)
        grad_mean_epoch = (diag_grad_mean_sum / diag_steps) if diag_steps > 0 else float("nan")
        print(
            f"[diag] epoch {epoch + 1:04d} | "
            f"grad_norm(mean/min/max)={grad_mean_epoch:.4f}/{diag_grad_min:.4f}/{diag_grad_max:.4f} | "
            f"pred_range=[{diag_pred_min:.4f}, {diag_pred_max:.4f}] | "
            f"target_range=[{diag_tgt_min:.4f}, {diag_tgt_max:.4f}]"
        )
        if wandb_enabled:
            try:
                wandb.log({
                    "train/epoch_loss": epoch_loss / n,
                    "train/epoch_sdf_loss": epoch_sdf / n,
                    "train/epoch_eikonal_loss": epoch_eik / n,
                    "diag/grad_norm_mean": grad_mean_epoch,
                    "diag/grad_norm_min": diag_grad_min,
                    "diag/grad_norm_max": diag_grad_max,
                    "diag/pred_min": diag_pred_min,
                    "diag/pred_max": diag_pred_max,
                    "diag/target_min": diag_tgt_min,
                    "diag/target_max": diag_tgt_max,
                    "train/epoch": epoch + 1,
                })
            except Exception:
                pass

        is_last = (epoch + 1) == args.epochs
        if is_last or (epoch + 1) % args.save_every == 0:
            ckpt_path = output_dir / f"sdf_head_epoch{epoch + 1:04d}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model": sdf_mlp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "meta": meta,
                "args": vars(args),
                "sdf_scale": sdf_scale,
            }, ckpt_path)

        if args.vis_every > 0 and (epoch + 1) % args.vis_every == 0:
            visualize_reconstructions(
                sdf_mlp=sdf_mlp,
                seen_dirs=vis_seen_dirs,
                unseen_dirs=vis_unseen_dirs,
                radius=radius,
                feature_reduction=feature_reduction,
                cache_dir=cache_dir,
                epoch=epoch + 1,
                output_dir=vis_output_dir,
                wandb_enabled=wandb_enabled,
                device=device,
                resolution=args.vis_resolution,
                n_freqs=n_freqs,
                elevation=args.elevation,
                fov=args.fov,
                image_size=args.image_size,
            )

    pbar.close()

    if wandb_enabled:
        try:
            wandb.finish()
        except Exception:
            pass
    print("\nTraining complete.")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    args = argparse.Namespace(
        dataset_dir    = DATASET_DIR,
        model          = MODEL,
        n_objects      = N_OBJECTS,
        azimuths_per_mesh = AZIMUTHS_PER_MESH,
        n_points       = N_POINTS,
        image_size     = IMAGE_SIZE,
        elevation      = ELEVATION,
        fov            = FOV,
        max_mesh_mb    = MAX_MESH_MB,
        verbose        = VERBOSE,
        output_dir     = OUTPUT_DIR,
        epochs         = EPOCHS,
        save_every     = SAVE_EVERY,
        hidden_dim     = HIDDEN_DIM,
        n_hidden       = N_HIDDEN,
        n_freqs        = N_FREQS,
        lr             = LR,
        eikonal_weight = EIKONAL_WEIGHT,
        sdf_clamp      = SDF_CLAMP,
        num_workers    = NUM_WORKERS,
        run_name       = RUN_NAME,
        weight_decay   = WEIGHT_DECAY,
        use_tanh_output = USE_TANH_OUTPUT,
        test_fraction  = TEST_FRACTION,
        test_view_fraction = TEST_VIEW_FRACTION,
        vis_every      = VIS_EVERY,
        vis_seen       = VIS_SEEN,
        vis_unseen     = VIS_UNSEEN,
        vis_resolution = VIS_RESOLUTION,
        batch_size     = BATCH_SIZE,
        resume         = RESUME,
    )

    if COMMAND == "precompute":
        run_precompute(args)
    elif COMMAND == "train":
        run_train(args)
    elif COMMAND == "both":
        run_precompute(args)
        run_train(args)
    else:
        print(f"Unknown COMMAND: {COMMAND!r}")


if __name__ == "__main__":
    main()
