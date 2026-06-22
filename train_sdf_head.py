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
import time
from datetime import timedelta
import itertools
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(0, str(Path(__file__).parent))

import argparse
import random
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
from einops import rearrange
from PIL import Image
from skimage.measure import marching_cubes
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from tsr.utils import get_activation, scale_tensor

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

COMMAND = "both"

# ── Shared ──────────────────────────────────────────────────────────────────
DATASET_DIR     = "/home/markiv/TripoSR/sdf_dataset"

# ── Precompute ───────────────────────────────────────────────────────────────
MODEL                 = "stabilityai/TripoSR"
N_OBJECTS             = 100
AZIMUTHS_PER_MESH     = 5              # × len(ELEVATIONS) = 10 views per object
ELEVATIONS            = [15.0, 30.0]   # two elevation bands; replaces single ELEVATION
N_POINTS              = 32768         # = 64^3
NEAR_SURFACE_FRACTION = 0.0            # 0 = fully uniform in [-radius, radius]³ (mimics MC inference grid);
                                       # near-surface supervision comes only from the training-time Pass-2 augment
IMAGE_SIZE            = 256
FOV                   = 40.0
MAX_MESH_MB           = 0.0
MAX_TRIANGLES         = 500_000       # skip meshes with more faces than this (BVH RAM guard)
VERBOSE               = False

# ── Train ────────────────────────────────────────────────────────────────────
OUTPUT_DIR      = "/home/markiv/TripoSR/sdf_checkpoints"
EPOCHS          = 500
SAVE_EVERY      = 25
HIDDEN_DIM      = 128
HIDDEN_DIM_NO_TRIPLANE = 256
N_HIDDEN        = 6
N_FREQS         = 6
LR              = 1e-3
LR_MIN          = 1e-5   # cosine annealing floor (only used when USE_ONECYCLE=False)
GRAD_CLIP       = 1.0    # max grad norm for clip_grad_norm_ (lower = more aggressive)
LOSS_REJECT_K   = 3.0    # drop points with per-point loss > mean + k·std (0 disables)
USE_ONECYCLE    = True   # OneCycleLR (warmup→anneal) instead of cosine annealing
ONECYCLE_PCT_START = 0.1 # fraction of total steps spent warming LR up to max
EIKONAL_WEIGHT        = 1e-3
SIGN_BCE_WEIGHT       = 0.1    # auxiliary sign-classification loss weight (0 disables)
SIGN_BCE_ALPHA        = 20.0   # sigmoid temperature: larger → sharper sign boundary
SIGN_BCE_EPSILON      = 0.02   # exclude |sdf_gt| < this (ambiguous near-surface sign)
SURFACE_LOSS_SIGMA    = 0.05   # exp(-|sdf|/sigma) weighting; ~5% of object scale
NEAR_SURFACE_SIGMA    = 0.02   # Gaussian noise std when perturbing near-surface pts
NEAR_SURFACE_THRESHOLD = 0.05  # |pred SDF| < this → point is considered near-surface
NEAR_SURFACE_RATIO = 1.0  # near-surface pts per sample = ratio × N_POINTS (1.0 ⇒ equal to uniform)
AUGMENT_BUDGET_SEC    = 20     # wall-clock budget for the per-step trimesh augment
NCCL_TIMEOUT_MIN      = 30     # DDP collective timeout (headroom for slow augment steps)
SDF_CLAMP             = 0.0
NUM_WORKERS     = 4
RUN_NAME        = "v0.52_100"
TEST_FRACTION   = 0.2        # fraction of meshes (UIDs) held out as unseen
TEST_VIEW_FRACTION = 0.2     # fraction of views held out per mesh
VIS_EVERY       = 25
VIS_SEEN        = 3
VIS_UNSEEN      = 3
VIS_AZIMUTHS_PER_OBJECT = 5
VIS_RESOLUTION  = 64
BATCH_SIZE        = 4096
SAMPLES_PER_BATCH = 4      # samples loaded per DataLoader step; points per step = SAMPLES_PER_BATCH × N_POINTS
RESUME            = None
WEIGHT_DECAY    = 1e-4
USE_TANH_OUTPUT = False
USE_TRIPLANE_FEATURES  = True   # ablation: set False to zero out triplane features (PE only)
USE_NERF_VIS           = True  # load TripoSR decoder + render NeRF mesh during visualization

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
    """Triplane features + optional Fourier PE -> scalar signed distance.

    The triplane block and the Fourier PE block are LayerNormed independently
    before being concatenated, so the two very-different-statistics inputs do
    not drown each other out inside a single LayerNorm.
    """

    def __init__(
        self,
        in_dim: int = 120,
        hidden_dim: int = 256,
        n_hidden: int = 3,
        use_tanh_output: bool = False,
        feat_dim: int | None = None,
        pe_dim: int = 0,
    ):
        super().__init__()
        if feat_dim is None:
            feat_dim = in_dim - pe_dim
        assert feat_dim + pe_dim == in_dim, (
            f"feat_dim ({feat_dim}) + pe_dim ({pe_dim}) must equal in_dim ({in_dim})"
        )
        self.feat_dim = feat_dim
        self.pe_dim = pe_dim
        self.feat_ln = nn.LayerNorm(feat_dim) if feat_dim > 0 else None
        self.pe_ln = nn.LayerNorm(pe_dim) if pe_dim > 0 else None

        layers: list[nn.Module] = [
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
        ]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        self.use_tanh_output = use_tanh_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.feat_ln is not None and self.pe_ln is not None:
            feats = self.feat_ln(x[..., : self.feat_dim])
            pe = self.pe_ln(x[..., self.feat_dim :])
            x = torch.cat([feats, pe], dim=-1)
        elif self.pe_ln is not None:
            x = self.pe_ln(x)
        else:
            x = self.feat_ln(x)
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

def surface_weighted_se(pred: torch.Tensor, target: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    """Per-point surface-weighted squared error (NOT reduced)."""
    weights = torch.exp(-target.abs() / sigma)
    weights = weights / weights.mean()
    return weights * (pred - target) ** 2


def surface_weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    return surface_weighted_se(pred, target, sigma).mean()


def sign_bce_loss(pred: torch.Tensor, target: torch.Tensor,
                  alpha: float = 20.0, epsilon: float = 0.02) -> torch.Tensor:
    """Auxiliary sign-classification loss: get inside/outside right so marching
    cubes produces correct topology. Points with |target| < epsilon are excluded
    (sign is ambiguous near the surface). label 1 = outside (target > 0)."""
    mask = target.abs() > epsilon
    if not mask.any():
        return pred.new_zeros(())
    logits = alpha * pred[mask]
    labels = (target[mask] > 0).float()
    return F.binary_cross_entropy_with_logits(logits, labels)


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
    """Copy ``raw``, apply centroid removal + unit-longest-edge scale (render_to_triposr convention)."""
    if len(raw.faces) == 0:
        raise ValueError("Mesh has no faces")
    centroid = np.asarray(raw.centroid, dtype=np.float64)
    longest = float(max(raw.extents) if max(raw.extents) > 0 else 1.0)
    mesh = raw.copy()
    mesh.apply_translation(-raw.centroid)
    mesh.apply_scale(1.0 / longest)
    return mesh, centroid, longest


def load_and_normalize_mesh(path: str, radius: float):
    raw = _load_trimesh(path)
    mesh, _, _ = _normalize_mesh_copy(raw, radius)
    return mesh


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
        axis=0,
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
            "train_sdf_head precompute: mesh centroid at origin, longest edge scaled to 1 "
            "(render_to_triposr convention). Camera is OpenGL camera-to-world."
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


def compute_sdf(mesh, points: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    """Signed distance: negative inside, positive outside.

    Avoids mesh.contains() (ray-casting, extremely memory-intensive) by deriving
    the sign from the dot product of (query − closest_surface) with the face
    normal at the closest triangle.  This is O(N) RAM instead of O(N × triangles).

    ProximityQuery is built once per call so the BVH is not rebuilt per batch.
    """
    import trimesh.proximity
    points = np.asarray(points, dtype=np.float64)
    prox = trimesh.proximity.ProximityQuery(mesh)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)

    sdf = np.empty(len(points), dtype=np.float32)
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        closest, distances, tri_ids = prox.on_surface(batch)
        # Sign: positive outside (dot > 0), negative inside (dot < 0)
        direction = batch - closest
        dot = np.einsum("ij,ij->i", direction, face_normals[tri_ids])
        sign = np.where(dot >= 0, 1.0, -1.0)
        sdf[i : i + batch_size] = (sign * distances).astype(np.float32)

    del prox  # release BVH memory explicitly
    return sdf


def sample_query_points(
    mesh,
    n_points: int,
    radius: float,
    near_surface_fraction: float = 0.5,
    near_surface_std: float | None = None,
) -> np.ndarray:
    """Sample query points in the full TripoSR scene volume [-radius, radius]³.

    With near_surface_fraction=0 (default for training), all points are drawn
    uniformly from the same [-radius, radius]³ cube that marching-cubes evaluates
    during inference — eliminating the training/inference distribution mismatch.
    """
    if near_surface_std is None:
        near_surface_std = float(radius * 0.04)
    n_near    = int(n_points * near_surface_fraction)
    n_uniform = n_points - n_near

    # Uniform samples span the full triplane query volume, not just mesh bounds.
    uniform = np.random.uniform(-radius, radius, (n_uniform, 3)).astype(np.float32)

    if n_near > 0 and len(mesh.vertices) > 0:
        idx   = np.random.choice(len(mesh.vertices), n_near, replace=True)
        verts = np.asarray(mesh.vertices[idx], dtype=np.float64)
        noise = np.random.normal(0.0, near_surface_std, (n_near, 3))
        near  = (verts + noise).astype(np.float32)
    else:
        near = np.zeros((0, 3), dtype=np.float32)

    pts = np.concatenate([uniform, near], axis=0) if n_near > 0 else uniform
    return np.clip(pts, -radius, radius)


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
    # Match render_to_triposr.py: with unit-longest-edge mesh, use fixed framing distance.
    distance = (0.7 / np.tan(fov_rad / 2.0))
    T_cam = _camera_pose(azimuth, elevation, distance)
    scene.add(pyrender.PerspectiveCamera(yfov=fov_rad, aspectRatio=1.0), pose=T_cam)
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0),
              pose=_camera_pose(azimuth + 20, elevation + 20, 1.0))
    scene.add(pyrender.DirectionalLight(color=[0.7, 0.8, 1.0], intensity=1.5),
              pose=_camera_pose(azimuth + 180, elevation - 10, 1.0))
    if extrinsics_json_path is not None:
        _write_camera_extrinsics_json(
            extrinsics_json_path,
            T_cam,
            azimuth_deg=float(azimuth),
            elevation_deg=float(elevation),
            distance=float(distance),
            fov_deg=float(fov),
        )
    # Match render_to_triposr.py: render at 2x and Lanczos-downsample for free AA.
    render_size = int(size) * 2
    r = pyrender.OffscreenRenderer(render_size, render_size)
    color, _ = r.render(scene)
    r.delete()
    scene.clear()
    del scene
    return Image.fromarray(color).resize((int(size), int(size)), Image.LANCZOS)


# ─── Objaverse helpers ────────────────────────────────────────────────────────

def get_objaverse_uid_pool(seed: int = 42, quiet: bool = False) -> list[str]:
    import objaverse
    if not quiet:
        print("[objaverse] Loading UID list...")
    all_uids = list(objaverse.load_uids())
    rng = random.Random(seed)
    rng.shuffle(all_uids)
    if not quiet:
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
    # Multi-GPU shard: each rank processes a disjoint subset of UIDs.
    # No dist.init_process_group needed — ranks write independently to shared disk.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank       = int(os.environ.get("RANK", 0))
    is_main    = (rank == 0)

    dataset_dir = Path(args.dataset_dir)
    samples_dir = dataset_dir / "samples"
    if samples_dir.exists() and any(samples_dir.iterdir()):
        existing = sum(1 for _ in samples_dir.iterdir() if not _.name.startswith("_tmp"))
        if world_size > 1:
            if is_main:
                print(f"Found {existing} existing samples; keeping existing work "
                      f"(delete manually to restart from scratch).")
        else:
            answer = input(
                f"Found {existing} existing samples in {samples_dir}.\n"
                f"Delete all and restart? [y/N] "
            ).strip().lower()
            if answer in ("y", "yes"):
                shutil.rmtree(samples_dir)
                print("Deleted existing samples.")
            else:
                print("Keeping existing samples (will skip already-done views).")

    from tsr.system import TSR

    if not args.verbose:
        import warnings
        warnings.filterwarnings("ignore")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main:
        suffix = f" across {world_size} GPUs" if world_size > 1 else ""
        print(f"Precompute on {device}{suffix}")

    if is_main:
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

    # Only rank 0 writes metadata (all ranks would write identical content).
    if is_main:
        metadata = {
            "radius": radius,
            "feature_reduction": feature_reduction,
            "feat_dim": feat_dim,
            "n_points": args.n_points,
            "near_surface_fraction": args.near_surface_fraction,
            "elevations": args.elevations,
        }
        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata -> {dataset_dir / 'metadata.json'}")

    uids = get_objaverse_uid_pool(quiet=not is_main)
    # Strided shard: rank r gets uids[r], uids[r+W], uids[r+2W], ...
    # Disjoint across ranks, same fixed-seed shuffle on all ranks.
    uids_for_rank  = uids[rank::world_size]
    target_objects = (args.n_objects + world_size - 1) // world_size  # ceil(N/W) per rank

    azimuths   = np.linspace(0, 360, args.azimuths_per_mesh, endpoint=False)
    elevations = list(args.elevations)
    view_pairs = list(itertools.product(azimuths, elevations))  # (az, el) × n_views
    n_views    = len(view_pairs)

    if is_main:
        print(f"View pairs ({n_views} per object): "
              f"{args.azimuths_per_mesh} azimuths × {len(elevations)} elevations")
        if world_size > 1:
            print(f"Distributed precompute: {world_size} ranks, "
                  f"~{target_objects} objects each (~{target_objects * world_size} total)")

    pbar = tqdm(total=target_objects, unit="obj", dynamic_ncols=True,
                desc=f"rank{rank}" if world_size > 1 else "precompute",
                position=rank, leave=True)
    obj_saved = obj_skipped = 0

    for uid in uids_for_rank:
        if obj_saved >= target_objects:
            break

        _done = [
            all((samples_dir / f"{uid}_az{int(az):03d}_el{int(el):03d}" / fn).exists()
                for fn in ("triplane.pt", "query_pts.pt", "sdf_gt.pt", "camera_extrinsics.json"))
            for az, el in view_pairs
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
                    if args.verbose:
                        tqdm.write(f"[skip mesh] {uid}: {size_mb:.1f} MB > limit {args.max_mesh_mb} MB")
                    obj_skipped += 1
                    pbar.set_postfix(objects=obj_saved, skipped=obj_skipped)
                    continue
            raw = _load_trimesh(mesh_path)
            mesh, _, _ = _normalize_mesh_copy(raw, radius)
            del raw  # raw is no longer needed; release before heavy SDF computation
            if not mesh.is_watertight:
                del mesh
                obj_skipped += 1
                pbar.set_postfix(objects=obj_saved, skipped=obj_skipped)
                continue
            if args.max_triangles > 0 and len(mesh.faces) > args.max_triangles:
                if args.verbose:
                    tqdm.write(f"[skip mesh] {uid}: {len(mesh.faces):,} faces > limit {args.max_triangles:,}")
                del mesh
                obj_skipped += 1
                pbar.set_postfix(objects=obj_saved, skipped=obj_skipped)
                continue
        except Exception as e:
            obj_skipped += 1
            pbar.set_postfix(objects=obj_saved, skipped=obj_skipped)
            if args.verbose:
                tqdm.write(f"[skip mesh] {uid}: {e}")
            continue  # silent skip by default

        # Precompute SDF once per object (query points are shared across views)
        query_pts_np = sample_query_points(
            mesh, args.n_points, radius,
            near_surface_fraction=args.near_surface_fraction,
        )
        sdf_gt_np = compute_sdf(mesh, query_pts_np)

        for (az, el), already_done in zip(view_pairs, _done):
            sample_id  = f"{uid}_az{int(az):03d}_el{int(el):03d}"
            sample_dir = samples_dir / sample_id

            if already_done:
                continue

            try:
                tmp_dir = sample_dir.parent / f"_tmp_{sample_id}"
                tmp_dir.mkdir(parents=True, exist_ok=True)

                # 1. Render to a temporary PNG and re-load as RGB array so the
                #    bytes are identical to what run.py / render_to_triposr.py produce.
                tmp_png = tmp_dir / "_render_tmp.png"
                pil_image = render_mesh_to_image(
                    mesh,
                    elevation=float(el),
                    fov=args.fov,
                    size=args.image_size,
                    azimuth=float(az),
                    extrinsics_json_path=tmp_dir / "camera_extrinsics.json",
                )
                pil_image.save(tmp_png)
                image_np = np.array(Image.open(tmp_png).convert("RGB"))
                tmp_png.unlink()  # not stored — only needed for TripoSR input
                del pil_image

                # 2. Run TripoSR to get the triplane (scene_codes).
                with torch.no_grad():
                    scene_codes = model([image_np], device=device)
                triplane = scene_codes[0].half().cpu()
                del image_np, scene_codes

                # 3. Persist core training tensors only (no source_mesh.obj, no input_view.png).
                torch.save(triplane, tmp_dir / "triplane.pt")
                torch.save(torch.from_numpy(query_pts_np), tmp_dir / "query_pts.pt")
                torch.save(torch.from_numpy(sdf_gt_np), tmp_dir / "sdf_gt.pt")

                tmp_dir.rename(sample_dir)
                pbar.set_postfix(objects=obj_saved, skipped=obj_skipped, uid=uid[:8])

                del triplane
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                if args.verbose:
                    import traceback
                    tqdm.write(
                        f"[skip sample] {sample_id}: {e}\n"
                        + traceback.format_exc()
                    )

        obj_saved += 1
        pbar.update(1)
        pbar.set_postfix(objects=obj_saved, skipped=obj_skipped)

        del mesh, query_pts_np, sdf_gt_np
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pbar.close()
    if is_main:
        total_samples = obj_saved * n_views
        print(f"\nPrecompute done — {obj_saved} objects ({total_samples} samples), "
              f"{obj_skipped} skipped -> {dataset_dir}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


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
                 sample_whitelist: set | None = None, cache_subdir: str = "_flat_cache"):
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

        # ── Flat cache (memory-mapped) ─────────────────────────────────────────
        # Built once from the precomputed samples; on all subsequent runs every
        # process mmaps the same files so only the pages touched per batch are
        # paged in, and the OS shares physical pages across DDP ranks.
        cache_dir  = root / cache_subdir
        cache_meta = cache_dir / "meta.json"
        sample_names = [p.parent.name for p in all_samples]

        cache_valid = False
        if cache_dir.exists() and cache_meta.exists():
            try:
                with open(cache_meta) as f:
                    cache_valid = json.load(f).get("sample_names") == sample_names
            except Exception:
                pass

        if not cache_valid:
            print(f"Building flat cache for {len(all_samples)} samples "
                  f"(one-time cost; mmap'd on all future runs)...")
            cache_dir.mkdir(exist_ok=True)

            ordered_dirs: list[Path] = []
            seen_dir: set[str] = set()
            for sp in all_samples:
                d = sp.parent
                if str(d) not in seen_dir:
                    seen_dir.add(str(d))
                    ordered_dirs.append(d)
            path_to_si = {pp: i for i, pp in enumerate(ordered_dirs)}
            R_rows = [load_R_world_from_recon_json_strict(d) for d in ordered_dirs]
            R_stack = torch.from_numpy(np.stack(R_rows, axis=0)).float()
            np.save(cache_dir / "R_stack.npy", R_stack.numpy())

            # ── Determine total points and feature dim from one sample ─────────
            # n_points is fixed per sample (set at precompute time); feat_dim
            # comes from the actual triplane query so we don't hard-code it.
            n_pts_per_sample: int = self.meta["n_points"]
            total_points = len(all_samples) * n_pts_per_sample
            sp0 = all_samples[0]
            _triplane0 = torch.load(sp0.parent / "triplane.pt", map_location="cpu",
                                    weights_only=False).float()
            _pts0 = torch.load(sp0.parent / "query_pts.pt", map_location="cpu",
                               weights_only=False).clamp(-float(radius), float(radius))
            with torch.no_grad():
                _feats0 = query_triplane_features(
                    _pts0 @ R_stack[path_to_si[sp0.parent]].T,
                    _triplane0, radius, feature_reduction,
                )
            feat_dim_actual = _feats0.shape[-1]
            del _triplane0, _pts0, _feats0
            gc.collect()

            print(f"  Allocating mmap: {total_points:,} pts × {feat_dim_actual}-dim feats "
                  f"({total_points * feat_dim_actual * 4 / 1e9:.1f} GB on disk)")

            # ── Pre-allocate output files as memory-mapped arrays ──────────────
            # open_memmap writes a proper .npy header so np.load(mmap_mode='r')
            # can read them back without any format conversion.
            mm_feats = np.lib.format.open_memmap(
                cache_dir / "feats.npy", mode="w+", dtype=np.float32,
                shape=(total_points, feat_dim_actual))
            mm_pts   = np.lib.format.open_memmap(
                cache_dir / "pts.npy",   mode="w+", dtype=np.float32,
                shape=(total_points, 3))
            mm_sdf   = np.lib.format.open_memmap(
                cache_dir / "sdf.npy",   mode="w+", dtype=np.float32,
                shape=(total_points,))
            mm_sid   = np.lib.format.open_memmap(
                cache_dir / "sid.npy",   mode="w+", dtype=np.int64,
                shape=(total_points,))

            # ── Single pass: write each sample directly into its mmap slice ───
            offset = 0
            for sp in tqdm(all_samples, desc="building cache", unit="sample", leave=False):
                p = sp.parent
                si = path_to_si[p]
                triplane = torch.load(p / "triplane.pt", map_location="cpu",
                                      weights_only=False).float()
                pts = torch.load(p / "query_pts.pt", map_location="cpu",
                                 weights_only=False).clamp(-float(radius), float(radius))
                sdf = torch.load(p / "sdf_gt.pt",    map_location="cpu",
                                 weights_only=False)
                with torch.no_grad():
                    feats = query_triplane_features(
                        pts @ R_stack[si].T, triplane, radius, feature_reduction)

                n = pts.shape[0]
                mm_feats[offset : offset + n] = feats.numpy()
                mm_pts  [offset : offset + n] = pts.numpy()
                mm_sdf  [offset : offset + n] = sdf.numpy()
                mm_sid  [offset : offset + n] = si
                offset += n

                del triplane, pts, sdf, feats

            mm_feats.flush(); mm_pts.flush(); mm_sdf.flush(); mm_sid.flush()
            del mm_feats, mm_pts, mm_sdf, mm_sid
            gc.collect()

            with open(cache_meta, "w") as f:
                json.dump({"sample_names": sample_names}, f)
            print(f"Cache built → {cache_dir}")

        # mmap: the OS pages in only what each batch touches, shared across ranks
        self.all_feats       = np.load(cache_dir / "feats.npy", mmap_mode="r")
        self.all_pts         = np.load(cache_dir / "pts.npy",   mmap_mode="r")
        self.all_sdf         = np.load(cache_dir / "sdf.npy",   mmap_mode="r")
        self.point_sample_id = np.load(cache_dir / "sid.npy",   mmap_mode="r")
        self.R_stack         = torch.from_numpy(
                                   np.load(cache_dir / "R_stack.npy"))
        self.sample_dirs     = [sp.parent for sp in all_samples]
        print(f"Dataset: {self.all_pts.shape[0]:,} points  "
              f"(mmap from {cache_dir.name}/)")

    def __len__(self) -> int:
        return self.all_pts.shape[0]

    def __getitem__(self, idx: int):
        # numpy memmap indexing returns ndarray; DataLoader collate converts to tensor
        return (self.all_feats[idx], self.all_pts[idx],
                self.all_sdf[idx],   self.point_sample_id[idx])


# ─── Lazy dataset (no flat cache) ────────────────────────────────────────────

class SDFLazyDataset(Dataset):
    """Per-sample lazy loader — no flat cache.

    Workers load raw tensors only (~6 MB per sample). Feature computation runs
    in the training loop on GPU to avoid filling system RAM with large feature
    tensors across many DataLoader worker processes.

    Each __getitem__ returns:
        pts_mesh : (N_POINTS, 3)          — query pts in normalized mesh frame
        sdf      : (N_POINTS,)            — GT SDF for uniform pts
        triplane : (3, Cp, Hp, Wp)        — raw triplane (queried on GPU in training loop)
        R        : (3, 3)                 — rotation mesh→triplane frame
        uid      : str                    — object UID (for loading mesh / BVH cache)
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

        self.radius: float = float(self.meta["radius"])
        self.feature_reduction: str = self.meta["feature_reduction"]
        self.sample_dirs: list[Path] = [p.parent for p in all_samples]
        self.R_list: list[np.ndarray] = [
            load_R_world_from_recon_json_strict(d) for d in self.sample_dirs
        ]

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int):
        p = self.sample_dirs[idx]
        R = torch.from_numpy(self.R_list[idx]).float()
        triplane = torch.load(p / "triplane.pt", map_location="cpu",
                              weights_only=False).float()
        pts = torch.load(p / "query_pts.pt", map_location="cpu",
                         weights_only=False).clamp(-self.radius, self.radius)
        sdf = torch.load(p / "sdf_gt.pt", map_location="cpu", weights_only=False)
        uid = p.name.split("_az")[0]
        # Feature computation happens in the training loop on GPU, not here.
        return pts, sdf, triplane, R, uid


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
    use_triplane_features: bool = True,
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
            if use_triplane_features:
                feats = query_triplane_features(batch, triplane_dev, radius, feature_reduction)
                if n_freqs > 0:
                    feats = torch.cat([feats, fourier_encode(batch, n_freqs)], dim=-1)
            else:
                feats = fourier_encode(batch, n_freqs) if n_freqs > 0 else batch
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


def reconstruct_mesh_nerf_decoder(
    triposr_decoder: nn.Module,
    triplane: torch.Tensor,
    radius: float,
    feature_reduction: str,
    density_activation: str,
    density_bias: float,
    resolution: int = 64,
    batch_size: int = 32768,
    threshold: float = 25.0,
    device: torch.device = None,
    R_world_from_trip: np.ndarray | None = None,
):
    """Run TripoSR's original NeRF decoder on a grid and extract a mesh via marching cubes.

    When R_world_from_trip is provided, applies verts @ R to rotate the marching-cubes
    output from TripoSR space into world space (same convention as reconstruct_mesh_from_triplane).
    """
    import trimesh as tr

    if device is None:
        device = next(triposr_decoder.parameters()).device

    coords = torch.linspace(-radius, radius, resolution)
    xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing="ij")
    grid_pts = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

    triplane_dev = triplane.to(device)
    act_fn = get_activation(density_activation)
    all_density: list[torch.Tensor] = []

    with torch.no_grad():
        for i in range(0, len(grid_pts), batch_size):
            batch = grid_pts[i : i + batch_size].to(device)
            feats = query_triplane_features(batch, triplane_dev, radius, feature_reduction)
            raw = triposr_decoder(feats)["density"].squeeze(-1)
            all_density.append(act_fn(raw + density_bias).cpu())

    density_vol = torch.cat(all_density).numpy().reshape(resolution, resolution, resolution)

    try:
        verts, faces, normals, _ = marching_cubes(-(density_vol - threshold), level=0.0)
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
    nerf_mesh=None,
) -> Path:
    from matplotlib.gridspec import GridSpec

    gt_renders = render_mesh_views(gt_mesh, phi_values=phi_values, theta_deg=theta_deg,
                                   base_color=(0.6, 0.7, 0.85))
    pred_renders = render_mesh_views(pred_mesh, phi_values=phi_values, theta_deg=theta_deg,
                                     base_color=(0.85, 0.6, 0.6))
    nerf_renders = (
        render_mesh_views(nerf_mesh, phi_values=phi_values, theta_deg=theta_deg,
                          base_color=(0.6, 0.85, 0.65))
        if nerf_mesh is not None else None
    )

    n = len(phi_values)
    n_mesh_rows = 2 + (1 if nerf_renders is not None else 0)
    n_rows = n_mesh_rows + (1 if input_image is not None else 0)
    fig = plt.figure(figsize=(4 * n, 4 * n_rows))
    gs = GridSpec(n_rows, n, figure=fig, hspace=0.35, wspace=0.05)

    row_offset = 0
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
        row_offset = 1

    gt_row   = row_offset
    nerf_row = row_offset + 1
    pred_row = row_offset + (2 if nerf_renders is not None else 1)

    for i, img in enumerate(gt_renders):
        ax = fig.add_subplot(gs[gt_row, i])
        ax.imshow(img)
        ax.set_title(f"GT  phi={phi_values[i]}", fontsize=10)
        ax.axis("off")

    if nerf_renders is not None:
        for i, img in enumerate(nerf_renders):
            ax = fig.add_subplot(gs[nerf_row, i])
            ax.imshow(img)
            ax.set_title(f"NeRF dec  phi={phi_values[i]}", fontsize=10)
            ax.axis("off")

    for i, img in enumerate(pred_renders):
        ax = fig.add_subplot(gs[pred_row, i])
        ax.imshow(img)
        ax.set_title(f"SDF MLP  phi={phi_values[i]}", fontsize=10)
        ax.axis("off")

    stats_lines = [
        f"GT:      {len(gt_mesh.vertices):,}v  {len(gt_mesh.faces):,}f",
        f"SDF MLP: {len(pred_mesh.vertices):,}v  {len(pred_mesh.faces):,}f",
    ]
    if nerf_mesh is not None:
        stats_lines.insert(1, f"NeRF dec:{len(nerf_mesh.vertices):,}v  {len(nerf_mesh.faces):,}f")
    fig.text(0.01, 0.01, "\n".join(stats_lines), fontsize=9, family="monospace",
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
    triposr_decoder: nn.Module | None = None,
    density_activation: str = "exp",
    density_bias: float = -1.0,
    nerf_threshold: float = 25.0,
    use_triplane_features: bool = True,
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
                    use_triplane_features=use_triplane_features,
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

                # Parse azimuth and elevation from sample name: {uid}_az{az:03d}_el{el:03d}
                suffix   = sample_dir.name.split("_az")[-1]  # e.g. "045_el015"
                az_part  = suffix.split("_el")[0]
                el_part  = suffix.split("_el")[1] if "_el" in suffix else None
                az_deg   = float(az_part) if az_part.isdigit() else 0.0
                el_deg   = float(el_part) if (el_part is not None and el_part.isdigit()) else elevation
                input_pil = render_mesh_to_image(
                    gt_mesh,
                    elevation=el_deg,
                    fov=fov,
                    size=image_size,
                    azimuth=az_deg,
                    extrinsics_json_path=None,
                )
                input_image = np.array(input_pil)

                # ── Optional NeRF decoder sanity mesh (USE_NERF_VIS=True only) ─
                nerf_mesh = None
                if triposr_decoder is not None:
                    try:
                        nerf_mesh = reconstruct_mesh_nerf_decoder(
                            triposr_decoder,
                            triplane,
                            radius,
                            feature_reduction,
                            density_activation,
                            density_bias,
                            resolution=resolution,
                            threshold=nerf_threshold,
                            device=device,
                            R_world_from_trip=R_np,
                        )
                        if nerf_mesh is None:
                            tqdm.write(f"[vis] NeRF decoder marching cubes failed for {uid}")
                    except Exception as e:
                        tqdm.write(f"[vis] NeRF decoder mesh failed for {uid}: {e}")

                save_path = output_dir / label / f"{uid}_epoch{epoch:04d}.png"
                create_mesh_comparison_visualization(
                    gt_mesh, pred_mesh,
                    title=f"{label} - {uid[:12]} - epoch {epoch}",
                    save_path=save_path,
                    input_image=input_image,
                    nerf_mesh=nerf_mesh,
                )

                if wandb_enabled:
                    try:
                        wandb.log({
                            f"mesh_reconstruction/{label}/{sample_dir.name}": wandb.Image(str(save_path)),
                            "mesh_reconstruction/epoch": epoch,
                        })
                    except Exception:
                        pass

                del triplane, pred_mesh, gt_mesh, nerf_mesh
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                tqdm.write(f"[vis] failed for {uid}: {e}")

    sdf_mlp.train()


# ─── TRAIN phase ──────────────────────────────────────────────────────────────

def run_train(args: argparse.Namespace) -> None:
    # ── DDP setup — works with torchrun --nproc_per_node=N; falls back to 1 GPU ─
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank       = int(os.environ.get("RANK", 0))
    is_ddp  = world_size > 1
    is_main = (rank == 0)

    if is_ddp:
        # Long timeout: the non-collective trimesh augment phase has variable
        # per-rank duration; a slow degenerate mesh must not trip the watchdog.
        dist.init_process_group(backend="nccl",
                                timeout=timedelta(minutes=NCCL_TIMEOUT_MIN))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if is_main:
        suffix = f" — DDP across {world_size} GPUs (effective batch {args.batch_size * world_size})" if is_ddp else ""
        print(f"Training on {device}{suffix}")

    # ── wandb ─────────────────────────────────────────────────────────────────
    wandb_enabled = False

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_dir = Path(args.dataset_dir)
    cache_dir = str(dataset_dir / "mesh_cache")

    # ── Mesh path index for near-surface point sampling ──────────────────────
    # BVHs are built fresh per outer training step (not cached) so memory stays
    # bounded at SAMPLES_PER_BATCH meshes in RAM at a time per rank.
    _uid_to_mesh_path: dict[str, str] = {}
    mesh_cache_path = dataset_dir / "mesh_cache"
    if mesh_cache_path.exists():
        for uid_dir in mesh_cache_path.iterdir():
            if uid_dir.is_dir():
                files = [f for f in uid_dir.iterdir() if f.is_file()]
                if files:
                    _uid_to_mesh_path[uid_dir.name] = str(files[0])

    def _build_batch_bvh(uids: list[str]) -> dict[str, tuple]:
        """Build (mesh, ProximityQuery) for each unique UID in this batch.
        Caller is responsible for deleting the returned dict when done."""
        import trimesh as _tr
        result: dict[str, tuple] = {}
        for uid in set(uids):
            if uid in result:
                continue
            path = _uid_to_mesh_path.get(uid)
            if path is None:
                continue
            try:
                mesh = load_and_normalize_mesh(path, radius)
                if not mesh.is_watertight:
                    continue
                result[uid] = (mesh, _tr.proximity.ProximityQuery(mesh))
            except Exception:
                pass
        return result

    # Small chunk: trimesh on_surface transiently materializes
    # (chunk × candidate-triangles-per-point) of data. Degenerate Objaverse
    # meshes can yield tens of thousands of candidates per point, so a large
    # chunk spikes tens of GB — fatal when 4 ranks query concurrently. The
    # spike scales linearly with chunk size; 512 keeps it ~1-2 GB/rank.
    _NEAR_SDF_CHUNK = 512

    def _compute_near_surface_sdf(mesh, prox, pts_np: np.ndarray,
                                  deadline: float | None = None):
        """Return (sdf, n_done). Stops early once ``deadline`` (time.monotonic)
        is passed so a pathologically slow mesh can't stall the rank — only the
        first ``n_done`` points are valid and the caller truncates to match."""
        face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
        sdf = np.empty(len(pts_np), dtype=np.float32)
        n_done = 0
        for i in range(0, len(pts_np), _NEAR_SDF_CHUNK):
            if deadline is not None and time.monotonic() > deadline:
                break
            batch = pts_np[i : i + _NEAR_SDF_CHUNK]
            closest, distances, tri_ids = prox.on_surface(batch)
            direction = batch - closest
            dot = np.einsum("ij,ij->i", direction, face_normals[tri_ids])
            sdf[i : i + _NEAR_SDF_CHUNK] = np.where(dot >= 0, distances, -distances).astype(np.float32)
            n_done = i + len(batch)
        return sdf[:n_done], n_done

    def _compute_features(pts_mesh_g: torch.Tensor, sid_g: torch.Tensor,
                          triplanes_gpu: list, R_gpu: torch.Tensor):
        """Lazily compute triplane features for one mini-batch of points.

        pts_mesh_g : (B, 3) GPU — points in normalized mesh frame
        sid_g      : (B,)   GPU — which sample (0..S-1) each point belongs to
        triplanes_gpu : list of S GPU triplanes (kept resident for the outer step)
        R_gpu      : (S, 3, 3) GPU rotation matrices

        Returns (feats, pts_trip) both on GPU. Points are grouped by sample so
        each triplane is queried once. Memory is O(B), never O(N_POINTS).
        """
        B = pts_mesh_g.shape[0]
        pts_trip = torch.empty(B, 3, device=device)
        feats    = torch.empty(B, feat_dim, device=device)
        for u in torch.unique(sid_g):
            m  = sid_g == u
            uu = int(u.item())
            pt = pts_mesh_g[m] @ R_gpu[uu].T
            pts_trip[m] = pt
            feats[m] = query_triplane_features(pt, triplanes_gpu[uu],
                                               float(radius), feature_reduction)
        return feats, pts_trip

    all_sample_paths = sorted((dataset_dir / "samples").glob("*/triplane.pt"))
    all_uids = sorted({p.parent.name.split("_az")[0] for p in all_sample_paths})

    # Cap to n_objects and azimuths_per_mesh (mirrors precompute behaviour)
    if len(all_uids) > args.n_objects:
        all_uids = all_uids[:args.n_objects]
    uid_set = set(all_uids)
    uid_az_seen: dict[str, set[str]] = {}
    allowed_names: set[str] = set()
    for p in all_sample_paths:
        name = p.parent.name
        uid = name.split("_az")[0]
        if uid not in uid_set:
            continue
        az = name.split("_az")[1].split("_el")[0]
        uid_az_seen.setdefault(uid, set())
        if len(uid_az_seen[uid]) < args.azimuths_per_mesh or az in uid_az_seen[uid]:
            uid_az_seen[uid].add(az)
            allowed_names.add(name)
    all_sample_paths = [p for p in all_sample_paths if p.parent.name in allowed_names]

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

    # No flat cache — each rank loads per-sample .pt files on the fly.
    dataset = SDFLazyDataset(args.dataset_dir, sample_whitelist=train_view_names)
    meta = dataset.meta
    radius: float = float(meta["radius"])
    feature_reduction: str = meta["feature_reduction"]
    feat_dim: int = meta["feat_dim"]
    n_pts_per_sample: int = meta["n_points"]

    # ── Test dataset (rank 0 only — eval never runs on other ranks) ───────────
    test_dataset: SDFLazyDataset | None = None
    test_loader = None
    if is_main:
        test_sample_names = (
            {s for s in all_sample_names if s.split("_az")[0] in test_uids}
            | test_view_names
        )
        if test_sample_names:
            test_dataset = SDFLazyDataset(args.dataset_dir, sample_whitelist=test_sample_names)
            test_loader = DataLoader(test_dataset, batch_size=args.samples_per_batch,
                                     shuffle=False, num_workers=args.num_workers, pin_memory=False)
            print(f"Test dataset: {len(test_dataset)} samples  "
                  f"({len(test_uids)} unseen UIDs + {len(test_view_names)} unseen views)")

    # ── Optionally load frozen TripoSR decoder for NeRF sanity-check mesh ────
    # Only rank 0 runs visualization, so other ranks skip the decoder entirely.
    triposr_decoder: nn.Module | None = None
    _density_activation = "exp"
    _density_bias = -1.0
    if is_main and args.use_nerf_vis:
        try:
            from tsr.system import TSR
            _full = TSR.from_pretrained(args.model, config_name="config.yaml", weight_name="model.ckpt")
            triposr_decoder = _full.decoder.to(device).eval()
            for p in triposr_decoder.parameters():
                p.requires_grad_(False)
            _density_activation = _full.renderer.cfg.density_activation
            _density_bias = float(_full.renderer.cfg.density_bias)
            del _full
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"TripoSR decoder loaded (density_activation={_density_activation}, bias={_density_bias})")
        except Exception as e:
            print(f"Warning: could not load TripoSR decoder ({e}). NeRF vis disabled.")
    elif is_main:
        print("[nerf_vis] Skipped TripoSR decoder load (USE_NERF_VIS=False).")

    if is_main:
        print(f"Dataset: {len(dataset)} samples ({len(dataset) * n_pts_per_sample:,} points) | "
              f"{n_test} unseen UIDs, {len(test_view_names)} unseen views | "
              f"radius={radius:.4f} | feat_dim={feat_dim}")

    if is_ddp:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=True, drop_last=True)
        loader = DataLoader(dataset, batch_size=args.samples_per_batch, sampler=sampler,
                            num_workers=args.num_workers, pin_memory=False, drop_last=True)
    else:
        sampler = None
        loader = DataLoader(dataset, batch_size=args.samples_per_batch, shuffle=True,
                            num_workers=args.num_workers, pin_memory=False, drop_last=True)

    # ── Visualization samples ─────────────────────────────────────────────────
    def _pick_vis_from_uids(uid_set: set, n: int) -> list:
        if not uid_set:
            return []
        chosen = random.Random(0).sample(sorted(uid_set), min(n, len(uid_set)))
        dirs = []
        for uid in chosen:
            candidates = sorted((dataset_dir / "samples").glob(f"{uid}_az*"))
            for c in candidates[: args.vis_azimuths_per_object]:
                dirs.append(c)
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
    if args.use_triplane_features:
        mlp_feat_dim = feat_dim
        mlp_hidden_dim = args.hidden_dim
    else:
        mlp_feat_dim = 0
        mlp_hidden_dim = args.hidden_dim_no_triplane
    mlp_in_dim: int = mlp_feat_dim + pe_dim
    if is_main:
        print(f"MLP input: {mlp_feat_dim}-dim triplane feats + {pe_dim}-dim PE = {mlp_in_dim}"
              + ("" if args.use_triplane_features else f"  [PE-only ablation, hidden_dim={mlp_hidden_dim}]"))

    global MLP_IN_DIM
    MLP_IN_DIM = mlp_in_dim

    sdf_mlp = SDFMLP(
        in_dim=mlp_in_dim,
        hidden_dim=mlp_hidden_dim,
        n_hidden=args.n_hidden,
        use_tanh_output=args.use_tanh_output,
        feat_dim=mlp_feat_dim,
        pe_dim=pe_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(sdf_mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Gradient-step counts (identical across ranks; see DDP note in the loop).
    steps_per_outer = max(1, (args.samples_per_batch * n_pts_per_sample) // args.batch_size)
    total_steps = args.epochs * len(loader) * steps_per_outer

    if args.use_onecycle:
        # OneCycle steps PER optimizer step (warmup then anneal); needs the
        # total step budget up front. step() is called in the inner loop.
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=total_steps,
            pct_start=args.onecycle_pct_start,
        )
    else:
        # Cosine steps PER epoch.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr_min,
        )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        sdf_mlp.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0)
        if is_main:
            print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # ── DDP wrapping (after resume so we load into the raw module) ─────────────
    if is_ddp:
        sdf_mlp = DDP(sdf_mlp, device_ids=[local_rank])

    # ── wandb (rank 0 only) ───────────────────────────────────────────────────
    _mlp_module = sdf_mlp.module if is_ddp else sdf_mlp
    if is_main:
        try:
            derived_meta = {
                "radius": radius,
                "feature_reduction": feature_reduction,
                "feat_dim": feat_dim,
                "mlp_in_dim": mlp_in_dim,
                "pe_dim": pe_dim,
                "total_dataset_points": len(dataset),
                "loader_batches_per_epoch": len(loader),
                "n_train_uids": len(train_uids),
                "n_test_uids": len(test_uids),
                "n_train_views": len(train_view_names),
                "n_test_views": len(test_view_names),
                "start_epoch": start_epoch,
                "world_size": world_size,
            }
            wb_config: dict = {
                "globals": wandb_collect_module_globals(),
                "args": {k: _wandb_jsonable(v) for k, v in vars(args).items()},
                "derived": {k: _wandb_jsonable(v) for k, v in derived_meta.items()},
            }
            wb_config.update(wandb_model_parameter_config(_mlp_module))
            wandb.init(
                project="simple-sdf",
                name=args.run_name,
                config=wb_config,
                settings=wandb.Settings(_disable_stats=True, console="off"),
            )
            wandb_log_model_parameter_table(_mlp_module)
            if len(loader) > 0:
                wandb.watch(
                    _mlp_module,
                    log="all",
                    log_freq=max(1, len(loader)),
                )
            wandb_enabled = True
        except Exception as e:
            print(f"Warning: wandb init failed: {e}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    # steps_per_outer / total_steps computed above (needed by the scheduler).
    pbar = tqdm(total=total_steps, initial=start_epoch * len(loader) * steps_per_outer,
                desc=f"Epoch 1/{args.epochs}", dynamic_ncols=True, unit="step") if is_main else None

    for epoch in range(start_epoch, args.epochs):
        sdf_mlp.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_loss = epoch_sdf = epoch_mse = epoch_eik = epoch_bce = 0.0
        diag_steps = 0
        diag_sign_acc_sum = 0.0
        diag_preclip_norm_sum = 0.0
        diag_reject_frac_sum = 0.0
        epoch_near_used_sum = 0
        epoch_near_target_sum = 0
        diag_grad_mean_sum = 0.0
        diag_grad_min = float("inf")
        diag_grad_max = float("-inf")
        diag_pred_min = float("inf")
        diag_pred_max = float("-inf")
        diag_tgt_min = float("inf")
        diag_tgt_max = float("-inf")
        if pbar is not None:
            pbar.set_description(f"Epoch {epoch + 1}/{args.epochs}")

        for sample_pts, sample_sdf, sample_triplane, sample_R, sample_uids in loader:
            # sample_pts:      (S, N, 3)           — uniform pts in mesh frame, CPU
            # sample_sdf:      (S, N)              — GT SDF for uniform pts, CPU
            # sample_triplane: (S, 3, Cp, Hp, Wp)  — raw triplane per sample, CPU
            # sample_R:        (S, 3, 3)           — rotation matrix per sample, CPU
            # sample_uids:     list[str]            — UID per sample
            S, N = sample_sdf.shape

            # Keep the S small triplanes + rotations resident on GPU for this
            # outer step. Features are computed per mini-batch from these (see
            # _compute_features) so we never materialize all N_POINTS features.
            triplanes_gpu = [sample_triplane[s].to(device) for s in range(S)]
            R_gpu = sample_R.to(device)  # (S, 3, 3)

            # Point pool, all in mesh frame on CPU. Features are NEVER stored
            # here — only raw 3-float points, scalar SDF, and a sample-id tag.
            pool_pts = sample_pts.reshape(S * N, 3)             # (S*N, 3) view
            pool_sdf = sample_sdf.reshape(S * N)                # (S*N,)
            pool_sid = torch.arange(S).repeat_interleave(N)     # (S*N,)

            # ── Pass 1: probe uniform pts in chunks → near-surface mask ───────
            # Forward-only, no gradient. Features computed per chunk on GPU.
            near_mask = torch.zeros(S * N, dtype=torch.bool)
            with torch.no_grad():
                for start in range(0, S * N, args.batch_size):
                    end = min(start + args.batch_size, S * N)
                    pm  = pool_pts[start:end].to(device)
                    sg  = pool_sid[start:end].to(device)
                    feats, pts_trip = _compute_features(pm, sg, triplanes_gpu, R_gpu)
                    if n_freqs > 0:
                        if args.use_triplane_features:
                            p1_in = torch.cat([feats, fourier_encode(pts_trip, n_freqs)], dim=-1)
                        else:
                            p1_in = fourier_encode(pts_trip, n_freqs)
                    else:
                        p1_in = feats if args.use_triplane_features else torch.empty(0)
                    pred = sdf_mlp(p1_in)
                    near_mask[start:end] = (pred.abs() < args.near_surface_threshold).cpu()

            # ── Pass 2: perturb near-surface pts, compute GT SDF, append to pool ─
            # Operates entirely in mesh frame; no features computed here. BVHs
            # are built per-step and freed immediately.
            batch_bvh = _build_batch_bvh(list(sample_uids))
            extra_pts: list[torch.Tensor] = []
            extra_sdf: list[torch.Tensor] = []
            extra_sid: list[torch.Tensor] = []
            # Wall-clock budget so a slow degenerate mesh can't desync DDP ranks.
            augment_deadline = time.monotonic() + args.augment_budget_sec
            for s in range(S):
                if time.monotonic() > augment_deadline:
                    break  # out of budget — skip remaining samples this step
                bvh = batch_bvh.get(sample_uids[s])
                if bvh is None:
                    continue
                mesh_s, prox_s = bvh
                s_mask  = near_mask[s * N : (s + 1) * N]
                ns_mesh = sample_pts[s][s_mask]        # (k, 3) flagged near-surface, mesh frame
                if ns_mesh.shape[0] == 0:
                    continue
                # Resample the flagged points (with replacement) up/down to a
                # target count = ratio × N (the per-sample uniform count). With
                # ratio=1 this requests as many near-surface points as uniform
                # ones. Independent noise below makes duplicates distinct.
                # NOTE: the per-step time budget may still truncate how many are
                # actually computed (see n_done), so the realised count can be
                # lower for slow meshes.
                target = max(1, int(N * args.near_surface_ratio))
                sel = torch.randint(0, ns_mesh.shape[0], (target,))
                ns_mesh = ns_mesh[sel]
                new_mesh = (ns_mesh + torch.randn_like(ns_mesh) * args.near_surface_sigma
                            ).clamp(-float(radius), float(radius))
                new_sdf_np, n_done = _compute_near_surface_sdf(
                    mesh_s, prox_s, new_mesh.numpy().astype(np.float64),
                    deadline=augment_deadline)
                if n_done == 0:
                    continue
                new_mesh = new_mesh[:n_done]  # truncate to points actually computed
                extra_pts.append(new_mesh)
                extra_sdf.append(torch.from_numpy(new_sdf_np))
                extra_sid.append(torch.full((new_mesh.shape[0],), s, dtype=torch.long))
            del batch_bvh
            gc.collect()

            # Near-surface points actually computed this step (the time budget
            # may truncate below the per-step target of S × ratio × N).
            near_used_step = int(sum(e.shape[0] for e in extra_pts))
            near_target_step = S * max(1, int(N * args.near_surface_ratio))
            epoch_near_used_sum   += near_used_step
            epoch_near_target_sum += near_target_step

            if extra_pts:
                pool_pts = torch.cat([pool_pts] + extra_pts, dim=0)
                pool_sdf = torch.cat([pool_sdf] + extra_sdf, dim=0)
                pool_sid = torch.cat([pool_sid] + extra_sid, dim=0)

            # ── Shuffle pool (cross-sample mixing) and train in mini-batches ──
            perm = torch.randperm(pool_pts.shape[0])
            pool_pts = pool_pts[perm]
            pool_sdf = pool_sdf[perm]
            pool_sid = pool_sid[perm]

            # CRITICAL for DDP: every rank must run the SAME number of backward
            # / gradient all-reduce calls. The pool size varies per rank (each
            # adds a different number of near-surface points), so we iterate a
            # FIXED count (steps_per_outer = S*N / batch_size) instead of over
            # the whole pool. The shuffle ensures the fixed window still samples
            # a proportional blend of uniform + near-surface points. Mismatched
            # collective counts across ranks would deadlock at the next
            # all-reduce — exactly the NCCL timeout we hit.
            for mb in range(steps_per_outer):
                start = mb * args.batch_size
                end   = start + args.batch_size
                pm     = pool_pts[start:end].to(device)
                sg     = pool_sid[start:end].to(device)
                sdf_gt = pool_sdf[start:end].to(device)

                # Features are detached (constant); gradient flows only through
                # the positional encoding of query_pts (matches prior semantics).
                with torch.no_grad():
                    base_feats, pts_trip = _compute_features(pm, sg, triplanes_gpu, R_gpu)
                query_pts = pts_trip.detach().requires_grad_(True)

                if n_freqs > 0:
                    if args.use_triplane_features:
                        model_feats = torch.cat(
                            [base_feats, fourier_encode(query_pts, n_freqs)], dim=-1)
                    else:
                        model_feats = fourier_encode(query_pts, n_freqs)
                else:
                    model_feats = base_feats if args.use_triplane_features else torch.empty(0)

                sdf_pred = sdf_mlp(model_feats)

                # Per-point data loss, with outlier rejection: drop points whose
                # loss exceeds mean + k·std (bad GT from degenerate meshes, etc.)
                # so a few outliers can't dominate the gradient. Rejection only
                # touches the squared-error term — BCE is bounded and doesn't
                # need it. Reducing over a mask does NOT change the number of
                # backward() calls, so DDP stays in lockstep.
                per_point = surface_weighted_se(sdf_pred, sdf_gt, sigma=args.surface_loss_sigma)
                reject_frac = 0.0
                if args.loss_reject_k > 0 and per_point.numel() > 1:
                    with torch.no_grad():
                        thr = per_point.mean() + args.loss_reject_k * per_point.std()
                        keep = per_point <= thr
                    if keep.any():
                        reject_frac = 1.0 - float(keep.float().mean().item())
                        sdf_loss = per_point[keep].mean()
                    else:
                        sdf_loss = per_point.mean()
                else:
                    sdf_loss = per_point.mean()

                bce_loss = sign_bce_loss(sdf_pred, sdf_gt,
                                         alpha=args.sign_bce_alpha, epsilon=args.sign_bce_epsilon)

                gradients = torch.autograd.grad(
                    outputs=sdf_pred,
                    inputs=query_pts,
                    grad_outputs=torch.ones_like(sdf_pred),
                    create_graph=True,
                    retain_graph=True,
                )[0]
                grad_norm = gradients.norm(dim=-1)
                eikonal_loss = ((grad_norm - 1.0) ** 2).mean()

                loss = (sdf_loss
                        + args.eikonal_weight * eikonal_loss
                        + args.sign_bce_weight * bce_loss)

                optimizer.zero_grad()
                loss.backward()
                # clip_grad_norm_ returns the TOTAL norm BEFORE clipping — log it
                # to see whether clipping is actually engaging.
                preclip_norm = float(
                    torch.nn.utils.clip_grad_norm_(sdf_mlp.parameters(), args.grad_clip))
                optimizer.step()
                if args.use_onecycle:
                    scheduler.step()   # OneCycle advances per optimizer step

                epoch_loss += loss.item()
                epoch_sdf  += sdf_loss.item()
                epoch_mse  += F.mse_loss(sdf_pred.detach(), sdf_gt).item()
                epoch_eik  += eikonal_loss.item()
                epoch_bce  += bce_loss.item()
                with torch.no_grad():
                    diag_sign_acc_sum += float(
                        (torch.sign(sdf_pred) == torch.sign(sdf_gt)).float().mean().item())
                diag_preclip_norm_sum += preclip_norm
                diag_reject_frac_sum += reject_frac
                diag_steps += 1

                grad_mean_val = float(grad_norm.mean().detach().item())
                grad_min_val  = float(grad_norm.min().detach().item())
                grad_max_val  = float(grad_norm.max().detach().item())
                pred_min_val  = float(sdf_pred.min().detach().item())
                pred_max_val  = float(sdf_pred.max().detach().item())
                tgt_min_val   = float(sdf_gt.min().detach().item())
                tgt_max_val   = float(sdf_gt.max().detach().item())
                diag_grad_mean_sum += grad_mean_val
                diag_grad_min = min(diag_grad_min, grad_min_val)
                diag_grad_max = max(diag_grad_max, grad_max_val)
                diag_pred_min = min(diag_pred_min, pred_min_val)
                diag_pred_max = max(diag_pred_max, pred_max_val)
                diag_tgt_min  = min(diag_tgt_min,  tgt_min_val)
                diag_tgt_max  = max(diag_tgt_max,  tgt_max_val)

                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(loss=f"{loss.item():.5f}",
                                     sdf=f"{sdf_loss.item():.5f}",
                                     eik=f"{eikonal_loss.item():.5f}")

                if is_main and wandb_enabled:
                    try:
                        wandb.log({
                            "train/loss": loss.item(),
                            "train/sdf_loss": sdf_loss.item(),
                            "train/eikonal_loss": eikonal_loss.item(),
                            "train/sign_bce_loss": bce_loss.item(),
                            "train/grad_norm_preclip": preclip_norm,
                            "train/lr": optimizer.param_groups[0]["lr"],
                        })
                    except Exception:
                        pass

            # Free GPU-resident triplanes before the next outer step
            del triplanes_gpu, R_gpu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── End of epoch ──────────────────────────────────────────────────────
        # Aggregate losses across ranks so rank-0 logs the global average.
        if is_ddp:
            t = torch.tensor([epoch_loss, epoch_sdf, epoch_mse, epoch_eik, epoch_bce], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            epoch_loss, epoch_sdf, epoch_mse, epoch_eik, epoch_bce = (t / world_size).tolist()

        if is_main:
            n = len(loader)
            grad_mean_epoch = (diag_grad_mean_sum / diag_steps) if diag_steps > 0 else float("nan")
            sign_acc_epoch = (diag_sign_acc_sum / diag_steps) if diag_steps > 0 else float("nan")
            preclip_epoch = (diag_preclip_norm_sum / diag_steps) if diag_steps > 0 else float("nan")
            reject_epoch = (diag_reject_frac_sum / diag_steps) if diag_steps > 0 else float("nan")
            # Near-surface points actually used (per rank, per outer step) and the
            # fraction of the requested target realised (1.0 = budget never truncated).
            near_used_epoch = (epoch_near_used_sum / n) if n > 0 else float("nan")
            near_frac_epoch = (epoch_near_used_sum / epoch_near_target_sum
                               if epoch_near_target_sum > 0 else float("nan"))
            if wandb_enabled:
                try:
                    wandb.log({
                        "train/epoch_loss": epoch_loss / n,
                        "train/epoch_sdf_loss": epoch_sdf / n,
                        "train/epoch_mse": epoch_mse / n,
                        "train/epoch_eikonal_loss": epoch_eik / n,
                        "train/epoch_sign_bce_loss": epoch_bce / n,
                        "diag/sign_accuracy": sign_acc_epoch,
                        "diag/near_points_used": near_used_epoch,
                        "diag/near_points_frac": near_frac_epoch,
                        "diag/grad_norm_preclip_mean": preclip_epoch,
                        "diag/reject_frac_mean": reject_epoch,
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

        # ── Test evaluation (rank 0 only) ─────────────────────────────────────
        if is_main and test_loader is not None:
            _mlp_module.eval()
            test_weighted_sum = test_mse_sum = test_sign_acc_sum = 0.0
            test_steps = 0
            with torch.no_grad():
                for t_pts, t_sdf, t_triplane, t_R, _ in test_loader:
                    tS, tN = t_sdf.shape
                    t_trip_gpu = [t_triplane[s].to(device) for s in range(tS)]
                    t_R_gpu    = t_R.to(device)
                    t_pool_pts = t_pts.reshape(tS * tN, 3)
                    t_pool_sdf = t_sdf.reshape(tS * tN)
                    t_pool_sid = torch.arange(tS).repeat_interleave(tN)
                    for ts_start in range(0, tS * tN, args.batch_size):
                        ts_end = min(ts_start + args.batch_size, tS * tN)
                        pm = t_pool_pts[ts_start:ts_end].to(device)
                        sg = t_pool_sid[ts_start:ts_end].to(device)
                        ms = t_pool_sdf[ts_start:ts_end].to(device)
                        mf, mp = _compute_features(pm, sg, t_trip_gpu, t_R_gpu)
                        if n_freqs > 0:
                            if args.use_triplane_features:
                                t_model_feats = torch.cat([mf, fourier_encode(mp, n_freqs)], dim=-1)
                            else:
                                t_model_feats = fourier_encode(mp, n_freqs)
                        else:
                            t_model_feats = mf if args.use_triplane_features else torch.empty(0)
                        t_pred = _mlp_module(t_model_feats)
                        test_weighted_sum += surface_weighted_mse_loss(t_pred, ms, sigma=args.surface_loss_sigma).item()
                        test_mse_sum += F.mse_loss(t_pred, ms).item()
                        test_sign_acc_sum += float((torch.sign(t_pred) == torch.sign(ms)).float().mean().item())
                        test_steps += 1
                    del t_trip_gpu, t_R_gpu
            _mlp_module.train()
            if wandb_enabled and test_steps > 0:
                try:
                    wandb.log({
                        "test/epoch_sdf_loss": test_weighted_sum / test_steps,
                        "test/epoch_mse": test_mse_sum / test_steps,
                        "test/sign_accuracy": test_sign_acc_sum / test_steps,
                        "train/epoch": epoch + 1,
                    })
                except Exception:
                    pass

        # Cosine steps per epoch; OneCycle already stepped per minibatch.
        if not args.use_onecycle:
            scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        if is_main and wandb_enabled:
            try:
                wandb.log({"train/epoch_lr": current_lr, "train/epoch": epoch + 1})
            except Exception:
                pass

        is_last = (epoch + 1) == args.epochs
        if is_main and (is_last or (epoch + 1) % args.save_every == 0):
            ckpt_path = output_dir / f"sdf_head_{RUN_NAME}_epoch{epoch + 1:04d}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch + 1,
                "model": _mlp_module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "meta": meta,
                "args": vars(args),
            }, ckpt_path)

        if is_main and args.vis_every > 0 and (epoch + 1) % args.vis_every == 0:
            visualize_reconstructions(
                sdf_mlp=_mlp_module,
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
                fov=args.fov,
                image_size=args.image_size,
                triposr_decoder=triposr_decoder,
                density_activation=_density_activation,
                density_bias=_density_bias,
                use_triplane_features=args.use_triplane_features,
            )

    if pbar is not None:
        pbar.close()

    if is_main:
        if wandb_enabled:
            try:
                wandb.finish()
            except Exception:
                pass
        print("\nTraining complete.")

    if is_ddp:
        dist.destroy_process_group()


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    # Allow train_sdf.sh (or any caller) to override the COMMAND constant via CLI.
    _p = argparse.ArgumentParser(add_help=False)
    _p.add_argument("--command", default=COMMAND,
                    choices=("precompute", "train", "both"))
    _cli, _ = _p.parse_known_args()
    command = _cli.command

    args = argparse.Namespace(
        dataset_dir    = DATASET_DIR,
        model          = MODEL,
        n_objects      = N_OBJECTS,
        azimuths_per_mesh = AZIMUTHS_PER_MESH,
        elevations     = ELEVATIONS,
        near_surface_fraction = NEAR_SURFACE_FRACTION,
        n_points       = N_POINTS,
        image_size     = IMAGE_SIZE,
        fov            = FOV,
        max_mesh_mb    = MAX_MESH_MB,
        max_triangles  = MAX_TRIANGLES,
        verbose        = VERBOSE,
        output_dir     = OUTPUT_DIR,
        epochs         = EPOCHS,
        save_every     = SAVE_EVERY,
        hidden_dim     = HIDDEN_DIM,
        hidden_dim_no_triplane = HIDDEN_DIM_NO_TRIPLANE,
        n_hidden       = N_HIDDEN,
        n_freqs        = N_FREQS,
        lr             = LR,
        lr_min         = LR_MIN,
        grad_clip          = GRAD_CLIP,
        loss_reject_k      = LOSS_REJECT_K,
        use_onecycle       = USE_ONECYCLE,
        onecycle_pct_start = ONECYCLE_PCT_START,
        eikonal_weight         = EIKONAL_WEIGHT,
        sign_bce_weight        = SIGN_BCE_WEIGHT,
        sign_bce_alpha         = SIGN_BCE_ALPHA,
        sign_bce_epsilon       = SIGN_BCE_EPSILON,
        surface_loss_sigma     = SURFACE_LOSS_SIGMA,
        near_surface_sigma     = NEAR_SURFACE_SIGMA,
        near_surface_threshold = NEAR_SURFACE_THRESHOLD,
        near_surface_ratio     = NEAR_SURFACE_RATIO,
        augment_budget_sec     = AUGMENT_BUDGET_SEC,
        sdf_clamp              = SDF_CLAMP,
        num_workers    = NUM_WORKERS,
        run_name       = RUN_NAME,
        weight_decay   = WEIGHT_DECAY,
        use_tanh_output = USE_TANH_OUTPUT,
        use_triplane_features = USE_TRIPLANE_FEATURES,
        test_fraction  = TEST_FRACTION,
        test_view_fraction = TEST_VIEW_FRACTION,
        vis_every      = VIS_EVERY,
        vis_seen       = VIS_SEEN,
        vis_unseen     = VIS_UNSEEN,
        vis_azimuths_per_object = VIS_AZIMUTHS_PER_OBJECT,
        vis_resolution = VIS_RESOLUTION,
        batch_size        = BATCH_SIZE,
        samples_per_batch = SAMPLES_PER_BATCH,
        resume         = RESUME,
        use_nerf_vis   = USE_NERF_VIS,
    )

    if command == "precompute":
        run_precompute(args)
    elif command == "train":
        run_train(args)
    elif command == "both":
        run_precompute(args)
        run_train(args)
    else:
        print(f"Unknown command: {command!r}")


if __name__ == "__main__":
    main()
