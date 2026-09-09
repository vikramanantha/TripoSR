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
from concurrent.futures import ThreadPoolExecutor
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
import math
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
# Precomputed data lives on the ws-frb NFS export (169 TB), NOT on the local
# disk — the 33k-object dataset alone is 2.2 TB and local was down to 3% free.
#
# The SAME export is visible at TWO different paths depending on where this
# runs: directly at /mnt/ws-frb on the host, and at /mnt/hostmnt/ws-frb inside
# the training container, whose /mnt bind-mount re-roots the host's /mnt.
# Resolve between them exactly the way objaverse_paths.py does for the shared
# Objaverse mirror, so one file works in both places with no hand-editing.
#
# GOTCHA: a plain bind mount captures the host's mount tree as it was when the
# container STARTED and does not pick up submounts added later. ws-frb was
# mounted on the host after this container was created, so /mnt/hostmnt/ws-frb
# was an empty placeholder until the container was restarted. If the resolver
# below raises inside the container while the host clearly has it mounted, the
# fix is `docker restart markiv` — not a path edit.
_WS_FRB_ROOTS = ("/mnt/ws-frb", "/mnt/hostmnt/ws-frb")
_PRECOMPUTED_REL = "users/markiv/sdfer/TripoSR/precomputed"


def _resolve_ws_frb_root() -> str:
    """First ws-frb path that is the REAL export, not a bare mountpoint dir.

    Presence of ``users/`` is the discriminator: an unpropagated mountpoint
    exists but is empty, so testing os.path.isdir(root) alone would happily
    return a path that silently reads as an empty dataset."""
    for _root in _WS_FRB_ROOTS:
        if os.path.isdir(os.path.join(_root, "users")):
            return _root
    raise FileNotFoundError(
        "ws-frb NFS export not found at " + " or ".join(_WS_FRB_ROOTS) + ". "
        "Inside the training container /mnt is bind-mounted to /mnt/hostmnt and "
        "only exposes host mounts that existed when the container STARTED — if "
        "the host has it mounted, `docker restart markiv` will expose it. "
        "Failing loudly rather than falling back to local disk on purpose: the "
        "local volume does not have room for these datasets."
    )


DATASET_DIR     = os.path.join(_resolve_ws_frb_root(), _PRECOMPUTED_REL)
# mesh_cache/ (downloaded + repaired meshes) is derived as DATASET_DIR/mesh_cache
# in both run_precompute and run_train, so it follows automatically.

# ── Precompute ───────────────────────────────────────────────────────────────
MODEL                 = "stabilityai/TripoSR"
N_OBJECTS             = 10000
AZIMUTHS_PER_MESH     = 5              # × len(ELEVATIONS) = 10 views per object
ELEVATIONS            = [15.0, 30.0]   # two elevation bands; replaces single ELEVATION
N_POINTS              = 32768         # = 64^3
NEAR_SURFACE_FRACTION = 0.25           # fraction of query pts drawn near mesh vertices (Gaussian jitter)
                                       # rather than fully uniform in [-radius, radius]³. Sampled ONCE here
                                       # at precompute (v0.41's 75/25 uniform/near-surface split) — no
                                       # training-time online resampling pass.
SHARP_EDGE_FRACTION   = 0.0            # fraction of query pts sampled near sharp mesh edges (Dora-style
                                       # salience sampling; uniform sampling loses geometric detail)
                                       # TEMPORARILY DISABLED (was 0.2) for debugging — restore to re-enable.
                                       # NOTE: takes effect at PRECOMPUTE; re-precompute to change the
                                       # point distribution of an existing dataset.
SHARP_EDGE_ANGLE_DEG  = 30.0           # dihedral angle above which an edge counts as sharp
REPAIR_MESHES         = True           # repair non-watertight meshes (fill holes → voxel remesh, TripoSG-style)
                                       # instead of skipping them
REPAIR_VOXEL_RES      = 128            # voxel remesh resolution along the longest edge (fallback repair)
REPAIR_VOXEL_METHOD   = "ray"          # trimesh voxelizer. DO NOT use "subdivide" (trimesh's DEFAULT):
                                       # it recursively splits triangles until edges < pitch/2, so a
                                       # 12-face box becomes 2.36M faces (196,608x) — measured — and real
                                       # Objaverse assets are full of large flat polygons, which is what
                                       # OOM-killed precompute. "ray" never subdivides (memory bounded by
                                       # the voxel grid): 4.6x less peak RSS, 6x faster, same watertight
                                       # result on that worst case.
IMAGE_SIZE            = 256
FOV                   = 40.0
MAX_MESH_MB           = 0.0
MAX_TRIANGLES         = 500_000       # skip meshes with more faces than this (BVH RAM guard)
SDF_BACKEND           = "kdtree"      # "kdtree" | "trimesh". Closest-point backend for GT SDF.
                                      # "trimesh" (ProximityQuery.on_surface) sizes its candidate set by
                                      # DISTANCE-TO-NEAREST-VERTEX, so far-field query points (75% of ours
                                      # once NEAR_SURFACE_FRACTION=0.25) pull in thousands of candidate
                                      # faces each — measured 6,528 faces/point on a 61K-face mesh, which
                                      # is BOTH the ~96 s/object time sink and a 3.5 GB RAM spike (x4 ranks).
                                      # "kdtree" bounds candidates to k per point regardless of distance:
                                      # measured 23x faster, and MORE accurate — where the two disagree,
                                      # brute force over all triangles confirms kdtree exact (0.00e+00)
                                      # and trimesh off by 1.6e-6 (its radius padding can miss the true
                                      # closest face). Keep "trimesh" only to reproduce older datasets.
SDF_KDTREE_K          = 32            # candidate faces per point for the kdtree backend's upper-bound
                                      # stage. A radius-refinement pass then makes the result exact, so
                                      # k only trades a little speed, never correctness.
SDF_KDTREE_CHUNK      = 8192          # points per kdtree chunk. The tree is built once and reused, so
                                      # this caps only the transient (chunk × k) triangle array — the
                                      # single sizeable allocation — without repeating any setup work.
SDF_QUERY_CHUNK       = 1024          # points per trimesh on_surface call. RAM GUARD, not a speed knob:
                                      # on_surface transiently materializes
                                      # (chunk × candidate-triangles-per-point), and degenerate Objaverse
                                      # meshes yield tens of thousands of candidates per point — the spike
                                      # scales LINEARLY with chunk size (~1-2 GB per 512 points, per process).
                                      # The old 8192 default could spike 16-32 GB on one bad mesh and get the
                                      # process OOM-killed. Lower this further if precompute still dies,
                                      # especially when sharding across GPUs (N processes spike concurrently).
VERBOSE               = False

# ── Train ────────────────────────────────────────────────────────────────────
OUTPUT_DIR      = "/home/markiv/TripoSR/sdf_checkpoints"
EPOCHS          = 500
SAVE_EVERY      = 10
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
NCCL_TIMEOUT_MIN      = 30     # DDP collective timeout
SDF_CLAMP             = 0.1   # TSDF clamp δ: pred & GT clamped to ±δ in the data loss (0 disables).
                              # DeepSDF-standard; focuses capacity near the surface. Sign-BCE and
                              # eikonal still steer badly-wrong far-field points.
                              # TEMPORARILY DISABLED (was 0.1) for debugging — restore to re-enable.
NORMAL_LOSS_WEIGHT    = 0.0   # surface-normal alignment loss weight (0 disables). IGR/TripoSG-style:
                              # align ∇f with the GT SDF gradient direction at near-surface points.
                              # TEMPORARILY DISABLED (was 0.1) for debugging — restore to re-enable.
NORMAL_LOSS_THRESHOLD = 0.05  # only points with |sdf_gt| < this get normal supervision
DATASET_SCAN_THREADS = 32   # threads for SDFLazyDataset's per-sample NFS scan (env SDFER_SCAN_THREADS)
NUM_WORKERS     = 4
RUN_NAME        = "v0.64_10k"
TEST_FRACTION   = 0.2        # fraction of meshes (UIDs) held out as unseen
TEST_VIEW_FRACTION = 0.2     # fraction of views held out per mesh
TEST_MAX_SAMPLES = 3600      # hard cap on test-set size (0 = no cap). The test set otherwise
                             # scales with N_OBJECTS (360 @100, 3.6k @1k, 36k @10k) and rank 0
                             # evaluates ALL of it every epoch while the other ranks sit blocked
                             # in the next epoch's DDP all-reduce -> at 10k objects that exceeded
                             # NCCL_TIMEOUT_MIN and killed the job. 3600 = the size the 1k runs
                             # used, so test curves stay comparable with those runs.
VIS_EVERY       = 25
VIS_SEEN        = 3
VIS_UNSEEN      = 3
VIS_AZIMUTHS_PER_OBJECT = 5
VIS_RESOLUTION  = 64
FSCORE_TAU            = 0.01   # F-score distance threshold (fraction of unit-normalized mesh scale)
MESH_METRIC_SAMPLES   = 50000  # surface samples for Chamfer / F-score; must be dense enough that
                               # the NN-distance floor sits well below FSCORE_TAU (identical meshes
                               # score F≈1.0 at 50k samples; at 10k the floor already eats τ=0.01)
BATCH_SIZE        = 4096
SAMPLES_PER_BATCH = 4      # samples loaded per DataLoader step; points per step = SAMPLES_PER_BATCH × N_POINTS
RESUME            = None
WEIGHT_DECAY    = 1e-4
USE_TANH_OUTPUT = False
USE_TRIPLANE_FEATURES  = True   # ablation: set False to zero out triplane features (PE only)
USE_NERF_VIS           = True  # load TripoSR decoder + render NeRF mesh during visualization
USE_TORCH_COMPILE      = False # INCOMPATIBLE with the eikonal loss — leave False while EIKONAL_WEIGHT > 0.
                               # The eikonal term does torch.autograd.grad(..., create_graph=True) and then
                               # loss.backward(), i.e. a DOUBLE BACKWARD, which aot_autograd rejects:
                               #   "torch.compile with aot_autograd does not currently support double backward"
                               # Note dynamo's suppress_errors=True does NOT rescue this: it only catches
                               # graph-CAPTURE failures, while this raises at runtime inside backward.
                               # run_train hard-disables it when eikonal_weight > 0 (see the guard there),
                               # so it is only usable in an eikonal-free ablation.

# ── LoRA fine-tuning ─────────────────────────────────────────────────────────
LORA_RANK              = 16
LORA_ALPHA             = 16.0   # LoRA scale = alpha / rank (1.0 = no extra scaling)
LORA_BLOCK_START       = 0      # first backbone block to adapt (0-indexed, inclusive); 0 = all layers
LORA_BLOCK_END         = 16     # exclusive end; 16 = through the last block (clamped to block count)
LORA_LR                = 1e-4   # separate LR for LoRA adapters + post_processor
LORA_WEIGHT_DECAY      = 1e-4

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

def surface_weighted_se(pred: torch.Tensor, target: torch.Tensor, sigma: float = 0.05,
                        weight_target: torch.Tensor | None = None) -> torch.Tensor:
    """Per-point surface-weighted squared error (NOT reduced).

    ``weight_target`` (default: ``target``) lets the caller weight by the
    UNCLAMPED distance while the residual uses TSDF-clamped values, so clamping
    does not flatten the near-surface emphasis."""
    w_src = target if weight_target is None else weight_target
    weights = torch.exp(-w_src.abs() / sigma)
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


# ─── LoRA ────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Frozen nn.Linear with a trainable low-rank perturbation ΔW = B·A·scale."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        d_out, d_in = base.weight.shape
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Toggled off by the stock_triposr() context manager. ``base`` holds the
        # ORIGINAL pretrained nn.Linear and is frozen for the whole run, so
        # skipping the low-rank term recovers stock TripoSR exactly — no need to
        # keep a second copy of the model around just to get a baseline.
        self.lora_enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.lora_enabled:
            out = out + (x @ self.lora_A.T @ self.lora_B.T) * self.scale
        return out


def _lora_attn(attn, rank: int, alpha: float) -> None:
    """Replace to_q/k/v/out in an Attention module with LoRALinear in-place."""
    attn.to_q = LoRALinear(attn.to_q, rank, alpha)
    if attn.to_k is not None:
        attn.to_k = LoRALinear(attn.to_k, rank, alpha)
    if attn.to_v is not None:
        attn.to_v = LoRALinear(attn.to_v, rank, alpha)
    attn.to_out[0] = LoRALinear(attn.to_out[0], rank, alpha)


def _lora_ff(ff, rank: int, alpha: float) -> None:
    """Apply LoRA to the gate projection and output projection in a GEGLU FFN."""
    gate = ff.net[0]
    if hasattr(gate, "proj"):
        gate.proj = LoRALinear(gate.proj, rank, alpha)
    out_proj = ff.net[2]
    if isinstance(out_proj, nn.Linear):
        ff.net[2] = LoRALinear(out_proj, rank, alpha)


def apply_lora_to_triposr(triposr_model, start_block: int, end_block: int,
                           rank: int, alpha: float) -> list:
    """Freeze all TripoSR params; inject LoRA into backbone blocks [start, end);
    fully unfreeze post_processor.  Returns the list of all trainable parameters."""
    for p in triposr_model.parameters():
        p.requires_grad_(False)

    blocks = triposr_model.backbone.transformer_blocks
    for i in range(start_block, min(end_block, len(blocks))):
        block = blocks[i]
        _lora_attn(block.attn1, rank, alpha)
        if block.attn2 is not None:
            _lora_attn(block.attn2, rank, alpha)
        _lora_ff(block.ff, rank, alpha)

    for p in triposr_model.post_processor.parameters():
        p.requires_grad_(True)

    # Stash the pristine post_processor so stock_triposr() can restore it. This
    # must happen BEFORE any fine-tuned weights are loaded/trained into it.
    snapshot_stock_post_processor(triposr_model)

    return [p for p in triposr_model.parameters() if p.requires_grad]


def snapshot_stock_post_processor(triposr_model) -> None:
    """Record the ORIGINAL post_processor weights for stock_triposr().

    apply_lora_to_triposr changes exactly two things relative to pretrained
    TripoSR: it adds low-rank deltas (whose ``base`` stays frozen at the
    pretrained value) and it fully unfreezes ``post_processor``. So the only
    state that must be snapshotted to reconstruct stock behaviour is the
    post_processor — and that is just 2 small tensors, not a second 427M model.
    """
    import copy
    triposr_model._stock_post_processor_state = copy.deepcopy(
        triposr_model.post_processor.state_dict())


@contextlib.contextmanager
def stock_triposr(triposr_model):
    """Temporarily run ``triposr_model`` as ORIGINAL, un-fine-tuned TripoSR.

    Disables every LoRA delta and swaps the pretrained post_processor back in,
    restoring the fine-tuned state on exit (even if the body raises). Used to
    produce the NeRF baseline so it answers "what does stock TripoSR do on this
    image?" — a real before/after against the fine-tuned model — rather than
    decoding a fine-tuned triplane with the original density head.

    A no-op (yields the model unchanged) if no LoRA was applied, so callers do
    not need to special-case stock checkpoints.
    """
    import copy
    lora_mods = [m for m in triposr_model.modules() if isinstance(m, LoRALinear)]
    stock_pp = getattr(triposr_model, "_stock_post_processor_state", None)
    prev = [m.lora_enabled for m in lora_mods]
    saved_pp = None
    try:
        for m in lora_mods:
            m.lora_enabled = False
        if stock_pp is not None:
            saved_pp = copy.deepcopy(triposr_model.post_processor.state_dict())
            triposr_model.post_processor.load_state_dict(stock_pp)
        yield triposr_model
    finally:
        for m, p in zip(lora_mods, prev):
            m.lora_enabled = p
        if saved_pp is not None:
            triposr_model.post_processor.load_state_dict(saved_pp)


def compute_cached_image_tokens(
    triposr_model, image_np: np.ndarray, device: torch.device,
) -> torch.Tensor:
    """Precompute-time: run the DINO image tokenizer once and cache its output.

    apply_lora_to_triposr only wraps backbone.transformer_blocks + unfreezes
    post_processor — image_tokenizer is NEVER touched by LoRA regardless of
    which blocks get fine-tuned. Its output is therefore a frozen, deterministic
    function of the image alone (ImagePreprocessor has no randomness either),
    safe to precompute once and reuse for the lifetime of training. Returns
    (n_tokens, C) with the batch dim dropped, for per-sample disk storage.
    """
    with torch.no_grad():
        rgb_cond = triposr_model.image_processor(
            [image_np], triposr_model.cfg.cond_image_size
        )[:, None].to(device)
        tokens = triposr_model.image_tokenizer(
            rearrange(rgb_cond, "B Nv H W C -> B Nv C H W", Nv=1)
        )
        tokens = rearrange(tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=1)
    return tokens[0].cpu()


def triposr_forward_from_cached_tokens(
    triposr_model, input_image_tokens: torch.Tensor,
) -> torch.Tensor:
    """Train-time: TSR.forward's tokenizer→backbone→post_processor chain,
    skipping the (frozen, precomputed) image tokenizer entirely.

    ``input_image_tokens`` is (B, n_tokens, C), already on the model's device.
    """
    batch_size = input_image_tokens.shape[0]
    tokens = triposr_model.tokenizer(batch_size)
    tokens = triposr_model.backbone(tokens, encoder_hidden_states=input_image_tokens)
    return triposr_model.post_processor(triposr_model.tokenizer.detokenize(tokens))


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


def _closest_point_kdtree(mesh, points: np.ndarray, k: int = SDF_KDTREE_K,
                          chunk: int = SDF_KDTREE_CHUNK):
    """Exact closest-point query via a cKDTree over face centroids.

    Same return contract as ``trimesh.proximity.ProximityQuery.on_surface``:
    ``(closest_points, distances, triangle_ids)``.

    Exactness argument (this is NOT a k-nearest approximation):
      Stage 1 takes the k nearest face centroids and computes true
      point-triangle distances to them. The best of those, ``d_ub``, is a
      distance actually achieved by a real triangle, hence a valid UPPER BOUND
      on the true closest distance.
      Stage 2 uses the fact that any triangle whose surface lies closer than
      ``d_ub`` must have its centroid within ``d_ub + R_T``, where ``R_T`` is
      that triangle's centroid->vertex radius. Triangles are split by ``R_T``:
      the handful of large ones are brute-forced, and the rest get a tight ball
      query at ``d_ub + r_thresh``. Every triangle that could possibly beat the
      bound is therefore examined.

    Verified against brute force over all triangles: exact (0.00e+00), while
    trimesh's on_surface was off by 1.6e-6 on the same points.
    """
    import trimesh  # module-local: train_sdf_head does not import trimesh globally
    from scipy.spatial import cKDTree

    tri = mesh.triangles
    if len(tri) == 0:
        raise ValueError("Mesh has no triangles")
    cent = tri.mean(axis=1)
    R = np.linalg.norm(tri - cent[:, None, :], axis=2).max(axis=1)

    # Split off the largest-radius triangles; they would otherwise force a huge
    # ball-query radius for every point (one giant triangle would negate the
    # whole bound). There are few of them, so brute force is cheap.
    r_thresh = float(np.percentile(R, 99.5))
    big = np.where(R > r_thresh)[0]
    small = np.where(R <= r_thresh)[0]
    if len(small) == 0:  # degenerate: all triangles "large"
        big, small = np.arange(len(tri)), np.arange(0)

    # The cKDTree is built ONCE and reused across point chunks. Chunking caps the
    # transient (chunk × k) triangle array — the only sizeable allocation here —
    # which matters because precompute runs one of these per rank concurrently.
    tree = cKDTree(cent[small]) if len(small) > 0 else None
    kk = int(min(k, len(small))) if tree is not None else 0

    cp_out = np.zeros((len(points), 3), dtype=np.float64)
    d_out = np.full(len(points), np.inf, dtype=np.float64)
    t_out = np.zeros(len(points), dtype=np.int64)

    for s0 in range(0, len(points), max(chunk, 1)):
        pc = points[s0 : s0 + max(chunk, 1)]
        ar = np.arange(len(pc))

        def _dists(cand_idx, _pc=pc):
            """Exact point-triangle distances for an (N, M) candidate index array."""
            n, m = cand_idx.shape
            q = np.repeat(_pc[:, None, :], m, axis=1).reshape(-1, 3)
            cp = trimesh.triangles.closest_point(tri[cand_idx.reshape(-1)], q)
            d = np.linalg.norm(q - cp, axis=1).reshape(n, m)
            return d, cp.reshape(n, m, 3)

        if tree is not None:
            _, idx = tree.query(pc, k=kk, workers=-1)
            cand = small[np.atleast_2d(idx.reshape(len(pc), kk))]
            d, cp = _dists(cand)
            b = d.argmin(axis=1)
            d_best, cp_best, t_best = d[ar, b], cp[ar, b], cand[ar, b]
        else:
            d_best = np.full(len(pc), np.inf)
            cp_best = np.zeros((len(pc), 3))
            t_best = np.zeros(len(pc), dtype=np.int64)

        if len(big) > 0:
            d_b, cp_b = _dists(np.tile(big, (len(pc), 1)))
            bb = d_b.argmin(axis=1)
            better = d_b[ar, bb] < d_best
            d_best = np.where(better, d_b[ar, bb], d_best)
            cp_best = np.where(better[:, None], cp_b[ar, bb], cp_best)
            t_best = np.where(better, big[bb], t_best)

        # Exact refinement: examine every small triangle that could still beat d_best.
        if tree is not None:
            for i, cl in enumerate(tree.query_ball_point(pc, d_best + r_thresh, workers=-1)):
                if len(cl) > kk:
                    ci = small[np.asarray(cl, dtype=np.int64)]
                    cp_i = trimesh.triangles.closest_point(
                        tri[ci], np.repeat(pc[i][None], len(ci), axis=0))
                    di = np.linalg.norm(pc[i] - cp_i, axis=1)
                    j = int(di.argmin())
                    if di[j] < d_best[i]:
                        d_best[i], cp_best[i], t_best[i] = di[j], cp_i[j], ci[j]

        cp_out[s0 : s0 + len(pc)] = cp_best
        d_out[s0 : s0 + len(pc)] = d_best
        t_out[s0 : s0 + len(pc)] = t_best

    return cp_out, d_out, t_out


def compute_sdf(mesh, points: np.ndarray, batch_size: int = SDF_QUERY_CHUNK,
                return_normals: bool = False, backend: str = SDF_BACKEND):
    """Signed distance: negative inside, positive outside.

    Avoids mesh.contains() (ray-casting, extremely memory-intensive) by deriving
    the sign from the dot product of (query − closest_surface) with the face
    normal at the closest triangle.  This is O(N) RAM instead of O(N × triangles).

    ProximityQuery is built once per call so the BVH is not rebuilt per batch.

    ``batch_size`` only applies to the "trimesh" backend, where it is a RAM
    guard rather than a speed knob (see SDF_QUERY_CHUNK): on_surface transiently
    materializes (batch_size × candidate-faces-per-point), and that candidate
    count grows with distance from the surface. The "kdtree" backend bounds
    candidates per point structurally, so it runs in a single pass.

    With ``return_normals=True`` also returns the GT SDF gradient direction per
    point: sign·(query − closest)/‖query − closest‖. For both inside and outside
    points this equals the outward surface direction (exact ∇SDF, correct even
    near edges/corners where face normals are ambiguous); falls back to the
    nearest face normal for points lying on the surface.
    """
    import trimesh.proximity
    points = np.asarray(points, dtype=np.float64)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)

    if backend == "kdtree":
        prox, step = None, len(points)          # candidate set is bounded per point
    elif backend == "trimesh":
        prox, step = trimesh.proximity.ProximityQuery(mesh), batch_size
    else:
        raise ValueError(f"SDF_BACKEND must be 'kdtree' or 'trimesh' — got {backend!r}")

    sdf = np.empty(len(points), dtype=np.float32)
    normals = np.empty((len(points), 3), dtype=np.float32) if return_normals else None
    for i in range(0, len(points), max(step, 1)):
        batch = points[i : i + max(step, 1)]
        if prox is None:
            closest, distances, tri_ids = _closest_point_kdtree(mesh, batch)
        else:
            closest, distances, tri_ids = prox.on_surface(batch)
        # Sign: positive outside (dot > 0), negative inside (dot < 0)
        direction = batch - closest
        dot = np.einsum("ij,ij->i", direction, face_normals[tri_ids])
        sign = np.where(dot >= 0, 1.0, -1.0)
        # Slice by len(batch), NOT batch_size: the kdtree backend steps by the
        # whole array, so batch_size would truncate the write-back.
        sdf[i : i + len(batch)] = (sign * distances).astype(np.float32)
        if return_normals:
            n = direction * (sign / np.maximum(distances, 1e-9))[:, None]
            on_surf = distances < 1e-6
            if on_surf.any():
                n[on_surf] = face_normals[tri_ids[on_surf]]
            normals[i : i + len(batch)] = n.astype(np.float32)

    del prox  # release BVH memory explicitly
    if return_normals:
        return sdf, normals
    return sdf


def sample_sharp_edge_points(
    mesh,
    n_points: int,
    radius: float,
    angle_deg: float = 30.0,
    noise_std: float = 0.02,
) -> np.ndarray:
    """Dora-style salience sampling: query points concentrated near sharp edges.

    Edges whose dihedral angle exceeds ``angle_deg`` are sampled proportionally
    to their length, then perturbed at two noise scales (σ and σ/4) so both the
    surface band and the fine-detail band around the edge are covered. Returns
    an empty array when the mesh has no sharp edges (caller pads with uniform).
    """
    empty = np.zeros((0, 3), dtype=np.float32)
    if n_points <= 0 or len(mesh.faces) == 0:
        return empty
    try:
        angles = np.asarray(mesh.face_adjacency_angles)   # radians, per adjacency
        edges  = np.asarray(mesh.face_adjacency_edges)    # (E, 2) vertex indices
    except Exception:
        return empty
    if angles.size == 0 or edges.shape[0] != angles.shape[0]:
        return empty
    sharp = angles > np.radians(angle_deg)
    if not sharp.any():
        return empty

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    v0, v1 = verts[edges[sharp, 0]], verts[edges[sharp, 1]]
    lengths = np.linalg.norm(v1 - v0, axis=1)
    total = float(lengths.sum())
    if total <= 0:
        return empty

    idx = np.random.choice(len(lengths), n_points, p=lengths / total)
    t = np.random.rand(n_points, 1)
    on_edge = v0[idx] * (1.0 - t) + v1[idx] * t
    std = np.where(np.random.rand(n_points, 1) < 0.5, noise_std, noise_std / 4.0)
    pts = on_edge + np.random.normal(0.0, 1.0, (n_points, 3)) * std
    return np.clip(pts.astype(np.float32), -radius, radius)


def sample_query_points(
    mesh,
    n_points: int,
    radius: float,
    near_surface_fraction: float = 0.5,
    near_surface_std: float | None = None,
    sharp_edge_fraction: float = 0.0,
    sharp_edge_angle_deg: float = 30.0,
) -> np.ndarray:
    """Sample query points in the full TripoSR scene volume [-radius, radius]³.

    With near_surface_fraction=0 (default for training), non-sharp points are
    drawn uniformly from the same [-radius, radius]³ cube that marching-cubes
    evaluates during inference — eliminating the training/inference distribution
    mismatch. ``sharp_edge_fraction`` additionally concentrates that fraction of
    points near sharp mesh edges (Dora-style); if the mesh has no sharp edges
    the budget falls back to uniform samples.
    """
    if near_surface_std is None:
        near_surface_std = float(radius * 0.04)
    n_near  = int(n_points * near_surface_fraction)
    n_sharp = int(n_points * sharp_edge_fraction)

    sharp = (sample_sharp_edge_points(mesh, n_sharp, radius,
                                      angle_deg=sharp_edge_angle_deg,
                                      noise_std=near_surface_std)
             if n_sharp > 0 else np.zeros((0, 3), dtype=np.float32))

    # Pad with uniform samples when the mesh yielded no sharp edges.
    n_uniform = n_points - n_near - sharp.shape[0]

    # Uniform samples span the full triplane query volume, not just mesh bounds.
    uniform = np.random.uniform(-radius, radius, (n_uniform, 3)).astype(np.float32)

    if n_near > 0 and len(mesh.vertices) > 0:
        idx   = np.random.choice(len(mesh.vertices), n_near, replace=True)
        verts = np.asarray(mesh.vertices[idx], dtype=np.float64)
        noise = np.random.normal(0.0, near_surface_std, (n_near, 3))
        near  = (verts + noise).astype(np.float32)
    else:
        near = np.zeros((0, 3), dtype=np.float32)

    parts = [p for p in (uniform, near, sharp) if p.shape[0] > 0]
    pts = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
    return np.clip(pts, -radius, radius)


def repair_mesh_watertight(mesh, voxel_res: int = 128,
                           voxel_method: str = REPAIR_VOXEL_METHOD):
    """Try to make ``mesh`` watertight instead of discarding it (TripoSG-style).

    1. Cheap topological repair: merge vertices, fill holes, fix normals.
    2. Fallback: voxelize + fill interior + marching cubes (TSDF-fusion-style
       remesh) — changes the triangulation but preserves the outer surface.

    ``voxel_method`` MUST NOT be trimesh's default "subdivide" — see
    REPAIR_VOXEL_METHOD. "subdivide" recursively splits every triangle until
    its edges are shorter than pitch/2, which for a normalized mesh at
    voxel_res=128 means edges < 0.0039: measured, a 12-triangle box explodes to
    2.36 MILLION faces (196,608x). Real Objaverse assets are full of large flat
    polygons (walls, ground planes, backdrops), so this is the common case, not
    the rare one — it is what OOM-killed precompute. "ray" never subdivides at
    all (memory bounded by the voxel grid), and measured 4.6x less peak RSS and
    6x faster on that same worst case, with equally watertight output.

    Returns (repaired_mesh, method) on success or (None, reason) on failure.
    """
    import trimesh
    if mesh.is_watertight:
        return mesh, "already-watertight"

    m = mesh.copy()
    try:
        m.merge_vertices()
        m.remove_unreferenced_vertices()
        trimesh.repair.fill_holes(m)
        trimesh.repair.fix_normals(m)
        if m.is_watertight and len(m.faces) > 0:
            return m, "hole-filled"
    except Exception:
        pass

    try:
        pitch = float(max(mesh.extents)) / float(voxel_res)
        vox = mesh.voxelized(pitch=pitch, method=voxel_method).fill()
        remeshed = vox.marching_cubes
        remeshed.apply_transform(vox.transform)  # MC output is in voxel-index space
        if remeshed.is_watertight and len(remeshed.faces) > 0:
            return remeshed, f"voxel-remeshed[{voxel_method}]"
    except (Exception, MemoryError) as e:
        return None, f"voxel-remesh-failed: {type(e).__name__}: {e}"
    return None, "unrepairable"


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
    from objaverse_paths import configure_objaverse
    configure_objaverse()
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
    from objaverse_paths import configure_objaverse
    configure_objaverse()
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
            "sharp_edge_fraction": args.sharp_edge_fraction,
            "sharp_edge_angle_deg": args.sharp_edge_angle_deg,
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
    obj_saved = obj_skipped = obj_repaired = 0

    for uid in uids_for_rank:
        if obj_saved >= target_objects:
            break

        _done = [
            all((samples_dir / f"{uid}_az{int(az):03d}_el{int(el):03d}" / fn).exists()
                for fn in ("triplane.pt", "query_pts.pt", "sdf_gt.pt", "normal_gt.pt",
                           "image_tokens.pt", "camera_extrinsics.json", "input_image.png"))
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
            # Triangle guard BEFORE repair so a pathological mesh can't stall voxelization.
            if args.max_triangles > 0 and len(mesh.faces) > args.max_triangles:
                if args.verbose:
                    tqdm.write(f"[skip mesh] {uid}: {len(mesh.faces):,} faces > limit {args.max_triangles:,}")
                del mesh
                obj_skipped += 1
                pbar.set_postfix(objects=obj_saved, skipped=obj_skipped)
                continue
            if not mesh.is_watertight:
                repaired = None
                if args.repair_meshes:
                    repaired, repair_method = repair_mesh_watertight(
                        mesh, voxel_res=args.repair_voxel_res,
                        voxel_method=args.repair_voxel_method)
                if repaired is None:
                    if args.verbose:
                        tqdm.write(f"[skip mesh] {uid}: not watertight, repair failed")
                    del mesh
                    obj_skipped += 1
                    pbar.set_postfix(objects=obj_saved, skipped=obj_skipped)
                    continue
                # Renormalize (voxel remesh can shift centroid/scale slightly) and
                # persist so the training-time near-surface augment builds its BVH
                # on the SAME geometry the GT SDF below is computed on.
                mesh, _, _ = _normalize_mesh_copy(repaired, radius)
                del repaired
                obj_repaired += 1
                try:
                    mesh.export(os.path.join(cache_dir, uid, "repaired.obj"))
                except Exception as e:
                    if args.verbose:
                        tqdm.write(f"[repair] {uid}: could not save repaired mesh ({e})")
                if args.verbose:
                    tqdm.write(f"[repair] {uid}: {repair_method}")
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
            sharp_edge_fraction=args.sharp_edge_fraction,
            sharp_edge_angle_deg=args.sharp_edge_angle_deg,
        )
        sdf_gt_np, normal_gt_np = compute_sdf(mesh, query_pts_np, return_normals=True)

        for (az, el), already_done in zip(view_pairs, _done):
            sample_id  = f"{uid}_az{int(az):03d}_el{int(el):03d}"
            sample_dir = samples_dir / sample_id

            if already_done:
                continue

            try:
                tmp_dir = sample_dir.parent / f"_tmp_{sample_id}"
                tmp_dir.mkdir(parents=True, exist_ok=True)

                # 1. Render to input_image.png (kept on disk for LoRA training).
                img_path = tmp_dir / "input_image.png"
                pil_image = render_mesh_to_image(
                    mesh,
                    elevation=float(el),
                    fov=args.fov,
                    size=args.image_size,
                    azimuth=float(az),
                    extrinsics_json_path=tmp_dir / "camera_extrinsics.json",
                )
                pil_image.save(img_path)
                image_np = np.array(Image.open(img_path).convert("RGB"))
                del pil_image

                # 2. Run TripoSR to get the triplane (scene_codes).
                with torch.no_grad():
                    scene_codes = model([image_np], device=device)
                triplane = scene_codes[0].half().cpu()

                # 2b. Cache the DINO image-tokenizer output (frozen regardless of
                # LoRA config — see compute_cached_image_tokens) so training can
                # skip re-running the ViT every step.
                image_tokens = compute_cached_image_tokens(model, image_np, device)
                del image_np, scene_codes

                # 3. Persist core training tensors only (no source_mesh.obj, no input_view.png).
                torch.save(triplane, tmp_dir / "triplane.pt")
                torch.save(torch.from_numpy(query_pts_np), tmp_dir / "query_pts.pt")
                torch.save(torch.from_numpy(sdf_gt_np), tmp_dir / "sdf_gt.pt")
                torch.save(image_tokens, tmp_dir / "image_tokens.pt")
                torch.save(torch.from_numpy(normal_gt_np), tmp_dir / "normal_gt.pt")

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

        del mesh, query_pts_np, sdf_gt_np, normal_gt_np
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pbar.close()
    if is_main:
        total_samples = obj_saved * n_views
        print(f"\nPrecompute done — {obj_saved} objects ({total_samples} samples), "
              f"{obj_repaired} repaired, {obj_skipped} skipped -> {dataset_dir}")

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
        # listdir, NOT glob("*/triplane.pt"): on the ~1M-dir NFS dataset the glob
        # stats every entry (>120 s, measured) vs 1.4 s for a readdir. The atomic
        # _tmp->final rename in precompute guarantees a final-named dir has its
        # triplane.pt, so the listing is exactly equivalent.
        all_samples = sorted(Path(root) / "samples" / _n / "triplane.pt"
            for _n in os.listdir(Path(root) / "samples")
            if not _n.startswith("_tmp"))
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
        nrm      : (N_POINTS, 3)          — GT SDF gradient direction (mesh frame);
                                            all-zeros for legacy datasets without
                                            normal_gt.pt (masked out of the loss)
        img      : (H, W, 3) uint8 array  — input image; used only if img_tokens
                                            is a placeholder (legacy dataset)
        img_tokens : (n_tokens, C) or (1,) — cached DINO image-tokenizer output;
                                            (1,) placeholder if this dataset has
                                            no image_tokens.pt (see has_cached_tokens)
        R        : (3, 3)                 — rotation mesh→triplane frame
        uid      : str                    — object UID (for loading mesh / BVH cache)
    """

    def __init__(self, dataset_dir: str, uid_whitelist: set | None = None,
                 sample_whitelist: set | None = None):
        root = Path(dataset_dir)
        # listdir, NOT glob("*/triplane.pt"): on the ~1M-dir NFS dataset the glob
        # stats every entry (>120 s, measured) vs 1.4 s for a readdir. The atomic
        # _tmp->final rename in precompute guarantees a final-named dir has its
        # triplane.pt, so the listing is exactly equivalent.
        all_samples = sorted(Path(root) / "samples" / _n / "triplane.pt"
            for _n in os.listdir(Path(root) / "samples")
            if not _n.startswith("_tmp"))
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
        # ONE parallel pass for the three per-sample NFS checks (extrinsics json,
        # input_image.png, image_tokens.pt). These used to be three sequential
        # sweeps of ~4 round-trips per sample; on the 1M-sample NFS dataset with
        # 4 DDP ranks contending that measured ~100 samples/s, i.e. ~2 HOURS of
        # startup per launch before a single training step. NFS I/O releases the
        # GIL, so a thread pool turns it into a few seconds. Semantics are
        # unchanged: same strict loader (exceptions propagate through map), same
        # order, same missing/has_cached_tokens results.
        def _scan(d: Path):
            return (load_R_world_from_recon_json_strict(d),
                    (d / "input_image.png").exists(),
                    (d / "image_tokens.pt").exists())
        _threads = int(os.environ.get("SDFER_SCAN_THREADS", DATASET_SCAN_THREADS))
        with ThreadPoolExecutor(max_workers=_threads) as _ex:
            _scanned = list(_ex.map(_scan, self.sample_dirs, chunksize=256))
        self.R_list: list[np.ndarray] = [r for r, _, _ in _scanned]

        missing = [d for d, (_, _img_ok, _) in zip(self.sample_dirs, _scanned) if not _img_ok]
        if missing:
            raise RuntimeError(
                f"{len(missing)} sample(s) are missing input_image.png "
                f"(e.g. {missing[0].name}). Re-run precompute to regenerate."
            )

        # Dataset-wide flag (not per-sample): a dataset is precomputed in one
        # pass, so it's either fully cached or not at all in practice. Training
        # branches on this ONCE per outer step rather than handling a mixed
        # batch, so a partially-upgraded dataset falls back to the slow path
        # entirely until fully re-precomputed.
        self.has_cached_tokens: bool = all(_tok for _, _, _tok in _scanned)

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int):
        p = self.sample_dirs[idx]
        R = torch.from_numpy(self.R_list[idx]).float()
        pts = torch.load(p / "query_pts.pt", map_location="cpu",
                         weights_only=False).clamp(-self.radius, self.radius)
        sdf = torch.load(p / "sdf_gt.pt", map_location="cpu", weights_only=False)
        nrm_path = p / "normal_gt.pt"
        if nrm_path.exists():
            nrm = torch.load(nrm_path, map_location="cpu", weights_only=False)
        else:
            nrm = torch.zeros(pts.shape[0], 3)  # legacy dataset: no normal supervision
        uid = p.name.split("_az")[0]
        img = np.array(Image.open(p / "input_image.png").convert("RGB"))
        if self.has_cached_tokens:
            img_tokens = torch.load(p / "image_tokens.pt", map_location="cpu", weights_only=False)
        else:
            img_tokens = torch.zeros(1)  # unused placeholder; training uses img instead
        return pts, sdf, nrm, img, img_tokens, R, uid


# ─── Mesh surface metrics ─────────────────────────────────────────────────────

def mesh_surface_metrics(gt_mesh, pred_mesh, n_samples: int = 50000,
                         fscore_tau: float = 0.01) -> dict:
    """Chamfer-L2 + F-score between surface samples of two ALIGNED meshes.

    Standard protocol (MeshLRM / TripoSG / Dora-bench): sample each surface,
    take bidirectional nearest-neighbour distances; Chamfer is the sum of both
    mean squared distances, F-score the harmonic mean of precision/recall at
    ``fscore_tau``. Complements SDF MSE, which is dominated by far-field error
    that marching cubes never sees.
    """
    from scipy.spatial import cKDTree
    gt_pts   = np.asarray(gt_mesh.sample(n_samples),   dtype=np.float64)
    pred_pts = np.asarray(pred_mesh.sample(n_samples), dtype=np.float64)
    d_pred_to_gt = cKDTree(gt_pts).query(pred_pts, workers=-1)[0]
    d_gt_to_pred = cKDTree(pred_pts).query(gt_pts, workers=-1)[0]
    chamfer   = float((d_pred_to_gt ** 2).mean() + (d_gt_to_pred ** 2).mean())
    precision = float((d_pred_to_gt < fscore_tau).mean())
    recall    = float((d_gt_to_pred < fscore_tau).mean())
    fscore    = (2 * precision * recall / (precision + recall)
                 if precision + recall > 0 else 0.0)
    return {"chamfer": chamfer, "fscore": fscore,
            "precision": precision, "recall": recall}


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
    # Do NOT pass skimage's `normals` as vertex_normals. For an SDF-like volume
    # (negative inside) marching_cubes returns normals pointing INWARD — measured
    # 100% inward on an analytic sphere — while the face WINDING is correct
    # (signed volume +0.267 vs the true 0.268). Handing pyrender inward normals
    # makes every surface shade as if facing away from the light, which is the
    # black/patchy look in the wandb rows. Omitting them lets trimesh derive
    # correct outward normals from the (correct) winding.
    return tr.Trimesh(vertices=verts, faces=faces)


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
    # Do NOT pass skimage's `normals` as vertex_normals. For an SDF-like volume
    # (negative inside) marching_cubes returns normals pointing INWARD — measured
    # 100% inward on an analytic sphere — while the face WINDING is correct
    # (signed volume +0.267 vs the true 0.268). Handing pyrender inward normals
    # makes every surface shade as if facing away from the light, which is the
    # black/patchy look in the wandb rows. Omitting them lets trimesh derive
    # correct outward normals from the (correct) winding.
    return tr.Trimesh(vertices=verts, faces=faces)


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
    triposr_model=None,  # required
    fscore_tau: float = 0.01,
    mesh_metric_samples: int = 50000,
) -> None:
    sdf_mlp.eval()
    triposr_model.eval()

    label_metrics: dict[str, list] = {"seen": [], "unseen": []}
    for label, sample_dirs in (("seen", seen_dirs), ("unseen", unseen_dirs)):
        for sample_dir in sample_dirs:
            uid = sample_dir.name.split("_az")[0]
            try:
                img_np = np.array(Image.open(sample_dir / "input_image.png").convert("RGB"))
                with torch.no_grad():
                    sc = triposr_model([img_np], device=device)
                triplane = sc[0].detach().float().cpu()
                # Second triplane from ORIGINAL TripoSR (LoRA off, pretrained
                # post_processor) so the NeRF row below is a genuine
                # stock-vs-fine-tuned comparison rather than the original
                # density head decoding an already-fine-tuned triplane.
                with torch.no_grad(), stock_triposr(triposr_model):
                    triplane_stock = triposr_model([img_np], device=device)[0].detach().float().cpu()

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
                # Prefer the repaired (watertight) mesh saved at precompute — it is
                # the geometry the GT SDF supervision was actually computed on.
                repaired_files = [f for f in mesh_files if f.stem == "repaired"]
                gt_mesh = load_and_normalize_mesh(
                    str(repaired_files[0] if repaired_files else mesh_files[0]), radius)

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
                            triplane_stock,   # ORIGINAL TripoSR, end to end
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

                # ── Surface metrics: Chamfer / F-score vs GT (meshes are aligned:
                # pred_mesh was rotated into world frame via R above) ───────────
                mesh_metrics = None
                try:
                    mesh_metrics = mesh_surface_metrics(
                        gt_mesh, pred_mesh,
                        n_samples=mesh_metric_samples, fscore_tau=fscore_tau)
                    label_metrics[label].append(mesh_metrics)
                    tqdm.write(f"[vis] {label}/{uid[:12]}: "
                               f"chamfer={mesh_metrics['chamfer']:.6f}  "
                               f"fscore@{fscore_tau}={mesh_metrics['fscore']:.3f}")
                except Exception as e:
                    tqdm.write(f"[vis] surface metrics failed for {uid}: {e}")

                title = f"{label} - {uid[:12]} - epoch {epoch}"
                if mesh_metrics is not None:
                    title += (f"  |  CD {mesh_metrics['chamfer']:.5f}"
                              f"  F@{fscore_tau} {mesh_metrics['fscore']:.2f}")
                save_path = output_dir / label / f"{uid}_epoch{epoch:04d}.png"
                create_mesh_comparison_visualization(
                    gt_mesh, pred_mesh,
                    title=title,
                    save_path=save_path,
                    input_image=input_image,
                    nerf_mesh=nerf_mesh,
                )

                if wandb_enabled:
                    try:
                        log_dict = {
                            f"mesh_reconstruction/{label}/{sample_dir.name}": wandb.Image(str(save_path)),
                            "mesh_reconstruction/epoch": epoch,
                        }
                        if mesh_metrics is not None:
                            log_dict.update({
                                f"mesh_metrics/{label}/{sample_dir.name}/chamfer": mesh_metrics["chamfer"],
                                f"mesh_metrics/{label}/{sample_dir.name}/fscore": mesh_metrics["fscore"],
                            })
                        wandb.log(log_dict)
                    except Exception:
                        pass

                del triplane, pred_mesh, gt_mesh, nerf_mesh
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                tqdm.write(f"[vis] failed for {uid}: {e}")

    # ── Per-split means (the headline numbers to watch across epochs) ─────────
    if wandb_enabled:
        try:
            mean_log: dict = {"mesh_reconstruction/epoch": epoch}
            for label, ms in label_metrics.items():
                if ms:
                    mean_log[f"mesh_metrics/{label}/chamfer_mean"] = float(
                        np.mean([m["chamfer"] for m in ms]))
                    mean_log[f"mesh_metrics/{label}/fscore_mean"] = float(
                        np.mean([m["fscore"] for m in ms]))
            wandb.log(mean_log)
        except Exception:
            pass

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

    # listdir, NOT glob("*/triplane.pt"): on the ~1M-dir NFS dataset the glob
    # stats every entry (>120 s, measured) vs 1.4 s for a readdir. The atomic
    # _tmp->final rename in precompute guarantees a final-named dir has its
    # triplane.pt, so the listing is exactly equivalent.
    all_sample_paths = sorted(Path(dataset_dir) / "samples" / _n / "triplane.pt"
        for _n in os.listdir(Path(dataset_dir) / "samples")
        if not _n.startswith("_tmp"))
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
    if True:  # built on EVERY rank now — eval is sharded, see the eval block below
        _unseen_uid_names  = {s for s in all_sample_names if s.split("_az")[0] in test_uids}
        _unseen_view_names = set(test_view_names) - _unseen_uid_names
        test_sample_names = _unseen_uid_names | _unseen_view_names
        _n_full = len(test_sample_names)

        # CAP THE TEST SET. It otherwise grows linearly with n_objects (360 @100
        # objects, 3.6k @1k, 36k @10k) and is evaluated by RANK 0 ALONE every
        # epoch with no collective inside. Past ~30 min of eval the other ranks,
        # already blocked in the next epoch's DDP gradient all-reduce, trip the
        # NCCL watchdog and the whole job dies — which is exactly what killed the
        # 10k run. Capping also stops three GPUs idling through the eval.
        # Sampling is STRATIFIED (keeps the unseen-UID : unseen-view ratio, since
        # those measure two different kinds of generalization) and uses a FIXED
        # seed, so the same subset is scored every epoch and across runs — test
        # curves stay comparable rather than jittering with a fresh draw.
        if 0 < args.test_max_samples < _n_full:
            _rng  = random.Random(1234)
            _frac = args.test_max_samples / _n_full
            _keep: set = set()
            for _grp in (_unseen_uid_names, _unseen_view_names):
                _g = sorted(_grp)
                if not _g:
                    continue
                _k = min(len(_g), max(1, round(len(_g) * _frac)))
                _keep |= set(_rng.sample(_g, _k))
            test_sample_names = _keep

        # Every rank derives test_sample_names from the SAME fixed seeds (42/43
        # for the splits, 1234 for the cap), so all ranks agree on the set and
        # the shards below partition it cleanly.
        if test_sample_names:
            test_dataset = SDFLazyDataset(args.dataset_dir, sample_whitelist=test_sample_names)
            if is_ddp:
                # shuffle=False -> deterministic, reproducible shards.
                # drop_last=False -> DistributedSampler PADS to equal length, so every
                # rank runs the same number of batches (a handful of samples may be
                # double-counted when the size is not divisible by world_size; that
                # is standard for eval and does not bias the mean materially).
                test_sampler = DistributedSampler(test_dataset, num_replicas=world_size,
                                                  rank=rank, shuffle=False, drop_last=False)
                test_loader = DataLoader(test_dataset, batch_size=args.samples_per_batch,
                                         sampler=test_sampler, num_workers=args.num_workers,
                                         pin_memory=True,
                                         persistent_workers=args.num_workers > 0)
            else:
                test_loader = DataLoader(test_dataset, batch_size=args.samples_per_batch,
                                         shuffle=False, num_workers=args.num_workers,
                                         pin_memory=True,
                                         persistent_workers=args.num_workers > 0)
            if is_main:
                _n_uid  = sum(1 for s in test_sample_names if s in _unseen_uid_names)
                _cap_note = (f"  [capped from {_n_full:,} by TEST_MAX_SAMPLES={args.test_max_samples:,}]"
                             if len(test_sample_names) < _n_full else "")
                _shard = f", sharded {world_size}-way (~{len(test_dataset)//world_size:,}/rank)" if is_ddp else ""
                print(f"Test dataset: {len(test_dataset):,} samples "
                      f"({_n_uid:,} unseen-UID + {len(test_sample_names) - _n_uid:,} unseen-view)"
                      f"{_cap_note}{_shard}")

    # ── LoRA: load full TripoSR, inject adapters, build lora_optimizer ────────
    from tsr.system import TSR
    if is_main:
        print("Loading TripoSR for LoRA fine-tuning...")
    triposr_model = TSR.from_pretrained(
        args.model, config_name="config.yaml", weight_name="model.ckpt"
    )
    # Inject LoRA BEFORE moving to device so the freshly-created lora_A/lora_B
    # Parameters get moved to the GPU too (.to() is in-place on Parameters, so
    # the returned references stay valid and now point at GPU tensors).
    lora_trainable_params = apply_lora_to_triposr(
        triposr_model,
        args.lora_block_start, args.lora_block_end,
        args.lora_rank, args.lora_alpha,
    )
    triposr_model.to(device)
    # All ranks must start from IDENTICAL LoRA weights. lora_A is randomly
    # initialised independently per process, so without this each rank would
    # train a different adapter even though grads are all-reduced every step
    # (matching grad updates from mismatched starting points still diverge).
    # Broadcast rank 0's trainable weights to everyone. (sdf_mlp gets this for
    # free from its DDP wrapper; this model is grad-synced manually instead.)
    if is_ddp:
        for p in lora_trainable_params:
            dist.broadcast(p.data, src=0)
    lora_optimizer = torch.optim.AdamW(
        lora_trainable_params,
        lr=args.lora_lr,
        weight_decay=args.lora_weight_decay,
    )
    _density_activation = triposr_model.renderer.cfg.density_activation
    _density_bias = float(triposr_model.renderer.cfg.density_bias)
    n_lora = sum(p.numel() for p in lora_trainable_params)
    n_total = sum(p.numel() for p in triposr_model.parameters())
    if is_main:
        print(f"LoRA ready: {n_lora:,} trainable / {n_total:,} total TripoSR params "
              f"(backbone blocks {args.lora_block_start}-{args.lora_block_end - 1} "
              f"+ post_processor) | rank={args.lora_rank} alpha={args.lora_alpha}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Optionally expose frozen TripoSR decoder for NeRF sanity-check mesh ──
    # Reuse the already-loaded model; decoder weights are still frozen.
    triposr_decoder: nn.Module | None = None
    if is_main and args.use_nerf_vis:
        try:
            triposr_decoder = triposr_model.decoder.eval()
            for p in triposr_decoder.parameters():
                p.requires_grad_(False)
            print(f"TripoSR decoder ready (density_activation={_density_activation}, bias={_density_bias})")
        except Exception as e:
            print(f"Warning: could not attach TripoSR decoder ({e}). NeRF vis disabled.")
    elif is_main:
        print("[nerf_vis] Skipped (USE_NERF_VIS=False).")

    if is_main:
        print(f"Dataset: {len(dataset)} samples ({len(dataset) * n_pts_per_sample:,} points) | "
              f"{n_test} unseen UIDs, {len(test_view_names)} unseen views | "
              f"radius={radius:.4f} | feat_dim={feat_dim}")

    if is_ddp:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=True, drop_last=True)
        loader = DataLoader(dataset, batch_size=args.samples_per_batch, sampler=sampler,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=args.num_workers > 0, drop_last=True)
    else:
        sampler = None
        loader = DataLoader(dataset, batch_size=args.samples_per_batch, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=args.num_workers > 0, drop_last=True)

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
        start_epoch = ckpt.get("epoch", 0)
        if "scheduler" in ckpt:
            # OneCycleLR bakes total_steps into its state_dict. Restoring a
            # checkpoint written under a DIFFERENT EPOCHS silently reinstates the
            # old budget, then dies mid-run with "Tried to step N times. The
            # specified number of total steps is M". Rebuild for the new budget
            # instead, fast-forwarded to the steps already consumed.
            _old_total = ckpt["scheduler"].get("total_steps")
            if args.use_onecycle and _old_total is not None and _old_total != total_steps:
                consumed = min(start_epoch * len(loader) * steps_per_outer, total_steps)
                if is_main:
                    print(f"[resume] EPOCHS changed since this checkpoint "
                          f"(schedule was {_old_total} steps, now {total_steps}). "
                          f"Rebuilding OneCycleLR, fast-forwarded to step {consumed}.")
                for gparam in optimizer.param_groups:
                    gparam.setdefault("initial_lr", args.lr / 25.0)  # OneCycle div_factor default
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=args.lr, total_steps=total_steps,
                    pct_start=args.onecycle_pct_start, last_epoch=consumed - 1,
                )
            else:
                scheduler.load_state_dict(ckpt["scheduler"])
        if "lora_model" in ckpt:
            triposr_model.load_state_dict(ckpt["lora_model"], strict=False)
        if "lora_optimizer" in ckpt:
            lora_optimizer.load_state_dict(ckpt["lora_optimizer"])
        # start_epoch was read above (the scheduler rebuild needs it).
        if is_main:
            print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # ── DDP wrapping (after resume so we load into the raw module) ─────────────
    if is_ddp:
        sdf_mlp = DDP(sdf_mlp, device_ids=[local_rank])

    # ── torch.compile (optional; suppress_errors falls back to eager per-graph
    # rather than crashing training on a compile failure) ──────────────────────
    # Hard guard: the eikonal term needs a double backward (autograd.grad with
    # create_graph=True, then loss.backward()), which aot_autograd rejects at
    # RUNTIME — dynamo's suppress_errors only covers graph-capture failures, so
    # it cannot save us here. Refuse rather than crash 1 step into training.
    _compile_ok = args.use_torch_compile and device.type == "cuda"
    if _compile_ok and args.eikonal_weight > 0:
        _compile_ok = False
        if is_main:
            print("[compile] DISABLED: eikonal_weight > 0 requires double backward, "
                  "which torch.compile/aot_autograd does not support.")
    if _compile_ok:
        # MUST be "import ... as": a bare `import torch._dynamo` binds the name
        # `torch` LOCALLY for this entire function, shadowing the module-level
        # import and making every earlier `torch.*` use (e.g. torch.device at the
        # top of run_train) raise UnboundLocalError. `import x.y as z` binds only z.
        import torch._dynamo as _dynamo
        _dynamo.config.suppress_errors = True
        sdf_mlp = torch.compile(sdf_mlp)
        if is_main:
            print("[compile] sdf_mlp wrapped with torch.compile (suppress_errors=True).")

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
        epoch_loss = epoch_sdf = epoch_mse = epoch_eik = epoch_bce = epoch_nrm = 0.0
        diag_steps = 0
        diag_sign_acc_sum = 0.0
        diag_preclip_norm_sum = 0.0
        diag_reject_frac_sum = 0.0
        diag_grad_mean_sum = 0.0
        diag_grad_min = float("inf")
        diag_grad_max = float("-inf")
        diag_pred_min = float("inf")
        diag_pred_max = float("-inf")
        diag_tgt_min = float("inf")
        diag_tgt_max = float("-inf")
        if pbar is not None:
            pbar.set_description(f"Epoch {epoch + 1}/{args.epochs}")

        for sample_pts, sample_sdf, sample_nrm, sample_imgs, sample_img_tokens, sample_R, _sample_uids in loader:
            # sample_pts:        (S, N, 3)       — uniform pts in mesh frame, CPU
            # sample_sdf:        (S, N)          — GT SDF, CPU
            # sample_nrm:        (S, N, 3)       — GT SDF gradient dir, mesh frame (zeros = legacy)
            # sample_imgs:       (S, H, W, 3)    — uint8 images; only used if no cached tokens
            # sample_img_tokens: (S, n_tok, C)   — cached DINO tokens, or (S, 1) placeholder
            # sample_R:          (S, 3, 3)       — rotation matrix per sample, CPU
            S, N = sample_sdf.shape
            R_gpu = sample_R.to(device)  # (S, 3, 3)

            # Run TripoSR with LoRA adapters to produce triplanes — ONE batched
            # forward pass for all S samples, not S sequential batch-1 calls.
            # TSR.forward derives its batch size from the length of the image
            # list, and there is no BatchNorm anywhere in the backbone (only
            # LayerNorm/GroupNorm, both per-sample), so this is numerically
            # equivalent to the old per-sample loop, just far better GPU
            # utilization for a batch this small.
            # If this dataset has cached DINO tokens (image_tokenizer is never
            # touched by LoRA — see compute_cached_image_tokens), skip the ViT
            # forward entirely and only run tokenizer→backbone→post_processor.
            # bf16 autocast covers only this forward pass (the expensive
            # LoRA-adapted transformer); the triplane is cast back to fp32
            # immediately after so the SDF MLP + eikonal double-backward stay
            # full precision, where numerical accuracy of the distance field
            # matters most and the compute cost is negligible anyway.
            triposr_model.train()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                if dataset.has_cached_tokens:
                    scene_codes = triposr_forward_from_cached_tokens(
                        triposr_model, sample_img_tokens.to(device))
                else:
                    img_batch = [sample_imgs[s].numpy() for s in range(S)]  # each (H, W, 3) uint8
                    scene_codes = triposr_model(img_batch, device=device)
            triplanes_gpu = [scene_codes[s].float() for s in range(S)]  # [3, 40, 64, 64] each

            # Detach each sample's triplane into a fresh leaf. Every minibatch
            # below backprops only into this leaf (cheap: bilinear sampling +
            # the small SDF MLP) — never into the expensive LoRA/TripoSR graph
            # directly. PyTorch accumulates each minibatch's contribution onto
            # the leaf's .grad for free, across all steps_per_outer minibatches.
            # The REAL upstream graph is backpropped only once per sample, at
            # the end of the outer step (search "graph split" below), using
            # that accumulated leaf gradient — mathematically identical total
            # gradient to backpropping the full graph every minibatch (proven:
            # same result by linearity of differentiation), but the expensive
            # part now runs S times per outer step instead of steps_per_outer
            # (~32) times.
            triplane_leaves = [t.detach().requires_grad_(True) for t in triplanes_gpu]

            # Point pool, all in mesh frame on CPU. Features are NEVER stored
            # here — only raw 3-float points, scalar SDF, and a sample-id tag.
            # Uniform/near-surface/sharp-edge mix is fixed at precompute time
            # (sample_query_points) — no online resampling pass here.
            pool_pts = sample_pts.reshape(S * N, 3)             # (S*N, 3) view
            pool_sdf = sample_sdf.reshape(S * N)                # (S*N,)
            pool_nrm = sample_nrm.reshape(S * N, 3)             # (S*N, 3) view
            pool_sid = torch.arange(S).repeat_interleave(N)     # (S*N,)

            # ── Shuffle pool (cross-sample mixing) and train in mini-batches ──
            perm = torch.randperm(pool_pts.shape[0])
            pool_pts = pool_pts[perm]
            pool_sdf = pool_sdf[perm]
            pool_nrm = pool_nrm[perm]
            pool_sid = pool_sid[perm]

            # LoRA grads accumulate across ALL mini-batches in this outer step,
            # then the lora_optimizer steps once at the end.  Zero them here so
            # the accumulation starts clean for this outer step.
            lora_optimizer.zero_grad()

            # CRITICAL for DDP: every rank must run the SAME number of backward
            # / gradient all-reduce calls. Pool size is now fixed at S*N (no
            # online augment appending a variable count), so steps_per_outer
            # (= S*N / batch_size) tiles through the whole shuffled pool once.
            for mb in range(steps_per_outer):
                start = mb * args.batch_size
                end   = start + args.batch_size
                pm     = pool_pts[start:end].to(device)
                sg     = pool_sid[start:end].to(device)
                sdf_gt = pool_sdf[start:end].to(device)
                nrm_gt = pool_nrm[start:end].to(device)

                # Gradient flows from sdf_pred through base_feats → triplane
                # LEAF only (never into LoRA/TripoSR here — see graph split
                # above). pts_trip is detached before being used as query_pts
                # so the eikonal ∂sdf/∂query_pts graph is independent too.
                base_feats, pts_trip = _compute_features(pm, sg, triplane_leaves, R_gpu)
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
                # TSDF clamp (DeepSDF): regress clamped values so capacity focuses
                # near the surface; the weighting still uses the UNCLAMPED GT so
                # near-surface emphasis is preserved. Sign-BCE (raw pred) and
                # eikonal keep steering badly-wrong far-field points.
                if args.sdf_clamp > 0:
                    _c = float(args.sdf_clamp)
                    per_point = surface_weighted_se(
                        sdf_pred.clamp(-_c, _c), sdf_gt.clamp(-_c, _c),
                        sigma=args.surface_loss_sigma, weight_target=sdf_gt)
                else:
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

                # ── Surface-normal alignment (IGR / TripoSG): align ∇f with the
                # GT SDF gradient direction at near-surface points. Normals are
                # stored in mesh frame; rotate into the triplane frame that the
                # gradients live in (n_trip = R @ n_mesh, same map as points).
                # Zero-norm normals (legacy datasets) are masked out.
                if args.normal_loss_weight > 0:
                    nrm_trip = torch.einsum("bij,bj->bi", R_gpu[sg], nrm_gt)
                    nrm_valid = ((sdf_gt.abs() < args.normal_loss_threshold)
                                 & (nrm_trip.norm(dim=-1) > 0.5))
                    if nrm_valid.any():
                        g_pred = F.normalize(gradients[nrm_valid], dim=-1, eps=1e-8)
                        g_gt   = F.normalize(nrm_trip[nrm_valid], dim=-1, eps=1e-8)
                        normal_loss = (1.0 - (g_pred * g_gt).sum(dim=-1)).mean()
                    else:
                        normal_loss = sdf_pred.new_zeros(())
                else:
                    normal_loss = sdf_pred.new_zeros(())

                loss = (sdf_loss
                        + args.eikonal_weight * eikonal_loss
                        + args.sign_bce_weight * bce_loss
                        + args.normal_loss_weight * normal_loss)

                optimizer.zero_grad()
                # No retain_graph needed: this minibatch's graph only reaches
                # back to triplane_leaves (a leaf) and the SDF MLP — nothing
                # here is shared with any other minibatch's graph, so it's
                # freed immediately as usual. The leaf's accumulated .grad is
                # what carries this minibatch's contribution to the LoRA
                # gradient forward to the graph-split backward after the loop.
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
                epoch_nrm  += normal_loss.item()
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
                            "train/normal_loss": normal_loss.item(),
                            "train/grad_norm_preclip": preclip_norm,
                            "train/lr": optimizer.param_groups[0]["lr"],
                        })
                    except Exception:
                        pass

            # ── Graph split: backprop the accumulated leaf gradients through
            # the REAL upstream (LoRA/TripoSR) graph — S backward calls here
            # instead of one per minibatch (steps_per_outer, ~32). All S slices
            # trace back to the SAME single batched TripoSR forward (item-4
            # batching), so retain_graph=True is needed until the last one
            # actually issued.
            leaf_grads = [(s, triplane_leaves[s].grad) for s in range(S)
                         if triplane_leaves[s].grad is not None]
            for i, (s, g) in enumerate(leaf_grads):
                triplanes_gpu[s].backward(g, retain_graph=(i < len(leaf_grads) - 1))

            # ── LoRA optimizer step (once per outer step, after all mini-batches) ──
            if is_ddp:
                # Manually sync accumulated LoRA grads across ranks before stepping.
                # all_reduce is a COLLECTIVE: every rank must issue exactly the same
                # number of calls in the same order or the mismatched ranks hang until
                # the NCCL timeout. Grads therefore must NOT be skipped when None —
                # and None is the norm, not the exception, since zero_grad() defaults
                # to set_to_none=True, so each outer step starts with every grad None
                # and they are only repopulated by the graph-split backward above.
                # Materialize zeros instead of skipping: a zero contribution is
                # mathematically correct and keeps all ranks in lockstep.
                for p in lora_trainable_params:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad.div_(world_size)
            torch.nn.utils.clip_grad_norm_(lora_trainable_params, args.grad_clip)
            lora_optimizer.step()

            # Free GPU-resident triplanes before the next outer step
            del triplanes_gpu, triplane_leaves, R_gpu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── End of epoch ──────────────────────────────────────────────────────
        # Aggregate losses across ranks so rank-0 logs the global average.
        if is_ddp:
            t = torch.tensor([epoch_loss, epoch_sdf, epoch_mse, epoch_eik, epoch_bce, epoch_nrm], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            epoch_loss, epoch_sdf, epoch_mse, epoch_eik, epoch_bce, epoch_nrm = (t / world_size).tolist()

        if is_main:
            n = len(loader)
            # Loss accumulators are summed per INNER minibatch, so normalise by
            # the inner-step count (diag_steps) to get a true mean — NOT by
            # len(loader) (outer steps), which would inflate by steps_per_outer.
            _ns = diag_steps if diag_steps > 0 else 1
            grad_mean_epoch = (diag_grad_mean_sum / diag_steps) if diag_steps > 0 else float("nan")
            sign_acc_epoch = (diag_sign_acc_sum / diag_steps) if diag_steps > 0 else float("nan")
            preclip_epoch = (diag_preclip_norm_sum / diag_steps) if diag_steps > 0 else float("nan")
            reject_epoch = (diag_reject_frac_sum / diag_steps) if diag_steps > 0 else float("nan")
            if wandb_enabled:
                try:
                    wandb.log({
                        "train/epoch_loss": epoch_loss / _ns,
                        "train/epoch_sdf_loss": epoch_sdf / _ns,
                        "train/epoch_mse": epoch_mse / _ns,
                        "train/epoch_eikonal_loss": epoch_eik / _ns,
                        "train/epoch_sign_bce_loss": epoch_bce / _ns,
                        "train/epoch_normal_loss": epoch_nrm / _ns,
                        "diag/sign_accuracy": sign_acc_epoch,
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

        # ── Test evaluation (SHARDED across ranks) ────────────────────────────
        # Previously rank 0 evaluated the whole test set alone while the other
        # ranks sat blocked in the next epoch's DDP gradient all-reduce; once
        # that exceeded NCCL_TIMEOUT_MIN the job died (the 10k-object failure).
        # Now every rank scores its own shard and the partial sums are reduced.
        #
        # CRITICAL: the accumulator and the all_reduce live OUTSIDE the
        # `test_loader is not None` guard. all_reduce is a collective — if any
        # rank skipped it we would deadlock in exactly the way this change is
        # meant to prevent. A rank with no shard simply contributes zeros.
        _test_acc = torch.zeros(5, device=device)   # [weighted, mse, mse_clamped, sign_acc, steps]
        if test_loader is not None:
            _mlp_module.eval()
            triposr_model.eval()
            test_weighted_sum = test_mse_sum = test_mse_clamped_sum = test_sign_acc_sum = 0.0
            test_steps = 0
            with torch.no_grad():
                for t_pts, t_sdf, _t_nrm, t_imgs, t_img_tokens, t_R, _ in test_loader:
                    tS, tN = t_sdf.shape
                    t_R_gpu = t_R.to(device)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                        enabled=(device.type == "cuda")):
                        if test_dataset.has_cached_tokens:
                            t_scene_codes = triposr_forward_from_cached_tokens(
                                triposr_model, t_img_tokens.to(device))
                        else:
                            t_img_batch = [t_imgs[s].numpy() for s in range(tS)]
                            t_scene_codes = triposr_model(t_img_batch, device=device)
                    t_trip_gpu = [t_scene_codes[s].float() for s in range(tS)]
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
                        if args.sdf_clamp > 0:
                            _tc = float(args.sdf_clamp)
                            test_mse_clamped_sum += F.mse_loss(
                                t_pred.clamp(-_tc, _tc), ms.clamp(-_tc, _tc)).item()
                        test_sign_acc_sum += float((torch.sign(t_pred) == torch.sign(ms)).float().mean().item())
                        test_steps += 1
                    del t_trip_gpu, t_R_gpu
            _mlp_module.train()
            _test_acc = torch.tensor(
                [test_weighted_sum, test_mse_sum, test_mse_clamped_sum,
                 test_sign_acc_sum, float(test_steps)], device=device)

        # Reduce partial sums from every shard. Unconditional under DDP so all
        # ranks issue exactly one collective here, in the same order, always.
        if is_ddp:
            dist.all_reduce(_test_acc, op=dist.ReduceOp.SUM)
        _tw, _tm, _tmc, _tsa, _tsteps = _test_acc.tolist()

        if is_main and wandb_enabled and _tsteps > 0:
            try:
                # Divide by the GLOBAL step count so the result is the mean over
                # the whole test set, not over one shard.
                _test_log = {
                    "test/epoch_sdf_loss": _tw / _tsteps,
                    "test/epoch_mse": _tm / _tsteps,
                    "test/sign_accuracy": _tsa / _tsteps,
                    "train/epoch": epoch + 1,
                }
                if args.sdf_clamp > 0:
                    _test_log["test/epoch_mse_clamped"] = _tmc / _tsteps
                wandb.log(_test_log)
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
            ckpt_dict = {
                "epoch": epoch + 1,
                "model": _mlp_module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "meta": meta,
                "args": vars(args),
            }
            ckpt_dict["lora_model"] = triposr_model.state_dict()
            ckpt_dict["lora_optimizer"] = lora_optimizer.state_dict()
            torch.save(ckpt_dict, ckpt_path)

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
                triposr_model=triposr_model,
                fscore_tau=args.fscore_tau,
                mesh_metric_samples=args.mesh_metric_samples,
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
        sharp_edge_fraction   = SHARP_EDGE_FRACTION,
        sharp_edge_angle_deg  = SHARP_EDGE_ANGLE_DEG,
        repair_meshes         = REPAIR_MESHES,
        repair_voxel_res      = REPAIR_VOXEL_RES,
        repair_voxel_method   = REPAIR_VOXEL_METHOD,
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
        sdf_clamp              = SDF_CLAMP,
        normal_loss_weight     = NORMAL_LOSS_WEIGHT,
        normal_loss_threshold  = NORMAL_LOSS_THRESHOLD,
        fscore_tau             = FSCORE_TAU,
        mesh_metric_samples    = MESH_METRIC_SAMPLES,
        num_workers    = NUM_WORKERS,
        run_name       = RUN_NAME,
        weight_decay   = WEIGHT_DECAY,
        use_tanh_output = USE_TANH_OUTPUT,
        use_triplane_features = USE_TRIPLANE_FEATURES,
        test_fraction  = TEST_FRACTION,
        test_view_fraction = TEST_VIEW_FRACTION,
        test_max_samples   = TEST_MAX_SAMPLES,
        vis_every      = VIS_EVERY,
        vis_seen       = VIS_SEEN,
        vis_unseen     = VIS_UNSEEN,
        vis_azimuths_per_object = VIS_AZIMUTHS_PER_OBJECT,
        vis_resolution = VIS_RESOLUTION,
        batch_size        = BATCH_SIZE,
        samples_per_batch = SAMPLES_PER_BATCH,
        resume         = RESUME,
        use_nerf_vis   = USE_NERF_VIS,
        use_torch_compile = USE_TORCH_COMPILE,
        lora_rank           = LORA_RANK,
        lora_alpha          = LORA_ALPHA,
        lora_block_start    = LORA_BLOCK_START,
        lora_block_end      = LORA_BLOCK_END,
        lora_lr             = LORA_LR,
        lora_weight_decay   = LORA_WEIGHT_DECAY,
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
