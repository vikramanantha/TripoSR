"""
train_sdf_head_fast.py  —  Ground-up rewrite of train_sdf_head.py's TRAIN phase
for wall-clock speed. SAME model architecture, SAME losses, SAME dataset format,
SAME checkpoint format — only the training-loop machinery is rebuilt.

It imports the architecture (SDFMLP, LoRA), losses, dataset class and the
visualization stack from train_sdf_head.py so the model stays identical by
construction — but ALL configuration lives HERE (nothing reads the old file's
constants), so editing/reverting the old script cannot change this one's
behaviour. Precompute is NOT reimplemented: `--command precompute` delegates to
train_sdf_head.run_precompute with the OLD script's own precompute settings
(Objaverse source; objaverse_paths.configure_objaverse resolves the shared
mirror both on the host, /mnt/workspace/datasets/objaverse, and inside the
docker container, /mnt/hostmnt/workspace/datasets/objaverse).

Launch (inside the TripoSR docker container, where the repo mounts at
~/TripoSR and the venv python is required for `import tsr`):

    # all GPUs
    ~/TripoSR/.venv/bin/python -m torch.distributed.run --standalone \
        --nnodes=1 --nproc_per_node=4 train_sdf_head_fast.py --command train
    # single GPU
    ~/TripoSR/.venv/bin/python train_sdf_head_fast.py --command train

Why the old loop was slow at 10k objects, and what changed
──────────────────────────────────────────────────────────
The per-step cost is dominated by the LoRA'd TripoSR backbone fwd+bwd (tens of
TFLOPs per outer step; the 103k-param SDF MLP is noise next to it), and that
backbone pass runs once per view per epoch. So the rewrite (a) squeezes more
learning out of every backbone pass, (b) makes the pass itself faster, and
(c) right-sizes the step budget:

1. ONE gradient step per outer step over the whole point pool (the old
   BATCH_SIZE=4096 inner loop ran 32 sequential mini-steps per pool, each with
   fixed kernel-launch/sync overhead for a tiny MLP — pure latency, no FLOPs).
   LR is sqrt-scaled for the bigger step (AUTO_SCALE_LR below).

2. ALL 32,768 precomputed points per sample per step. The backbone pass costs
   the same whether the MLP sees 4,096 or 32,768 points, so this multiplies
   supervision per backbone pass for ~free.

3. EPOCHS = 100 (was 500, tuned at 1k objects). Steps/epoch scale linearly
   with object count, so 500 epochs at 10k objects is 10x the gradient updates
   that schedule was tuned for — that IS the "many days". 100 epochs at 10k
   objects still runs ~2x the total updates of a 500-epoch 1k-object run, with
   double the points per update on top. OneCycleLR derives its schedule from
   EPOCHS, so any override (SDFER_EPOCHS=...) yields a complete schedule.
   This is the single biggest wall-clock lever.

4. torch.compile on the TripoSR backbone. The old script couldn't compile
   anything because the eikonal loss needs a double backward — but the graph
   split (triplane leaf detach) keeps that double backward entirely downstream
   of the leaf, so the BACKBONE graph only ever sees one ordinary backward and
   is safe to compile. The SDF MLP stays eager.

5. Batched triplane sampling: one grid_sample over all S samples' 3 planes,
   replacing the per-sample-id loop (torch.unique + .item() = ~S forced device
   syncs per step) and the pointless cross-sample pool shuffle (a shuffle
   permutes a set that a single mean-reduced step averages over anyway).

6. One flat LoRA gradient all-reduce (~25 MB, 1 NCCL call) instead of one
   all_reduce per LoRA tensor (~320 latency-bound calls per step).

7. CUDA-stream prefetcher: next batch's H2D copies overlap current compute.

8. Fused AdamW, TF32 matmuls for the fp32 MLP path, cudnn.benchmark.
   (Flash/SDPA attention was already on: tsr's Attention uses AttnProcessor2_0.)

9. Eval is SHARDED across ranks (the committed script ran the whole test set
   on rank 0 alone every epoch — at 10k objects that once outran the NCCL
   timeout and killed the job) and runs every EVAL_EVERY epochs.

Memory note (4x A6000 48 GB): backbone activations are ~3.2 GB/sample in bf16
without gradient checkpointing, so S=8 -> ~32 GB held across the step. Raising
SAMPLES_PER_BATCH beyond ~10-11 needs GRADIENT_CHECKPOINTING=True (~25%
throughput tax). The backbone is compute-bound, so a larger S mostly amortizes
fixed per-step overhead rather than adding throughput — the spare memory is
better spent on the full 32k-point pool (item 2), which is already the default.
"""

import argparse
import contextlib
import math
import os
import re
import random
import sys
import time
from datetime import timedelta
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from einops import rearrange
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Architecture / losses / dataset / vis come from the ORIGINAL script so the
# model stays identical by construction. Only long-lived, committed symbols are
# imported — never its config constants (all configuration lives in THIS file).
import train_sdf_head as base
from train_sdf_head import (
    SDFMLP,
    SDFLazyDataset,
    apply_lora_to_triposr,
    fourier_encode,
    sign_bce_loss,
    surface_weighted_mse_loss,
    surface_weighted_se,
    triposr_forward_from_cached_tokens,
)
from tsr.utils import scale_tensor

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  (self-contained; train_sdf_head.py's constants are never read)
# ═══════════════════════════════════════════════════════════════════════════════

COMMAND         = "train"

# Precomputed data lives on the ws-frb NFS export, NOT local disk: this run
# targets 1,000,000 samples (~5.1 TB) and the local volume has ~115 GB free.
# The same export appears at two paths — /mnt/ws-frb on the host, and
# /mnt/hostmnt/ws-frb inside this container, whose /mnt bind re-roots the
# host's /mnt. Resolved the same way objaverse_paths.py resolves the shared
# Objaverse mirror (and the same way train_sdf_head.py now does).
# GOTCHA: a bind mount captures the host mount tree as of container START and
# does not pick up later submounts — if this raises inside the container while
# the host has it mounted, `docker restart markiv`, not a path edit.
_WS_FRB_ROOTS = ("/mnt/ws-frb", "/mnt/hostmnt/ws-frb")


def _resolve_ws_frb_root() -> str:
    """First ws-frb path that is the REAL export, not a bare mountpoint dir.
    Presence of ``users/`` is the discriminator: an unpropagated mountpoint
    exists but is empty, so isdir(root) alone would return a path that reads
    as an empty dataset."""
    for _root in _WS_FRB_ROOTS:
        if os.path.isdir(os.path.join(_root, "users")):
            return _root
    raise FileNotFoundError(
        "ws-frb NFS export not found at " + " or ".join(_WS_FRB_ROOTS) +
        ". Inside the container /mnt is bind-mounted to /mnt/hostmnt and only "
        "exposes host mounts that existed when it STARTED; try "
        "`docker restart markiv`. Failing loudly rather than falling back to "
        "local disk on purpose - local does not have room for this dataset."
    )


DATASET_DIR     = os.path.join(_resolve_ws_frb_root(),
                               "users/markiv/sdfer/TripoSR/precomputed")
OUTPUT_DIR      = "/home/markiv/TripoSR/sdf_checkpoints"
RUN_NAME        = "v0.67_100k"   # warm start from v0.64_10k ep50 (best live head, see prof/compare_ckpts.py)
MODEL           = "stabilityai/TripoSR"

# ── Scale / schedule ─────────────────────────────────────────────────────────
EPOCHS          = 100    # header item 3 — the dominant wall-clock knob.
N_OBJECTS       = 100000
AZIMUTHS_PER_MESH = 5    # train-side view cap; 5 az x 2 elevations = 10 views/object
                         # NOTE: the cap keeps the LOWEST azimuths on disk (0,72,144,...),
                         # not a spread around the object.
SAVE_EVERY      = 2
EVAL_EVERY      = 5      # sharded eval every N epochs (last epoch always evals)
VIS_EVERY       = 25

# ── Batch geometry ───────────────────────────────────────────────────────────
SAMPLES_PER_BATCH       = 8      # distinct views per outer step per rank (x4 ranks =
                                 # 32 views per gradient step). Compute is linear in
                                 # this; see the memory note before raising past ~11.
TRAIN_POINTS_PER_SAMPLE = 0      # 0 = ALL precomputed points (32,768). Header item 2.
TEST_POINTS_PER_SAMPLE  = 0      # 0 = all points, matching the committed baseline's
                                 # eval convention so test curves stay comparable.
EVAL_POINT_CHUNK        = 131072 # points per MLP forward during eval (no_grad).

# ── Optimization ─────────────────────────────────────────────────────────────
# LR lineage: committed v0.64 used 1e-3 at a 4,096-point minibatch. This loop
# takes ONE step per pool, so the reference here is that same tuning expressed
# at a 131,072-point pool: 1e-3 x sqrt(131072/4096) = 5.66e-3 -> 5e-3 (rounded
# down for margin; sqrt not linear because AdamW normalizes by the gradient's
# second moment). AUTO_SCALE_LR then sqrt-scales from this reference to the
# ACTUAL pool (262,144 by default -> lr ~7.1e-3).
LR              = 5e-3
LR_REF_POINTS   = 131072
# LoRA optimizer semantics are unchanged (accumulate over the pool, step once);
# committed v0.64 used 1e-4 at a 131,072-pt pool with S=4. S=8 halves the
# update count per epoch, so the base is nudged to 1.5e-4 to cover the same
# ground in fewer steps (kept small — adapting a pretrained backbone is the
# delicate half of this model), then sqrt-scaled with the pool like LR.
LORA_LR         = 1.5e-4
AUTO_SCALE_LR   = True
LR_MIN          = 1e-5
WEIGHT_DECAY    = 1e-4
LORA_WEIGHT_DECAY = 1e-4
GRAD_CLIP       = 1.0
LOSS_REJECT_K   = 3.0
USE_ONECYCLE    = True
ONECYCLE_PCT_START = 0.1

EIKONAL_WEIGHT        = 1e-3
EIKONAL_FRACTION      = 0.25  # fraction of the pool the eikonal term is evaluated on, each step a
                              # fresh random subset. The eikonal needs a create_graph double-backward
                              # through the MLP, measured at 96 ms of the MLP step's 129 ms (grid_sample's
                              # scatter is only 3 ms; forward-mode AD via torch.func was 3x SLOWER).
                              # It is a 1e-3-weighted regularizer, so a random-subset estimate of the
                              # same expectation is the standard IGR/SIREN move. 1.0 = old behaviour.
                              # Forced to 1.0 while NORMAL_LOSS_WEIGHT > 0 (that loss needs per-point
                              # gradients at near-surface points across the whole pool).
SIGN_BCE_WEIGHT       = 0.1
SIGN_BCE_ALPHA        = 20.0
SIGN_BCE_EPSILON      = 0.02
SURFACE_LOSS_SIGMA    = 0.05
SDF_CLAMP             = 0.1
NORMAL_LOSS_WEIGHT    = 0.0
NORMAL_LOSS_THRESHOLD = 0.05

# ── Model (identical architecture to train_sdf_head.py) ──────────────────────
HIDDEN_DIM      = 128
HIDDEN_DIM_NO_TRIPLANE = 256
N_HIDDEN        = 6
N_FREQS         = 6
USE_TANH_OUTPUT = False
USE_TRIPLANE_FEATURES = True

LORA_RANK        = 16
LORA_ALPHA       = 16.0
LORA_BLOCK_START = 0   # adapt ALL 16 backbone blocks (reverted 2026-09-06 by request:
                       # fine-tune the whole transformer, not just the top blocks).
                       # SPEED COST, measured (S=8, bf16, compiled; backbone fwd+bwd ms):
                       #   blocks 0-16 786 | 8-16 520 | 10-16 449 | 12-16 385 | 14-16 316 | none 253
                       # Backward is ~linear in adapted depth; forward has a 253 ms floor.
                       # At 0-16 a step is ~849 ms vs ~452 ms at 12-16 — i.e. this gives up
                       # most of the 2x speedup in exchange for adapting every block.
                       # EIKONAL_FRACTION (the other optimization) is unaffected and still
                       # saves ~63 ms/step. Set 8 or 10 here for a middle ground.
LORA_BLOCK_END   = 16
LORA_TARGETS     = "all"   # which linears get adapters in each adapted block:
                           #   "all"  - q,k,v,out on both attentions + FF gate/out (10/block)
                           #   "attn" - q,k,v,out on both attentions (8/block, no FF)
                           #   "qv"   - q and v on both attentions (4/block, classic LoRA)

# ── Speed machinery ──────────────────────────────────────────────────────────
COMPILE_BACKBONE       = True   # header item 4. First step pays a ~1-2 min compile;
                                # suppress_errors falls back to eager per-graph.
GRADIENT_CHECKPOINTING = False  # only to push SAMPLES_PER_BATCH past ~11.
FUSED_ADAMW            = True
ALLOW_TF32             = True   # fp32 MLP/grid_sample matmuls on tensor cores.
                                # TF32 keeps fp32 accumulation; at 128-wide layers
                                # the rounding sits far below the supervision noise.
PREFETCH               = True
NUM_WORKERS            = 6
PREFETCH_FACTOR        = 4

# ── Splits / eval / vis (identical seeds => identical splits as old runs) ────
TEST_FRACTION      = 0.2
TEST_VIEW_FRACTION = 0.2
TEST_MAX_SAMPLES   = 3600
VIS_SEEN           = 3
VIS_UNSEEN         = 3
VIS_AZIMUTHS_PER_OBJECT = 5
VIS_RESOLUTION     = 64
USE_NERF_VIS       = True
FSCORE_TAU         = 0.01
MESH_METRIC_SAMPLES = 50000
IMAGE_SIZE         = 256
FOV                = 40.0
DIAG_EVERY         = 25
NCCL_TIMEOUT_MIN   = 30
# ── Starting from an existing checkpoint ────────────────────────────────────
# Two DIFFERENT things, deliberately separate knobs (env: SDFER_RESUME /
# SDFER_INIT_FROM). Setting both is an error.
#
#   RESUME    — continue an INTERRUPTED run of the same experiment. Restores
#               weights + both optimizers + the LR schedule and picks up at the
#               checkpoint's epoch. Use when a run died and you want it back.
#
#   INIT_FROM — WARM-START a NEW run from trained weights. Loads weights ONLY;
#               optimizers and the OneCycle schedule start fresh at epoch 0.
#               Use when the thing being trained changed — new dataset, new
#               epoch budget, new LoRA depth — and the old optimizer moments /
#               LR position would be actively wrong to carry over.
#
# Both load tolerantly and PRINT exactly what matched, what was dropped, and
# what config drifted. A checkpoint path that does not exist is a hard error,
# never a silent fresh start.
RESUME             = None   # the v0.66 ep10 head is collapsed (constant output); do not resume it
INIT_FROM          = "/home/markiv/TripoSR/sdf_checkpoints/sdf_head_v0.64_10k_epoch0050.pt"

# ═══════════════════════════════════════════════════════════════════════════════


# Stand-in for the input image when cached DINO tokens make it unnecessary.
# Module-level so no per-__getitem__ allocation; shape keeps default_collate happy.
_IMG_PLACEHOLDER = np.zeros((1, 1, 3), dtype=np.uint8)


class FastSDFDataset(SDFLazyDataset):
    """SDFLazyDataset + (a) optional load-time point subsampling and (b) no PNG
    decode when cached DINO tokens exist (the base class decodes the input
    image in every worker on every step even though the cached-token path
    never reads it).

    Subsampling is STRATIFIED over the [uniform, near-surface] blocks that
    sample_query_points concatenates in order on disk — a naive pts[:k] would
    return 100% uniform points and silently destroy the 75/25 mix. Train draws
    fresh indices per __getitem__ (all points still seen across epochs); test
    seeds per sample index so the same subset is scored every epoch.
    """

    def __init__(self, dataset_dir: str, uid_whitelist: set | None = None,
                 sample_whitelist: set | None = None, n_points_out: int = 0,
                 deterministic_subsample: bool = False):
        super().__init__(dataset_dir, uid_whitelist=uid_whitelist,
                         sample_whitelist=sample_whitelist)
        n_in = int(self.meta["n_points"])
        self.n_points_in = n_in
        self.n_points_out = n_in if n_points_out <= 0 else min(int(n_points_out), n_in)
        self.deterministic_subsample = bool(deterministic_subsample)
        self._blocks: list[tuple[int, int, int]] | None = None
        if self.n_points_out < n_in:
            nsf = float(self.meta.get("near_surface_fraction", 0.0))
            sef = float(self.meta.get("sharp_edge_fraction", 0.0))
            if sef == 0.0:
                # Mirrors sample_query_points' block sizes. With sharp-edge
                # points present the per-mesh block offsets are unrecoverable;
                # fall back to an unstratified draw (mix preserved in
                # expectation, hypergeometric).
                n_near = int(n_in * nsf)
                n_unif = n_in - n_near
                k_near = int(round(n_near * self.n_points_out / n_in))
                k_unif = self.n_points_out - k_near
                if 0 <= k_unif <= n_unif and 0 <= k_near <= n_near:
                    self._blocks = [(0, n_unif, k_unif), (n_unif, n_in, k_near)]

    def _subsample_idx(self, idx: int) -> torch.Tensor | None:
        if self.n_points_out >= self.n_points_in:
            return None
        gen = None
        if self.deterministic_subsample:
            gen = torch.Generator()
            gen.manual_seed(idx)
        if self._blocks is None:
            return torch.randperm(self.n_points_in, generator=gen)[: self.n_points_out]
        return torch.cat([
            torch.randperm(hi - lo, generator=gen)[:k] + lo
            for lo, hi, k in self._blocks
        ])

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
            nrm = torch.zeros(pts.shape[0], 3)  # legacy dataset: masked out of loss
        sub = self._subsample_idx(idx)
        if sub is not None:
            pts, sdf, nrm = pts[sub], sdf[sub], nrm[sub]
        uid = p.name.split("_az")[0]
        if self.has_cached_tokens:
            # The cached-token path never reads the image — skip the PNG decode
            # that the base class pays in every worker on every step.
            img = _IMG_PLACEHOLDER
            img_tokens = torch.load(p / "image_tokens.pt", map_location="cpu",
                                    weights_only=False)
        else:
            from PIL import Image
            img = np.array(Image.open(p / "input_image.png").convert("RGB"))
            img_tokens = torch.zeros(1)
        return pts, sdf, nrm, img, img_tokens, R, uid


def query_triplane_features_batched(
    pts_trip: torch.Tensor,      # (S, N, 3) — triplane-frame positions
    triplanes: torch.Tensor,     # (S, 3, C, H, W)
    radius: float,
) -> torch.Tensor:               # (S, N, 3*C)
    """Batched equivalent of base.query_triplane_features (feature_reduction
    "concat"): one grid_sample over all S*3 planes, zero device syncs.
    Feature ordering is (plane, channel) — bit-identical to the original's
    "N (Np Cp)" rearrange (verified: max|diff| = 0 against the per-sample path)."""
    S, N, _ = pts_trip.shape
    C, H, W = triplanes.shape[-3:]
    norm = scale_tensor(pts_trip, (-radius, radius), (-1, 1))
    idx2d = torch.stack(
        (norm[..., [0, 1]], norm[..., [0, 2]], norm[..., [1, 2]]), dim=1
    )                                                    # (S, 3, N, 2)
    out = F.grid_sample(
        triplanes.reshape(S * 3, C, H, W),
        idx2d.reshape(S * 3, 1, N, 2),
        align_corners=False,
        mode="bilinear",
    )                                                    # (S*3, C, 1, N)
    return rearrange(out, "(s p) c () n -> s n (p c)", p=3)


class CudaPrefetcher:
    """Iterate a DataLoader while copying the NEXT batch host->device on a side
    CUDA stream, so H2D transfer overlaps the current step's compute.
    Yields (pts, sdf, nrm, img_cpu, img_tokens, R, uid) with all tensors except
    img already on `device`. Falls back to plain blocking copies on CPU."""

    def __init__(self, loader: DataLoader, device: torch.device, enabled: bool = True):
        self.loader = loader
        self.device = device
        self.enabled = enabled and device.type == "cuda"
        self.stream = torch.cuda.Stream(device) if self.enabled else None

    def __len__(self) -> int:
        return len(self.loader)

    def _to_device(self, batch, non_blocking: bool):
        pts, sdf, nrm, img, tok, R, uid = batch
        d, nb = self.device, non_blocking
        return (pts.to(d, non_blocking=nb), sdf.to(d, non_blocking=nb),
                nrm.to(d, non_blocking=nb), img,
                tok.to(d, non_blocking=nb), R.to(d, non_blocking=nb), uid)

    def __iter__(self):
        if not self.enabled:
            for b in self.loader:
                yield self._to_device(b, non_blocking=False)
            return
        it = iter(self.loader)
        nxt = self._preload(it)
        main = torch.cuda.current_stream(self.device)
        while nxt is not None:
            main.wait_stream(self.stream)
            cur = nxt
            for t in cur:
                if isinstance(t, torch.Tensor) and t.is_cuda:
                    # Tell the allocator this side-stream tensor is consumed on
                    # the main stream, so its memory isn't reused too early.
                    t.record_stream(main)
            nxt = self._preload(it)   # enqueue next H2D before compute starts
            yield cur

    def _preload(self, it):
        try:
            batch = next(it)
        except StopIteration:
            return None
        with torch.cuda.stream(self.stream):
            return self._to_device(batch, non_blocking=True)


def flat_all_reduce_grads(params: list, world_size: int) -> None:
    """Average grads across ranks with ONE collective instead of len(params).
    Params with .grad None contribute zeros (mathematically correct, and keeps
    every rank issuing identical collectives — see the old file's DDP notes)."""
    grads = [(p.grad if p.grad is not None else torch.zeros_like(p)) for p in params]
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(world_size)
    offset = 0
    for p, g in zip(params, grads):
        n = g.numel()
        p.grad = flat[offset:offset + n].view_as(p)
        offset += n


def _make_adamw(params, lr: float, weight_decay: float, device: torch.device):
    """Fused AdamW when available (single multi-tensor kernel per step —
    matters most for the ~320 small LoRA tensors); plain AdamW otherwise."""
    if FUSED_ADAMW and device.type == "cuda":
        try:
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=True)
        except (RuntimeError, TypeError):
            pass
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


class StageTimers:
    """CUDA-event stage timing, sampled every `every` steps, read back with a
    single sync per epoch. Cheap enough to leave on permanently."""

    STAGES = ("backbone_fwd", "mlp_step", "backbone_bwd_opt")

    def __init__(self, device: torch.device, every: int):
        self.on = device.type == "cuda"
        self.every = max(1, every)
        self.pairs: dict[str, list] = {s: [] for s in self.STAGES}
        self._step = 0
        self.active = False

    def step_begin(self) -> None:
        self._step += 1
        self.active = self.on and (self._step % self.every == 0)

    @contextlib.contextmanager
    def stage(self, name: str):
        if not self.active:
            yield
            return
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        yield
        e1.record()
        self.pairs[name].append((e0, e1))

    def epoch_summary_ms(self) -> dict[str, float]:
        if not self.on:
            return {}
        torch.cuda.synchronize()
        out = {}
        for name, pairs in self.pairs.items():
            if pairs:
                out[name] = float(np.mean([a.elapsed_time(b) for a, b in pairs]))
            pairs.clear()
        return out


# ─── TRAIN ────────────────────────────────────────────────────────────────────

def _lora_blocks_in(keys) -> set:
    """Backbone block indices that carry LoRA adapter tensors in a state_dict."""
    out = set()
    for k in keys:
        if ".lora_A" in k or ".lora_B" in k:
            m = re.match(r"backbone\.transformer_blocks\.(\d+)\.", k)
            if m:
                out.add(int(m.group(1)))
    return out


def _load_state_report(module, state: dict, label: str, is_main: bool) -> None:
    """load_state_dict(strict=False) that first drops shape-mismatched tensors
    and then REPORTS what happened. The report is the point: a silent partial
    load is how trained weights disappear without anyone noticing."""
    own = module.state_dict()
    filtered, shape_bad = {}, []
    for k, v in state.items():
        if k not in own:
            continue
        if own[k].shape == v.shape:
            filtered[k] = v
        else:
            shape_bad.append((k, tuple(v.shape), tuple(own[k].shape)))
    res = module.load_state_dict(filtered, strict=False)
    unexpected = [k for k in state if k not in own]
    if not is_main:
        return
    print(f"  [{label}] loaded {len(filtered)}/{len(own)} tensors")
    if res.missing_keys:
        print(f"  [{label}] {len(res.missing_keys)} NOT in checkpoint -> kept current init"
              f" (e.g. {res.missing_keys[0]})")
    if unexpected:
        print(f"  [{label}] {len(unexpected)} in checkpoint but not in this model -> DISCARDED"
              f" (e.g. {unexpected[0]})")
    for k, cs, os_ in shape_bad:
        print(f"  [{label}] SHAPE MISMATCH {k}: ckpt {cs} vs model {os_} -> skipped")


def _report_config_drift(ckpt_args: dict, args, is_main: bool) -> None:
    """Warn on checkpoint-vs-now differences that change what the weights mean."""
    if not is_main or not ckpt_args:
        return
    watch = ("lora_block_start", "lora_block_end", "lora_rank", "lora_targets",
             "hidden_dim", "n_hidden", "n_freqs", "use_triplane_features",
             "samples_per_batch", "eikonal_fraction", "epochs")
    diffs = [(k, ckpt_args.get(k), getattr(args, k, None)) for k in watch
             if k in ckpt_args and ckpt_args.get(k) != getattr(args, k, None)]
    if diffs:
        print("  [config drift] checkpoint -> now:")
        for k, was, now in diffs:
            print(f"      {k}: {was!r} -> {now!r}")


def load_from_checkpoint(path: str, mode: str, sdf_mlp, triposr_model, args,
                         is_main: bool, device) -> tuple:
    """Shared loader for RESUME and INIT_FROM. Returns (ckpt, start_epoch).

    Called BEFORE the DDP wrap so weights land in the raw module."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{mode} checkpoint not found: {path}\n"
            "Refusing to silently start from scratch — fix the path or unset the knob.")
    if is_main:
        print(f"[{mode}] loading {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    _report_config_drift(ckpt.get("args", {}), args, is_main)

    _load_state_report(sdf_mlp, ckpt["model"], "sdf_mlp", is_main)

    if "lora_model" in ckpt:
        ck_blocks = _lora_blocks_in(ckpt["lora_model"].keys())
        now_blocks = _lora_blocks_in(triposr_model.state_dict().keys())
        dropped = sorted(ck_blocks - now_blocks)
        if dropped and is_main:
            n_lost = sum(1 for k in ckpt["lora_model"]
                         if (".lora_A" in k or ".lora_B" in k)
                         and (m := re.match(r"backbone\.transformer_blocks\.(\d+)\.", k))
                         and int(m.group(1)) in set(dropped))
            print(f"  [WARNING] checkpoint has TRAINED LoRA adapters for blocks {dropped} "
                  f"but this run only adapts {sorted(now_blocks)}.")
            print(f"            {n_lost} trained adapter tensors will be DISCARDED; those "
                  f"blocks revert to stock pretrained TripoSR.")
            print(f"            To keep them, set LORA_BLOCK_START={min(ck_blocks)} "
                  f"(SDFER_LORA_BLOCK_START={min(ck_blocks)}).")
        _load_state_report(triposr_model, ckpt["lora_model"], "triposr/lora", is_main)

    start_epoch = int(ckpt.get("epoch", 0)) if mode == "RESUME" else 0
    if mode == "INIT_FROM" and is_main:
        print(f"  [INIT_FROM] weights only — optimizers and LR schedule start FRESH "
              f"at epoch 0 (checkpoint was at epoch {ckpt.get('epoch', '?')}).")
    return ckpt, start_epoch


def apply_lora_selective(triposr_model, start_block: int, end_block: int,
                         rank: int, alpha: float, targets: str) -> list:
    """base.apply_lora_to_triposr with a target-set switch. "all" is byte-for-byte
    the original (same module names -> same checkpoint keys)."""
    if targets == "all":
        return base.apply_lora_to_triposr(triposr_model, start_block, end_block, rank, alpha)
    for p in triposr_model.parameters():
        p.requires_grad_(False)
    blocks = triposr_model.backbone.transformer_blocks
    for i in range(start_block, min(end_block, len(blocks))):
        for attn in (blocks[i].attn1, blocks[i].attn2):
            if attn is None:
                continue
            if targets == "attn":
                base._lora_attn(attn, rank, alpha)
            elif targets == "qv":
                attn.to_q = base.LoRALinear(attn.to_q, rank, alpha)
                if attn.to_v is not None:
                    attn.to_v = base.LoRALinear(attn.to_v, rank, alpha)
            else:
                raise ValueError(f"LORA_TARGETS={targets!r} not in (all, attn, qv)")
    for p in triposr_model.post_processor.parameters():
        p.requires_grad_(True)
    base.snapshot_stock_post_processor(triposr_model)
    return [p for p in triposr_model.parameters() if p.requires_grad]


def run_train_fast(args: argparse.Namespace) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank       = int(os.environ.get("RANK", 0))
    is_ddp  = world_size > 1
    is_main = (rank == 0)

    if is_ddp:
        dist.init_process_group(backend="nccl",
                                timeout=timedelta(minutes=NCCL_TIMEOUT_MIN))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ── Global perf switches ─────────────────────────────────────────────────
    if ALLOW_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    if is_main:
        suffix = f" — DDP across {world_size} GPUs" if is_ddp else ""
        print(f"[fast] Training on {device}{suffix}")
        print("[config] " + "  ".join(
            f"{k}={getattr(args, k)!r}" for k in (
                "run_name", "epochs", "save_every", "eval_every", "vis_every",
                "n_objects", "azimuths_per_mesh", "samples_per_batch",
                "lora_block_start", "lora_block_end", "lora_targets",
                "eikonal_fraction", "lr", "lora_lr", "compile_backbone", "resume", "init_from")),
              flush=True)

    wandb_enabled = False
    dataset_dir = Path(args.dataset_dir)
    cache_dir = str(dataset_dir / "mesh_cache")

    # ── Sample discovery + splits (IDENTICAL logic & seeds to train_sdf_head) ─
    # listdir, NOT glob("*/triplane.pt"): on the ~1M-dir NFS dataset the glob
    # stats every entry (>120 s, measured) vs 1.4 s for a readdir. The atomic
    # _tmp->final rename in precompute guarantees a final-named dir has its
    # triplane.pt, so the listing is exactly equivalent.
    all_sample_paths = sorted(Path(dataset_dir) / "samples" / _n / "triplane.pt"
        for _n in os.listdir(Path(dataset_dir) / "samples")
        if not _n.startswith("_tmp"))
    all_uids = sorted({p.parent.name.split("_az")[0] for p in all_sample_paths})
    if len(all_uids) > args.n_objects:
        all_uids = all_uids[: args.n_objects]
    uid_set = set(all_uids)
    uid_az_seen: dict[str, set] = {}
    allowed_names: set = set()
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

    # ── Datasets ─────────────────────────────────────────────────────────────
    dataset = FastSDFDataset(args.dataset_dir, sample_whitelist=train_view_names,
                             n_points_out=args.train_points_per_sample)
    meta = dataset.meta
    radius: float = float(meta["radius"])
    feature_reduction: str = meta["feature_reduction"]
    feat_dim: int = meta["feat_dim"]
    assert feature_reduction == "concat", (
        "the batched triplane sampler implements feature_reduction='concat' only "
        f"(dataset says {feature_reduction!r})")
    n_pts_per_sample: int = dataset.n_points_out
    points_per_step = args.samples_per_batch * n_pts_per_sample
    args.batch_size = points_per_step  # recorded in the checkpoint for bookkeeping

    if not dataset.has_cached_tokens and is_main:
        print("[fast] WARNING: dataset has no cached image_tokens.pt — falling back "
              "to running the DINO ViT every step (slow). Re-run precompute to fix.")

    # ── Test dataset (same cap / stratification / seeds as the old script) ───
    test_dataset = None
    test_loader = None
    _unseen_uid_names  = {s for s in all_sample_names if s.split("_az")[0] in test_uids}
    _unseen_view_names = set(test_view_names) - _unseen_uid_names
    test_sample_names = _unseen_uid_names | _unseen_view_names
    _n_full = len(test_sample_names)
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
    if test_sample_names:
        test_dataset = FastSDFDataset(args.dataset_dir, sample_whitelist=test_sample_names,
                                      n_points_out=args.test_points_per_sample,
                                      deterministic_subsample=True)
        _kw = dict(batch_size=args.samples_per_batch, num_workers=args.num_workers,
                   pin_memory=True, persistent_workers=args.num_workers > 0)
        if args.num_workers > 0:
            _kw["prefetch_factor"] = PREFETCH_FACTOR
        if is_ddp:
            # shuffle=False -> deterministic shards; drop_last=False pads so all
            # ranks run the same number of batches (standard for eval).
            test_sampler = DistributedSampler(test_dataset, num_replicas=world_size,
                                              rank=rank, shuffle=False, drop_last=False)
            test_loader = DataLoader(test_dataset, sampler=test_sampler, **_kw)
        else:
            test_loader = DataLoader(test_dataset, shuffle=False, **_kw)
        if is_main:
            _n_uid = sum(1 for s in test_sample_names if s in _unseen_uid_names)
            print(f"Test dataset: {len(test_dataset):,} samples "
                  f"({_n_uid:,} unseen-UID + {len(test_sample_names) - _n_uid:,} unseen-view)"
                  + (f"  [capped from {_n_full:,}]" if len(test_sample_names) < _n_full else "")
                  + (f", sharded {world_size}-way" if is_ddp else ""))

    # ── TripoSR + LoRA (identical setup to train_sdf_head) ───────────────────
    from tsr.system import TSR
    if is_main:
        print("Loading TripoSR for LoRA fine-tuning...")
    triposr_model = TSR.from_pretrained(args.model, config_name="config.yaml",
                                        weight_name="model.ckpt")
    lora_trainable_params = apply_lora_selective(
        triposr_model, args.lora_block_start, args.lora_block_end,
        args.lora_rank, args.lora_alpha, args.lora_targets)
    triposr_model.to(device)
    # All ranks must start from IDENTICAL LoRA weights (lora_A is randomly
    # initialized per process); broadcast rank 0's.
    if is_ddp:
        for p in lora_trainable_params:
            dist.broadcast(p.data, src=0)

    if args.gradient_checkpointing:
        triposr_model.backbone.gradient_checkpointing = True
        if is_main:
            print("[ckpt] Gradient checkpointing ON for the backbone blocks.")

    # torch.compile the backbone forward. Bound-method assignment keeps the
    # module's state_dict untouched (no _orig_mod. prefix), so checkpoints stay
    # byte-compatible with train_sdf_head.py. Skipped under gradient
    # checkpointing (the two interact badly) — see header item 4.
    if args.compile_backbone and device.type == "cuda" and not args.gradient_checkpointing:
        import torch._dynamo as _dynamo
        _dynamo.config.suppress_errors = True   # fall back to eager per-graph
        triposr_model.backbone.forward = torch.compile(triposr_model.backbone.forward)
        if is_main:
            print("[compile] TripoSR backbone wrapped with torch.compile "
                  "(first training step pays the compile cost).")

    _density_activation = triposr_model.renderer.cfg.density_activation
    _density_bias = float(triposr_model.renderer.cfg.density_bias)
    n_lora = sum(p.numel() for p in lora_trainable_params)
    if is_main:
        print(f"LoRA ready: {n_lora:,} trainable params | rank={args.lora_rank} "
              f"alpha={args.lora_alpha} blocks=[{args.lora_block_start},{args.lora_block_end})")

    triposr_decoder = None
    if is_main and args.use_nerf_vis:
        try:
            triposr_decoder = triposr_model.decoder.eval()
            for p in triposr_decoder.parameters():
                p.requires_grad_(False)
        except Exception as e:
            print(f"Warning: could not attach TripoSR decoder ({e}). NeRF vis disabled.")

    # ── Train loader ─────────────────────────────────────────────────────────
    _lkw = dict(batch_size=args.samples_per_batch, num_workers=args.num_workers,
                pin_memory=True, persistent_workers=args.num_workers > 0,
                drop_last=True)
    if args.num_workers > 0:
        _lkw["prefetch_factor"] = PREFETCH_FACTOR
    if is_ddp:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=True, drop_last=True)
        loader = DataLoader(dataset, sampler=sampler, **_lkw)
    else:
        sampler = None
        loader = DataLoader(dataset, shuffle=True, **_lkw)

    # ── Vis samples (same picks as the old script — same seed) ───────────────
    def _pick_vis_from_uids(uids: set, n: int) -> list:
        if not uids:
            return []
        chosen = random.Random(0).sample(sorted(uids), min(n, len(uids)))
        dirs = []
        for uid in chosen:
            # In-memory prefix match, not glob(): a glob is a full scan of the
            # 1M-entry samples dir on NFS, and this ran once per vis object.
            candidates = sorted(dataset_dir / "samples" / n for n in all_sample_names
                                if n.startswith(uid + "_az"))
            dirs.extend(candidates[: args.vis_azimuths_per_object])
        return dirs

    def _pick_vis_from_names(names: set, n: int) -> list:
        if not names:
            return []
        chosen = random.Random(0).sample(sorted(names), min(n, len(names)))
        return [dataset_dir / "samples" / name for name in chosen]

    vis_seen_dirs   = _pick_vis_from_uids(train_uids, args.vis_seen)
    vis_unseen_dirs = (_pick_vis_from_uids(test_uids, args.vis_unseen)
                       + _pick_vis_from_names(test_view_names, args.vis_unseen))
    vis_output_dir  = Path(args.output_dir) / "vis"

    # ── SDF MLP (identical architecture) ─────────────────────────────────────
    n_freqs: int = args.n_freqs
    pe_dim: int = (3 + 6 * n_freqs) if n_freqs > 0 else 0
    if args.use_triplane_features:
        mlp_feat_dim, mlp_hidden_dim = feat_dim, args.hidden_dim
    else:
        mlp_feat_dim, mlp_hidden_dim = 0, args.hidden_dim_no_triplane
    mlp_in_dim = mlp_feat_dim + pe_dim
    sdf_mlp = SDFMLP(in_dim=mlp_in_dim, hidden_dim=mlp_hidden_dim,
                     n_hidden=args.n_hidden, use_tanh_output=args.use_tanh_output,
                     feat_dim=mlp_feat_dim, pe_dim=pe_dim).to(device)

    # ── LR scaling + optimizers + schedule ───────────────────────────────────
    if args.auto_scale_lr:
        _scale = math.sqrt(points_per_step / LR_REF_POINTS)
        args.lr = args.lr * _scale
        args.lora_lr = args.lora_lr * _scale
        if is_main:
            print(f"[lr] pool {points_per_step:,} pts vs reference {LR_REF_POINTS:,} "
                  f"-> sqrt scale {_scale:.3f}: lr={args.lr:.2e}  lora_lr={args.lora_lr:.2e}")

    optimizer = _make_adamw(sdf_mlp.parameters(), args.lr, args.weight_decay, device)
    lora_optimizer = _make_adamw(lora_trainable_params, args.lora_lr,
                                 args.lora_weight_decay, device)

    # ONE gradient step per outer step, by construction (no inner minibatching).
    total_steps = args.epochs * len(loader)
    if args.use_onecycle:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=total_steps,
            pct_start=args.onecycle_pct_start)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr_min)

    if is_main:
        print(f"Pool/step: {args.samples_per_batch} samples x {n_pts_per_sample:,} pts "
              f"= {points_per_step:,} points, 1 gradient step each")
        print(f"Dataset: {len(dataset)} samples | {len(loader)} steps/epoch/rank | "
              f"{args.epochs} epochs -> {total_steps:,} total steps/rank | "
              f"{n_test} unseen UIDs, {len(test_view_names)} unseen views | "
              f"radius={radius:.4f} feat_dim={feat_dim} mlp_in={mlp_in_dim}")

    # ── Start from an existing checkpoint (RESUME vs INIT_FROM) ─────────────
    start_epoch = 0
    if args.resume and args.init_from:
        raise ValueError(
            "Set RESUME or INIT_FROM, not both: RESUME continues an interrupted run "
            "(restores optimizers + LR position), INIT_FROM warm-starts a new one "
            "(weights only, fresh schedule).")

    if args.init_from:
        load_from_checkpoint(args.init_from, "INIT_FROM", sdf_mlp, triposr_model,
                             args, is_main, device)

    elif args.resume:
        ckpt, start_epoch = load_from_checkpoint(
            args.resume, "RESUME", sdf_mlp, triposr_model, args, is_main, device)
        # Optimizer moments are only meaningful for the SAME parameter set. The
        # LoRA param list changes with block depth/targets, so guard that load.
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except ValueError as e:
            if is_main:
                print(f"  [RESUME] sdf_mlp optimizer state incompatible ({e}); starting fresh.")
        if "lora_optimizer" in ckpt:
            try:
                lora_optimizer.load_state_dict(ckpt["lora_optimizer"])
            except ValueError as e:
                if is_main:
                    print(f"  [RESUME] LoRA optimizer state incompatible ({e}); starting fresh. "
                          "Expected when lora_block_start/targets changed.")
        if "scheduler" in ckpt:
            _old_total = ckpt["scheduler"].get("total_steps")
            if args.use_onecycle and _old_total is not None and _old_total != total_steps:
                consumed = min(start_epoch * len(loader), total_steps)
                if is_main:
                    print(f"  [RESUME] step budget changed ({_old_total} -> {total_steps}); "
                          f"rebuilding OneCycleLR at step {consumed}.")
                for g in optimizer.param_groups:
                    g.setdefault("initial_lr", args.lr / 25.0)
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=args.lr, total_steps=total_steps,
                    pct_start=args.onecycle_pct_start, last_epoch=consumed - 1)
            else:
                try:
                    scheduler.load_state_dict(ckpt["scheduler"])
                except Exception as e:
                    if is_main:
                        print(f"  [RESUME] scheduler state not restorable ({e}); fresh schedule.")
        if is_main:
            print(f"[RESUME] continuing at epoch {start_epoch}")

    if is_ddp:
        sdf_mlp = DDP(sdf_mlp, device_ids=[local_rank])
    _mlp_module = sdf_mlp.module if is_ddp else sdf_mlp

    # ── wandb (rank 0; scalars only — no watch/param-table overhead) ─────────
    if is_main:
        try:
            wandb.init(
                project="simple-sdf",
                name=args.run_name,
                config={
                    "args": {k: base._wandb_jsonable(v) for k, v in vars(args).items()},
                    "derived": {
                        "radius": radius, "feat_dim": feat_dim, "mlp_in_dim": mlp_in_dim,
                        "points_per_step": points_per_step,
                        "steps_per_epoch": len(loader), "total_steps": total_steps,
                        "world_size": world_size, "start_epoch": start_epoch,
                        "script": "train_sdf_head_fast",
                    },
                },
                settings=wandb.Settings(_disable_stats=True, console="off"),
            )
            wandb_enabled = True
        except Exception as e:
            print(f"Warning: wandb init failed: {e}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefetch = CudaPrefetcher(loader, device, enabled=PREFETCH)
    test_prefetch = (CudaPrefetcher(test_loader, device, enabled=PREFETCH)
                     if test_loader is not None else None)
    # Benchmark mode: SDFER_BENCH_STEPS=N -> run N steps with stage timers
    # sampled EVERY step, print the breakdown + median wall ms/step, and return.
    # Steps 0-4 (torch.compile + allocator warmup) are excluded from both.
    bench_steps = int(os.environ.get("SDFER_BENCH_STEPS", "0"))
    timers = StageTimers(device, every=(1 if bench_steps else args.diag_every))
    _bench_wall: list = []
    _bench_n = 0
    _bench_t0 = time.perf_counter()

    pbar = tqdm(total=total_steps, initial=start_epoch * len(loader),
                desc=f"Epoch 1/{args.epochs}", dynamic_ncols=True,
                unit="step") if is_main else None

    for epoch in range(start_epoch, args.epochs):
        sdf_mlp.train()
        triposr_model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_t0 = time.perf_counter()
        data_wait = 0.0
        diag_steps = 0
        # GPU-side accumulators, ONE host sync per epoch (each .item() in the
        # loop would drain the CUDA queue and stall the CPU's run-ahead).
        # slots: loss, sdf, mse, eik, bce, nrm, sign_acc, preclip, reject, gradmean
        _acc  = torch.zeros(10, device=device)
        _mins = torch.full((3,), float("inf"),  device=device)
        _maxs = torch.full((3,), float("-inf"), device=device)
        if pbar is not None:
            pbar.set_description(f"Epoch {epoch + 1}/{args.epochs}")

        _fetch_t0 = time.perf_counter()
        for pts, sdf_gt_s, nrm_s, imgs_cpu, img_tokens, R_gpu, _uids in prefetch:
            data_wait += time.perf_counter() - _fetch_t0
            # pts (S,N,3) | sdf_gt_s (S,N) | nrm_s (S,N,3) | img_tokens (S,nt,C)
            # R_gpu (S,3,3) — all already on device via the prefetcher.
            timers.step_begin()

            # ── 1. Backbone forward (bf16), batched over all S views ─────────
            with timers.stage("backbone_fwd"):
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=(device.type == "cuda")):
                    if dataset.has_cached_tokens:
                        scene_codes = triposr_forward_from_cached_tokens(
                            triposr_model, img_tokens)
                    else:
                        img_batch = [imgs_cpu[s].numpy() for s in range(imgs_cpu.shape[0])]
                        scene_codes = triposr_model(img_batch, device=device)
                trip = scene_codes.float()               # (S, 3, C, H, W)

            # Graph split (identical math to the old script): the MLP/eikonal
            # backward only reaches this detached leaf; the expensive backbone
            # graph gets ONE ordinary backward at step end. This is also what
            # makes compiling the backbone legal despite the double backward.
            leaf = trip.detach().requires_grad_(True)

            # ── 2. Features + MLP + losses + backward (one step per pool) ────
            with timers.stage("mlp_step"):
                pts_trip = torch.einsum("snj,sij->sni", pts, R_gpu)  # p @ R.T
                feats = query_triplane_features_batched(pts_trip, leaf, radius)
                B = pts.shape[0] * pts.shape[1]                      # S*N points
                query_pts = pts_trip.reshape(B, 3).detach().requires_grad_(True)
                flat_feats = feats.reshape(B, -1)
                if n_freqs > 0:
                    pe = fourier_encode(query_pts, n_freqs)
                    model_in = (torch.cat([flat_feats, pe], dim=-1)
                                if args.use_triplane_features else pe)
                else:
                    model_in = flat_feats if args.use_triplane_features else query_pts

                sdf_pred = sdf_mlp(model_in)
                sdf_gt = sdf_gt_s.reshape(B)

                # TSDF clamp + surface weighting + outlier rejection: identical
                # formulas to train_sdf_head.py.
                if args.sdf_clamp > 0:
                    _c = float(args.sdf_clamp)
                    per_point = surface_weighted_se(
                        sdf_pred.clamp(-_c, _c), sdf_gt.clamp(-_c, _c),
                        sigma=args.surface_loss_sigma, weight_target=sdf_gt)
                else:
                    per_point = surface_weighted_se(sdf_pred, sdf_gt,
                                                    sigma=args.surface_loss_sigma)
                reject_frac = sdf_pred.new_zeros(())
                if args.loss_reject_k > 0 and per_point.numel() > 1:
                    with torch.no_grad():
                        thr = per_point.mean() + args.loss_reject_k * per_point.std()
                        keep = per_point <= thr
                    if keep.any():
                        reject_frac = 1.0 - keep.float().mean()
                        sdf_loss = per_point[keep].mean()
                    else:
                        sdf_loss = per_point.mean()
                else:
                    sdf_loss = per_point.mean()

                bce_loss = sign_bce_loss(sdf_pred, sdf_gt, alpha=args.sign_bce_alpha,
                                         epsilon=args.sign_bce_epsilon)

                # Eikonal on a random subset of the pool (see EIKONAL_FRACTION). The
                # subset gets its own small forward so the double-backward only
                # spans K points; feats are gathered WITHOUT detach so the (tiny)
                # eikonal->triplane gradient path is preserved exactly as before.
                _eik_K = int(B * args.eikonal_fraction)
                if 0 < _eik_K < B and args.normal_loss_weight <= 0:
                    _e_idx = torch.randperm(B, device=device)[:_eik_K]
                    _q_e = query_pts[_e_idx].detach().requires_grad_(True)
                    _pe_e = fourier_encode(_q_e, n_freqs)
                    _in_e = (torch.cat([flat_feats[_e_idx], _pe_e], dim=-1)
                             if args.use_triplane_features else _pe_e)
                    _pred_e = sdf_mlp(_in_e)
                    gradients = torch.autograd.grad(
                        outputs=_pred_e, inputs=_q_e,
                        grad_outputs=torch.ones_like(_pred_e),
                        create_graph=True)[0]
                else:
                    gradients = torch.autograd.grad(
                        outputs=sdf_pred, inputs=query_pts,
                        grad_outputs=torch.ones_like(sdf_pred),
                        create_graph=True, retain_graph=True)[0]
                grad_norm = gradients.norm(dim=-1)
                eikonal_loss = ((grad_norm - 1.0) ** 2).mean()

                if args.normal_loss_weight > 0:
                    nrm_trip = torch.einsum(
                        "sij,snj->sni", R_gpu, nrm_s).reshape(B, 3)
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

                optimizer.zero_grad(set_to_none=True)
                lora_optimizer.zero_grad(set_to_none=True)
                loss.backward()   # -> MLP grads (DDP all-reduced) + leaf.grad
                preclip_norm = torch.nn.utils.clip_grad_norm_(
                    sdf_mlp.parameters(), args.grad_clip)
                optimizer.step()
                if args.use_onecycle:
                    scheduler.step()

            # ── 3. Backbone backward + LoRA step ─────────────────────────────
            with timers.stage("backbone_bwd_opt"):
                if leaf.grad is not None:
                    torch.autograd.backward(trip, leaf.grad)
                if is_ddp:
                    flat_all_reduce_grads(lora_trainable_params, world_size)
                torch.nn.utils.clip_grad_norm_(lora_trainable_params, args.grad_clip)
                lora_optimizer.step()

            if bench_steps:
                torch.cuda.synchronize()
                _now = time.perf_counter()
                if _bench_n == 5:
                    timers.epoch_summary_ms()          # discard warmup samples
                if _bench_n >= 5:
                    _bench_wall.append(_now - _bench_t0)
                _bench_t0 = _now
                _bench_n += 1
                if _bench_n >= bench_steps:
                    ms = timers.epoch_summary_ms()
                    if is_main:
                        import statistics
                        # bench_steps <= 5 leaves no post-warmup samples; report
                        # the raw steps rather than crash on an empty median.
                        w = (statistics.median(_bench_wall) * 1000 if _bench_wall
                             else float("nan"))
                        print(f"\n[bench] {len(_bench_wall)} timed steps | "
                              f"median wall {w:.0f} ms/step | "
                              + " | ".join(f"{k} {v:.0f}" for k, v in ms.items())
                              + f"\n[bench] S={pts.shape[0]} N={pts.shape[1]} "
                              f"pts/step={pts.shape[0]*pts.shape[1]:,}", flush=True)
                    return

            # ── Diagnostics (on-device; one sync per epoch) ──────────────────
            with torch.no_grad():
                _acc += torch.stack([
                    loss.detach(), sdf_loss.detach(),
                    F.mse_loss(sdf_pred.detach(), sdf_gt),
                    eikonal_loss.detach(), bce_loss.detach(), normal_loss.detach(),
                    (torch.sign(sdf_pred) == torch.sign(sdf_gt)).float().mean(),
                    preclip_norm.detach(), reject_frac.detach(),
                    grad_norm.mean().detach(),
                ])
                _mins = torch.minimum(_mins, torch.stack(
                    [grad_norm.min().detach(), sdf_pred.min().detach(), sdf_gt.min()]))
                _maxs = torch.maximum(_maxs, torch.stack(
                    [grad_norm.max().detach(), sdf_pred.max().detach(), sdf_gt.max()]))
            diag_steps += 1

            if pbar is not None:
                pbar.update(1)
                if diag_steps % max(1, args.diag_every) == 0:
                    pbar.set_postfix(loss=f"{loss.item():.5f}",
                                     sdf=f"{sdf_loss.item():.5f}",
                                     eik=f"{eikonal_loss.item():.5f}")
            if is_main and wandb_enabled and diag_steps % max(1, args.diag_every) == 0:
                try:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/sdf_loss": sdf_loss.item(),
                        "train/eikonal_loss": eikonal_loss.item(),
                        "train/sign_bce_loss": bce_loss.item(),
                        "train/normal_loss": normal_loss.item(),
                        "train/grad_norm_preclip": preclip_norm.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                    })
                except Exception:
                    pass

            del trip, leaf, scene_codes
            _fetch_t0 = time.perf_counter()

        # ── End of epoch: single host sync for all statistics ────────────────
        (epoch_loss, epoch_sdf, epoch_mse, epoch_eik, epoch_bce, epoch_nrm,
         sign_acc_sum, preclip_sum, reject_sum, grad_mean_sum) = _acc.tolist()
        gmin, pmin, tmin = _mins.tolist()
        gmax, pmax, tmax = _maxs.tolist()

        if is_ddp:
            t = torch.tensor([epoch_loss, epoch_sdf, epoch_mse,
                              epoch_eik, epoch_bce, epoch_nrm], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            (epoch_loss, epoch_sdf, epoch_mse,
             epoch_eik, epoch_bce, epoch_nrm) = (t / world_size).tolist()

        epoch_wall = time.perf_counter() - epoch_t0
        if is_main:
            _ns = max(diag_steps, 1)
            stage_ms = timers.epoch_summary_ms()
            steps_s = diag_steps / epoch_wall if epoch_wall > 0 else 0.0
            eta_h = (args.epochs - epoch - 1) * epoch_wall / 3600.0
            _stages = "  ".join(f"{k}={v:.0f}ms" for k, v in stage_ms.items())
            tqdm.write(
                f"[epoch {epoch + 1}] {epoch_wall:.1f}s  "
                f"{steps_s:.2f} steps/s  {steps_s * points_per_step / 1e6:.2f} Mpts/s  "
                f"data_wait={data_wait:.1f}s  {_stages}  ETA {eta_h:.1f}h")
            if wandb_enabled:
                try:
                    wandb.log({
                        "train/epoch_loss": epoch_loss / _ns,
                        "train/epoch_sdf_loss": epoch_sdf / _ns,
                        "train/epoch_mse": epoch_mse / _ns,
                        "train/epoch_eikonal_loss": epoch_eik / _ns,
                        "train/epoch_sign_bce_loss": epoch_bce / _ns,
                        "train/epoch_normal_loss": epoch_nrm / _ns,
                        "diag/sign_accuracy": sign_acc_sum / _ns,
                        "diag/grad_norm_preclip_mean": preclip_sum / _ns,
                        "diag/reject_frac_mean": reject_sum / _ns,
                        "diag/grad_norm_mean": grad_mean_sum / _ns,
                        "diag/grad_norm_min": gmin, "diag/grad_norm_max": gmax,
                        "diag/pred_min": pmin, "diag/pred_max": pmax,
                        "diag/target_min": tmin, "diag/target_max": tmax,
                        "perf/epoch_seconds": epoch_wall,
                        "perf/steps_per_sec": steps_s,
                        "perf/points_per_sec": steps_s * points_per_step,
                        "perf/data_wait_seconds": data_wait,
                        **{f"perf/{k}_ms": v for k, v in stage_ms.items()},
                        "train/epoch": epoch + 1,
                    })
                except Exception:
                    pass

        # ── Sharded eval. The all_reduce stays OUTSIDE the test_loader guard:
        # it is a collective, and every rank must issue it in the same order
        # always — a rank with no shard (or a skipped epoch) reduces zeros. ───
        _run_eval = ((epoch + 1) % max(1, args.eval_every) == 0
                     or (epoch + 1) == args.epochs)
        _test_acc = torch.zeros(5, device=device)  # weighted, mse, mse_clamped, sign, steps
        if test_loader is not None and _run_eval:
            _mlp_module.eval()
            triposr_model.eval()
            tw = tm = tmc = tsa = 0.0
            tsteps = 0
            with torch.no_grad():
                for t_pts, t_sdf, _t_nrm, t_imgs, t_tok, t_R, _ in test_prefetch:
                    tS, tN = t_sdf.shape
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                        enabled=(device.type == "cuda")):
                        if test_dataset.has_cached_tokens:
                            t_codes = triposr_forward_from_cached_tokens(
                                triposr_model, t_tok)
                        else:
                            t_codes = triposr_model(
                                [t_imgs[s].numpy() for s in range(tS)], device=device)
                    t_trip = t_codes.float()
                    t_ptrip = torch.einsum("snj,sij->sni", t_pts, t_R)
                    t_feats = query_triplane_features_batched(t_ptrip, t_trip, radius)
                    fp = t_ptrip.reshape(tS * tN, 3)
                    ff = t_feats.reshape(tS * tN, -1)
                    fs = t_sdf.reshape(tS * tN)
                    for c0 in range(0, tS * tN, EVAL_POINT_CHUNK):
                        c1 = min(c0 + EVAL_POINT_CHUNK, tS * tN)
                        if n_freqs > 0:
                            pe = fourier_encode(fp[c0:c1], n_freqs)
                            mi = (torch.cat([ff[c0:c1], pe], dim=-1)
                                  if args.use_triplane_features else pe)
                        else:
                            mi = ff[c0:c1] if args.use_triplane_features else fp[c0:c1]
                        pred = _mlp_module(mi)
                        ms = fs[c0:c1]
                        tw += surface_weighted_mse_loss(
                            pred, ms, sigma=args.surface_loss_sigma).item()
                        tm += F.mse_loss(pred, ms).item()
                        if args.sdf_clamp > 0:
                            _tc = float(args.sdf_clamp)
                            tmc += F.mse_loss(pred.clamp(-_tc, _tc),
                                              ms.clamp(-_tc, _tc)).item()
                        tsa += float((torch.sign(pred) == torch.sign(ms)).float().mean())
                        tsteps += 1
                    del t_trip, t_codes
            _mlp_module.train()
            triposr_model.train()
            _test_acc = torch.tensor([tw, tm, tmc, tsa, float(tsteps)], device=device)
        if is_ddp:
            dist.all_reduce(_test_acc, op=dist.ReduceOp.SUM)
        _tw, _tm, _tmc, _tsa, _tsteps = _test_acc.tolist()
        if is_main and wandb_enabled and _tsteps > 0:
            try:
                _log = {
                    "test/epoch_sdf_loss": _tw / _tsteps,
                    "test/epoch_mse": _tm / _tsteps,
                    "test/sign_accuracy": _tsa / _tsteps,
                    "train/epoch": epoch + 1,
                }
                if args.sdf_clamp > 0:
                    _log["test/epoch_mse_clamped"] = _tmc / _tsteps
                wandb.log(_log)
            except Exception:
                pass

        if not args.use_onecycle:
            scheduler.step()

        # ── Checkpoint (identical dict layout to train_sdf_head.py) ──────────
        is_last = (epoch + 1) == args.epochs
        if is_main and (is_last or (epoch + 1) % args.save_every == 0):
            ckpt_path = output_dir / f"sdf_head_{args.run_name}_epoch{epoch + 1:04d}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model": _mlp_module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "meta": meta,
                "args": vars(args),
                "lora_model": triposr_model.state_dict(),
                "lora_optimizer": lora_optimizer.state_dict(),
            }, ckpt_path)

        # Vis is diagnostics: never let a vis crash kill a multi-day run.
        if is_main and args.vis_every > 0 and (epoch + 1) % args.vis_every == 0:
            try:
                base.visualize_reconstructions(
                    sdf_mlp=_mlp_module,
                    seen_dirs=vis_seen_dirs, unseen_dirs=vis_unseen_dirs,
                    radius=radius, feature_reduction=feature_reduction,
                    cache_dir=cache_dir, epoch=epoch + 1,
                    output_dir=vis_output_dir, wandb_enabled=wandb_enabled,
                    device=device, resolution=args.vis_resolution, n_freqs=n_freqs,
                    fov=args.fov, image_size=args.image_size,
                    triposr_decoder=triposr_decoder,
                    density_activation=_density_activation, density_bias=_density_bias,
                    use_triplane_features=args.use_triplane_features,
                    triposr_model=triposr_model,
                    fscore_tau=args.fscore_tau,
                    mesh_metric_samples=args.mesh_metric_samples,
                )
            except Exception as e:
                tqdm.write(f"[vis] visualization failed (continuing training): {e}")
            triposr_model.train()

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

def build_train_args() -> argparse.Namespace:
    return argparse.Namespace(
        dataset_dir    = DATASET_DIR,
        model          = MODEL,
        n_objects      = N_OBJECTS,
        azimuths_per_mesh = AZIMUTHS_PER_MESH,
        image_size     = IMAGE_SIZE,
        fov            = FOV,
        output_dir     = OUTPUT_DIR,
        run_name       = RUN_NAME,
        epochs         = EPOCHS,
        save_every     = SAVE_EVERY,
        eval_every     = EVAL_EVERY,
        vis_every      = VIS_EVERY,
        samples_per_batch       = SAMPLES_PER_BATCH,
        train_points_per_sample = TRAIN_POINTS_PER_SAMPLE,
        test_points_per_sample  = TEST_POINTS_PER_SAMPLE,
        batch_size     = 0,   # derived (= samples_per_batch x points/sample)
        hidden_dim     = HIDDEN_DIM,
        hidden_dim_no_triplane = HIDDEN_DIM_NO_TRIPLANE,
        n_hidden       = N_HIDDEN,
        n_freqs        = N_FREQS,
        lr             = LR,
        lora_lr        = LORA_LR,
        auto_scale_lr  = AUTO_SCALE_LR,
        lr_min         = LR_MIN,
        grad_clip      = GRAD_CLIP,
        loss_reject_k  = LOSS_REJECT_K,
        use_onecycle   = USE_ONECYCLE,
        onecycle_pct_start = ONECYCLE_PCT_START,
        eikonal_weight        = EIKONAL_WEIGHT,
        eikonal_fraction = float(os.environ.get("SDFER_EIKONAL_FRACTION", EIKONAL_FRACTION)),
        sign_bce_weight       = SIGN_BCE_WEIGHT,
        sign_bce_alpha        = SIGN_BCE_ALPHA,
        sign_bce_epsilon      = SIGN_BCE_EPSILON,
        surface_loss_sigma    = SURFACE_LOSS_SIGMA,
        sdf_clamp             = SDF_CLAMP,
        normal_loss_weight    = NORMAL_LOSS_WEIGHT,
        normal_loss_threshold = NORMAL_LOSS_THRESHOLD,
        weight_decay   = WEIGHT_DECAY,
        lora_weight_decay = LORA_WEIGHT_DECAY,
        use_tanh_output = USE_TANH_OUTPUT,
        use_triplane_features = USE_TRIPLANE_FEATURES,
        lora_rank        = LORA_RANK,
        lora_alpha       = LORA_ALPHA,
        lora_block_start = LORA_BLOCK_START,
        lora_block_end   = LORA_BLOCK_END,
        lora_targets = os.environ.get("SDFER_LORA_TARGETS", LORA_TARGETS),
        test_fraction  = TEST_FRACTION,
        test_view_fraction = TEST_VIEW_FRACTION,
        test_max_samples   = TEST_MAX_SAMPLES,
        vis_seen       = VIS_SEEN,
        vis_unseen     = VIS_UNSEEN,
        vis_azimuths_per_object = VIS_AZIMUTHS_PER_OBJECT,
        vis_resolution = VIS_RESOLUTION,
        use_nerf_vis   = USE_NERF_VIS,
        fscore_tau     = FSCORE_TAU,
        mesh_metric_samples = MESH_METRIC_SAMPLES,
        diag_every     = DIAG_EVERY,
        num_workers    = NUM_WORKERS,
        gradient_checkpointing = GRADIENT_CHECKPOINTING,
        compile_backbone       = COMPILE_BACKBONE,
        resume         = os.environ.get("SDFER_RESUME", RESUME) or None,
        init_from      = os.environ.get("SDFER_INIT_FROM", INIT_FROM) or None,
    )


# ─── FAST PRECOMPUTE ──────────────────────────────────────────────────────────
# Ground-up rewrite of base.run_precompute, built from a 12-object stage
# profile of the original (2026-09-04, writing to ws-frb NFS; 16.25 s/object):
#
#   gc_collect          4.72 s/obj  29%  <- 21 manual collects/object at 224 ms
#                                          each on a torch+trimesh-sized heap
#   compute_sdf         4.43 s/obj  27%  <- real CPU work
#   tsr_forward (x10)   2.85 s/obj  18%  <- fp32; bf16 measured 4.0x faster and
#                                          the triplane is stored .half() anyway
#   render_view (x10)   2.50 s/obj  15%  <- a NEW OffscreenRenderer (EGL context)
#                                          per view + from_trimesh per view
#   repair_watertight   1.35 s/obj   8%
#   download_mesh       0.80 s/obj   7%  <- objaverse re-gunzips the 800k-entry
#                                          object-paths index EVERY call (0.64 s)
#   everything else     <0.5 s/obj
#
# Also measured: batch-10 TSR inference is NOT faster than 10x batch-1 (3072
# triplane tokens saturate the GPU at batch 1), so the win is OVERLAP, not
# batching; and the _done check costs 70 NFS stats = 460 ms per already-done
# object, i.e. ~an hour of pure stat traffic per rank on every restart.
#
# Design: a spawn ProcessPool per rank does all CPU work (mesh copy/load/repair,
# point sampling, SDF, all 10 renders through ONE long-lived EGL renderer per
# worker) and returns ~3 MB of arrays; the main process keeps the GPU busy with
# bf16 TSR + fp32 DINO forwards and hands file writes to a small thread pool.
# DINO tokens stay fp32 ON PURPOSE: they are the cached TRAINING INPUT and must
# stay distribution-identical to the 331k samples already on disk; the triplane
# tolerates bf16 because training recomputes triplanes from the tokens anyway.
# Manual gc is gone entirely (workers churn, their auto-GC copes). Resume-skip
# is ONE readdir into a set: the _tmp -> final rename is atomic, so a dir being
# listed under its final name proves it is complete.

_PREP_STATE: dict = {}   # per-WORKER-process globals (renderer)


def _prep_worker_init(egl_device: int) -> None:
    """Runs in each spawned prep worker before any task. EGL_DEVICE_ID must be
    set before the first pyrender import (which happens lazily in the first
    task), so renders spread across the rank's own GPU.

    The address-space rlimit is the OOM firewall. Measured 2026-09-05: one
    pathological mesh drove a single worker to 18.6 GB RSS and the fleet to
    ~117/125 GB; the night before that, 32 workers OOM-killed the whole box
    INCLUDING the user's tmux server. With the cap, a monster mesh raises
    MemoryError inside that one worker, _prep_object catches it, and the uid
    is recorded in skipped_uids.txt as a deterministic failure — exactly what
    should happen to a mesh that needs >16 GB to voxel-repair."""
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["EGL_DEVICE_ID"] = str(egl_device)
    import resource
    # RLIMIT_DATA, not RLIMIT_AS: measured VmSize of an IDLE worker here is
    # ~15 GB (torch's library mappings + 64 kdtree-thread stacks + glibc
    # arenas) against ~2 GB RSS, so an address-space cap fires on healthy
    # meshes — it briefly poisoned the skip cache with MemoryError entries.
    # RLIMIT_DATA covers the actual heap (brk + private mmaps, kernel >= 4.7)
    # and idles at ~1-2 GB, so 10 GB of headroom is real headroom.
    _cap = 10 * 1024**3
    try:
        resource.setrlimit(resource.RLIMIT_DATA, (_cap, _cap))
    except (ValueError, OSError):
        pass


def _render_all_views(mesh, view_pairs, image_size: int, fov: float):
    """All views of one object through ONE renderer and ONE uploaded mesh.

    Replicates base.render_mesh_to_image exactly (2x render + Lanczos down,
    same lights/camera/framing) EXCEPT that the OffscreenRenderer (an EGL
    context, ~250 ms to create) lives for the whole worker process and
    pyrender.Mesh.from_trimesh (vertex-normal smoothing) runs once per object
    instead of once per view. Returns (uint8 images, extrinsics json strings).
    """
    import tempfile
    import pyrender
    from PIL import Image as _Image

    render_size = int(image_size) * 2
    r = _PREP_STATE.get("renderer")
    if r is None:
        r = pyrender.OffscreenRenderer(render_size, render_size)
        _PREP_STATE["renderer"] = r
    try:
        pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    except Exception:
        import trimesh as _tr
        pr_mesh = pyrender.Mesh.from_trimesh(
            _tr.Trimesh(vertices=mesh.vertices, faces=mesh.faces))

    fov_rad = np.radians(fov)
    distance = 0.7 / np.tan(fov_rad / 2.0)
    images, extrs = [], []
    for az, el in view_pairs:
        scene = pyrender.Scene(bg_color=[0.5, 0.5, 0.5, 1.0],
                               ambient_light=[0.25, 0.25, 0.25])
        scene.add(pr_mesh)
        T_cam = base._camera_pose(float(az), float(el), distance)
        scene.add(pyrender.PerspectiveCamera(yfov=fov_rad, aspectRatio=1.0), pose=T_cam)
        scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0),
                  pose=base._camera_pose(float(az) + 20, float(el) + 20, 1.0))
        scene.add(pyrender.DirectionalLight(color=[0.7, 0.8, 1.0], intensity=1.5),
                  pose=base._camera_pose(float(az) + 180, float(el) - 10, 1.0))
        color, _ = r.render(scene)
        img = _Image.fromarray(color).resize((int(image_size),) * 2, _Image.LANCZOS)
        images.append(np.asarray(img, dtype=np.uint8))
        # Reuse the committed writer for byte-identical extrinsics json.
        with tempfile.NamedTemporaryFile("w", suffix=".json", dir="/dev/shm",
                                         delete=False) as tf:
            tmp_path = tf.name
        base._write_camera_extrinsics_json(
            tmp_path, T_cam, azimuth_deg=float(az), elevation_deg=float(el),
            distance=float(distance), fov_deg=float(fov))
        with open(tmp_path) as fh:
            extrs.append(fh.read())
        os.unlink(tmp_path)
        scene.clear()
    return images, extrs


def _prep_object(task: dict) -> dict:
    """Everything CPU for one object; runs in a prep worker. Never touches CUDA."""
    import shutil as _shutil
    uid = task["uid"]
    try:
        cached_dir = os.path.join(task["cache_dir"], uid)
        dst = os.path.join(cached_dir, os.path.basename(task["src"]))
        if not os.path.exists(dst):
            os.makedirs(cached_dir, exist_ok=True)
            _shutil.copy2(task["src"], dst)

        raw = base._load_trimesh(dst)
        mesh, _, _ = base._normalize_mesh_copy(raw, task["radius"])
        del raw
        if task["max_triangles"] > 0 and len(mesh.faces) > task["max_triangles"]:
            return {"status": "skip", "uid": uid,
                    "reason": f"{len(mesh.faces):,} faces > limit"}
        repaired = False
        if not mesh.is_watertight:
            if not task["repair_meshes"]:
                return {"status": "skip", "uid": uid, "reason": "not watertight"}
            rep, _method = base.repair_mesh_watertight(
                mesh, voxel_res=task["repair_voxel_res"],
                voxel_method=task["repair_voxel_method"])
            if rep is None:
                return {"status": "skip", "uid": uid, "reason": "repair failed"}
            mesh, _, _ = base._normalize_mesh_copy(rep, task["radius"])
            del rep
            repaired = True
            try:
                mesh.export(os.path.join(cached_dir, "repaired.obj"))
            except Exception:
                pass

        pts = base.sample_query_points(
            mesh, task["n_points"], task["radius"],
            near_surface_fraction=task["near_surface_fraction"],
            sharp_edge_fraction=task["sharp_edge_fraction"],
            sharp_edge_angle_deg=task["sharp_edge_angle_deg"])
        sdf, nrm = base.compute_sdf(mesh, pts, return_normals=True)
        images, extrs = _render_all_views(
            mesh, task["view_pairs"], task["image_size"], task["fov"])
        return {"status": "ok", "uid": uid, "repaired": repaired,
                "pts": pts.astype(np.float32), "sdf": sdf.astype(np.float32),
                "nrm": nrm.astype(np.float32), "images": images, "extrs": extrs}
    except Exception as e:  # one bad mesh must never take down the run
        return {"status": "skip", "uid": uid, "reason": repr(e)[:200]}


def _write_view(samples_dir: Path, sample_id: str, image_np, extr_json: str,
                triplane, image_tokens, pts_t, sdf_t, nrm_t) -> None:
    """Write one view's 7 files into _tmp then atomically rename (same protocol
    as the original, so resume-by-listdir stays sound). Runs on writer threads;
    pure IO, releases the GIL."""
    from PIL import Image as _Image
    tmp_dir = samples_dir / f"_tmp_{sample_id}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    _Image.fromarray(image_np).save(tmp_dir / "input_image.png")
    with open(tmp_dir / "camera_extrinsics.json", "w") as fh:
        fh.write(extr_json)
    torch.save(triplane, tmp_dir / "triplane.pt")
    torch.save(pts_t, tmp_dir / "query_pts.pt")
    torch.save(sdf_t, tmp_dir / "sdf_gt.pt")
    torch.save(image_tokens, tmp_dir / "image_tokens.pt")
    torch.save(nrm_t, tmp_dir / "normal_gt.pt")
    os.replace(tmp_dir, samples_dir / sample_id)


def run_precompute_fast(args: argparse.Namespace) -> None:
    import concurrent.futures as cf
    import itertools
    import json
    import multiprocessing as mp
    import shutil  # noqa: F401  (parity with base import block)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank       = int(os.environ.get("RANK", 0))
    is_main    = (rank == 0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    dataset_dir = Path(args.dataset_dir)
    samples_dir = dataset_dir / "samples"
    cache_dir   = str(dataset_dir / "mesh_cache")
    samples_dir.mkdir(parents=True, exist_ok=True)

    from tsr.system import TSR
    if is_main:
        print(f"[fast-precompute] {device} x{world_size} | prep_workers={args.prep_workers}"
              f" prefetch={args.prefetch}")
        print("Loading TripoSR...")
    model = TSR.from_pretrained(args.model, config_name="config.yaml",
                                weight_name="model.ckpt")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    radius = float(model.renderer.cfg.radius)

    if is_main:
        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump({
                "radius": radius,
                "feature_reduction": model.renderer.cfg.feature_reduction,
                "feat_dim": model.decoder.cfg.in_channels,
                "n_points": args.n_points,
                "near_surface_fraction": args.near_surface_fraction,
                "sharp_edge_fraction": args.sharp_edge_fraction,
                "sharp_edge_angle_deg": args.sharp_edge_angle_deg,
                "elevations": args.elevations,
                "data_source": args.data_source,
            }, f, indent=2)

    # Objaverse index ONCE (the library re-gunzips it per load_objects call).
    import objaverse
    from objaverse_paths import configure_objaverse
    configure_objaverse()
    obj_paths = objaverse._load_object_paths()
    mirror_root = objaverse._VERSIONED_PATH

    uids = base.get_objaverse_uid_pool(quiet=not is_main)
    uids_for_rank  = uids[rank::world_size]
    target_objects = (args.n_objects + world_size - 1) // world_size

    azimuths   = np.linspace(0, 360, args.azimuths_per_mesh, endpoint=False)
    view_pairs = [(float(a), float(e)) for a, e in
                  itertools.product(azimuths, list(args.elevations))]
    view_ids   = [f"az{int(a):03d}_el{int(e):03d}" for a, e in view_pairs]

    # Resume set: ONE readdir instead of 70 stats/object (460 ms each on NFS).
    done_names = {n for n in os.listdir(samples_dir) if not n.startswith("_tmp")}

    # Persistent skip cache. Without it every restart RE-ATTEMPTS every uid that
    # ever failed (download + load + repair-fail, several worker-seconds each) —
    # measured ~34% of traversed uids fail, i.e. thousands of re-attempts per
    # rank per restart. Failures are deterministic (too many faces, repair
    # impossible), so record them once and never submit again. All ranks append
    # to one file; they share a kernel, so short O_APPEND writes don't interleave.
    # Pool-crash skips are deliberately NOT recorded (transient, uid may be fine).
    skip_file = dataset_dir / "skipped_uids.txt"
    known_bad: set = set()
    if skip_file.exists():
        with open(skip_file) as fh:
            known_bad = {ln.split("\t")[0].strip() for ln in fh if ln.strip()}
    if is_main and known_bad:
        print(f"[fast-precompute] skip cache: {len(known_bad):,} known-bad uids")

    def _record_skip(uid: str, reason: str) -> None:
        with open(skip_file, "a") as fh:
            fh.write(f"{uid}\t{reason}\n")

    task_const = dict(cache_dir=cache_dir, radius=radius,
                      max_triangles=args.max_triangles,
                      repair_meshes=args.repair_meshes,
                      repair_voxel_res=args.repair_voxel_res,
                      repair_voxel_method=args.repair_voxel_method,
                      n_points=args.n_points,
                      near_surface_fraction=args.near_surface_fraction,
                      sharp_edge_fraction=args.sharp_edge_fraction,
                      sharp_edge_angle_deg=args.sharp_edge_angle_deg,
                      view_pairs=view_pairs, image_size=args.image_size,
                      fov=args.fov)

    ctx = mp.get_context("spawn")   # parent holds CUDA; fork would be unsafe
    # max_tasks_per_child: workers accumulate RSS across meshes (trimesh caches
    # + allocator fragmentation; measured creep 56 -> 117 GB fleet-wide over an
    # hour). Recycling every 16 tasks bounds it; the ~15 s respawn amortizes to
    # <1 s/object.
    pool = cf.ProcessPoolExecutor(max_workers=args.prep_workers, mp_context=ctx,
                                  initializer=_prep_worker_init,
                                  initargs=(local_rank,),
                                  max_tasks_per_child=16)
    writer = cf.ThreadPoolExecutor(max_workers=4)
    write_futures: list = []

    pbar = tqdm(total=target_objects, unit="obj", dynamic_ncols=True,
                desc=f"rank{rank}" if world_size > 1 else "precompute",
                position=rank, leave=True)
    obj_saved = obj_skipped = obj_repaired = 0
    inflight: dict = {}          # future -> uid
    uid_iter = iter(uids_for_rank)
    exhausted = False

    def _respawn_pool():
        nonlocal pool
        pool = cf.ProcessPoolExecutor(max_workers=args.prep_workers, mp_context=ctx,
                                      initializer=_prep_worker_init,
                                      initargs=(local_rank,),
                                      max_tasks_per_child=16)

    def _submit_more():
        nonlocal exhausted, obj_saved, obj_skipped
        while not exhausted and len(inflight) < args.prefetch \
                and obj_saved + len(inflight) < target_objects:
            uid = next(uid_iter, None)
            if uid is None:
                exhausted = True
                return
            if uid in known_bad:
                obj_skipped += 1
                continue
            if all(f"{uid}_{v}" in done_names for v in view_ids):
                obj_saved += 1
                pbar.update(1)
                continue
            rel = obj_paths.get(uid)
            if rel is None:
                obj_skipped += 1
                _record_skip(uid, "missing from objaverse index")
                continue
            t = dict(task_const, uid=uid, src=os.path.join(mirror_root, rel))
            # submit() raises BrokenProcessPool too — this is what killed rank 3
            # 12 h into the 2026-09-05 run: an over-rlimit malloc inside a C
            # extension takes the worker down ABRUPTLY (no catchable
            # MemoryError), and the .result() handler below never sees it if
            # the break is first observed here. Respawn and retry once; if the
            # fresh pool is also broken, something machine-level is wrong and
            # raising is correct.
            try:
                fut = pool.submit(_prep_object, t)
            except cf.process.BrokenProcessPool:
                tqdm.write(f"[prep pool broken at submit ({uid}); respawning]")
                for f2 in list(inflight):
                    inflight.pop(f2)
                    obj_skipped += 1
                _respawn_pool()
                fut = pool.submit(_prep_object, t)
            inflight[fut] = uid

    _submit_more()
    while inflight:
        done, _ = cf.wait(list(inflight), return_when=cf.FIRST_COMPLETED)
        for fut in done:
            # pop with default: when the pool breaks, EVERY pending future in
            # this wait() batch resolves at once and the handler below already
            # drained inflight — the siblings would otherwise KeyError here.
            uid = inflight.pop(fut, None)
            if uid is None:
                continue
            try:
                res = fut.result()
            except Exception as e:   # includes BrokenProcessPool
                if isinstance(e, cf.process.BrokenProcessPool):
                    tqdm.write(f"[prep pool died on {uid}: respawning] {e!r}")
                    for f2 in list(inflight):   # siblings died with it
                        inflight.pop(f2)
                        obj_skipped += 1
                    _respawn_pool()
                res = {"status": "skip", "uid": uid, "reason": repr(e)[:200]}
            if res["status"] != "ok":
                obj_skipped += 1
                if "BrokenProcessPool" not in res["reason"]:
                    _record_skip(res["uid"], res["reason"])
                if args.verbose:
                    tqdm.write(f"[skip mesh] {res['uid']}: {res['reason']}")
            else:
                pts_t = torch.from_numpy(res["pts"])
                sdf_t = torch.from_numpy(res["sdf"])
                nrm_t = torch.from_numpy(res["nrm"])
                with torch.no_grad():
                    for vid, img_np, extr in zip(view_ids, res["images"], res["extrs"]):
                        sample_id = f"{res['uid']}_{vid}"
                        if sample_id in done_names:
                            continue
                        # bf16 for the backbone (4.0x, output stored .half()
                        # regardless); tokens OUTSIDE autocast: fp32, cached
                        # training input, must match the existing dataset.
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                            enabled=(device.type == "cuda")):
                            scene_codes = model([img_np], device=device)
                        triplane = scene_codes[0].float().half().cpu()
                        tokens = base.compute_cached_image_tokens(model, img_np, device)
                        write_futures.append(writer.submit(
                            _write_view, samples_dir, sample_id, img_np, extr,
                            triplane, tokens, pts_t, sdf_t, nrm_t))
                obj_saved += 1
                obj_repaired += int(res["repaired"])
                pbar.update(1)
            pbar.set_postfix(objects=obj_saved, skipped=obj_skipped,
                             repaired=obj_repaired, prep=len(inflight))
            if len(write_futures) > 200:   # drain + surface IO errors early
                mid = len(write_futures) // 2
                for wf in write_futures[:mid]:
                    wf.result()
                write_futures = write_futures[mid:]
        _submit_more()

    for wf in write_futures:
        wf.result()
    writer.shutdown(wait=True)
    pool.shutdown(wait=True)
    pbar.close()
    if is_main:
        print(f"\n[fast-precompute] done — {obj_saved} objects "
              f"({obj_repaired} repaired, {obj_skipped} skipped) -> {dataset_dir}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_precompute_args() -> argparse.Namespace:
    """Dataset SCALE (n_objects, azimuths_per_mesh) comes from THIS file's
    config so one edit here changes both precompute and training. The
    remaining per-sample generation settings (points, elevations, repair, ...)
    still come from train_sdf_head.py's own config; getattr defaults keep this
    working across versions of that file. data_source is forced to objaverse."""
    g = lambda name, default: getattr(base, name, default)
    return argparse.Namespace(
        dataset_dir    = DATASET_DIR,
        data_source    = "objaverse",
        hy3d_mesh_dir  = g("HY3D_MESH_DIR", ""),
        model          = MODEL,
        n_objects      = N_OBJECTS,
        azimuths_per_mesh = AZIMUTHS_PER_MESH,
        elevations     = g("ELEVATIONS", [15.0, 30.0]),
        near_surface_fraction = g("NEAR_SURFACE_FRACTION", 0.25),
        sharp_edge_fraction   = g("SHARP_EDGE_FRACTION", 0.0),
        sharp_edge_angle_deg  = g("SHARP_EDGE_ANGLE_DEG", 30.0),
        repair_meshes         = g("REPAIR_MESHES", True),
        repair_voxel_res      = g("REPAIR_VOXEL_RES", 128),
        repair_voxel_method   = g("REPAIR_VOXEL_METHOD", "ray"),
        n_points       = g("N_POINTS", 32768),
        image_size     = g("IMAGE_SIZE", IMAGE_SIZE),
        fov            = g("FOV", FOV),
        max_mesh_mb    = g("MAX_MESH_MB", 0.0),
        max_triangles  = g("MAX_TRIANGLES", 500_000),
        verbose        = g("VERBOSE", False),
        # fast-precompute knobs (env-tunable; more workers = more RAM in
        # flight, each holds a mesh + voxel repair at peak)
        prep_workers   = int(os.environ.get("SDFER_PREP_WORKERS", "4")),
        prefetch       = int(os.environ.get("SDFER_PREFETCH", "12")),
    )


def main() -> None:
    _p = argparse.ArgumentParser(add_help=False)
    _p.add_argument("--command", default=COMMAND,
                    choices=("precompute", "train", "both"))
    _cli, _ = _p.parse_known_args()
    args = build_train_args()

    # Env overrides for sweeps, same convention as train_sdf_head.py:
    #   SDFER_EPOCHS=50 SDFER_SAMPLES_PER_BATCH=10 torchrun ... --command train
    for _k in ("samples_per_batch", "epochs", "eval_every", "train_points_per_sample",
               "test_points_per_sample", "n_objects", "num_workers", "vis_every",
               "save_every", "test_max_samples", "azimuths_per_mesh",
               "compile_backbone", "auto_scale_lr", "gradient_checkpointing",
               "lora_block_start", "lora_block_end", "lora_rank"):
        _v = os.environ.get("SDFER_" + _k.upper())
        if _v is not None:
            setattr(args, _k, int(_v))
            print(f"[env] {_k} = {int(_v)}")

    if _cli.command in ("precompute", "both"):
        # SDFER_PRECOMPUTE_LEGACY=1 falls back to the original serial pipeline.
        if os.environ.get("SDFER_PRECOMPUTE_LEGACY"):
            base.run_precompute(build_precompute_args())
        else:
            run_precompute_fast(build_precompute_args())
    if _cli.command in ("train", "both"):
        run_train_fast(args)


if __name__ == "__main__":
    main()
