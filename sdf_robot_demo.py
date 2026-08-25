"""
sdf_robot_demo.py — SDF-guided robot obstacle-avoidance demo.

One image of an object is fed through the fine-tuned TripoSR + SDF head ONCE to
produce a triplane; a Franka Panda in PyBullet then sweeps its end effector past
the object while querying the learned SDF at every control tick, steering around
the obstacle with a potential-field/sliding local planner. See ROBOT_DEMO.md for
the full system design and the honesty caveats.

Run inside the TripoSR docker container with .venv active (same as
train_sdf_head.py / infer_sdf_mesh.py):

    python sdf_robot_demo.py                              # all defaults
    python sdf_robot_demo.py --uid <objaverse-uid>
    python sdf_robot_demo.py --mesh path/to/object.glb
    python sdf_robot_demo.py --mode floating --no-baseline

Outputs (in --output-dir, default robot_demo_output/):
    demo_sdf_avoidance.mp4      avoidance run (SDF in the loop)
    demo_baseline_straight.mp4  baseline run (no SDF -> collides)
    sdf_slices.png              predicted SDF slices + GT contour + executed path
    clearance.png               predicted vs ground-truth clearance over time
    input_render.png            the single image the model saw
    sdf_mlp_mesh.obj            marching-cubes surface of the predicted SDF
    metrics.json                min clearance, query latency, collision flags

IMPORTANT — TSDF sensing horizon: checkpoints trained with SDF clamping
(v0.64: delta = 0.1 normalized units) saturate beyond delta, so the model can
only *measure* distances up to H = delta * object_scale metres from the
surface. The demo therefore (a) senses gripper clearance by sampling the SDF
ON the gripper sphere's surface rather than subtracting the radius from a
saturating field, and (b) auto-shrinks the avoidance buffer to fit inside H
(with a loud warning). Bigger objects buy a bigger horizon.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYOPENGL_USE_ACCELERATE", "0")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw, ImageFont

from train_sdf_head import (
    compute_sdf,
    fourier_encode,
    load_and_normalize_mesh,
    load_R_world_from_recon_json_strict,
    query_triplane_features,
    reconstruct_mesh_from_triplane,
    render_mesh_to_image,
)
from infer_sdf_mesh import (
    apply_finetuned_lora,
    fetch_objaverse_glb,
    load_sdf_mlp_from_checkpoint,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — defaults for anything not passed on the command line
# ═══════════════════════════════════════════════════════════════════════════════

CHECKPOINT = str(SCRIPT_DIR / "sdf_checkpoints" / "sdf_head_v0.64_1k_epoch0425.pt")
MODEL      = "stabilityai/TripoSR"
UID        = "97a038cd7a304bce81890c118fadd793"   # chunky watertight blob — a good
                                                  # obstacle. (infer_sdf_mesh's default
                                                  # 85739db9... is a figure glued to a
                                                  # huge flat panel that walls off the
                                                  # whole workspace and traps the arm.)
OUTPUT_DIR = str(SCRIPT_DIR / "robot_demo_output")

# Input render — MUST match the training/inference domain (see infer_sdf_mesh).
AZIMUTH, ELEVATION, FOV, IMAGE_SIZE = 0.0, 30.0, 40.0, 256
MC_RESOLUTION = 128

# Scene geometry (metres). Panda base at world origin; reach is 0.855 m.
# Scene sizing is a KINEMATIC constraint, not just taste: the Panda must be able
# to wrap the EE around the object's outer flank with the gripper down, inside
# its 0.855 m reach. At scale 0.5 / x 0.5 there is NO feasible detour (outer
# route ~0.85 m, over-the-top ~0.90 m) and the arm wedges on the object forever.
OBJECT_SCALE = 0.35          # metres per normalized unit (= longest edge)
OBJECT_XY    = (0.45, 0.00)  # object center, dead-center on the EE path
PATH_START_Y, PATH_GOAL_Y = -0.40, 0.40   # EE sweeps along +y at x = OBJECT_XY[0]

# Planner (requested values; the effective buffer is auto-fitted to the sensing
# horizon in main() and stored back into these globals).
ROBOT_RADIUS = 0.03   # gripper-tip sphere (m)
# Hand envelope: sphere offsets above the EE (gripper points down) and radii,
# covering fingers, hand bar, and wrist. Conservative on purpose — v4 showed a
# single thin sphere lets the hand body graze the obstacle.
HAND_SPHERES = [(0.00, 0.03), (0.06, 0.06), (0.12, 0.065)]
FOREARM_LINK, FOREARM_RADIUS = 5, 0.07   # panda_link5, tracked live each tick
D_SAFE       = 0.09   # buffer: start steering when clearance drops below this (m)
                      # generous: the sliding equilibrium settles ~2-3 cm inside
                      # the buffer, and the real hand geometry pokes a few cm
                      # beyond the control spheres
D_STOP       = 0.03   # emergency: pure retreat below this clearance (m)
V_MAX        = 0.20   # EE speed (m/s)
GOAL_TOL     = 0.02   # stop when within this of the goal (m)
FD_STEP      = 0.005  # finite-difference step for SDF gradient (m)
Z_MIN, Z_MAX = 0.05, 0.80   # EE height limits (floor / reach)

# Sim / video
SIM_HZ        = 240
TICKS_PER_SEC = 30            # planner + camera rate
MAX_SECONDS   = 30
FRAME_W, FRAME_H = 1280, 720

# ═══════════════════════════════════════════════════════════════════════════════

_AXES6 = np.array([[1, 0, 0], [-1, 0, 0],
                   [0, 1, 0], [0, -1, 0],
                   [0, 0, 1], [0, 0, -1]], dtype=np.float64)


def gt_sdf_signed(mesh, pts: np.ndarray) -> np.ndarray:
    """GT signed distance with a robust sign.

    compute_sdf's nearest-face-normal sign heuristic misfires near concave
    edges (verified: isolated -11.9 cm readings at points that are clearly
    outside). Magnitude from compute_sdf, sign from ray-parity containment,
    which is robust for watertight meshes. NOTE: the training pipeline uses
    the raw heuristic for its labels — the same artifact exists there.
    """
    d = np.abs(compute_sdf(mesh, pts))
    inside = mesh.contains(pts)
    d[inside] *= -1.0
    return d


# ─── SDF oracle: world-frame metric queries against the learned model ─────────

class SDFOracle:
    """Wraps triplane + SDF MLP as a metric, world-frame distance oracle.

    world -> normalized:  p_norm = (p_world - center) / scale
    normalized -> recon:  p_trip = p_norm @ R.T          (training convention)
    metres:               sdf_norm * scale

    Outside the triplane cube [-radius, radius]^3 the model is undefined: the
    query is clamped into the cube and the clamp distance added on top (exact
    when the nearest surface lies along the clamp direction; fine everywhere
    the demo actually goes — see ROBOT_DEMO.md flag #4).
    """

    def __init__(self, sdf_mlp, triplane, R_np, radius, feature_reduction,
                 n_freqs, use_triplane, scale_m, center_world, device):
        self.sdf_mlp = sdf_mlp
        self.triplane = triplane.to(device)
        self.R = (torch.from_numpy(np.asarray(R_np, dtype=np.float32)).to(device)
                  if R_np is not None else None)
        self.radius = float(radius)
        self.feature_reduction = feature_reduction
        self.n_freqs = int(n_freqs)
        self.use_triplane = bool(use_triplane)
        self.scale = float(scale_m)
        self.center = np.asarray(center_world, dtype=np.float64)
        self.device = device

    @torch.no_grad()
    def _query_norm(self, pts_norm: torch.Tensor) -> torch.Tensor:
        lim = self.radius - 1e-4
        clamped = pts_norm.clamp(-lim, lim)
        box_excess = (pts_norm - clamped).norm(dim=-1)
        p = clamped @ self.R.T if self.R is not None else clamped
        if self.use_triplane:
            feats = query_triplane_features(p, self.triplane, self.radius,
                                            self.feature_reduction)
            if self.n_freqs > 0:
                feats = torch.cat([feats, fourier_encode(p, self.n_freqs)], dim=-1)
        else:
            feats = fourier_encode(p, self.n_freqs) if self.n_freqs > 0 else p
        return self.sdf_mlp(feats) + box_excess

    def query_world(self, pts_world: np.ndarray) -> np.ndarray:
        """pts_world (N,3) metres -> predicted signed distance (N,) metres."""
        pts_norm = (np.asarray(pts_world, dtype=np.float32) - self.center) / self.scale
        t = torch.from_numpy(pts_norm.astype(np.float32)).to(self.device)
        return (self._query_norm(t) * self.scale).cpu().numpy()

    # ── ESDF: propagate the narrow-band model belief into a full distance field
    #
    # Empirically (see ROBOT_DEMO.md "narrow band" flag) the trained head is a
    # SURFACE DETECTOR, not a metric SDF: values are accurate only in a ~±3 cm
    # shell around the surface; the far field AND the deep interior saturate to
    # large positive values (the interior reads as free space!). Raw queries
    # are therefore unusable for planning. The standard robotics remedy for a
    # narrow-band TSDF (Voxblox et al.) is one-time ESDF propagation: evaluate
    # the field on a grid, take occupancy = the (hole-filled) region enclosed
    # by the predicted shell, and distance-transform. Still 100% derived from
    # the single input image; queries become trilinear lookups (~µs).

    def build_esdf(self, resolution: int = 128, tau_norm: float = 0.03,
                   ground_z_norm: float | None = None) -> None:
        from scipy import ndimage
        lim = self.radius
        axes = torch.linspace(-lim, lim, resolution)
        grid = torch.stack(torch.meshgrid(axes, axes, axes, indexing="ij"),
                           dim=-1).reshape(-1, 3).to(self.device)
        vals = []
        with torch.no_grad():
            for i in range(0, grid.shape[0], 262144):
                vals.append(self._query_norm(grid[i:i + 262144]).cpu())
        G = torch.cat(vals).reshape(resolution, resolution, resolution).numpy()

        occ = G < tau_norm                    # the predicted shell (thickened by tau)
        if ground_z_norm is not None:
            # The predicted shell is reliably OPEN at the unseen underside, so
            # hole-filling alone leaks and the interior reads as free space
            # (v6 tunneled straight through the object because of this). The
            # object rests on the ground plane, so stamping the ground slab
            # into occupancy seals the bottom topologically — and is physically
            # true regardless: the robot can't go below the floor.
            k = int(np.searchsorted(axes.numpy(), ground_z_norm + 0.01))
            occ[:, :, :max(k, 1)] = True
        occ = ndimage.binary_closing(occ, iterations=3)   # seal small shell gaps
        occ = ndimage.binary_fill_holes(occ)              # ... then fill interior
        voxel = 2 * lim / (resolution - 1)
        esdf = (ndimage.distance_transform_edt(~occ)
                - ndimage.distance_transform_edt(occ)) * voxel
        self._esdf = esdf.astype(np.float32)
        self._esdf_res = resolution
        frac = float(occ.mean())
        print(f"        [esdf] occupancy fraction {frac:.3f} "
              f"(shell sealed + interior filled + ground slab)")

    def query_esdf_world(self, pts_world: np.ndarray) -> np.ndarray:
        """(N,3) metres -> distance to the believed surface (metres), from the
        propagated ESDF. Out-of-cube points get the clamp distance added."""
        from scipy import ndimage
        pts_norm = (np.asarray(pts_world, dtype=np.float64) - self.center) / self.scale
        lim = self.radius
        clamped = np.clip(pts_norm, -lim, lim)
        box_excess = np.linalg.norm(pts_norm - clamped, axis=-1)
        idx = ((clamped + lim) / (2 * lim) * (self._esdf_res - 1)).T
        d = ndimage.map_coordinates(self._esdf, idx, order=1, mode="nearest")
        return ((d + box_excess) * self.scale).astype(np.float32)

    def sense(self, spheres: list[tuple[np.ndarray, float]]
              ) -> tuple[float, np.ndarray]:
        """-> (min clearance over control spheres, gradient).

        THIS is what the planner consumes, and all it consumes: trilinear
        interpolation into the precomputed ESDF array. No triplane sampling, no
        MLP, no GPU — the model was evaluated 2.1M times to BUILD that array,
        once, in build_esdf(). The raw per-point model query is timed and logged
        separately in run_episode() so the two costs never get conflated.

        ``spheres`` is [(absolute world center, radius), ...] — the EE envelope
        plus any tracked arm-link spheres. Clearance = ESDF(center) − radius per
        sphere (exact for spheres, since the ESDF is a true distance field).
        Gradient = central differences on the ESDF at the closest sphere's
        center.
        """
        h = FD_STEP
        centers = np.stack([c for c, _ in spheres])
        probes = np.concatenate([centers] + [centers + h * a for a in _AXES6])
        d = self.query_esdf_world(probes)
        k = len(spheres)
        clear = d[:k] - np.array([r for _, r in spheres])
        i = int(np.argmin(clear))
        # probe layout: k centers, then 6 blocks of k (one per axis offset)
        gx = (d[k + 0 * k + i] - d[k + 1 * k + i]) / (2 * h)
        gy = (d[k + 2 * k + i] - d[k + 3 * k + i]) / (2 * h)
        gz = (d[k + 4 * k + i] - d[k + 5 * k + i]) / (2 * h)
        grad = np.array([gx, gy, gz])
        gn = np.linalg.norm(grad)
        n_dir = grad / gn if gn > 1e-9 else np.array([0.0, 0.0, 1.0])
        return float(clear[i]), n_dir


# ─── Simulation wrapper (Panda arm or floating gripper) ───────────────────────

PANDA_EE_LINK = 11          # panda_grasptarget
PANDA_ARM_JOINTS = list(range(7))
PANDA_FINGERS = [9, 10]
PANDA_REST = [0.0, -0.6, 0.0, -2.2, 0.0, 2.0, 0.785]
# Null-space IK limits/ranges/rest (7 arm + 2 finger DOF): biases solutions
# toward the elbow-up rest pose so the forearm stays high, away from the object.
PANDA_LL = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0]
PANDA_UL = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04]
PANDA_JR = [u - l for l, u in zip(PANDA_LL, PANDA_UL)]
PANDA_RP = PANDA_REST + [0.04, 0.04]


class Sim:
    def __init__(self, mode: str, obj_path: str, obj_scale: float,
                 obj_center: np.ndarray, ghost_path: str | None):
        import pybullet as pb
        import pybullet_data
        self.pb = pb
        self.mode = mode
        self.client = pb.connect(pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        # GPU rendering if available (same recipe as pybullet_to_triposr.py);
        # the tiny renderer at 720p over ~900 frames is painfully slow otherwise.
        self.use_egl = False
        try:
            import pkgutil
            egl = pkgutil.get_loader("eglRenderer")
            if egl:
                pb.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
                self.use_egl = True
        except Exception:
            pass
        pb.setGravity(0, 0, -9.81)
        pb.setTimeStep(1.0 / SIM_HZ)
        pb.loadURDF("plane.urdf")

        # Ground-truth object: the SAME normalized mesh the model saw, static,
        # concave collision so the baseline run visibly hits the real shape.
        col = pb.createCollisionShape(
            pb.GEOM_MESH, fileName=obj_path, meshScale=[obj_scale] * 3,
            flags=pb.GEOM_FORCE_CONCAVE_TRIMESH)
        vis = pb.createVisualShape(
            pb.GEOM_MESH, fileName=obj_path, meshScale=[obj_scale] * 3,
            rgbaColor=[0.62, 0.62, 0.62, 1.0])
        self.obj = pb.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                      baseVisualShapeIndex=vis,
                                      basePosition=obj_center.tolist())

        # Translucent orange ghost of the PREDICTED SDF surface (visual only):
        # the robot navigates w.r.t. this belief, not the grey GT.
        if ghost_path is not None and os.path.exists(ghost_path):
            gvis = pb.createVisualShape(
                pb.GEOM_MESH, fileName=ghost_path, meshScale=[obj_scale] * 3,
                rgbaColor=[1.0, 0.42, 0.20, 0.5])
            pb.createMultiBody(baseMass=0, baseVisualShapeIndex=gvis,
                               basePosition=obj_center.tolist())

        self.down_orn = pb.getQuaternionFromEuler([np.pi, 0.0, 0.0])
        if mode == "panda":
            self.robot = pb.loadURDF("franka_panda/panda.urdf", [0, 0, 0],
                                     useFixedBase=True)
            for j, q in zip(PANDA_ARM_JOINTS, PANDA_REST):
                pb.resetJointState(self.robot, j, q)
            for j in PANDA_FINGERS:
                pb.resetJointState(self.robot, j, 0.04)
        else:  # floating gripper sphere, moved kinematically
            fvis = pb.createVisualShape(pb.GEOM_SPHERE, radius=ROBOT_RADIUS,
                                        rgbaColor=[0.15, 0.45, 0.9, 1.0])
            self.robot = pb.createMultiBody(baseMass=0, baseVisualShapeIndex=fvis,
                                            basePosition=[0, 0, 0.3])

    # -- end-effector control ------------------------------------------------

    def teleport_ee(self, pos: np.ndarray) -> None:
        pb = self.pb
        if self.mode == "panda":
            for _ in range(8):  # iterate IK to convergence, then hard-reset
                q = pb.calculateInverseKinematics(
                    self.robot, PANDA_EE_LINK, pos.tolist(), self.down_orn,
                    lowerLimits=PANDA_LL, upperLimits=PANDA_UL,
                    jointRanges=PANDA_JR, restPoses=PANDA_RP,
                    maxNumIterations=100, residualThreshold=1e-5)
                for j, qj in zip(PANDA_ARM_JOINTS, q[:7]):
                    pb.resetJointState(self.robot, j, qj)
            for j in PANDA_FINGERS:
                pb.resetJointState(self.robot, j, 0.04)
        else:
            pb.resetBasePositionAndOrientation(self.robot, pos.tolist(), [0, 0, 0, 1])

    def track_ee(self, target: np.ndarray) -> None:
        pb = self.pb
        if self.mode == "panda":
            q = pb.calculateInverseKinematics(
                self.robot, PANDA_EE_LINK, target.tolist(), self.down_orn,
                lowerLimits=PANDA_LL, upperLimits=PANDA_UL,
                jointRanges=PANDA_JR, restPoses=PANDA_RP,
                maxNumIterations=60, residualThreshold=1e-5)
            for j, qj in zip(PANDA_ARM_JOINTS, q[:7]):
                pb.setJointMotorControl2(self.robot, j, pb.POSITION_CONTROL,
                                         targetPosition=qj, force=240,
                                         maxVelocity=2.0)
            for j in PANDA_FINGERS:
                pb.setJointMotorControl2(self.robot, j, pb.POSITION_CONTROL,
                                         targetPosition=0.04, force=30)
        else:
            pb.resetBasePositionAndOrientation(self.robot, target.tolist(),
                                               [0, 0, 0, 1])

    def ee_pos(self) -> np.ndarray:
        if self.mode == "panda":
            st = self.pb.getLinkState(self.robot, PANDA_EE_LINK)
            return np.array(st[0])
        pos, _ = self.pb.getBasePositionAndOrientation(self.robot)
        return np.array(pos)

    def link_pos(self, link: int) -> np.ndarray | None:
        if self.mode != "panda":
            return None
        return np.array(self.pb.getLinkState(self.robot, link)[0])

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self.pb.stepSimulation()

    def contact_links(self) -> list[str]:
        """Names of robot links currently touching the object (empty = none)."""
        pts = self.pb.getContactPoints(self.robot, self.obj)
        names = set()
        for c in pts:
            li = c[3]  # linkIndexA (robot)
            if self.mode != "panda" or li == -1:
                names.add("base")
            else:
                names.add(self.pb.getJointInfo(self.robot, li)[12].decode())
        return sorted(names)

    # -- camera --------------------------------------------------------------

    def frame(self, target) -> np.ndarray:
        pb = self.pb
        view = pb.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target, distance=1.55, yaw=55, pitch=-28,
            roll=0, upAxisIndex=2)
        proj = pb.computeProjectionMatrixFOV(fov=50, aspect=FRAME_W / FRAME_H,
                                             nearVal=0.02, farVal=6.0)
        if self.use_egl:  # EGL plugin renders regardless of the renderer flag
            _, _, rgba, _, _ = pb.getCameraImage(FRAME_W, FRAME_H, view, proj)
        else:
            _, _, rgba, _, _ = pb.getCameraImage(FRAME_W, FRAME_H, view, proj,
                                                 renderer=pb.ER_TINY_RENDERER)
        return np.asarray(rgba, dtype=np.uint8).reshape(FRAME_H, FRAME_W, 4)[..., :3]

    def close(self) -> None:
        self.pb.disconnect(self.client)


# ─── Frame annotation ─────────────────────────────────────────────────────────

_FONTS: dict = {}


def _font(size: int):
    if size not in _FONTS:
        try:
            from matplotlib import font_manager
            _FONTS[size] = ImageFont.truetype(
                font_manager.findfont("DejaVu Sans"), size)
        except Exception:
            _FONTS[size] = ImageFont.load_default()
    return _FONTS[size]


def annotate(frame: np.ndarray, title: str, lines: list[str],
             color=(255, 255, 255)) -> np.ndarray:
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle([0, 0, FRAME_W, 46 + 30 * len(lines)], fill=(0, 0, 0, 170))
    draw.text((16, 8), title, fill=color, font=_font(26))
    f = _font(21)
    for i, ln in enumerate(lines):
        draw.text((16, 46 + 30 * i), ln, fill=(230, 230, 230), font=f)
    return np.asarray(img)


# ─── One episode (avoidance ON or OFF) ────────────────────────────────────────

def run_episode(sim: Sim, oracle: SDFOracle, gt_mesh: trimesh.Trimesh,
                start: np.ndarray, goal: np.ndarray, avoid: bool,
                cam_target, side_bias: np.ndarray) -> dict:
    dt = 1.0 / TICKS_PER_SEC
    substeps = SIM_HZ // TICKS_PER_SEC
    up = np.array([0.0, 0.0, 1.0])
    sim.teleport_ee(start)
    sim.step(SIM_HZ // 4)  # settle

    target = sim.ee_pos().copy()
    frames = []
    log = {"t": [], "pred": [], "raw": [], "gt": [], "ee": [],
           "esdf_us": [], "raw_ms": []}
    collided, reached = False, False
    first_contact_tick = None
    contact_links: set[str] = set()
    title = ("SDF avoidance ON  -  planner queries the learned SDF every tick"
             if avoid else "BASELINE  -  straight line, no SDF checking")

    for tick in range(MAX_SECONDS * TICKS_PER_SEC):
        ee = sim.ee_pos()
        spheres = [(ee + np.array([0.0, 0.0, off]), r) for off, r in HAND_SPHERES]
        fore = sim.link_pos(FOREARM_LINK)
        if fore is not None:
            spheres.append((fore, FOREARM_RADIUS))

        # What the planner actually runs on: trilinear ESDF lookups (CPU).
        t0 = time.perf_counter()
        d, n = oracle.sense(spheres)
        esdf_us = (time.perf_counter() - t0) * 1e6

        # Logging only: one batch-1 triplane+MLP evaluation, for the raw-vs-ESDF
        # curve in clearance.png. Latency-bound (~1.6 ms idle) and further
        # inflated by contention with the EGL frame capture on the same GPU —
        # it is NOT the cost of sensing, and is reported on its own line.
        t0 = time.perf_counter()
        raw = float(oracle.query_world(ee[None, :])[0])
        raw_ms = (time.perf_counter() - t0) * 1e3

        # ground-truth clearance: exact SDF on the GT mesh (sphere => exact)
        centers = np.stack([(c - oracle.center) / oracle.scale
                            for c, _ in spheres])
        gt_sdf = gt_sdf_signed(gt_mesh, centers.astype(np.float64)) * oracle.scale
        gt = float(min(s - r for s, (_, r) in zip(gt_sdf, spheres)))

        to_goal = goal - ee
        dist_goal = float(np.linalg.norm(to_goal))
        reached = dist_goal < GOAL_TOL
        dir_goal = to_goal / max(dist_goal, 1e-9)

        # stall watchdog: no net EE progress for ~3 s -> boost the side bias to
        # escape whatever equilibrium the blend has found
        stalled = (tick > 3 * TICKS_PER_SEC and np.linalg.norm(
            ee - np.array(log["ee"][-3 * TICKS_PER_SEC])) < 0.03)

        if avoid and d < D_SAFE and not reached:
            w = float(np.clip((D_SAFE - d) / max(D_SAFE - D_STOP, 1e-6), 0.0, 1.5))
            slide = dir_goal - np.dot(dir_goal, n) * n
            sn = np.linalg.norm(slide)
            slide = slide / sn if sn > 1e-6 else up.copy()
            if d < D_STOP:
                v_dir = n                                   # emergency retreat
            else:
                # blend goal->slide, push out along the gradient, and bias to a
                # consistent side so a head-on approach (degenerate slide)
                # resolves the same way every tick instead of deadlocking
                w_side = 0.5 * w * (3.0 if stalled else 1.0)
                v_dir = ((1 - min(w, 1)) * dir_goal + min(w, 1) * slide
                         + 1.3 * w * n + w_side * side_bias)
                v_dir /= max(np.linalg.norm(v_dir), 1e-9)
            speed = V_MAX * (0.35 + 0.65 * float(np.clip(d / D_SAFE, 0, 1)))
        else:
            v_dir, speed = dir_goal, V_MAX
        speed = 0.0 if reached else min(speed, dist_goal / dt)

        target = target + v_dir * speed * dt
        target[2] = float(np.clip(target[2], Z_MIN, Z_MAX))
        # leash the IK target to the actual EE so a briefly-stuck arm doesn't
        # accumulate integrator windup and overshoot when it breaks free
        err = target - ee
        err_n = np.linalg.norm(err)
        if err_n > 0.05:
            target = ee + err * (0.05 / err_n)
        sim.track_ee(target)
        sim.step(substeps)

        links = sim.contact_links()
        if links:
            collided = True
            contact_links.update(links)
            if first_contact_tick is None:
                first_contact_tick = tick

        log["t"].append(tick * dt)
        log["pred"].append(d)
        log["raw"].append(raw)
        log["gt"].append(gt)
        log["ee"].append(ee.tolist())
        log["esdf_us"].append(esdf_us)
        log["raw_ms"].append(raw_ms)

        status = ("COLLISION" if collided else
                  "goal reached" if reached else
                  f"steering (SDF < {D_SAFE * 100:.0f} cm buffer)"
                  if avoid and d < D_SAFE else "cruising")
        col = ((255, 80, 60) if (collided or d < 0) else
               (255, 190, 80) if d < D_SAFE else (140, 255, 140))
        frames.append(annotate(
            sim.frame(cam_target), title,
            [f"predicted clearance: {d * 100:6.1f} cm   (true: {gt * 100:6.1f} cm)"
             f"   buffer: {D_SAFE * 100:.1f} cm",
             f"ESDF query: {esdf_us:5.1f} \u00b5s"
             f"   (raw model ref, logging only: {raw_ms:4.1f} ms)"
             f"   tick {tick:4d}   {status}"],
            color=col))

        if reached:
            break
        # a collided baseline just grinds against the static object — record a
        # couple more seconds for the video, then stop
        if (not avoid and first_contact_tick is not None
                and tick - first_contact_tick > 2 * TICKS_PER_SEC):
            break

    log.update(reached=bool(reached), collided=bool(collided),
               contact_links=sorted(contact_links),
               first_contact_tick=first_contact_tick)
    return {"frames": frames, "log": log}


# ─── Plots ────────────────────────────────────────────────────────────────────

def _slice_panel(ax, oracle, gt_mesh, ee_paths, plane: str, coord: float,
                 lims_a, lims_b, labels):
    """One SDF slice: predicted field (filled) + GT zero contour + paths."""
    A, B = np.meshgrid(np.linspace(*lims_a, 200), np.linspace(*lims_b, 200))
    if plane == "z":            # top-down: vary x,y at height coord
        P = np.stack([A.ravel(), B.ravel(), np.full(A.size, coord)], axis=-1)
        proj = (0, 1)
    else:                       # side view: vary y,z at x = coord
        P = np.stack([np.full(A.size, coord), A.ravel(), B.ravel()], axis=-1)
        proj = (1, 2)
    D_pred = oracle.query_esdf_world(P).reshape(A.shape)   # planner's belief
    P_norm = (P - oracle.center) / oracle.scale
    D_gt = (gt_sdf_signed(gt_mesh, P_norm.astype(np.float64)) * oracle.scale
            ).reshape(A.shape)

    im = ax.pcolormesh(A, B, D_pred, cmap="RdBu", vmin=-0.15, vmax=0.15,
                       shading="auto")
    ax.contour(A, B, D_pred, levels=[0.0], colors="darkorange", linewidths=2.5)
    ax.contour(A, B, D_pred, levels=[D_SAFE], colors="orange",
               linewidths=1.2, linestyles=":")
    ax.contour(A, B, D_gt, levels=[0.0], colors="black", linewidths=1.5,
               linestyles="--")
    for path, style, lbl in ee_paths:
        ax.plot(path[:, proj[0]], path[:, proj[1]], style, lw=2.2, label=lbl)
    ax.set_aspect("equal")
    ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1])
    return im


def make_plots(oracle: SDFOracle, gt_mesh, run_log: dict, base_log: dict | None,
               start, goal, obj_x: float, z_slice: float, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ee = np.array(run_log["ee"])
    paths = [(ee, "b-", "executed (SDF avoidance)")]
    if base_log is not None:
        paths.append((np.array(base_log["ee"]), "r--", "baseline (no SDF)"))

    y_lims = (min(start[1], goal[1]) - 0.1, max(start[1], goal[1]) + 0.1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    im = _slice_panel(axes[0], oracle, gt_mesh, paths, "z", z_slice,
                      (oracle.center[0] - 0.55, oracle.center[0] + 0.55), y_lims,
                      ("x (m)", "y (m)"))
    axes[0].set_title(f"top-down slice, z = {z_slice:.2f} m")
    _slice_panel(axes[1], oracle, gt_mesh, paths, "x", obj_x,
                 y_lims, (0.0, 0.9), ("y (m)", "z (m)"))
    axes[1].set_title(f"side slice, x = {obj_x:.2f} m (path plane)")
    for ax, s_idx, g_idx in [(axes[0], (0, 1), (0, 1)), (axes[1], (1, 2), (1, 2))]:
        ax.plot(start[s_idx[0]], start[s_idx[1]], "g^", ms=11, label="start")
        ax.plot(goal[g_idx[0]], goal[g_idx[1]], "r*", ms=15, label="goal")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.colorbar(im, ax=axes, label="believed distance / ESDF (m)", shrink=0.75)
    fig.suptitle("Model belief: ESDF propagated from the predicted surface "
                 "(fill, orange contour) vs GT surface (black dashed)")
    fig.savefig(out_dir / "sdf_slices.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(run_log["t"], np.array(run_log["pred"]) * 100, lw=2,
            color="tab:blue", label="believed clearance (ESDF, used by planner)")
    ax.plot(run_log["t"], np.array(run_log["gt"]) * 100, lw=1.6, ls="--",
            color="black", label="true clearance (GT mesh)")
    ax.plot(run_log["t"], np.array(run_log["raw"]) * 100, lw=1,
            color="grey", alpha=0.7,
            label="raw model SDF at EE (narrow-band, for reference)")
    ax.axhline(D_SAFE * 100, color="orange", ls=":",
               label=f"buffer ({D_SAFE * 100:.1f} cm)")
    ax.axhline(0, color="red", lw=1, label="contact")
    ax.set_ylim(-6, 32)
    ax.set_xlabel("time (s)"); ax.set_ylabel("gripper clearance (cm)")
    ax.set_title("SDF-avoidance run: believed vs true clearance")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "clearance.png", dpi=160)
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    global D_SAFE, D_STOP, ROBOT_RADIUS
    ap = argparse.ArgumentParser(
        description="SDF-guided robot obstacle-avoidance demo (see ROBOT_DEMO.md).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--uid", default=UID, help="Objaverse UID for the obstacle.")
    src.add_argument("--mesh", default=None, help="Local mesh file instead of a UID.")
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--mode", choices=["panda", "floating"], default="panda")
    ap.add_argument("--object-scale", type=float, default=OBJECT_SCALE,
                    help="Longest edge of the object in metres (hard-coded scale; "
                         "also sets the TSDF sensing horizon).")
    ap.add_argument("--object-x", type=float, default=OBJECT_XY[0])
    ap.add_argument("--object-y", type=float, default=OBJECT_XY[1])
    ap.add_argument("--buffer", type=float, default=D_SAFE,
                    help="Requested clearance buffer (m); auto-shrunk to fit the "
                         "checkpoint's TSDF sensing horizon if needed.")
    ap.add_argument("--robot-radius", type=float, default=ROBOT_RADIUS)
    ap.add_argument("--azimuth", type=float, default=AZIMUTH)
    ap.add_argument("--elevation", type=float, default=ELEVATION)
    ap.add_argument("--mc-resolution", type=int, default=MC_RESOLUTION)
    ap.add_argument("--avoid-side", choices=["outer", "inner", "over"],
                    default="outer",
                    help="Which way to bias around the obstacle: outer (+x, away "
                         "from the robot base), inner (-x), or over the top.")
    ap.add_argument("--no-baseline", action="store_true",
                    help="Skip the straight-line (collision) baseline run.")
    ap.add_argument("--no-ghost", action="store_true",
                    help="Do not show the predicted-SDF ghost mesh in the scene.")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    timing: dict = {}

    # ── 1. models ──────────────────────────────────────────────────────────
    print(f"[1/6] Loading SDF checkpoint + TripoSR on {device} ...")
    sdf_mlp, meta, sargs = load_sdf_mlp_from_checkpoint(args.checkpoint, device)
    radius = float(meta["radius"])
    feat_red = meta["feature_reduction"]
    n_freqs = int(getattr(sargs, "n_freqs", 0))
    use_trip = bool(getattr(sargs, "use_triplane_features", True))
    sdf_clamp = float(getattr(sargs, "sdf_clamp", 0.0))

    from tsr.system import TSR
    triposr = TSR.from_pretrained(args.model, config_name="config.yaml",
                                  weight_name="model.ckpt")
    triposr.renderer.set_chunk_size(8192)
    triposr.to(device).eval()
    for p in triposr.parameters():
        p.requires_grad_(False)
    apply_finetuned_lora(triposr, args.checkpoint, device)

    # ── narrow-band bookkeeping (TSDF clamp) ───────────────────────────────
    # The raw field is only trustworthy in a ~sdf_clamp*S band at the surface;
    # the planner therefore runs on a propagated ESDF (built in step 4), which
    # has no horizon limit — the clamp only determines the band the ESDF's
    # occupancy is extracted from.
    S = float(args.object_scale)
    ROBOT_RADIUS = args.robot_radius
    D_SAFE = args.buffer
    D_STOP = min(D_STOP, 0.4 * D_SAFE)
    if sdf_clamp > 0:
        print(f"        TSDF clamp delta={sdf_clamp} -> raw field is a "
              f"{sdf_clamp * S * 100:.1f} cm narrow band; planning on the ESDF")

    # ── 2. object mesh -> normalized greyscale copy ────────────────────────
    print("[2/6] Loading + normalizing object mesh ...")
    if args.mesh:
        mesh_path = os.path.abspath(os.path.expanduser(args.mesh))
    else:
        mesh_path = fetch_objaverse_glb(args.uid, str(out / "objaverse_cache"))
    gt_mesh = load_and_normalize_mesh(mesh_path, radius)
    gt_mesh = trimesh.Trimesh(vertices=gt_mesh.vertices, faces=gt_mesh.faces,
                              process=False)          # greyscale, as in training
    obj_norm_path = str(out / "object_norm.obj")
    gt_mesh.export(obj_norm_path)
    # Winding-repaired copy for the GT metric ONLY: compute_sdf derives the
    # sign from nearest-face normals, and raw GLBs with inconsistent winding
    # produce spurious "deep inside" readings (v6 logged -17.9 cm true
    # clearance at a point the arm never entered). process=True merges
    # vertices so adjacency exists, then fix_normals makes winding consistent.
    gt_metric_mesh = trimesh.Trimesh(vertices=gt_mesh.vertices,
                                     faces=gt_mesh.faces, process=True)
    trimesh.repair.fix_normals(gt_metric_mesh)

    # ── 3. single input image -> triplane (the ONE model pass) ─────────────
    print("[3/6] Rendering input image + extracting triplane ...")
    render_path = str(out / "input_render.png")
    t0 = time.perf_counter()
    render_mesh_to_image(
        gt_mesh, elevation=args.elevation, fov=FOV, size=IMAGE_SIZE,
        azimuth=args.azimuth,
        extrinsics_json_path=str(out / "camera_extrinsics.json"),
    ).save(render_path)
    timing["render_input_s"] = time.perf_counter() - t0
    image_np = np.array(Image.open(render_path).convert("RGB"))
    R_np = load_R_world_from_recon_json_strict(out)

    t0 = time.perf_counter()
    with torch.no_grad():
        triplane = triposr([image_np], device=device)[0].float()
    timing["triplane_s"] = time.perf_counter() - t0
    print(f"        triplane {tuple(triplane.shape)} in {timing['triplane_s']:.2f}s")

    # ── 4. scene registration (hard-coded scale, see ROBOT_DEMO.md #3) ─────
    b0, _b1 = gt_mesh.bounds
    center = np.array([args.object_x, args.object_y, -b0[2] * S])  # rest on plane
    oracle = SDFOracle(sdf_mlp, triplane, R_np, radius, feat_red, n_freqs,
                       use_trip, S, center, device)
    oracle.query_world(np.zeros((4, 3)))                     # CUDA warm-up

    t0 = time.perf_counter()
    oracle.build_esdf(resolution=args.mc_resolution,
                      ground_z_norm=float(b0[2]))
    timing["esdf_build_s"] = time.perf_counter() - t0
    print(f"        ESDF propagated ({args.mc_resolution}^3) in "
          f"{timing['esdf_build_s']:.2f}s")

    ghost_path = None
    if not args.no_ghost:
        print("[4/6] Marching cubes on the predicted SDF (ghost mesh) ...")
        sdf_mesh = reconstruct_mesh_from_triplane(
            sdf_mlp, triplane, radius, feat_red, resolution=args.mc_resolution,
            device=device, n_freqs=n_freqs, R_world_from_trip=R_np,
            use_triplane_features=use_trip)
        if sdf_mesh is not None:
            ghost_path = str(out / "sdf_mlp_mesh.obj")
            sdf_mesh.export(ghost_path)
        else:
            print("        WARNING: no zero crossing — ghost skipped")

    z_path = center[2]                                        # object mid-height
    start = np.array([args.object_x, PATH_START_Y, z_path])
    goal  = np.array([args.object_x, PATH_GOAL_Y, z_path])
    cam_target = [args.object_x - 0.05, 0.0, max(z_path - 0.02, 0.05)]
    side_bias = {"outer": np.array([1.0, 0.0, 0.0]),
                 "inner": np.array([-1.0, 0.0, 0.0]),
                 "over":  np.array([0.0, 0.0, 1.0])}[args.avoid_side]

    # ── 5. episodes ────────────────────────────────────────────────────────
    import imageio
    print("[5/6] Running avoidance episode ...")
    sim = Sim(args.mode, obj_norm_path, S, center, ghost_path)
    res = run_episode(sim, oracle, gt_metric_mesh, start, goal, avoid=True,
                      cam_target=cam_target, side_bias=side_bias)
    sim.close()
    imageio.mimwrite(out / "demo_sdf_avoidance.mp4", res["frames"],
                     fps=TICKS_PER_SEC, quality=8)
    lg = res["log"]
    print(f"        reached={lg['reached']} collided={lg['collided']} "
          f"min pred clearance={min(lg['pred']) * 100:.1f}cm "
          f"min true clearance={min(lg['gt']) * 100:.1f}cm "
          f"esdf={np.mean(lg['esdf_us']):.1f}us avg, "
          f"raw model ref={np.mean(lg['raw_ms']):.2f}ms avg")

    base_log = None
    if not args.no_baseline:
        print("        Running straight-line baseline ...")
        sim = Sim(args.mode, obj_norm_path, S, center, ghost_path)
        base = run_episode(sim, oracle, gt_metric_mesh, start, goal, avoid=False,
                           cam_target=cam_target, side_bias=side_bias)
        sim.close()
        imageio.mimwrite(out / "demo_baseline_straight.mp4", base["frames"],
                         fps=TICKS_PER_SEC, quality=8)
        base_log = base["log"]
        print(f"        baseline collided={base_log['collided']}")

    # ── 6. figures + metrics ───────────────────────────────────────────────
    print("[6/6] Figures + metrics ...")
    make_plots(oracle, gt_metric_mesh, lg, base_log, start, goal,
               args.object_x, z_path, out)

    metrics = {
        "checkpoint": os.path.basename(args.checkpoint),
        "object": args.mesh or args.uid,
        "object_scale_m": S,
        "mode": args.mode,
        "sdf_clamp_delta": sdf_clamp,
        "raw_narrow_band_m": (sdf_clamp * S) if sdf_clamp > 0 else None,
        "buffer_m": D_SAFE,
        "robot_radius_m": ROBOT_RADIUS,
        "triplane_extraction_s": timing["triplane_s"],
        "esdf_build_s": timing["esdf_build_s"],
        "esdf_query_us_mean": float(np.mean(lg["esdf_us"])),
        "esdf_query_us_p95": float(np.percentile(lg["esdf_us"], 95)),
        "raw_model_query_ms_mean": float(np.mean(lg["raw_ms"])),
        "raw_model_query_ms_p95": float(np.percentile(lg["raw_ms"], 95)),
        "esdf_grid_evals": int(args.mc_resolution ** 3),
        "avoidance": {
            "reached_goal": lg["reached"], "collided": lg["collided"],
            "contact_links": lg["contact_links"],
            "first_contact_tick": lg["first_contact_tick"],
            "min_pred_clearance_m": float(min(lg["pred"])),
            "min_true_clearance_m": float(min(lg["gt"])),
            "duration_s": lg["t"][-1] if lg["t"] else 0.0,
        },
        "baseline": (None if base_log is None else {
            "reached_goal": base_log["reached"], "collided": base_log["collided"],
            "contact_links": base_log["contact_links"],
            "min_true_clearance_m": float(min(base_log["gt"])),
        }),
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out / "run_log.json", "w") as f:
        json.dump({"avoidance": lg, "baseline": base_log}, f)
    print(json.dumps(metrics, indent=2))
    print(f"\nDone. Outputs in {out}/")


if __name__ == "__main__":
    main()
