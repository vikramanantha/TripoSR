#!/usr/bin/env python3
"""Seamlessly-looping GIF: Panda EE sweeps a closed circle over uneven terrain,
with the live EE->surface distance shown in a panel on the left.

Distances are exact geometry (trimesh nearest-point on the terrain mesh) — no
model involved. Runs on the HOST venv (no `tsr` import):

    TripoSR/.venv/bin/python terrain_circle_gif.py
"""
from __future__ import annotations

import os
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

OUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "terrain_gif_output")
FRAME_W, FRAME_H = 880, 560
N_FRAMES  = 96          # exactly one revolution, no duplicated end frame
FPS       = 24

# The EE traces a circle centred on the base axis. A fixed-base Panda cannot
# rotate through 360 deg (joint 1 spans +-166 deg), so the arm is turntable-
# mounted: the base yaw carries a fixed reaching-down posture all the way
# round. The loop is then exactly periodic and the circle exactly a circle.
CIRCLE_C  = np.array([0.00, 0.00])
CIRCLE_R  = 0.60
BASE_Z    = 0.22        # pedestal height
EE_MARGIN = 0.06        # EE plane clears the highest terrain under the path by this

# Terrain
TER_HALF  = 1.00        # metres, square patch centred at origin
TER_N     = 181         # grid resolution
TER_TOP   = -0.02       # highest terrain point (all terrain sits below z=0)
TER_AMP   = 0.46        # peak-to-peak height variation


# ── terrain ──────────────────────────────────────────────────────────────────

# Bumps and hollows placed around the traced circle, so the EE->surface
# distance swings hard over one revolution. No bump reaches the EE plane, so
# nothing ever obstructs the path.
_BUMPS = [(0.00, 1.00), (0.90, -0.75), (1.75, 0.60), (2.60, -1.00),
          (3.45, 0.85), (4.35, -0.55), (5.25, 0.70), (5.80, -0.85)]


def height(x, y):
    """Smooth, obstacle-free undulation, strongest along the traced circle."""
    h = (0.85 * np.sin(1.55 * x + 0.4) * np.cos(1.30 * y - 0.9)
         + 0.45 * np.sin(2.40 * y + 1.7) * np.cos(2.05 * x + 0.2)
         + 0.20 * np.sin(4.30 * x - 2.1) * np.cos(3.80 * y + 1.1))
    for phi, amp in _BUMPS:
        bx = CIRCLE_C[0] + CIRCLE_R * np.cos(phi)
        by = CIRCLE_C[1] + CIRCLE_R * np.sin(phi)
        h = h + amp * np.exp(-((x - bx) ** 2 + (y - by) ** 2) / 0.055)
    return h


def build_terrain(path_obj: str, path_png: str) -> trimesh.Trimesh:
    ax = np.linspace(-TER_HALF, TER_HALF, TER_N)
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    H = height(X, Y)
    lo, hi = H.min(), H.max()
    scaled = lambda h: (h - hi) / (hi - lo) * TER_AMP + TER_TOP
    H = scaled(H)

    V = np.stack([X.ravel(), Y.ravel(), H.ravel()], axis=1)
    idx = np.arange(TER_N * TER_N).reshape(TER_N, TER_N)
    a, b = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
    c, d = idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    F = [np.stack([a, b, c], 1), np.stack([a, c, d], 1)]

    # UVs for the top surface live in the upper V0..1 band of the texture; the
    # bottom band is a flat neutral colour used by the skirt.
    V0 = 0.12
    UV = [np.stack([np.repeat(np.linspace(0, 1, TER_N), TER_N),
                    V0 + (1 - V0) * np.tile(np.linspace(0, 1, TER_N), TER_N)], 1)]

    # Skirt + bottom, so the patch reads as a solid block of ground rather than
    # a floating sheet. Both windings, so it is visible from any camera.
    z_base = float(H.min() - 0.12)
    walls = [idx[0, :], idx[:, -1], idx[-1, ::-1], idx[::-1, 0]]
    verts, faces = [V], F
    n = TER_N * TER_N
    for w in walls:
        top = V[w]
        bot = top.copy(); bot[:, 2] = z_base
        i0 = n; n += len(top); i1 = n; n += len(bot)
        verts += [top, bot]
        UV.append(np.full((2 * len(top), 2), 0.05))
        k = np.arange(len(top) - 1)
        q = [np.stack([i0 + k, i0 + k + 1, i1 + k + 1], 1),
             np.stack([i0 + k, i1 + k + 1, i1 + k], 1)]
        faces += q + [t[:, ::-1] for t in q]
    corners = np.array([[-TER_HALF, -TER_HALF, z_base], [TER_HALF, -TER_HALF, z_base],
                        [TER_HALF, TER_HALF, z_base], [-TER_HALF, TER_HALF, z_base]])
    verts.append(corners)
    UV.append(np.full((4, 2), 0.05))
    q = [np.array([[n, n + 1, n + 2]]), np.array([[n, n + 2, n + 3]])]
    faces += q + [t[:, ::-1] for t in q]

    V = np.concatenate(verts)
    F = np.concatenate(faces)
    UV = np.concatenate(UV)
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)

    with open(path_obj, "w") as f:
        f.write("\n".join(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}" for p in V))
        f.write("\n")
        f.write("\n".join(f"vt {t[0]:.5f} {t[1]:.5f}" for t in UV))
        f.write("\n")
        f.write("\n".join(
            f"f {t[0]+1}/{t[0]+1} {t[1]+1}/{t[1]+1} {t[2]+1}/{t[2]+1}" for t in F))
        f.write("\n")

    # texture: height colour-ramp + contour lines above, neutral skirt below
    import matplotlib
    matplotlib.use("Agg")
    Ht = H.T[::-1]
    t = (Ht - Ht.min()) / (Ht.max() - Ht.min())
    rgb = matplotlib.colormaps["YlGnBu_r"](0.28 + 0.46 * t)[..., :3]
    rgb = 0.55 * rgb + 0.45 * np.array([0.90, 0.91, 0.93])
    band = np.abs(((Ht - Ht.min()) / 0.060) % 1.0 - 0.5) < 0.045
    rgb[band] *= 0.90
    top = np.array(Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
                   .resize((1024, int(1024 * (1 - V0))), Image.BILINEAR))
    skirt = np.full((1024 - top.shape[0], 1024, 3), (150, 146, 138), np.uint8)
    img = np.concatenate([top, skirt], axis=0)
    Image.fromarray(img).save(path_png)
    return mesh, scaled


# ── sim ──────────────────────────────────────────────────────────────────────

PANDA_EE_LINK = 11
ARM = list(range(7))
LL = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0]
UL = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04]
JR = [u - l for l, u in zip(LL, UL)]
RP = [0.0, -0.4, 0.0, -2.0, 0.0, 1.7, 0.785, 0.04, 0.04]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    obj_p = os.path.join(OUT_DIR, "terrain.obj")
    png_p = os.path.join(OUT_DIR, "terrain_tex.png")
    mesh, scaled = build_terrain(obj_p, png_p)

    # constant EE height: clear the highest terrain point under the traced path
    _th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    _hz = scaled(height(CIRCLE_C[0] + CIRCLE_R * np.cos(_th),
                        CIRCLE_C[1] + CIRCLE_R * np.sin(_th)))
    ee_z = float(_hz.max() + EE_MARGIN)
    print(f"[path] terrain under circle: {_hz.min():.3f} .. {_hz.max():.3f} m "
          f"-> EE plane z = {ee_z:.3f} m")
    print(f"[terrain] {len(mesh.faces)} faces, z in "
          f"[{mesh.vertices[:,2].min():.3f}, {mesh.vertices[:,2].max():.3f}] m")

    import pybullet as pb
    import pybullet_data
    pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    try:
        import pkgutil
        egl = pkgutil.get_loader("eglRenderer")
        if egl:
            pb.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
    except Exception:
        pass
    pb.setGravity(0, 0, 0)
    pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 1)

    vis = pb.createVisualShape(pb.GEOM_MESH, fileName=obj_p,
                               rgbaColor=[1, 1, 1, 1])
    ter = pb.createMultiBody(baseMass=0, baseVisualShapeIndex=vis)
    pb.changeVisualShape(ter, -1, textureUniqueId=pb.loadTexture(png_p))

    # pedestal carrying the turntable base down to the terrain
    z_bot = float(mesh.vertices[:, 2].min()) - 0.02
    ped = pb.createVisualShape(pb.GEOM_CYLINDER, radius=0.068,
                               length=BASE_Z - z_bot,
                               rgbaColor=[0.42, 0.44, 0.48, 1.0])
    pb.createMultiBody(baseMass=0, baseVisualShapeIndex=ped,
                       basePosition=[0, 0, 0.5 * (BASE_Z + z_bot)])
    foot = pb.createVisualShape(pb.GEOM_CYLINDER, radius=0.14, length=0.04,
                                rgbaColor=[0.34, 0.36, 0.40, 1.0])
    pb.createMultiBody(baseMass=0, baseVisualShapeIndex=foot,
                       basePosition=[0, 0, z_bot + 0.06])

    robot = pb.loadURDF("franka_panda/panda.urdf", [0, 0, BASE_Z],
                        useFixedBase=True)
    down = pb.getQuaternionFromEuler([np.pi, 0, 0])

    # nearest-point marker + dotted drop line
    mk = pb.createVisualShape(pb.GEOM_SPHERE, radius=0.024,
                              rgbaColor=[1.0, 0.42, 0.10, 1.0])
    marker = pb.createMultiBody(baseMass=0, baseVisualShapeIndex=mk)
    dot = pb.createVisualShape(pb.GEOM_SPHERE, radius=0.010,
                               rgbaColor=[1.0, 0.55, 0.20, 1.0])
    dots = [pb.createMultiBody(baseMass=0, baseVisualShapeIndex=dot)
            for _ in range(9)]

    # ── posture: solve IK once at yaw 0, then spin the base ─────────────────
    th = 2 * np.pi * np.arange(N_FRAMES) / N_FRAMES
    p0 = np.array([CIRCLE_R, 0.0, ee_z])
    for _ in range(12):
        q = pb.calculateInverseKinematics(
            robot, PANDA_EE_LINK, p0.tolist(), down,
            lowerLimits=LL, upperLimits=UL, jointRanges=JR, restPoses=RP,
            maxNumIterations=400, residualThreshold=1e-7)
        for j_, qj in zip(ARM, q[:7]):
            pb.resetJointState(robot, j_, qj)
    for j_ in (9, 10):
        pb.resetJointState(robot, j_, 0.04)
    ee0 = np.array(pb.getLinkState(robot, PANDA_EE_LINK,
                                   computeForwardKinematics=1)[0])
    print(f"[ik] EE tracking error {np.linalg.norm(ee0 - p0)*1e3:.2f} mm  "
          f"(achieved radius {np.hypot(*ee0[:2]):.3f} m, z {ee0[2]:.3f} m)")

    # The base yaw carries that fixed posture around, so the EE traces an exact
    # circle and frame N wraps onto frame 0 with no discontinuity at all.
    c, s_ = np.cos(th), np.sin(th)
    ee = np.stack([c * ee0[0] - s_ * ee0[1],
                   s_ * ee0[0] + c * ee0[1],
                   np.full(N_FRAMES, ee0[2])], axis=1)

    # ── exact EE->surface distance (batched nearest point on the mesh) ──────
    pq = trimesh.proximity.ProximityQuery(mesh)
    near, dist, _ = pq.on_surface(ee)
    print(f"[dist] {dist.min():.3f} – {dist.max():.3f} m")

    # ── render ──────────────────────────────────────────────────────────────
    # Compose so the arm never sweeps behind the readout panel: offset the
    # camera target along its own left vector, pushing the subject right.
    cam = dict(distance=2.45, yaw=45, pitch=-42, roll=0, upAxisIndex=2)
    v0 = np.array(pb.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0, 0, 0], **cam)).reshape(4, 4, order="F")
    right = v0[0, :3]
    view = pb.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=(-0.26 * right + np.array([0, 0, 0.05])).tolist(), **cam)
    proj = pb.computeProjectionMatrixFOV(fov=48, aspect=FRAME_W / FRAME_H,
                                         nearVal=0.02, farVal=8.0)

    frames = []
    for i in range(N_FRAMES):
        pb.resetBasePositionAndOrientation(
            robot, [0, 0, BASE_Z],
            pb.getQuaternionFromEuler([0, 0, float(th[i])]))
        pb.resetBasePositionAndOrientation(marker, near[i].tolist(), [0, 0, 0, 1])
        for k, body in enumerate(dots):
            f = (k + 1) / (len(dots) + 1)
            pb.resetBasePositionAndOrientation(
                body, (ee[i] + f * (near[i] - ee[i])).tolist(), [0, 0, 0, 1])
        _, _, rgba, _, _ = pb.getCameraImage(
            FRAME_W, FRAME_H, view, proj, shadow=1,
            lightDirection=[0.85, 1.25, 0.95], lightColor=[1.0, 0.99, 0.96],
            lightAmbientCoeff=0.30, lightDiffuseCoeff=0.95,
            lightSpecularCoeff=0.10)
        rgb = np.asarray(rgba, dtype=np.uint8).reshape(FRAME_H, FRAME_W, 4)[..., :3]
        frames.append(panel(rgb, dist[i]))
        if i % 24 == 0:
            print(f"  frame {i}/{N_FRAMES}")

    # One global palette for every frame: no inter-frame palette flicker, and
    # PIL is explicit about the per-frame delay (imageio silently dropped it).
    pal_src = Image.fromarray(np.concatenate(frames[::8], axis=1))
    pal = pal_src.quantize(colors=255, method=Image.MEDIANCUT)
    qs_ = [Image.fromarray(f).quantize(palette=pal, dither=Image.FLOYDSTEINBERG)
           for f in frames]

    gif = os.path.join(OUT_DIR, "ee_terrain_distance.gif")
    qs_[0].save(gif, save_all=True, append_images=qs_[1:], loop=0,
                duration=int(round(1000 / FPS)), disposal=2, optimize=True)

    import imageio.v2 as imageio
    imageio.mimsave(os.path.join(OUT_DIR, "ee_terrain_distance.mp4"),
                    frames, fps=FPS, quality=8)
    print(f"[out] {gif}  ({os.path.getsize(gif)/1e6:.1f} MB, "
          f"{N_FRAMES} frames @ {FPS} fps, {1000//FPS} ms/frame, loops forever)")
    pb.disconnect()


# ── left-centre readout panel ────────────────────────────────────────────────

_F: dict = {}


def font(size: int, bold: bool = False):
    key = (size, bold)
    if key not in _F:
        try:
            from matplotlib import font_manager
            _F[key] = ImageFont.truetype(font_manager.findfont(
                "DejaVu Sans:bold" if bold else "DejaVu Sans"), size)
        except Exception:
            _F[key] = ImageFont.load_default()
    return _F[key]


def panel(frame: np.ndarray, d: float) -> np.ndarray:
    img = Image.fromarray(frame).convert("RGB")
    dr = ImageDraw.Draw(img, "RGBA")
    x0, w, h = 26, 268, 132
    y0 = FRAME_H // 2 - h // 2
    dr.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=14,
                         fill=(14, 16, 20, 205), outline=(255, 120, 40, 220),
                         width=2)
    dr.text((x0 + 22, y0 + 20), "DISTANCE TO SURFACE",
            fill=(190, 196, 205), font=font(16, True))
    dr.text((x0 + 22, y0 + 48), f"{d:.3f}", fill=(255, 255, 255), font=font(52, True))
    dr.text((x0 + 172, y0 + 74), "m", fill=(255, 150, 70), font=font(26, True))
    return np.asarray(img)


if __name__ == "__main__":
    main()
