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
    print("[objaverse] Loading UID list…")
    uids_list = list(objaverse.load_uids())
    if not (0 <= index < len(uids_list)):
        raise IndexError(f"--uid-index {index} out of range ({len(uids_list)} objects)")
    uid = uids_list[index]
    print(f"[objaverse] UID at index {index}: {uid}")
    return uid


def fetch_objaverse_glb(uid: str, cache_dir: str) -> str:
    import objaverse
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
    ]
    print(f"[triposr] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    mesh_path = os.path.join(output_dir, "0", "mesh.obj")
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"TripoSR did not produce a mesh at {mesh_path}")
    print(f"[triposr] Mesh saved → {mesh_path}")
    return mesh_path


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

    # Step 1 – render
    render_mesh(
        mesh_path, render_path,
        azimuth=args.azimuth,
        elevation=args.elevation,
        fov=args.fov,
        size=args.size,
    )

    # Step 2 – TripoSR reconstruction
    out_mesh = run_triposr(render_path, output_dir)

    # Step 3 – view (overlay uses camera_extrinsics.json next to render for recon rotation)
    extrinsics_path = os.path.join(output_dir, "camera_extrinsics.json")
    launch_viewer(
        out_mesh,
        mesh_path,
        render_path,
        output_dir,
        port=args.port,
        listen=args.listen,
        camera_extrinsics_path=extrinsics_path if os.path.isfile(extrinsics_path) else None,
    )


if __name__ == "__main__":
    main()
