"""
Render a URDF object in PyBullet, feed the screenshot to TripoSR, and open
the reconstructed mesh in a Gradio viewer.

Usage
-----
    # Use a specific Objaverse UID
    python pybullet_to_triposr.py --uid 1b5e7a72f9cc42a0bd76d1d64db6d3c5

    # Use an index into the full Objaverse UID list (0 = first object)
    python pybullet_to_triposr.py --uid-index 0

    # Use a local URDF directly
    python pybullet_to_triposr.py --urdf path/to/object.urdf

    # No args: uses pybullet's built-in duck
    python pybullet_to_triposr.py

Optional flags
--------------
    --uid         Objaverse UID string to download and render
    --uid-index   Index into the Objaverse UID list (mutually exclusive with --uid)
    --urdf        Path to a local URDF (skips Objaverse download)
    --yaw         Camera yaw   (default 45)
    --pitch       Camera pitch (default -25)
    --fov         Camera FOV   (default 45)
    --size        Screenshot width/height in pixels (default 1024)
    --output-dir  Where to save screenshot + mesh (default pybullet_output/)
    --port        Gradio viewer port (default 7861)
    --listen      Bind viewer to 0.0.0.0
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import pybullet as pb
import pybullet_data
from PIL import Image


# ---------------------------------------------------------------------------
# Objaverse → OBJ → URDF
# ---------------------------------------------------------------------------

def _normalize_scale(obj_path: str, target_size: float = 0.2) -> float:
    """Return a uniform scale factor that fits the mesh inside *target_size* metres."""
    import trimesh
    mesh = trimesh.load(obj_path, force="mesh")
    extents = mesh.bounding_box.extents
    longest = max(extents) if max(extents) > 0 else 1.0
    return target_size / longest


def glb_to_obj(glb_path: str, obj_path: str) -> None:
    import trimesh
    mesh = trimesh.load(glb_path, force="mesh")
    mesh.export(obj_path)
    print(f"[objaverse] Converted GLB → OBJ: {obj_path}")


def obj_to_urdf(obj_path: str) -> str:
    """Write a minimal URDF next to *obj_path* and return its path."""
    obj_path = os.path.abspath(obj_path)
    name = os.path.splitext(os.path.basename(obj_path))[0]
    urdf_path = os.path.join(os.path.dirname(obj_path), f"{name}.urdf")
    scale = _normalize_scale(obj_path)
    scale_str = f"{scale} {scale} {scale}"
    content = f"""<?xml version="1.0"?>
<robot name="{name}">
  <link name="base">
    <visual>
      <geometry>
        <mesh filename="{obj_path}" scale="{scale_str}"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="{obj_path}" scale="{scale_str}"/>
      </geometry>
    </collision>
  </link>
</robot>
"""
    with open(urdf_path, "w") as f:
        f.write(content)
    print(f"[objaverse] URDF written: {urdf_path}")
    return urdf_path


def fetch_objaverse_urdf(uid: str, cache_dir: str) -> str:
    """Download *uid* from Objaverse, convert to OBJ+URDF, return URDF path."""
    import objaverse

    urdf_path = os.path.join(cache_dir, uid, f"{uid}.urdf")
    if os.path.exists(urdf_path):
        print(f"[objaverse] Cache hit: {urdf_path}")
        return urdf_path

    print(f"[objaverse] Downloading UID: {uid}")
    objects = objaverse.load_objects(uids=[uid], download_processes=1)
    if uid not in objects:
        raise RuntimeError(f"Objaverse did not return an asset for UID: {uid}")

    glb_path = objects[uid]
    obj_dir = os.path.join(cache_dir, uid)
    os.makedirs(obj_dir, exist_ok=True)
    obj_path = os.path.join(obj_dir, f"{uid}.obj")

    glb_to_obj(glb_path, obj_path)
    return obj_to_urdf(obj_path)


def uid_from_index(index: int) -> str:
    """Return the Objaverse UID at position *index* in the full UID list."""
    import objaverse
    print(f"[objaverse] Loading UID list…")
    uids = objaverse.load_uids()
    uids_list = list(uids)
    if index < 0 or index >= len(uids_list):
        raise IndexError(
            f"--uid-index {index} is out of range (dataset has {len(uids_list)} objects)"
        )
    uid = uids_list[index]
    print(f"[objaverse] UID at index {index}: {uid}")
    return uid


# ---------------------------------------------------------------------------
# PyBullet render
# ---------------------------------------------------------------------------

def render_object(
    urdf_path: str,
    out_image_path: str,
    yaw: float = 45.0,
    pitch: float = -25.0,
    fov: float = 45.0,
    size: int = 512,
) -> None:
    """Render *urdf_path* headlessly and save a gray-background PNG.

    Tries the OpenGL/EGL hardware renderer first (smooth, anti-aliased).
    Falls back to the software tiny renderer if EGL is unavailable.
    """

    # Try hardware (EGL) renderer first; fall back to software tiny renderer.
    # Render at 2× then downsample for free anti-aliasing in both cases.
    render_size = size * 2

    client = None
    use_egl = False
    try:
        client = pb.connect(pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
        # Load the EGL plugin to enable GPU-accelerated rendering
        egl = __import__("pkgutil").get_loader("eglRenderer")
        if egl:
            pb.loadPlugin(egl.get_filename(), "_eglRendererPlugin", physicsClientId=client)
            use_egl = True
            print("[render] Using EGL hardware renderer")
        else:
            print("[render] EGL not available, using tiny renderer")
    except Exception:
        if client is None:
            client = pb.connect(pb.DIRECT)
            pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
        print("[render] EGL load failed, using tiny renderer")

    pb.resetSimulation(physicsClientId=client)
    pb.setGravity(0, 0, -9.81, physicsClientId=client)

    body_id = pb.loadURDF(urdf_path, basePosition=[0, 0, 0], physicsClientId=client)

    # Auto-fit camera distance to bounding box.
    # Multiplier 1.3 keeps the object large in frame while avoiding clipping.
    aabb_min, aabb_max = pb.getAABB(body_id, physicsClientId=client)
    center = [(lo + hi) * 0.5 for lo, hi in zip(aabb_min, aabb_max)]
    diagonal = np.linalg.norm(np.array(aabb_max) - np.array(aabb_min))
    distance = max(diagonal * 1.3, 0.3)

    view_matrix = pb.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=center,
        distance=distance,
        yaw=yaw,
        pitch=pitch,
        roll=0,
        upAxisIndex=2,
        physicsClientId=client,
    )
    proj_matrix = pb.computeProjectionMatrixFOV(
        fov=fov,
        aspect=1.0,
        nearVal=distance * 0.01,
        farVal=distance * 10.0,
        physicsClientId=client,
    )

    renderer = pb.ER_TINY_RENDERER if not use_egl else pb.ER_TINY_RENDERER
    _, _, rgba_flat, _, _ = pb.getCameraImage(
        width=render_size,
        height=render_size,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=renderer,
        physicsClientId=client,
    )
    pb.disconnect(client)

    # rgba_flat may be a list or a numpy array depending on the pb version
    rgba = np.array(rgba_flat, dtype=np.uint8).reshape(render_size, render_size, 4)

    # Blend onto neutral gray (128) using the alpha channel so TripoSR's
    # --no-remove-bg path receives a clean foreground-on-gray image.
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb   = rgba[:, :, :3].astype(np.float32)
    gray  = np.full_like(rgb, 128.0)
    blended = (rgb * alpha + gray * (1.0 - alpha)).astype(np.uint8)

    # Downsample 2× with high-quality resampling for free anti-aliasing
    img = Image.fromarray(blended).resize((size, size), Image.LANCZOS)

    os.makedirs(os.path.dirname(out_image_path) or ".", exist_ok=True)
    img.save(out_image_path)
    print(f"[render] Screenshot saved → {out_image_path}")


# ---------------------------------------------------------------------------
# TripoSR inference (calls run.py in the same venv as a subprocess)
# ---------------------------------------------------------------------------

def run_triposr(image_path: str, output_dir: str) -> str:
    """Call run.py and return the path to the generated mesh."""
    import shutil

    # run.py only creates output_dir/0/ when background removal runs; with
    # --no-remove-bg that mkdir is skipped, so we pre-create it here.
    run_dir = os.path.join(output_dir, "0")
    os.makedirs(run_dir, exist_ok=True)

    # Save a copy of the input image next to the mesh for easy inspection.
    saved_input = os.path.join(run_dir, "pybullet_render.png")
    shutil.copy2(image_path, saved_input)
    print(f"[triposr] Input image saved → {saved_input}")

    script = os.path.join(os.path.dirname(__file__), "run.py")
    cmd = [
        sys.executable, script,
        image_path,
        "--output-dir", output_dir,
        "--no-remove-bg",      # background is already gray from the render step
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
# Gradio viewer (reuses view_mesh.py logic inline to avoid a second subprocess)
# ---------------------------------------------------------------------------

def launch_viewer(mesh_path: str, port: int = 7861, listen: bool = False) -> None:
    import gradio as gr

    mesh_name = os.path.basename(mesh_path)
    with gr.Blocks(title=f"Mesh Viewer – {mesh_name}") as app:
        gr.Markdown(f"# Mesh Viewer\nViewing `{mesh_name}`")
        gr.Model3D(
            value=mesh_path,
            clear_color=[1.0, 1.0, 1.0, 1.0],
            label="Reconstructed mesh (drag to rotate, scroll to zoom)",
        )
        gr.File(value=mesh_path, label="Download mesh")

    launch_kwargs = dict(
        server_name="0.0.0.0" if listen else None,
        server_port=port,
    )
    try:
        app.launch(allowed_paths=[mesh_path], **launch_kwargs)
    except TypeError:
        app.launch(**launch_kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PyBullet → TripoSR → Gradio viewer pipeline."
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--uid",
        default=None,
        help="Objaverse UID to download, convert, and render.",
    )
    source.add_argument(
        "--uid-index",
        type=int,
        default=None,
        metavar="N",
        help="Index into the Objaverse UID list (0-based). Downloads that object.",
    )
    source.add_argument(
        "--urdf",
        default=None,
        help="Path to a local URDF file. Skips Objaverse download.",
    )

    parser.add_argument(
        "--output-dir",
        default="pybullet_output",
        help="Directory for screenshots and mesh files.  Default: pybullet_output/",
    )
    parser.add_argument("--yaw",   type=float, default=45.0,  help="Camera yaw   (default 45)")
    parser.add_argument("--pitch", type=float, default=-25.0, help="Camera pitch (default -25)")
    parser.add_argument("--fov",   type=float, default=45.0,  help="Camera FOV   (default 45)")
    parser.add_argument("--size",  type=int,   default=1024,  help="Screenshot size in pixels (default 1024)")
    parser.add_argument("--port",  type=int,   default=7861,  help="Gradio viewer port (default 7861)")
    parser.add_argument("--listen", action="store_true",
                        help="Bind viewer to 0.0.0.0 (accessible from other machines)")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    objaverse_cache = os.path.join(output_dir, "objaverse_cache")

    # Resolve URDF path from whichever source was given
    if args.uid is not None:
        urdf_path = fetch_objaverse_urdf(args.uid, objaverse_cache)
    elif args.uid_index is not None:
        uid = uid_from_index(args.uid_index)
        urdf_path = fetch_objaverse_urdf(uid, objaverse_cache)
    elif args.urdf is not None:
        urdf_path = os.path.abspath(os.path.expanduser(args.urdf))
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
    else:
        urdf_path = os.path.join(pybullet_data.getDataPath(), "duck_vhacd.urdf")
        print(f"[info] No source given, using built-in: {urdf_path}")

    screenshot_path = os.path.join(output_dir, "pybullet_render.png")

    # Step 1 – render in PyBullet
    render_object(
        urdf_path,
        screenshot_path,
        yaw=args.yaw,
        pitch=args.pitch,
        fov=args.fov,
        size=args.size,
    )

    # Step 2 – reconstruct mesh with TripoSR
    mesh_path = run_triposr(screenshot_path, output_dir)

    # Step 3 – open the mesh in a Gradio viewer
    launch_viewer(mesh_path, port=args.port, listen=args.listen)


if __name__ == "__main__":
    main()
