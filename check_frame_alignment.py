"""
check_frame_alignment.py — Verify that TripoSR's triplane coordinate frame
aligns with the query-point coordinate frame used for SDF ground truth.

For a single precomputed sample:
  1. Load the saved triplane and run TripoSR's own decoder + marching cubes
     to produce a "TripoSR mesh".
  2. Load the original cached mesh, normalize + rotate it the same way
     precompute did, to produce the "GT mesh".
  3. Save both as .ply files side-by-side for visual comparison.
  4. Print bounding-box stats so misalignments are obvious.

Usage:
    python check_frame_alignment.py                       # auto-pick first sample
    python check_frame_alignment.py --uid <uid> --az 36   # specific sample
"""

import argparse
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.modules["torchmcubes"] = MagicMock()
sys.modules["rembg"] = MagicMock()
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

from tsr.system import TSR
from tsr.utils import scale_tensor


def load_and_normalize_mesh(path: str, radius: float):
    import trimesh
    loaded = trimesh.load(path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No triangle geometry in {path}")
        mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise ValueError(f"Unsupported geometry type: {type(loaded)}")
    if len(mesh.faces) == 0:
        raise ValueError("Mesh has no faces")
    mesh.apply_translation(-mesh.centroid)
    longest = max(mesh.extents) if max(mesh.extents) > 0 else 1.0
    mesh.apply_scale((2.0 * radius) / longest)
    return mesh


def rotate_mesh_z(mesh, angle_deg: float):
    import trimesh
    R = trimesh.transformations.rotation_matrix(np.radians(angle_deg), [0, 0, 1])
    rotated = mesh.copy()
    rotated.apply_transform(R)
    return rotated


def extract_triposr_mesh(model, triplane, device, resolution=256, threshold=25.0):
    """Run TripoSR's own decoder + MC on a saved triplane."""
    from tsr.models.isosurface import MarchingCubeHelper
    from tsr.utils import scale_tensor

    scene_code = triplane.float().to(device)
    if scene_code.dim() == 4:
        scene_code_batch = scene_code.unsqueeze(0)
    else:
        scene_code_batch = scene_code

    model.set_marching_cubes_resolution(resolution)
    radius = model.renderer.cfg.radius

    with torch.no_grad():
        grid_pts = scale_tensor(
            model.isosurface_helper.grid_vertices.to(device),
            model.isosurface_helper.points_range,
            (-radius, radius),
        )
        result = model.renderer.query_triplane(model.decoder, grid_pts, scene_code_batch[0])
        density = result["density_act"]

    print(f"  Density stats: min={density.min():.4f}  max={density.max():.4f}  "
          f"mean={density.mean():.4f}  median={density.median():.4f}")
    print(f"  Density > {threshold}: {(density > threshold).sum().item()} / {density.numel()} voxels")

    if (density > threshold).sum() == 0:
        pct = 90.0
        auto_thresh = float(torch.quantile(density.float(), pct / 100.0).item())
        print(f"  WARNING: No voxels above threshold {threshold}. "
              f"Trying auto threshold at p{pct:.0f} = {auto_thresh:.4f}")
        threshold = auto_thresh
        print(f"  Density > {threshold}: {(density > threshold).sum().item()} / {density.numel()} voxels")

    from skimage.measure import marching_cubes
    sdf_vol = -(density.cpu().numpy() - threshold)
    sdf_vol = sdf_vol.reshape(resolution, resolution, resolution)
    try:
        verts, faces, _, _ = marching_cubes(sdf_vol, level=0.0)
    except ValueError:
        print("  ERROR: Marching cubes found no surface even with auto threshold.")
        return None

    voxel_size = 1.0 / (resolution - 1)
    verts_norm = verts * voxel_size
    verts_world = verts_norm * (2 * radius) - radius

    import trimesh
    return trimesh.Trimesh(vertices=verts_world, faces=faces)


def print_bbox(name, mesh):
    verts = np.asarray(mesh.vertices)
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    ctr = (lo + hi) / 2
    ext = hi - lo
    print(f"  [{name}]  verts={len(mesh.vertices):,}  faces={len(mesh.faces):,}")
    print(f"    bbox min : {lo}")
    print(f"    bbox max : {hi}")
    print(f"    center   : {ctr}")
    print(f"    extents  : {ext}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(Path(__file__).parent / "sdf_dataset"))
    parser.add_argument("--model", default="stabilityai/TripoSR")
    parser.add_argument("--uid", default=None, help="Objaverse UID (auto-detect if omitted)")
    parser.add_argument("--az", type=int, default=None, help="Azimuth in degrees (auto-detect if omitted)")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=25.0)
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "frame_check"))
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    samples_dir = dataset_dir / "samples"
    cache_dir = dataset_dir / "mesh_cache"

    with open(dataset_dir / "metadata.json") as f:
        meta = json.load(f)
    radius = meta["radius"]
    print(f"Dataset radius: {radius}")

    # Pick a sample
    if args.uid and args.az is not None:
        sample_name = f"{args.uid}_az{args.az:03d}"
    else:
        sample_name = sorted(os.listdir(samples_dir))[0]
    sample_dir = samples_dir / sample_name
    uid = sample_name.split("_az")[0]
    az_str = sample_name.split("_az")[1]
    az_deg = float(az_str)
    print(f"Sample: {sample_name}  (uid={uid}, az={az_deg})")

    # Load saved triplane
    triplane = torch.load(sample_dir / "triplane.pt", map_location="cpu", weights_only=False).float()
    print(f"Triplane shape: {triplane.shape}")

    # Load query points + GT SDF for reference stats
    query_pts = torch.load(sample_dir / "query_pts.pt", map_location="cpu", weights_only=False)
    sdf_gt = torch.load(sample_dir / "sdf_gt.pt", map_location="cpu", weights_only=False)
    print(f"Query pts range: [{query_pts.min():.4f}, {query_pts.max():.4f}]")
    print(f"SDF GT range:    [{sdf_gt.min():.4f}, {sdf_gt.max():.4f}]")

    # Load TripoSR model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading TripoSR on {device}...")
    model = TSR.from_pretrained(args.model, config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(8192)
    model.to(device).eval()

    print(f"Renderer radius: {model.renderer.cfg.radius}")
    print(f"Feature reduction: {model.renderer.cfg.feature_reduction}")

    # Extract mesh using TripoSR's own decoder
    print(f"Running TripoSR decoder + marching cubes (res={args.resolution})...")
    triposr_mesh = extract_triposr_mesh(model, triplane, device,
                                         resolution=args.resolution,
                                         threshold=args.threshold)

    if triposr_mesh is None:
        print("Cannot proceed — no TripoSR mesh extracted.")
        return

    # Load + rotate GT mesh the same way precompute did
    mesh_files = list(Path(cache_dir / uid).glob("*"))
    if not mesh_files:
        print(f"ERROR: No cached mesh for uid {uid}")
        return
    gt_mesh = load_and_normalize_mesh(str(mesh_files[0]), radius)
    gt_mesh_rot = rotate_mesh_z(gt_mesh, az_deg)

    # Print comparison
    print("\n" + "=" * 60)
    print("BOUNDING BOX COMPARISON")
    print("=" * 60)
    print_bbox("TripoSR decoder mesh", triposr_mesh)
    print_bbox(f"GT mesh (rotated az={az_deg})", gt_mesh_rot)
    print()

    # Check rough alignment
    tri_center = np.mean([triposr_mesh.bounds[0], triposr_mesh.bounds[1]], axis=0)
    gt_center = np.mean([gt_mesh_rot.bounds[0], gt_mesh_rot.bounds[1]], axis=0)
    offset = np.linalg.norm(tri_center - gt_center)
    tri_ext = triposr_mesh.bounds[1] - triposr_mesh.bounds[0]
    gt_ext = gt_mesh_rot.bounds[1] - gt_mesh_rot.bounds[0]
    scale_ratio = np.mean(tri_ext / (gt_ext + 1e-8))
    print(f"Center offset:  {offset:.4f}")
    print(f"Scale ratio:    {scale_ratio:.4f} (1.0 = perfect)")
    if offset > 0.1:
        print("WARNING: Significant center offset — likely a frame mismatch!")
    if abs(scale_ratio - 1.0) > 0.3:
        print("WARNING: Significant scale mismatch!")

    # Save meshes
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    triposr_path = out_dir / f"{sample_name}_triposr.ply"
    gt_path = out_dir / f"{sample_name}_gt_rotated.ply"
    triposr_mesh.export(str(triposr_path))
    gt_mesh_rot.visual = None
    gt_mesh_rot.export(str(gt_path))
    print(f"\nSaved: {triposr_path}")
    print(f"Saved: {gt_path}")
    print("\nOpen both in MeshLab/Blender to visually confirm alignment.")


if __name__ == "__main__":
    main()
