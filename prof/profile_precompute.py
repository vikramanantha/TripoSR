"""Stage-level wall-time profile of run_precompute, run against the REAL NFS
destination pattern (separate prof_tmp dir) so file-IO costs are realistic.
Monkeypatches every pipeline stage with a timing wrapper; GPU stages get a
cuda.synchronize inside the wrapper so async launch cost is attributed to the
stage that caused it, not to whoever syncs next."""
import argparse, functools, gc, sys, time, collections
sys.path.insert(0, "/home/markiv/TripoSR")
import torch

import train_sdf_head as base
from tsr.system import TSR

ACC = collections.defaultdict(float)
CNT = collections.defaultdict(int)

def timed(name, fn, sync=False):
    @functools.wraps(fn)
    def w(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            if sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            ACC[name] += time.perf_counter() - t0
            CNT[name] += 1
    return w

# CPU stages
base.download_mesh          = timed("download_mesh", base.download_mesh)
base._load_trimesh          = timed("trimesh_load", base._load_trimesh)
base._normalize_mesh_copy   = timed("normalize", base._normalize_mesh_copy)
base.repair_mesh_watertight = timed("repair_watertight", base.repair_mesh_watertight)
base.sample_query_points    = timed("sample_points", base.sample_query_points)
base.compute_sdf            = timed("compute_sdf", base.compute_sdf)
base.render_mesh_to_image   = timed("render_view", base.render_mesh_to_image)
# watertight property check
import trimesh
_wt = trimesh.Trimesh.is_watertight
class _TimedWT:
    def __get__(self, obj, objtype=None):
        t0 = time.perf_counter()
        try:
            return _wt.__get__(obj, objtype)
        finally:
            ACC["watertight_check"] += time.perf_counter() - t0
            CNT["watertight_check"] += 1
trimesh.Trimesh.is_watertight = _TimedWT()
# GPU stages
TSR.forward                    = timed("tsr_forward_b1", TSR.forward, sync=True)
base.compute_cached_image_tokens = timed("dino_tokens_b1", base.compute_cached_image_tokens, sync=True)
# IO + cleanup
torch.save            = timed("torch_save_nfs", torch.save)
torch.cuda.empty_cache = timed("empty_cache", torch.cuda.empty_cache)
gc.collect            = timed("gc_collect", gc.collect)
base.gc.collect       = gc.collect
from pathlib import Path
_ren = Path.rename
Path.rename           = timed("nfs_rename", _ren)
from PIL import Image
Image.Image.save      = timed("png_save", Image.Image.save)

N_OBJ = int(sys.argv[1]) if len(sys.argv) > 1 else 10
args = argparse.Namespace(
    dataset_dir="/mnt/hostmnt/ws-frb/users/markiv/sdfer/TripoSR/prof_tmp",
    data_source="objaverse", hy3d_mesh_dir="", model="stabilityai/TripoSR",
    n_objects=N_OBJ, azimuths_per_mesh=5, elevations=[15.0, 30.0],
    near_surface_fraction=0.25, sharp_edge_fraction=0.0, sharp_edge_angle_deg=30.0,
    repair_meshes=True, repair_voxel_res=128, repair_voxel_method="ray",
    n_points=32768, image_size=256, fov=40.0, max_mesh_mb=0.0,
    max_triangles=500_000, verbose=False,
)

t0 = time.perf_counter()
base.run_precompute(args)
total = time.perf_counter() - t0

print("\n" + "=" * 74)
print(f"TOTAL wall: {total:.1f} s for {N_OBJ} saved objects "
      f"({total/N_OBJ:.2f} s/object incl. model load)")
print(f"{'stage':22s} {'total s':>9s} {'calls':>7s} {'ms/call':>9s} {'% of wall':>10s}")
acct = 0.0
for name, secs in sorted(ACC.items(), key=lambda kv: -kv[1]):
    n = CNT[name]
    print(f"{name:22s} {secs:9.2f} {n:7d} {1000*secs/max(n,1):9.1f} {100*secs/total:9.1f}%")
    acct += secs
print("-" * 74)
print(f"{'attributed':22s} {acct:9.2f} {'':7s} {'':9s} {100*acct/total:9.1f}%")
print(f"{'unattributed/other':22s} {total-acct:9.2f}")
