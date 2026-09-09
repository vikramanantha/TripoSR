import argparse, sys, time, os
sys.path.insert(0, "/home/markiv/TripoSR")
import train_sdf_head_fast as fast

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
args = argparse.Namespace(
    dataset_dir="/mnt/hostmnt/ws-frb/users/markiv/sdfer/TripoSR/prof_tmp_fast",
    data_source="objaverse", hy3d_mesh_dir="", model="stabilityai/TripoSR",
    n_objects=N, azimuths_per_mesh=5, elevations=[15.0, 30.0],
    near_surface_fraction=0.25, sharp_edge_fraction=0.0, sharp_edge_angle_deg=30.0,
    repair_meshes=True, repair_voxel_res=128, repair_voxel_method="ray",
    n_points=32768, image_size=256, fov=40.0, max_mesh_mb=0.0,
    max_triangles=500_000, verbose=True,
    prep_workers=int(os.environ.get("SDFER_PREP_WORKERS", "4")),
    prefetch=int(os.environ.get("SDFER_PREFETCH", "12")),
)
if __name__ == "__main__":   # REQUIRED: spawn workers re-import the main module
    t0 = time.perf_counter()
    fast.run_precompute_fast(args)
    dt = time.perf_counter() - t0
    print(f"\nFAST TOTAL: {dt:.1f} s for {N} objects = {dt/N:.2f} s/object (incl model load + worker spawn)")
