import sys, os
sys.path.insert(0, "/home/markiv/TripoSR")
import train_sdf_head_fast as f
a = f.build_train_args()
print("  import OK")
print("  DATASET_DIR  =", f.DATASET_DIR)
print("  samples/ present =", os.path.isdir(os.path.join(f.DATASET_DIR, "samples")))
print("  RUN_NAME     =", f.RUN_NAME, "| OUTPUT_DIR =", f.OUTPUT_DIR)
print(f"  lora         = blocks [{a.lora_block_start},{a.lora_block_end}) targets={a.lora_targets} rank={a.lora_rank}")
print(f"  eikonal_frac = {a.eikonal_fraction} | S={a.samples_per_batch} | epochs={a.epochs} | n_objects={a.n_objects}")
print("  init_from    =", a.init_from, "| exists =", os.path.exists(a.init_from) if a.init_from else "n/a")
print("  resume       =", a.resume)
print("  bench mode   =", os.environ.get("SDFER_BENCH_STEPS", "unset -> normal training"))
for fn in ("apply_lora_selective", "load_from_checkpoint", "_load_state_report",
           "run_train_fast", "run_precompute_fast"):
    assert hasattr(f, fn), fn
print("  all new functions present")
