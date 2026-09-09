"""Functional check of a trained checkpoint: rebuild the exact training-time
model (TripoSR + LoRA on blocks 0-16 + SDFMLP), strict-load the checkpoint,
run it on unseen-UID test samples through the SAME forward path the trainer
uses, and report SDF metrics. Compares against the warm-start checkpoint so
"works" has a reference. Run inside the docker container:
    .venv/bin/python prof/validate_ckpt.py
"""
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/markiv/TripoSR")
import torch
import torch.nn.functional as F

import train_sdf_head as base
import train_sdf_head_fast as fast
from tsr.system import TSR

CKPTS = {
    "v0.66_100k_ep10": "/home/markiv/TripoSR/sdf_checkpoints/sdf_head_v0.66_100k_epoch0010.pt",
    "v0.64_10k_ep70 (warm start)": "/home/markiv/TripoSR/sdf_checkpoints/sdf_head_v0.64_10k_epoch0070.pt",
}
N_SAMPLES = 24
dev = torch.device("cuda:0")
torch.backends.cuda.matmul.allow_tf32 = True

samples_dir = os.path.join(fast.DATASET_DIR, "samples")
meta = json.load(open(os.path.join(fast.DATASET_DIR, "metadata.json")))
radius = float(meta["radius"])

# Same unseen-UID split as training (seed 42 over the first 100k uids).
names = [n for n in os.listdir(samples_dir) if not n.startswith("_tmp")]
uids = sorted({n.split("_az")[0] for n in names})[:100000]
r = random.Random(42); sh = list(uids); r.shuffle(sh)
test_uids = set(sh[:int(len(sh) * 0.2)])
test_names = sorted(n for n in names if n.split("_az")[0] in test_uids)
picks = random.Random(99).sample(test_names, N_SAMPLES)
print(f"{len(test_names):,} unseen-UID views; evaluating {N_SAMPLES} random ones")


def load_sample(nm):
    d = Path(samples_dir) / nm
    pts = torch.load(d / "query_pts.pt", map_location="cpu", weights_only=False).clamp(-radius, radius)
    sdf = torch.load(d / "sdf_gt.pt", map_location="cpu", weights_only=False)
    tok = torch.load(d / "image_tokens.pt", map_location="cpu", weights_only=False)
    R = torch.from_numpy(base.load_R_world_from_recon_json_strict(d)).float()
    return pts, sdf, tok, R


batches = [load_sample(n) for n in picks]

print("building TripoSR + LoRA[0,16) + SDFMLP (same constructors as training)...")
model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
fast.apply_lora_selective(model, 0, 16, 16, 16.0, "all")
model.to(dev).eval()
mlp = base.SDFMLP(in_dim=159, hidden_dim=128, n_hidden=6, use_tanh_output=False,
                  feat_dim=120, pe_dim=39).to(dev).eval()


@torch.no_grad()
def evaluate(tag):
    agg = {"mse_clamped": 0.0, "mse": 0.0, "surface_weighted": 0.0, "sign_acc": 0.0}
    nonfinite = 0
    for pts, sdf, tok, R in batches:
        pts, sdf, tok, R = pts.to(dev), sdf.to(dev), tok.to(dev), R.to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            codes = fast.triposr_forward_from_cached_tokens(model, tok[None])
        trip = codes.float()
        pts_trip = pts @ R.T
        feats = fast.query_triplane_features_batched(pts_trip[None], trip, radius)[0]
        pe = fast.fourier_encode(pts_trip, 6)
        pred = mlp(torch.cat([feats, pe], dim=-1))
        nonfinite += int((~torch.isfinite(pred)).sum())
        agg["mse_clamped"] += F.mse_loss(pred.clamp(-0.1, 0.1), sdf.clamp(-0.1, 0.1)).item()
        agg["mse"] += F.mse_loss(pred, sdf).item()
        agg["surface_weighted"] += base.surface_weighted_mse_loss(pred, sdf, sigma=0.05).item()
        agg["sign_acc"] += (torch.sign(pred) == torch.sign(sdf)).float().mean().item()
    n = len(batches)
    print(f"  {tag:30s} " + "  ".join(f"{k}={v / n:.5f}" for k, v in agg.items())
          + f"  non-finite preds={nonfinite}")


for tag, path in CKPTS.items():
    t0 = time.time()
    ck = torch.load(path, map_location=dev, weights_only=False)
    r1 = model.load_state_dict(ck["lora_model"], strict=True)
    r2 = mlp.load_state_dict(ck["model"], strict=True)
    print(f"\n[{tag}] epoch={ck['epoch']} strict-load OK (triposr: {len(ck['lora_model'])} keys, "
          f"mlp: {len(ck['model'])} keys) in {time.time() - t0:.1f}s")
    evaluate(tag)
    del ck
print("\ndone")
