"""Rank stored checkpoints as FULL models (each one's own LoRA adapters + its
own SDF head) on unseen-UID views, through the same forward path as training.
Metrics chosen to separate a real SDF from an "everything is outside" prior:
sign accuracy INSIDE and OUTSIDE separately (balanced acc), occupancy IoU,
correlation with GT within +-0.05 of the surface, MAE inside the +-0.1 band,
surface-weighted MSE (the training objective), and prediction std.
Run inside the container:  .venv/bin/python prof/compare_ckpts.py [N_SAMPLES]
"""
import json, os, random, sys
from pathlib import Path
sys.path.insert(0, "/home/markiv/TripoSR")
import torch
import torch.nn.functional as F
import train_sdf_head as base
import train_sdf_head_fast as fast
from tsr.system import TSR

N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
CK = ["sdf_head_v0.64_1k_epoch0100.pt", "sdf_head_v0.64_1k_epoch0425.pt", "sdf_head_v0.64_1k_epoch0500.pt",
      "sdf_head_v0.64_10k_epoch0030.pt", "sdf_head_v0.64_10k_epoch0050.pt", "sdf_head_v0.64_10k_epoch0070.pt",
      "sdf_head_v0.65_1k_epoch0010.pt", "sdf_head_v0.65_1k_epoch0020.pt", "sdf_head_v0.66_100k_epoch0010.pt"]
dev = torch.device("cuda:0")
torch.backends.cuda.matmul.allow_tf32 = True
sd = Path(fast.DATASET_DIR) / "samples"
radius = float(json.load(open(Path(fast.DATASET_DIR) / "metadata.json"))["radius"])
names = [n for n in os.listdir(sd) if not n.startswith("_tmp")]
uids = sorted({n.split("_az")[0] for n in names})[:100000]
r = random.Random(42); sh = list(uids); r.shuffle(sh)
test_uids = set(sh[:int(len(sh) * 0.2)])
seen_by_small_runs = set(uids[:10000])   # the 1k/10k runs trained on prefixes of the sorted uid list
picks = random.Random(7).sample(sorted(n for n in names if n.split("_az")[0] in test_uids
                                       and n.split("_az")[0] not in seen_by_small_runs), N)

def load_sample(nm):
    d = sd / nm
    return (torch.load(d / "query_pts.pt", map_location=dev, weights_only=False).clamp(-radius, radius),
            torch.load(d / "sdf_gt.pt", map_location=dev, weights_only=False),
            torch.load(d / "image_tokens.pt", map_location=dev, weights_only=False),
            torch.from_numpy(base.load_R_world_from_recon_json_strict(d)).float().to(dev))
batches = [load_sample(n) for n in picks]
print(f"{N} unseen-UID views; inside fraction = "
      f"{torch.cat([b[1] for b in batches]).lt(0).float().mean():.3%}", flush=True)

model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
fast.apply_lora_selective(model, 0, 16, 16, 16.0, "all")
model.to(dev).eval()
lora_B = [p for n, p in model.named_parameters() if n.endswith("lora_B")]
mlp = base.SDFMLP(in_dim=159, hidden_dim=128, n_hidden=6, use_tanh_output=False, feat_dim=120, pe_dim=39).to(dev).eval()

@torch.no_grad()
def run(tag):
    M = {k: 0.0 for k in ("sw_mse", "mse_clamp", "sign_in", "sign_out", "iou", "corr_near", "mae_band", "pred_std")}
    for pts, gt, tok, R in batches:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            trip = fast.triposr_forward_from_cached_tokens(model, tok[None]).float()
        q = pts @ R.T
        feats = fast.query_triplane_features_batched(q[None], trip, radius)[0]
        p = mlp(torch.cat([feats, fast.fourier_encode(q, 6)], -1))
        inside, near, band = gt < 0, gt.abs() < 0.05, gt.abs() < 0.1
        pin = p < 0
        M["sw_mse"] += base.surface_weighted_mse_loss(p, gt, sigma=0.05).item()
        M["mse_clamp"] += F.mse_loss(p.clamp(-.1, .1), gt.clamp(-.1, .1)).item()
        M["sign_in"] += pin[inside].float().mean().item() if inside.any() else float("nan")
        M["sign_out"] += (~pin[~inside]).float().mean().item()
        M["iou"] += ((pin & inside).sum() / ((pin | inside).sum().clamp(min=1))).item()
        M["corr_near"] += torch.corrcoef(torch.stack([p[near], gt[near]]))[0, 1].item() if near.sum() > 2 else float("nan")
        M["mae_band"] += (p[band] - gt[band]).abs().mean().item() if band.any() else float("nan")
        M["pred_std"] += p.std().item()
    M = {k: v / len(batches) for k, v in M.items()}
    M["balanced"] = (M["sign_in"] + M["sign_out"]) / 2
    print(f"{tag:30s} sw_mse={M['sw_mse']:.4f} mse_c={M['mse_clamp']:.5f} sign_in={M['sign_in']:6.1%} "
          f"sign_out={M['sign_out']:6.1%} bal={M['balanced']:6.1%} IoU={M['iou']:.3f} "
          f"corr_near={M['corr_near']:+.3f} mae_band={M['mae_band']:.4f} pred_std={M['pred_std']:.3f}", flush=True)

for c in CK:
    ck = torch.load(f"/home/markiv/TripoSR/sdf_checkpoints/{c}", map_location=dev, weights_only=False)
    blocks = fast._lora_blocks_in(ck["lora_model"].keys())
    with torch.no_grad():
        for b in lora_B: b.zero_()                   # blocks absent from this ckpt revert to stock
    res = model.load_state_dict(ck["lora_model"], strict=False)
    assert not res.unexpected_keys, res.unexpected_keys
    mlp.load_state_dict(ck["model"], strict=True)
    tag = c.replace("sdf_head_", "").replace(".pt", "") + f" [lora {min(blocks)}-{max(blocks)+1}]"
    run(tag)
    del ck
print("done")
