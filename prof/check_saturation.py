"""Is the SDF head saturated past the TSDF clamp? For real samples and each
checkpoint: prediction distribution, fraction with |pred| > SDF_CLAMP (where the
clamped loss has ZERO gradient), sign accuracy inside/outside, correlation with
GT near the surface, and the fraction of points whose per-point clamped loss
still has a nonzero gradient w.r.t. the prediction."""
import os, random, sys, json
from pathlib import Path
sys.path.insert(0, "/home/markiv/TripoSR")
import torch
import train_sdf_head as base
import train_sdf_head_fast as fast

dev = torch.device("cuda:0")
sd = Path(fast.DATASET_DIR) / "samples"
meta = json.load(open(Path(fast.DATASET_DIR) / "metadata.json")); radius = float(meta["radius"])
names = [n for n in os.listdir(sd) if not n.startswith("_tmp")]
picks = random.Random(5).sample(names, 6)
CK = sys.argv[1:] or ["sdf_head_v0.64_1k_epoch0425.pt", "sdf_head_v0.64_10k_epoch0070.pt", "sdf_head_v0.66_100k_epoch0010.pt"]
C = 0.1
mlps = {}
for c in CK:
    m = base.SDFMLP(in_dim=159, hidden_dim=128, n_hidden=6, use_tanh_output=False, feat_dim=120, pe_dim=39).to(dev).eval()
    m.load_state_dict(torch.load(f"/home/markiv/TripoSR/sdf_checkpoints/{c}", map_location=dev, weights_only=False)["model"], strict=True)
    mlps[c] = m

agg = {c: [] for c in CK}
for nm in picks:
    d = sd / nm
    pts = torch.load(d / "query_pts.pt", map_location=dev, weights_only=False).clamp(-radius, radius)
    gt = torch.load(d / "sdf_gt.pt", map_location=dev, weights_only=False)
    R = torch.from_numpy(base.load_R_world_from_recon_json_strict(d)).float().to(dev)
    trip = torch.load(d / "triplane.pt", map_location=dev, weights_only=False).float()[None]
    q = pts @ R.T
    feats = fast.query_triplane_features_batched(q[None], trip, radius)[0]
    x = torch.cat([feats, fast.fourier_encode(q, 6)], -1)
    for c, m in mlps.items():
        p = m(x.clone().requires_grad_(True))
        pe = m(x)
        with torch.no_grad():
            inside = gt < 0; near = gt.abs() < 0.05
            sat = (pe.abs() > C).float().mean()
            sa_in = (torch.sign(pe[inside]) == -1).float().mean() if inside.any() else torch.tensor(float('nan'))
            sa_out = (torch.sign(pe[~inside]) == 1).float().mean()
            corr = torch.corrcoef(torch.stack([pe[near], gt[near]]))[0, 1] if near.sum() > 2 else torch.tensor(float('nan'))
        # gradient of the clamped, surface-weighted per-point loss w.r.t. the prediction
        per_point = base.surface_weighted_se(p.clamp(-C, C), gt.clamp(-C, C), sigma=0.05, weight_target=gt)
        gp = torch.autograd.grad(per_point.sum(), p)[0]
        live = (gp != 0).float().mean()
        agg[c].append((pe.min().item(), pe.max().item(), pe.std().item(), sat.item(), sa_in.item(), sa_out.item(),
                       corr.item(), live.item(), inside.float().mean().item(), (gt.abs() < C).float().mean().item()))
print(f"{'checkpoint':34s} {'pred_min':>9} {'pred_max':>9} {'pred_std':>8} {'|p|>0.1':>8} {'sign_in':>8} {'sign_out':>8} {'corr_near':>9} {'grad_live':>9}")
for c in CK:
    a = torch.tensor(agg[c]).mean(0).tolist()
    print(f"{c:34s} {a[0]:9.3f} {a[1]:9.3f} {a[2]:8.3f} {a[3]:8.1%} {a[4]:8.1%} {a[5]:8.1%} {a[6]:9.3f} {a[7]:9.1%}")
a = torch.tensor(agg[CK[0]]).mean(0).tolist()
print(f"\nGT reference over these samples: {a[8]:.1%} of points inside, {a[9]:.1%} within the +-0.1 clamp band")
print("grad_live = fraction of points whose clamped SDF loss still passes gradient to the prediction")
