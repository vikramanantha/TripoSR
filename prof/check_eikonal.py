"""Offline check of the eikonal gradient path: for real samples, compute
d(sdf)/d(position) through the Fourier-PE branch exactly as run_train_fast
does (feats gathered at non-grad positions, PE from a grad-enabled copy), plus
finite differences (a) with feats held fixed = same quantity, and (b) with feats
re-sampled = the TRUE field gradient. Run inside the container:
    .venv/bin/python prof/check_eikonal.py
"""
import os, random, sys, json
from pathlib import Path
sys.path.insert(0, "/home/markiv/TripoSR")
import torch
import train_sdf_head as base
import train_sdf_head_fast as fast

dev = torch.device("cuda:0")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
sd = Path(fast.DATASET_DIR) / "samples"
meta = json.load(open(Path(fast.DATASET_DIR) / "metadata.json")); radius = float(meta["radius"])
names = [n for n in os.listdir(sd) if not n.startswith("_tmp")]
picks = random.Random(5).sample(names, 3)

CK = ["sdf_head_v0.64_1k_epoch0425.pt", "sdf_head_v0.64_10k_epoch0070.pt", "sdf_head_v0.66_100k_epoch0010.pt"]
mlps = {}
for c in CK:
    m = base.SDFMLP(in_dim=159, hidden_dim=128, n_hidden=6, use_tanh_output=False, feat_dim=120, pe_dim=39).to(dev)
    m.load_state_dict(torch.load(f"/home/markiv/TripoSR/sdf_checkpoints/{c}", map_location=dev, weights_only=False)["model"], strict=True)
    m.train()  # same mode as the training loop
    mlps[c] = m

def feats_at(q, trip):
    return fast.query_triplane_features_batched(q[None], trip, radius)[0]

for nm in picks:
    d = sd / nm
    pts = torch.load(d / "query_pts.pt", map_location=dev, weights_only=False).clamp(-radius, radius)
    R = torch.from_numpy(base.load_R_world_from_recon_json_strict(d)).float().to(dev)
    trip = torch.load(d / "triplane.pt", map_location=dev, weights_only=False).float()[None]  # stock triplane on disk
    q0 = (pts @ R.T)[:16384]
    feats = feats_at(q0, trip)
    print(f"\n== sample {nm}  (stock triplane, 16k pts)")
    for c, m in mlps.items():
        # (1) exactly the training-loop eikonal path
        q = q0.detach().requires_grad_(True)
        pred = m(torch.cat([feats, fast.fourier_encode(q, 6)], dim=-1))
        g = torch.autograd.grad(pred, q, torch.ones_like(pred), create_graph=True)[0]
        gn = g.norm(dim=-1)
        # (2) finite differences, feats FIXED (should match (1))
        eps = 1e-3; fd_fixed = []; fd_full = []
        with torch.no_grad():
            for ax in range(3):
                e = torch.zeros(3, device=dev); e[ax] = eps
                fp = m(torch.cat([feats, fast.fourier_encode(q0 + e, 6)], -1)); fm = m(torch.cat([feats, fast.fourier_encode(q0 - e, 6)], -1))
                fd_fixed.append((fp - fm) / (2 * eps))
                fp = m(torch.cat([feats_at(q0 + e, trip), fast.fourier_encode(q0 + e, 6)], -1)); fm = m(torch.cat([feats_at(q0 - e, trip), fast.fourier_encode(q0 - e, 6)], -1))
                fd_full.append((fp - fm) / (2 * eps))
        gfix = torch.stack(fd_fixed, -1).norm(dim=-1); gfull = torch.stack(fd_full, -1).norm(dim=-1)
        print(f"  {c:34s} autograd|g_pe|: mean={gn.mean():.3e} med={gn.median():.3e} max={gn.max():.3e} eik={((gn-1)**2).mean():.4f}"
              f" | FD fixed-feats mean={gfix.mean():.3e} | FD full-field mean={gfull.mean():.3e}")
