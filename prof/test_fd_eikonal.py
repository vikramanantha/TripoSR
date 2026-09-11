"""Verify the v0.69 finite-difference eikonal/normal terms (fd_gradient_terms).
Run in the container:  .venv/bin/python prof/test_fd_eikonal.py [checkpoint] [N_VIEWS]

1. FD gradient vs the EXACT full-field autograd gradient (first-order through
   grid_sample IS supported): median cosine at eps in {0.02,0.01,0.005,0.002,0.0005}.
2. Gradient flow: leaf.grad nonzero fraction, every MLP tensor has a grad, and after
   backward(trip, leaf.grad) every LoRA tensor has a grad.
3. Normal loss on real band points vs random directions (expect ~0.5 vs ~1.0).
4. Cosine of the OLD PE-only autograd gradient vs the true one (the no-op diagnosis).
"""
import argparse, json, os, random, sys, types
from pathlib import Path
sys.path.insert(0, "/home/markiv/TripoSR")
import torch, torch.nn.functional as F
import train_sdf_head as base
import train_sdf_head_fast as fast
from tsr.system import TSR

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint", nargs="?", default="sdf_checkpoints/ablation/sdf_head_abl_control_epoch0025.pt")
ap.add_argument("n_views", nargs="?", type=int, default=8)
a = ap.parse_args()
dev = torch.device("cuda:0")
torch.manual_seed(0)
ck = torch.load(a.checkpoint, map_location=dev, weights_only=False)
sd = Path(fast.DATASET_DIR) / "samples"
radius = float(json.load(open(Path(fast.DATASET_DIR) / "metadata.json"))["radius"])
names = [n for n in os.listdir(sd) if not n.startswith("_tmp")]
uids = sorted({n.split("_az")[0] for n in names})[:10000]
r = random.Random(42); sh = list(uids); r.shuffle(sh)
test_uids = set(sh[:2000])
picks = random.Random(7).sample(sorted(n for n in names if n.split("_az")[0] in test_uids), a.n_views)

model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
lora_params = fast.apply_lora_selective(model, 0, 16, 16, 16.0, "all")
model.load_state_dict(ck["lora_model"], strict=False); model.to(dev).train()
mlp = base.SDFMLP(in_dim=159, hidden_dim=128, n_hidden=6, use_tanh_output=False, feat_dim=120, pe_dim=39).to(dev)
mlp.load_state_dict(ck["model"])
args = types.SimpleNamespace(eikonal_fd_points=4096, eikonal_band_only=True, sdf_clamp=0.1,
                             use_triplane_features=True, normal_loss_weight=1e-3, normal_loss_threshold=0.05,
                             eikonal_fd_eps_start=0.01, eikonal_fd_eps_end=0.002)

pts, gts, nrms, toks, Rs = [], [], [], [], []
for nm in picks:
    d = sd / nm
    pts.append(torch.load(d / "query_pts.pt", map_location=dev, weights_only=False).clamp(-radius, radius))
    gts.append(torch.load(d / "sdf_gt.pt", map_location=dev, weights_only=False))
    nrms.append(torch.load(d / "normal_gt.pt", map_location=dev, weights_only=False))
    toks.append(torch.load(d / "image_tokens.pt", map_location=dev, weights_only=False))
    Rs.append(torch.from_numpy(base.load_R_world_from_recon_json_strict(d)).float().to(dev))
pts, gt_s, nrm_s, tok, R = map(torch.stack, (pts, gts, nrms, toks, Rs))
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    trip = fast.triposr_forward_from_cached_tokens(model, tok).float()
leaf = trip.detach().requires_grad_(True)
pts_trip = torch.einsum("snj,sij->sni", pts, R)
S, N, _ = pts_trip.shape

# ── 1. FD vs exact autograd (full field) on band points ──────────────────────
band = gt_s.abs() < 0.1
def exact_grad(q):  # q (M,3) in triplane frame -> true df/dq incl. triplane branch
    q = q.detach().clone().requires_grad_(True)
    feats = fast.query_triplane_features_batched(q[None], leaf[:1].detach(), radius)[0]
    f = mlp(torch.cat([feats, fast.fourier_encode(q, 6)], -1))
    return torch.autograd.grad(f.sum(), q)[0]
def fd_grad(q, eps):
    offs = torch.eye(3, device=dev) * eps
    qpm = torch.cat([q[:, None, :] + offs, q[:, None, :] - offs], 1).reshape(-1, 3)
    with torch.no_grad():
        feats = fast.query_triplane_features_batched(qpm[None], leaf[:1].detach(), radius)[0]
        f = mlp(torch.cat([feats, fast.fourier_encode(qpm, 6)], -1)).reshape(-1, 6)
    return (f[:, :3] - f[:, 3:]) / (2 * eps)
qb = pts_trip[0][band[0]][:4096]
g_true = exact_grad(qb)
print(f"[1] view 0: {qb.shape[0]} band points | true |grad| mean {g_true.norm(dim=-1).mean():.3f} "
      f"median {g_true.norm(dim=-1).median():.3f}")
for eps in (0.02, 0.01, 0.005, 0.002, 0.0005):
    g = fd_grad(qb, eps)
    cos = F.cosine_similarity(g, g_true, dim=-1)
    print(f"    eps={eps:<7} median cos {cos.median():.3f}  mean cos {cos.mean():.3f}  "
          f"|grad| median {g.norm(dim=-1).median():.3f}")

# ── 4. old PE-only autograd gradient vs true ─────────────────────────────────
q = qb.detach().clone().requires_grad_(True)
with torch.no_grad():
    feats_fixed = fast.query_triplane_features_batched(qb[None], leaf[:1], radius)[0]
f = mlp(torch.cat([feats_fixed, fast.fourier_encode(q, 6)], -1))
g_pe = torch.autograd.grad(f.sum(), q)[0]
cos = F.cosine_similarity(g_pe, g_true, dim=-1)
print(f"[4] OLD PE-only autograd gradient: median cos to true {cos.median():.3f}, "
      f"|grad_pe| median {g_pe.norm(dim=-1).median():.3f} vs true {g_true.norm(dim=-1).median():.3f}")

# ── 2. gradient flow through the real term ───────────────────────────────────
mlp.zero_grad(); model.zero_grad(); leaf.grad = None
eik, nrm_loss, gn = fast.fd_gradient_terms(mlp, leaf, pts_trip, gt_s, nrm_s, R, radius, 6, 0.01, args)
(1e-3 * eik + 1e-3 * nrm_loss).backward()
print(f"[2] eikonal {eik.item():.4f}  normal {nrm_loss.item():.4f}  |grad| on band median {gn.median():.3f} | "
      f"leaf.grad nonzero {float((leaf.grad != 0).float().mean()):.1%} | "
      f"MLP tensors with grad {sum(p.grad is not None for p in mlp.parameters())}/{sum(1 for _ in mlp.parameters())}")
torch.autograd.backward(trip, leaf.grad)
print(f"    LoRA tensors with nonzero grad {sum(p.grad is not None and p.grad.abs().sum() > 0 for p in lora_params)}/{len(lora_params)}")

# ── 3. normal loss vs random directions ──────────────────────────────────────
rand_n = F.normalize(torch.randn_like(nrm_s), dim=-1)
_, nrm_rand, _ = fast.fd_gradient_terms(mlp, leaf.detach(), pts_trip, gt_s, rand_n, R, radius, 6, 0.01, args)
print(f"[3] normal loss real normals {nrm_loss.item():.3f} vs random directions {nrm_rand.item():.3f}")
