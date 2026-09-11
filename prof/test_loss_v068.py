"""Verify the v0.68 loss fix (see LOSS_FIX_HANDOFF / sdf_loss_terms in train_sdf_head_fast.py).
Run inside the container:  .venv/bin/python prof/test_loss_v068.py [checkpoint] [N_SAMPLES]

1. sign_bce_loss_v2(clamp=0, balanced=False) == base.sign_bce_loss (bit-identical).
2. Synthetic inside point gt=-0.03 predicted +0.28: old SDF-term gradient is 0, new is > 0.
3. Real pool from a checkpoint's own LoRA + head on unseen-UID views: per-point
   d(loss)/d(pred) under the OLD objective (pred-clamped MSE + unclamped BCE +
   mean+3sd rejection) vs the NEW one: inside points with zero gradient,
   wrong-sign inside points pushed down, outside points beyond +0.1 pushed up.
"""
import argparse, json, os, random, sys, types
from pathlib import Path
sys.path.insert(0, "/home/markiv/TripoSR")
import torch
import torch.nn.functional as F
import train_sdf_head as base
import train_sdf_head_fast as fast

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint", nargs="?", default="sdf_checkpoints/sdf_head_v0.64_10k_epoch0050.pt")
ap.add_argument("n_samples", nargs="?", type=int, default=8)
a = ap.parse_args()
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ── 1. equivalence ───────────────────────────────────────────────────────────
torch.manual_seed(0)
pred = torch.randn(50000) * 0.3
gt = torch.randn(50000) * 0.2
old = base.sign_bce_loss(pred, gt, alpha=20.0, epsilon=0.02)
new = fast.sign_bce_loss_v2(pred, gt, alpha=20.0, epsilon=0.02, clamp=0.0, balanced=False)
print(f"[1] base.sign_bce_loss={old.item():.8f}  v2(clamp=0,balanced=False)={new.item():.8f}  "
      f"-> {'IDENTICAL' if torch.equal(old, new) else 'DIFFERENT (max|d|=%.2e)' % (old - new).abs().item()}")

# ── 2. synthetic absorbing-state point ──────────────────────────────────────
args_new = types.SimpleNamespace(sdf_clamp=0.1, surface_loss_sigma=0.05, loss_reject_k=0.0,
                                 sign_bce_alpha=20.0, sign_bce_epsilon=0.005,
                                 sign_bce_balanced=True, sign_bce_clamp_logits=True)
args_old = types.SimpleNamespace(sdf_clamp=0.1, surface_loss_sigma=0.05, loss_reject_k=3.0,
                                 sign_bce_alpha=20.0, sign_bce_epsilon=0.02,
                                 sign_bce_balanced=False, sign_bce_clamp_logits=False)

def old_sdf_term(p, g, args):
    c = args.sdf_clamp
    return base.surface_weighted_se(p.clamp(-c, c), g.clamp(-c, c), sigma=args.surface_loss_sigma,
                                    weight_target=g)

def old_terms(p, g, args):
    per = old_sdf_term(p, g, args)
    with torch.no_grad():
        keep = per <= per.mean() + args.loss_reject_k * per.std() if args.loss_reject_k > 0 else torch.ones_like(per, dtype=torch.bool)
    sdf = per[keep].mean() if keep.any() else per.mean()
    bce = base.sign_bce_loss(p, g, alpha=args.sign_bce_alpha, epsilon=args.sign_bce_epsilon)
    return sdf, bce

p = torch.tensor([0.28], requires_grad=True); g = torch.tensor([-0.03])
go = torch.autograd.grad(old_sdf_term(p, g, args_old).mean(), p)[0].item()
p2 = torch.tensor([0.28], requires_grad=True)
gn = torch.autograd.grad(fast.sdf_loss_terms(p2, g, args_new)[0], p2)[0].item()
print(f"[2] inside pt gt=-0.03 pred=+0.28: old SDF grad {go:+.4f}, new {gn:+.4f}  "
      f"-> {'OK' if go == 0.0 and gn > 0 else 'UNEXPECTED'}")

# ── 3. real pool ─────────────────────────────────────────────────────────────
from tsr.system import TSR
ck = torch.load(a.checkpoint, map_location=dev, weights_only=False)
sd = Path(fast.DATASET_DIR) / "samples"
radius = float(json.load(open(Path(fast.DATASET_DIR) / "metadata.json"))["radius"])
names = [n for n in os.listdir(sd) if not n.startswith("_tmp")]
uids = sorted({n.split("_az")[0] for n in names})[:10000]
r = random.Random(42); sh = list(uids); r.shuffle(sh)
test_uids = set(sh[:int(len(sh) * 0.2)])
picks = random.Random(7).sample(sorted(n for n in names if n.split("_az")[0] in test_uids), a.n_samples)

model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
fast.apply_lora_selective(model, 0, 16, 16, 16.0, "all")
model.load_state_dict(ck["lora_model"], strict=False)
model.to(dev).eval()
mlp = base.SDFMLP(in_dim=159, hidden_dim=128, n_hidden=6, use_tanh_output=False, feat_dim=120, pe_dim=39).to(dev).eval()
mlp.load_state_dict(ck["model"])

preds, gts = [], []
with torch.no_grad():
    for nm in picks:
        d = sd / nm
        pts = torch.load(d / "query_pts.pt", map_location=dev, weights_only=False).clamp(-radius, radius)
        gt = torch.load(d / "sdf_gt.pt", map_location=dev, weights_only=False)
        tok = torch.load(d / "image_tokens.pt", map_location=dev, weights_only=False)
        R = torch.from_numpy(base.load_R_world_from_recon_json_strict(d)).float().to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            trip = fast.triposr_forward_from_cached_tokens(model, tok[None]).float()
        q = pts @ R.T
        feats = fast.query_triplane_features_batched(q[None], trip, radius)[0]
        preds.append(mlp(torch.cat([feats, fast.fourier_encode(q, 6)], -1)))
        gts.append(gt)
P = torch.cat(preds); G = torch.cat(gts)
print(f"[3] pool: {P.numel():,} pts from {len(picks)} unseen-UID views | pred std {P.std():.4f} "
      f"mean {P.mean():+.4f} | inside frac {(G < 0).float().mean():.3%} | "
      f"sign acc {(torch.sign(P) == torch.sign(G)).float().mean():.3f} "
      f"inside sign acc {((P < 0) & (G < 0)).float().sum() / (G < 0).float().sum():.3f}")

def grads(args, use_old):
    p = P.detach().clone().requires_grad_(True)
    if use_old:
        sdf, bce = old_terms(p, G, args)
    else:
        sdf, bce, _ = fast.sdf_loss_terms(p, G, args)
    loss = sdf + 0.1 * bce
    return torch.autograd.grad(loss, p)[0], sdf.item(), bce.item()

inside = G < 0; wrong_in = inside & (P > 0); out_beyond = (G > 0) & (P > 0.1)
for label, args, use_old in (("OLD", args_old, True), ("NEW", args_new, False)):
    gr, sdfv, bcev = grads(args, use_old)
    print(f"    {label}: sdf_loss={sdfv:.5f} bce={bcev:.5f} | inside pts with ZERO grad "
          f"{(gr[inside] == 0).float().mean():.1%} | wrong-sign inside pushed DOWN "
          f"{(gr[wrong_in] > 0).float().mean():.1%} | outside beyond +0.1 pushed UP "
          f"{(gr[out_beyond] < 0).float().mean():.1%}")
