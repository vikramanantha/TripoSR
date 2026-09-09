"""MLP-step micro-profile: the exact op sequence of run_train_fast's stage 2 on
synthetic data at the real size (S=8, N=32768 -> 262,144 pts), sub-staged with
cuda syncs. Then: is forward-mode AD viable for the eikonal gradient?"""
import sys, time, torch, torch.nn.functional as F
sys.path.insert(0, "/home/markiv/TripoSR")
import train_sdf_head as base
import train_sdf_head_fast as fast
dev = torch.device("cuda:0")
torch.backends.cuda.matmul.allow_tf32 = True
S, N, radius, n_freqs = 8, 32768, 0.87, 6
B = S * N
mlp = base.SDFMLP(in_dim=159, hidden_dim=128, n_hidden=6, feat_dim=120, pe_dim=39).to(dev)
opt = torch.optim.AdamW(mlp.parameters(), lr=5e-3, weight_decay=1e-4, fused=True)
trip = torch.randn(S, 3, 40, 64, 64, device=dev)
pts = (torch.rand(S, N, 3, device=dev) * 2 - 1) * radius
R = torch.eye(3, device=dev).expand(S, 3, 3).contiguous()
sdf_gt = torch.randn(B, device=dev) * 0.3

def sync(): torch.cuda.synchronize()
def run(times=None):
    T = {}
    def mark(k, t0): sync(); T[k] = T.get(k, 0) + (time.perf_counter() - t0)
    leaf = trip.detach().requires_grad_(True)
    t0 = time.perf_counter()
    pts_trip = torch.einsum("snj,sij->sni", pts, R)
    feats = fast.query_triplane_features_batched(pts_trip, leaf, radius)
    query_pts = pts_trip.reshape(B, 3).detach().requires_grad_(True)
    pe = base.fourier_encode(query_pts, n_freqs)
    model_in = torch.cat([feats.reshape(B, -1), pe], dim=-1)
    mark("features+pe", t0); t0 = time.perf_counter()
    sdf_pred = mlp(model_in)
    mark("mlp_fwd", t0); t0 = time.perf_counter()
    per_point = base.surface_weighted_se(sdf_pred.clamp(-.1,.1), sdf_gt.clamp(-.1,.1), sigma=0.05, weight_target=sdf_gt)
    with torch.no_grad():
        thr = per_point.mean() + 3.0 * per_point.std(); keep = per_point <= thr
    sdf_loss = per_point[keep].mean() if keep.any() else per_point.mean()
    bce = base.sign_bce_loss(sdf_pred, sdf_gt, alpha=20.0, epsilon=0.02)
    mark("losses", t0); t0 = time.perf_counter()
    grads = torch.autograd.grad(sdf_pred, query_pts, torch.ones_like(sdf_pred), create_graph=True, retain_graph=True)[0]
    eik = ((grads.norm(dim=-1) - 1.0) ** 2).mean()
    mark("eikonal_grad(create_graph)", t0); t0 = time.perf_counter()
    loss = sdf_loss + 1e-3 * eik + 0.1 * bce
    opt.zero_grad(set_to_none=True)
    loss.backward()
    mark("loss.backward(double)", t0); t0 = time.perf_counter()
    torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0); opt.step()
    mark("clip+adamw", t0)
    return T
for _ in range(3): run()
acc = {}
for _ in range(10):
    for k, v in run().items(): acc[k] = acc.get(k, 0) + v
tot = sum(acc.values()) / 10
print(f"MLP STEP TOTAL: {tot*1000:.0f} ms  (B={B:,})")
for k, v in acc.items(): print(f"  {k:28s} {v/10*1000:7.1f} ms  {100*v/10/tot:5.1f}%")

# ---- forward-mode AD feasibility for the eikonal gradient ----
print("\n--- forward-mode (jvp) eikonal feasibility ---")
try:
    from torch.func import jvp
    leaf = trip.detach()
    def f(q):                         # q: (B,3) -> sdf (B,)
        qs = q.reshape(S, N, 3)
        fe = fast.query_triplane_features_batched(qs, leaf, radius).reshape(B, -1)
        return mlp(torch.cat([fe, base.fourier_encode(q, n_freqs)], -1))
    q0 = pts.reshape(B, 3)
    sync(); t0 = time.perf_counter()
    cols = []
    for d in range(3):
        tang = torch.zeros_like(q0); tang[:, d] = 1.0
        _, jd = jvp(f, (q0,), (tang,))
        cols.append(jd)
    g_fwd = torch.stack(cols, -1)
    eik2 = ((g_fwd.norm(dim=-1) - 1.0) ** 2).mean()
    eik2.backward()
    sync(); t_j = time.perf_counter() - t0
    # reference via reverse double-backward
    q1 = q0.clone().requires_grad_(True)
    y = f(q1); g_rev = torch.autograd.grad(y, q1, torch.ones_like(y), create_graph=True)[0]
    print(f"  jvp path works: 3 jvps + backward = {t_j*1000:.0f} ms | max|g_fwd-g_rev| = {(g_fwd-g_rev).abs().max().item():.2e}")
except Exception as e:
    print(f"  jvp path FAILED: {type(e).__name__}: {str(e)[:160]}")
