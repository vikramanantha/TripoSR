"""Split the MLP step's 116 ms of gradient work, mirroring the REAL loop (feats
from non-grad positions; query_pts detached -> eikonal only via the PE branch):
  full         : as trained today (data->leaf scatter + MLP double-backward)
  no_eik       : drop eikonal  -> isolates data-path backward (incl. scatter)
  no_leaf_grad : leaf frozen   -> isolates MLP fwd/bwd/double-bwd (no scatter)
  jvp          : eikonal via 3 forward-mode jvps, NO create_graph
  jvp+compile  : same with the MLP torch.compiled (now legal: no double backward)"""
import sys, time, torch
sys.path.insert(0, "/home/markiv/TripoSR")
import train_sdf_head as base, train_sdf_head_fast as fast
from torch.func import jvp
dev = torch.device("cuda:0"); torch.backends.cuda.matmul.allow_tf32 = True
S, N, radius, nf = 8, 32768, 0.87, 6; B = S*N
trip = torch.randn(S,3,40,64,64, device=dev); pts = (torch.rand(S,N,3,device=dev)*2-1)*radius
sdf_gt = torch.randn(B, device=dev)*0.3
def sync(): torch.cuda.synchronize()
def make(compiled=False):
    m = base.SDFMLP(in_dim=159, hidden_dim=128, n_hidden=6, feat_dim=120, pe_dim=39).to(dev)
    o = torch.optim.AdamW(m.parameters(), lr=5e-3, weight_decay=1e-4, fused=True)
    return (torch.compile(m) if compiled else m), o
def variant(v, mlp, opt):
    leaf = trip.detach().requires_grad_(v != "no_leaf_grad")
    feats = fast.query_triplane_features_batched(pts, leaf, radius).reshape(B, -1)
    q = pts.reshape(B, 3).detach().requires_grad_(True)
    pred = mlp(torch.cat([feats, base.fourier_encode(q, nf)], -1))
    data = base.surface_weighted_se(pred.clamp(-.1,.1), sdf_gt.clamp(-.1,.1), sigma=.05, weight_target=sdf_gt).mean() \
           + 0.1 * base.sign_bce_loss(pred, sdf_gt, alpha=20., epsilon=.02)
    if v == "no_eik":
        loss = data
    elif v in ("full", "no_leaf_grad"):
        g = torch.autograd.grad(pred, q, torch.ones_like(pred), create_graph=True, retain_graph=True)[0]
        loss = data + 1e-3 * ((g.norm(dim=-1) - 1) ** 2).mean()
    else:  # jvp variants: eikonal through PE branch only (as today), forward-mode
        fc = feats.detach()
        f = lambda qq: mlp(torch.cat([fc, base.fourier_encode(qq, nf)], -1))
        q0 = q.detach()
        cols = []
        for d in range(3):
            t = torch.zeros_like(q0); t[:, d] = 1.0
            cols.append(jvp(f, (q0,), (t,))[1])
        g = torch.stack(cols, -1)
        loss = data + 1e-3 * ((g.norm(dim=-1) - 1) ** 2).mean()
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return g if v not in ("no_eik",) else None
def bench(v, compiled=False):
    mlp, opt = make(compiled)
    for _ in range(3): variant(v, mlp, opt)
    ts = []
    for _ in range(8):
        sync(); t0 = time.perf_counter(); variant(v, mlp, opt); sync(); ts.append(time.perf_counter() - t0)
    ts.sort(); return ts[4] * 1000
res = {}
for v in ("full", "no_eik", "no_leaf_grad", "jvp"):
    try: res[v] = bench(v); print(f"{v:14s} {res[v]:7.1f} ms", flush=True)
    except Exception as e: print(f"{v:14s} FAILED {type(e).__name__}: {str(e)[:100]}", flush=True)
try: res["jvp+compile"] = bench("jvp", compiled=True); print(f"{'jvp+compile':14s} {res['jvp+compile']:7.1f} ms", flush=True)
except Exception as e: print(f"{'jvp+compile':14s} FAILED {type(e).__name__}: {str(e)[:100]}", flush=True)
if "full" in res and "no_eik" in res and "no_leaf_grad" in res:
    print(f"\n=> eikonal double-backward cost ~ {res['full']-res['no_eik']:.0f} ms | grid_sample scatter (leaf grad) ~ {res['full']-res['no_leaf_grad']:.0f} ms")
# correctness of jvp gradient vs reverse-mode
mlp, opt = make(); leaf = trip.detach(); feats = fast.query_triplane_features_batched(pts, leaf, radius).reshape(B,-1).detach()
q = pts.reshape(B,3).detach().requires_grad_(True); pred = mlp(torch.cat([feats, base.fourier_encode(q, nf)], -1))
g_rev = torch.autograd.grad(pred, q, torch.ones_like(pred))[0]
f = lambda qq: mlp(torch.cat([feats, base.fourier_encode(qq, nf)], -1)); q0 = q.detach()
g_fwd = torch.stack([jvp(f, (q0,), (torch.eye(3, device=dev)[d].expand(B,3).contiguous(),))[1] for d in range(3)], -1)
print(f"jvp vs reverse gradient: max|diff| = {(g_fwd-g_rev).abs().max().item():.2e} (rel {(g_fwd-g_rev).abs().max().item()/g_rev.abs().max().item():.2e})")
