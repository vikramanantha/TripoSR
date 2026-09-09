"""Where does the eikonal double-backward spend its 116 ms? Variants:
  V0 current: one forward, eikonal grad + loss.backward through everything
  V1 eikonal path uses leaf.detach() (no 2nd derivative into the triplane)
  V2 eikonal path uses pe-only input (isolates grid_sample's role entirely; NOT a
     candidate - just tells us what grid_sample's double-backward costs)"""
import sys, time, torch
sys.path.insert(0, "/home/markiv/TripoSR")
import train_sdf_head as base, train_sdf_head_fast as fast
dev = torch.device("cuda:0"); torch.backends.cuda.matmul.allow_tf32 = True
S, N, radius, nf = 8, 32768, 0.87, 6; B = S*N
mlp = base.SDFMLP(in_dim=159, hidden_dim=128, n_hidden=6, feat_dim=120, pe_dim=39).to(dev)
mlp_pe = base.SDFMLP(in_dim=39, hidden_dim=256, n_hidden=6, feat_dim=0, pe_dim=39).to(dev)
trip = torch.randn(S,3,40,64,64, device=dev); pts = (torch.rand(S,N,3,device=dev)*2-1)*radius
sdf_gt = torch.randn(B, device=dev)*0.3
def sync(): torch.cuda.synchronize()
def variant(v):
    leaf = trip.detach().requires_grad_(True)
    q = pts.reshape(B,3).detach().requires_grad_(True)
    pe = base.fourier_encode(q, nf)
    feats = fast.query_triplane_features_batched(pts, leaf, radius).reshape(B,-1)
    pred = mlp(torch.cat([feats, pe], -1))
    data = base.surface_weighted_se(pred.clamp(-.1,.1), sdf_gt.clamp(-.1,.1), sigma=.05, weight_target=sdf_gt).mean()
    if v == 0:
        g = torch.autograd.grad(pred, q, torch.ones_like(pred), create_graph=True, retain_graph=True)[0]
    elif v == 1:
        feats_d = fast.query_triplane_features_batched(q.reshape(S,N,3), leaf.detach(), radius).reshape(B,-1)
        pred_e = mlp(torch.cat([feats_d, base.fourier_encode(q, nf)], -1))
        g = torch.autograd.grad(pred_e, q, torch.ones_like(pred_e), create_graph=True)[0]
    else:
        pred_e = mlp_pe(base.fourier_encode(q, nf))
        g = torch.autograd.grad(pred_e, q, torch.ones_like(pred_e), create_graph=True)[0]
    eik = ((g.norm(dim=-1)-1)**2).mean()
    loss = data + 1e-3*eik
    sync(); t0 = time.perf_counter(); loss.backward(); sync()
    return time.perf_counter()-t0, leaf.grad.abs().sum().item()
for v, name in ((0,"V0 current (double-bwd thru grid_sample+triplane)"),(1,"V1 eikonal on leaf.detach() (2 fwds)"),(2,"V2 eikonal thru PE-only MLP (diagnostic)")):
    for _ in range(3): variant(v)
    ts = sorted(variant(v)[0] for _ in range(8))
    print(f"{name:52s} loss.backward = {ts[4]*1000:6.1f} ms")
# whole-step timing for V0 vs V1 (fwd+grad+bwd)
def full(v):
    sync(); t0=time.perf_counter(); variant(v); sync(); return time.perf_counter()-t0
for v in (0,1):
    for _ in range(2): full(v)
    ts = sorted(full(v) for _ in range(8)); print(f"V{v} full MLP step (fwd+eik+bwd): {ts[4]*1000:6.1f} ms")
