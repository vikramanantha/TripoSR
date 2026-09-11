"""Time fd_gradient_terms fwd+bwd at S=8 for K in {2048,4096,8192} (container)."""
import sys, time, types, torch
sys.path.insert(0, "/home/markiv/TripoSR")
import train_sdf_head as base, train_sdf_head_fast as fast
dev = torch.device("cuda:0"); torch.backends.cuda.matmul.allow_tf32 = True
S, N, radius = 8, 32768, 0.87
mlp = base.SDFMLP(in_dim=159, hidden_dim=128, n_hidden=6, feat_dim=120, pe_dim=39).to(dev)
leaf = torch.randn(S, 3, 40, 64, 64, device=dev, requires_grad=True)
pts = (torch.rand(S, N, 3, device=dev) * 2 - 1) * radius
gt = torch.rand(S, N, device=dev) * 0.4 - 0.2
nrm = torch.nn.functional.normalize(torch.randn(S, N, 3, device=dev), dim=-1)
R = torch.eye(3, device=dev).expand(S, 3, 3)
for K in (2048, 4096, 8192):
    args = types.SimpleNamespace(eikonal_fd_points=K, eikonal_band_only=True, sdf_clamp=0.1,
                                 use_triplane_features=True, normal_loss_weight=1e-3, normal_loss_threshold=0.05)
    for i in range(13):
        if i == 3: torch.cuda.synchronize(); t0 = time.perf_counter()
        e, n, _ = fast.fd_gradient_terms(mlp, leaf, pts, gt, nrm, R, radius, 6, 0.01, args)
        (1e-3 * e + 1e-3 * n).backward()
    torch.cuda.synchronize(); print(f"K={K}: {(time.perf_counter() - t0) / 10 * 1000:.0f} ms fwd+bwd (S=8)")
