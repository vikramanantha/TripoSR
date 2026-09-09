"""Backbone fwd+bwd cost per LoRA configuration, isolated from the dataset/MLP.
Same conditions as run_train_fast: S=8 cached DINO tokens, bf16 autocast,
torch.compile on backbone.forward, backward through the backbone graph."""
import sys, time, math, torch, torch.nn as nn
sys.path.insert(0, "/home/markiv/TripoSR")
import train_sdf_head as base
from tsr.system import TSR

dev = torch.device("cuda:0")
torch.backends.cuda.matmul.allow_tf32 = True
S, NT, C = 8, 1025, 768
tokens = torch.randn(S, NT, C, device=dev)

def lora_attn_qv(attn, rank, alpha):
    attn.to_q = base.LoRALinear(attn.to_q, rank, alpha)
    if attn.to_v is not None:
        attn.to_v = base.LoRALinear(attn.to_v, rank, alpha)

def build(start, end, mode):
    m = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
    for p in m.parameters(): p.requires_grad_(False)
    blocks = m.backbone.transformer_blocks
    for i in range(start, min(end, len(blocks))):
        b = blocks[i]
        if mode == "all":
            base._lora_attn(b.attn1, 16, 16.0)
            if b.attn2 is not None: base._lora_attn(b.attn2, 16, 16.0)
            base._lora_ff(b.ff, 16, 16.0)
        elif mode == "qv":
            lora_attn_qv(b.attn1, 16, 16.0)
            if b.attn2 is not None: lora_attn_qv(b.attn2, 16, 16.0)
        elif mode == "attn":   # q,k,v,out on both attentions, no FF
            base._lora_attn(b.attn1, 16, 16.0)
            if b.attn2 is not None: base._lora_attn(b.attn2, 16, 16.0)
    for p in m.post_processor.parameters(): p.requires_grad_(True)
    m.to(dev).train()
    import torch._dynamo as d; d.config.suppress_errors = True
    m.backbone.forward = torch.compile(m.backbone.forward)
    n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return m, n_tr

def step(m):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sc = base.triposr_forward_from_cached_tokens(m, tokens)
    trip = sc.float()
    torch.cuda.synchronize(); t1 = time.perf_counter()
    trip.sum().backward()
    torch.cuda.synchronize(); t2 = time.perf_counter()
    return t1, t2

CONFIGS = [
    ("A: blocks 0-16, all 10 linears (CURRENT)", 0, 16, "all"),
    ("B: blocks 8-16, all 10 linears",          8, 16, "all"),
    ("C: blocks 0-16, q+v only",                0, 16, "qv"),
    ("D: blocks 8-16, q+v only",                8, 16, "qv"),
    ("E: blocks 0-16, attn only (no FF)",       0, 16, "attn"),
    ("F: blocks 12-16, all 10 linears",        12, 16, "all"),
    ("G: NO LoRA (post_processor only) = floor",16, 16, "all"),
]
print(f"{'config':46s} {'fwd ms':>7s} {'bwd ms':>7s} {'total':>7s} {'trainable':>10s}")
for name, s0, s1, mode in CONFIGS:
    m, n_tr = build(s0, s1, mode)
    for _ in range(3): step(m)                       # warmup + compile
    fw, bw = [], []
    for _ in range(10):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        t1, t2 = step(m)
        fw.append(t1 - t0); bw.append(t2 - t1)
    fw.sort(); bw.sort()
    f, b = fw[5] * 1000, bw[5] * 1000
    print(f"{name:46s} {f:7.0f} {b:7.0f} {f+b:7.0f} {n_tr:10,d}", flush=True)
    del m; torch.cuda.empty_cache()
    import torch._dynamo as d; d.reset()
