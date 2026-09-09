import sys, time, torch
sys.path.insert(0, "/home/markiv/TripoSR")
import train_sdf_head as base
from tsr.system import TSR
dev = torch.device("cuda:0"); torch.backends.cuda.matmul.allow_tf32 = True
tokens = torch.randn(8, 1025, 768, device=dev)
def qv(attn, r, a):
    attn.to_q = base.LoRALinear(attn.to_q, r, a)
    if attn.to_v is not None: attn.to_v = base.LoRALinear(attn.to_v, r, a)
def build(s0, s1, mode, cmode):
    m = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
    for p in m.parameters(): p.requires_grad_(False)
    for i in range(s0, s1):
        b = m.backbone.transformer_blocks[i]
        if mode == "all":
            base._lora_attn(b.attn1,16,16.); base._lora_attn(b.attn2,16,16.); base._lora_ff(b.ff,16,16.)
        else:
            qv(b.attn1,16,16.); qv(b.attn2,16,16.)
    for p in m.post_processor.parameters(): p.requires_grad_(True)
    m.to(dev).train()
    import torch._dynamo as d; d.config.suppress_errors = True
    m.backbone.forward = torch.compile(m.backbone.forward, mode=cmode) if cmode else torch.compile(m.backbone.forward)
    return m
def step(m):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sc = base.triposr_forward_from_cached_tokens(m, tokens)
    t = sc.float(); torch.cuda.synchronize(); t1=time.perf_counter(); t.sum().backward(); torch.cuda.synchronize(); return t1, time.perf_counter()
CF = [("10-16 all  default",10,16,"all",None),("12-16 qv   default",12,16,"qv",None),("14-16 all  default",14,16,"all",None),
      ("12-16 all  default",12,16,"all",None),("12-16 all  reduce-overhead",12,16,"all","reduce-overhead"),("12-16 all  max-autotune",12,16,"all","max-autotune-no-cudagraphs")]
print(f"{'config':30s} {'fwd':>6s} {'bwd':>6s} {'total':>6s}")
for name,s0,s1,mode,cm in CF:
    try:
        m = build(s0,s1,mode,cm)
        for _ in range(3): step(m)
        fw,bw=[],[]
        for _ in range(10):
            torch.cuda.synchronize(); t0=time.perf_counter(); t1,t2=step(m); fw.append(t1-t0); bw.append(t2-t1)
        fw.sort(); bw.sort(); print(f"{name:30s} {fw[5]*1000:6.0f} {bw[5]*1000:6.0f} {(fw[5]+bw[5])*1000:6.0f}", flush=True)
    except Exception as e: print(f"{name:30s} FAILED {type(e).__name__}: {str(e)[:80]}", flush=True)
    import torch._dynamo as d; d.reset(); torch.cuda.empty_cache()
