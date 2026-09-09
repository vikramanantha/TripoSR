import sys, re, torch
sys.path.insert(0, "/home/markiv/TripoSR")
import train_sdf_head as base
P = "/home/markiv/TripoSR/sdf_checkpoints/sdf_head_v0.64_10k_epoch0070.pt"
c = torch.load(P, map_location="cpu", weights_only=False)
print("  keys:", sorted(c.keys()))
print("  epoch:", c.get("epoch"), "| meta n_points:", c["meta"].get("n_points"), "| feat_dim:", c["meta"].get("feat_dim"))
a = c.get("args", {})
print("  trained with:", {k: a.get(k) for k in ("lora_block_start","lora_block_end","lora_rank","hidden_dim","n_hidden","n_freqs","use_triplane_features","run_name")})
# MLP: does the ckpt match what the fast file will build (in=159, hidden=128, n_hidden=6)?
mlp = base.SDFMLP(in_dim=159, hidden_dim=128, n_hidden=6, feat_dim=120, pe_dim=39)
own = mlp.state_dict(); ck = c["model"]
shape_ok = all(k in ck and ck[k].shape == v.shape for k, v in own.items())
print(f"  sdf_mlp: {len(ck)} ckpt keys vs {len(own)} model keys | all present + same shapes: {shape_ok}")
# LoRA coverage
blocks = sorted({int(m.group(1)) for k in c["lora_model"] if "lora_" in k and (m := re.match(r"backbone\.transformer_blocks\.(\d+)\.", k))})
n_lora = sum(1 for k in c["lora_model"] if ".lora_A" in k or ".lora_B" in k)
trained = sum(1 for k, v in c["lora_model"].items() if k.endswith("lora_B") and v.abs().max() > 0)
print(f"  lora_model: {len(c['lora_model'])} keys | adapted blocks {blocks[0]}..{blocks[-1]} ({len(blocks)}) | {n_lora} adapter tensors, {trained} lora_B trained (non-zero)")
print(f"  => with LORA_BLOCK_START=0/END=16/targets=all: expect clean 869/869 load, no discard: {blocks == list(range(16)) and len(c['lora_model']) == 869}")
