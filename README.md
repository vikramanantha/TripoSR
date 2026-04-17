# TripoSR SDF Head Training

Train a lightweight MLP to predict signed distance fields (SDF) from frozen TripoSR triplane features.

## Scripts

### `train_sdf.sh` — Main entry point

Wrapper script that runs the full pipeline or individual phases.

```bash
./train_sdf.sh                  # precompute + align + train (default)
./train_sdf.sh --precompute     # precompute only
./train_sdf.sh --train          # train only (dataset must exist)
```

Configuration is done by editing the constants at the top of `train_sdf_head.py` (e.g. `N_OBJECTS`, `EPOCHS`, `LR`, `HIDDEN_DIM`, etc.).

### `train_sdf_head.py` — Precompute, align, and train

The main script containing all three pipeline phases. Can also be run directly:

```bash
.venv/bin/python train_sdf_head.py
```

Key configuration variables (top of file):

| Variable | Default | Description |
|---|---|---|
| `COMMAND` | `"both"` | `"precompute"`, `"align"`, `"train"`, or `"both"` |
| `N_OBJECTS` | `10` | Number of Objaverse objects to use |
| `AZIMUTHS_PER_MESH` | `10` | Camera views per object |
| `N_POINTS` | `262144` | Query points sampled per view |
| `EPOCHS` | `500` | Training epochs |
| `LR` | `1e-5` | Learning rate (AdamW) |
| `HIDDEN_DIM` | `64` | MLP hidden layer width |
| `N_HIDDEN` | `10` | Number of hidden layers |
| `EIKONAL_WEIGHT` | `0.0` | Eikonal regularization weight |

**Dataset structure** (created by precompute):
```
sdf_dataset/
├── metadata.json
├── alignments.json          # per-sample ICP transforms (created by align)
├── mesh_cache/<uid>/        # downloaded Objaverse meshes
└── samples/<uid>_az<NNN>/
    ├── triplane.pt          # TripoSR triplane features
    ├── query_pts.pt         # sampled 3D query points
    └── sdf_gt.pt            # ground-truth SDF values
```

### `check_frame_alignment.py` — Verify coordinate frame alignment

Extracts TripoSR's decoder mesh for each sample and ICP-aligns it to the GT mesh. Saves per-azimuth alignment transforms and `.ply` meshes for visual inspection.

```bash
# First precomputed UID, all azimuths
.venv/bin/python check_frame_alignment.py --index 0

# Range of UIDs [0, 5)
.venv/bin/python check_frame_alignment.py --index-range 0 5

# All precomputed UIDs
.venv/bin/python check_frame_alignment.py --all

# Specific UID
.venv/bin/python check_frame_alignment.py --uid <uid>

# Single azimuth
.venv/bin/python check_frame_alignment.py --index 0 --az 36
```

**Output** (per UID):
```
frame_check/<uid>/
├── alignments.json              # per-azimuth transforms
├── az000_gt.ply                 # GT mesh (rotated)
├── az000_triposr.ply            # TripoSR decoder mesh
├── az000_gt_aligned.ply         # GT mesh after ICP alignment
└── ...
```

### `check_multi_azimuth.py` — Analyze orientation across azimuths

Tests whether TripoSR's output tracks input rotation by comparing PCA axes and scale ratios across all azimuths for one UID.

```bash
.venv/bin/python check_multi_azimuth.py                          # auto-pick first UID
.venv/bin/python check_multi_azimuth.py --uid <uid>
.venv/bin/python check_multi_azimuth.py --uid <uid> --resolution 64   # faster
```

### `run.py` — TripoSR image-to-mesh (original)

The original TripoSR inference script. Takes one or more input images and produces 3D meshes.

```bash
.venv/bin/python run.py path/to/image.png
.venv/bin/python run.py image.png --output-dir output/ --model-save-format glb
.venv/bin/python run.py image.png --no-remove-bg --mc-resolution 256
.venv/bin/python run.py image.png --render                       # also save NeRF video
.venv/bin/python run.py image.png --bake-texture                 # bake texture atlas
```

### `render_to_triposr.py` — Mesh render + TripoSR + viewer

End-to-end pipeline: load a mesh (from Objaverse or local file), render it, feed to TripoSR, then open a Gradio viewer comparing original and reconstructed meshes.

```bash
# By Objaverse UID
.venv/bin/python render_to_triposr.py --uid <uid>

# By index into Objaverse UID list
.venv/bin/python render_to_triposr.py --uid-index 0

# Local mesh file
.venv/bin/python render_to_triposr.py --mesh path/to/object.glb

# Custom camera
.venv/bin/python render_to_triposr.py --uid-index 0 --azimuth 90 --elevation 20
```

Opens a Gradio viewer on `localhost:7861` with side-by-side original and reconstructed meshes.

### `extract_triposr_features.py` — Extract raw triplane features

Standalone script to extract the 120-dim triplane features from TripoSR for a given input image, without running the full mesh extraction.

```bash
.venv/bin/python extract_triposr_features.py --image input.png
.venv/bin/python extract_triposr_features.py --image input.png --resolution 64 --output features.pt
```

### `view_mesh.py` — Gradio mesh viewer

Opens any mesh file in a browser-based 3D viewer.

```bash
.venv/bin/python view_mesh.py path/to/mesh.obj
.venv/bin/python view_mesh.py mesh.glb --port 8080
.venv/bin/python view_mesh.py mesh.obj --listen    # accessible from other machines
```

### `render_alignment_plots.py` — Save alignment comparison images

Renders the same 4-column comparison plots as the notebook but applies the **pre-computed** `alignments.json` transforms (identical to how the training pipeline applies them) instead of re-running ICP. Saves one PNG per UID into `frame_check/<uid>/`.

```bash
# All UIDs
.venv/bin/python render_alignment_plots.py

# One UID
.venv/bin/python render_alignment_plots.py --uid <uid>

# Limit to first 3 azimuths per UID (faster)
.venv/bin/python render_alignment_plots.py --max-azimuths 3
```

**Output**: `frame_check/<uid>/alignment_comparison.png` — one row per azimuth with columns: GT | TripoSR | Raw overlay | Aligned overlay.

### `view_meshes.ipynb` — Jupyter notebook for mesh comparison

Visualizes `.ply` files generated by `check_frame_alignment.py`. For each azimuth pair, shows four columns:

1. **GT** — ground truth mesh
2. **TripoSR** — TripoSR decoder mesh
3. **Raw overlay** — both meshes without alignment
4. **ICP-aligned overlay** — GT scaled + rotated + translated to best match TripoSR

```bash
jupyter notebook view_meshes.ipynb
```

## Typical Workflow

```bash
# 1. Full pipeline: precompute + align + train
./train_sdf.sh

# 2. Inspect alignment quality
.venv/bin/python check_frame_alignment.py --all
.venv/bin/python render_alignment_plots.py

# 3. If alignment looks good, train (or retrain)
./train_sdf.sh --train

# 4. Quick test: render a mesh through TripoSR and view
.venv/bin/python render_to_triposr.py --uid-index 0
```
