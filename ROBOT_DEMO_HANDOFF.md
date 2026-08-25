# Robot SDF-Avoidance Demo — Handoff / Continuation Notes

Written 2026-08-19 by Claude at the end of the session that built the demo.
Purpose: everything needed to pick this up cold in a new chat — what exists,
why it looks the way it does, what was learned (including two research-relevant
findings), and how to run/extend it.

Companion doc: [`ROBOT_DEMO.md`](ROBOT_DEMO.md) — the system *design* (planner
choice, scale/extrinsics handling, FLAGS section). This file is the *history +
operations* side. Read both.

---

## 1. What was built

**Goal** (from `journal.md`, "Demo with a robot [Aug 19 2026]"): a video demo
where a robot arm end effector moves from one side of an object to the other,
continuously querying the TripoSR-clone SDF model (one image → triplane → SDF
head) to keep a safety buffer from the obstacle. For Vikram's research
presentation.

**Deliverables** (all in `TripoSR/`):

| File | What |
|---|---|
| `sdf_robot_demo.py` | The whole demo pipeline, one script (~750 lines) |
| `ROBOT_DEMO.md` | Design doc: planner, scale/extrinsics answers, FLAGS |
| `robot_demo_output/` | Final working run (v8): videos, figures, metrics, logs |
| `robot_demo_output_v1..v7log/` | Archived failed iterations (see §4 — some are useful failure visuals for a talk) |

**Final result (v8, all defaults):** goal reached in **6.0 s, zero contacts**,
min believed clearance **6.3 cm** vs min true clearance **5.9 cm** (belief
tracks ground truth to ~4 mm across the whole run). Triplane extraction 0.95 s
+ ESDF propagation 0.65 s, once per scene; per-tick sensing ~9 ms (almost all
of that is the raw-MLP *reference* query kept for logging; the ESDF lookups
the planner actually uses are microseconds). Baseline run without SDF checking
collides (contacts: `panda_hand`, `panda_link5`).

Outputs of a run:
- `demo_sdf_avoidance.mp4` / `demo_baseline_straight.mp4` — 30 fps, 1280×720,
  every frame annotated with believed clearance, true clearance, buffer, query
  latency, tick, status (colour-coded green/orange/red).
- `sdf_slices.png` — top-down + side slice of the ESDF belief (fill), believed
  surface (orange contour), GT surface (black dashed), executed + baseline
  paths. This is the registration-correctness proof: orange vs dashed contours
  coincide to ~voxel accuracy.
- `clearance.png` — believed vs true vs raw-model clearance over time. The
  single most convincing figure: blue (belief) and black-dashed (truth) overlap
  within ~1 cm; grey (raw model output) floats 10–20 cm above them, telling the
  narrow-band story at a glance.
- `metrics.json` — everything quantitative (timings, min clearances, contact
  links, first contact tick).
- `run_log.json` — full per-tick arrays (t, pred, raw, gt, ee position,
  query_ms) for both episodes. Added for debugging; keep it, it's cheap.
- `input_render.png` — the single image the model saw; `camera_extrinsics.json`
  — the R used for recon→world; `object_norm.obj` — normalized greyscale mesh
  (loaded into PyBullet, so sim geometry ≡ model input geometry);
  `sdf_mlp_mesh.obj` — marching-cubes surface of the predicted field (shown in
  the sim as a translucent orange "ghost" over the grey GT object).

---

## 2. How to run

Same environment ritual as `train_sdf_head.py` / `infer_sdf_mesh.py`:

```bash
cd TripoSR/docker && ./run.sh        # enter container (running one is named "markiv")
cd ~/TripoSR && source .venv/bin/activate
python sdf_robot_demo.py                         # all defaults
python sdf_robot_demo.py --uid <objaverse-uid>   # different object
python sdf_robot_demo.py --mesh path/to/obj.glb
python sdf_robot_demo.py --mode floating         # no arm, floating sphere EE
python sdf_robot_demo.py --avoid-side over       # detour over the top instead of +x
python sdf_robot_demo.py --no-baseline --no-ghost
```

Non-interactive from the host (how this session ran everything):

```bash
docker exec markiv bash -c 'cd /home/markiv/TripoSR && CUDA_VISIBLE_DEVICES=1 .venv/bin/python sdf_robot_demo.py'
```

**Environment facts that will bite you:**
- The **host cannot run it**: `import tsr` → torchmcubes → needs
  `libcudart.so.13`, which only exists in the container. PyBullet/pyrender/
  matplotlib parts DO run on the host venv (`TripoSR/.venv/bin/python`) — handy
  for analyzing outputs (reading `run_log.json`, extracting video frames).
- The container mounts `TripoSR/` at `/home/markiv/TripoSR`; host path is
  `/home/markiv/sdfer/TripoSR`. Same files, different prefix. The script uses
  paths relative to its own location, so it works in both.
- EGL GPU rendering works in both host and container (NVIDIA A6000s). The
  script auto-loads the eglRenderer plugin; without it the tiny renderer makes
  video capture painfully slow.
- A run is **deterministic**: no seeds anywhere, PyBullet fixed timestep, fixed
  scene → identical trajectories bit-for-bit on re-run. Great for debugging
  (add logging, rerun, same trajectory).
- Full run takes ~2–4 min wall clock (video capture dominates).
- Everything ran on `CUDA_VISIBLE_DEVICES=1`; any free GPU works (~2 GB).

---

## 3. Architecture of `sdf_robot_demo.py`

```
ONCE per scene:
  mesh (Objaverse UID / local file)
    → load_and_normalize_mesh (centroid→origin, longest edge→1)  [train_sdf_head]
    → greyscale copy (bare Trimesh — matches training domain)
    → render_mesh_to_image (pyrender, grey bg, az=0 el=30 fov=40, 256px)
        writes camera_extrinsics.json → R (recon→world)          [train_sdf_head]
    → TripoSR forward w/ LoRA weights from checkpoint → triplane [infer_sdf_mesh helpers]
    → SDFOracle(triplane, R, scale S, world center c)
    → oracle.build_esdf():  raw field on 128³ grid → occupancy (raw < τ=0.03)
        → stamp ground slab below object base → binary_closing(3)
        → binary_fill_holes → signed EDT → ESDF grid (normalized units)
    → reconstruct_mesh_from_triplane → ghost mesh                [train_sdf_head]

EVERY control tick (30 Hz; 8 physics substeps at 240 Hz):
  control spheres = hand envelope at EE + [0, +6, +12] cm (r = 3/6/6.5 cm)
                    + forearm sphere at panda_link5's live position (r = 7 cm)
  oracle.sense(): batched trilinear ESDF lookups → min clearance d, gradient n
                  (+ one raw-MLP query, logging only)
  GT check: gt_sdf_signed() on the same sphere centers (magnitude compute_sdf,
            sign from mesh.contains — see finding B)
  planner:  v = blend(goal-dir, slide, 1.3·w·n repulsion, side-bias)
            slide = goal-dir projected onto SDF level-set tangent plane
            w = (d_safe − d)/(d_safe − d_stop);  speed scaled down near obstacle
            d < d_stop → pure retreat along n
            stall watchdog: <3 cm net EE motion in 3 s → side-bias ×3
  target += v·dt, leashed to ≤5 cm from actual EE (anti-windup)
  IK (null-space, elbow-up rest pose) → position control → step physics
  capture + annotate frame
```

Coordinate transforms (the part that must never change silently):
`p_norm = (p_world − c)/S`, then `p_trip = p_norm @ Rᵀ` (training convention,
same as `infer_sdf_mesh.compute_pointwise_sdf_mse`), metric distances = value
× S. Out-of-cube queries (|p_norm| > 0.87) are clamped into the cube with the
clamp distance added on top. Registration is exact **by construction** because
the sim loads the exact normalized mesh at scale S / position c. On a real
robot S, c, R must be estimated — unsolved, flagged in the design doc.

Key tunables (top of the script): `OBJECT_SCALE=0.35`, `OBJECT_XY=(0.45,0)`,
`PATH_START_Y/GOAL_Y=∓0.40`, `D_SAFE=0.09`, `D_STOP=0.03`, `V_MAX=0.20`,
`HAND_SPHERES`, `FOREARM_LINK/RADIUS`, `MC_RESOLUTION=128` (also the ESDF grid),
`UID=97a038cd7a304bce81890c118fadd793` (chunky watertight blob).
Checkpoint default: `sdf_checkpoints/sdf_head_v0.64_1k_epoch0425.pt`.

---

## 4. The debugging history (8 iterations) — read this before "improving" anything

Each failure taught something load-bearing. Archived outputs let you replay.

| Ver | Outcome | Root cause / lesson |
|---|---|---|
| v1 | Arm trapped, pred clearance never <16 cm while truth −51 cm | Default `infer_sdf_mesh` UID (`85739db9…`) is a figure glued to a **huge flat panel** spanning the whole workspace. Also GT sign unreliable on that non-watertight mesh. Lesson: object choice is scene design. |
| v2 | "Reached" but only by physical deflection; pred stuck ≥17 cm | **Finding A** (below): raw field is a narrow band; values 2 cm outside the surface already read 20 cm. Sphere-surface sampling of the raw field can't fix a field that lies. |
| v3 | Belief finally tracked truth (±2 mm) after ESDF; stalled, arm wedged | ESDF works. But scene at scale 0.5 m / x 0.5 is **kinematically infeasible** (outer detour ≈ 0.85 m = reach limit; over-the-top ≈ 0.90 m; even the start pose drapes the forearm across the object). Up-bias deadlocked EE-vs-wrist. |
| v4 | Same stall; learned contacts = `panda_link5` + `panda_hand` | Contact-link logging added. A single thin wrist sphere under-covers the hand; the forearm isn't covered at all. |
| v5 | Same stall (scene still infeasible) | Hand envelope + live forearm sphere + null-space elbow-up IK are right, but no planner fixes impossible kinematics. **Size the scene first**: ‖object center‖ + half-width + buffer + hand < 0.855 m. |
| v6 | Reached in 9 s but **tunneled through the object interior** (true −17.9 cm at one point… or so it seemed) | Two things: (i) predicted shell is **open at the unseen underside** → `binary_fill_holes` leaks → interior believed free; inside the shell the "retreat" gradient points *inward*. (ii) the −17.9 turned out to be partly **Finding B** (GT sign artifact). |
| v7 | Bit-identical to v6 (deterministic sim; ESDF change didn't touch the traversed region) | Added ground-slab stamping (object rests on the floor → seal the bottom before hole-fill; also physically true). Added `run_log.json` for per-tick forensics. |
| v8 | **Works.** 6.0 s, zero contacts, belief ≈ truth ±4 mm | Final pieces: buffer 0.07→0.09 (sliding equilibrium settles ~2–3 cm inside the buffer; real hand pokes past the spheres), radial gain 0.8→1.3, GT metric fixed (Finding B). |

### Finding A — the trained field is a narrow-band surface detector, not a metric SDF

Measured on `sdf_head_v0.64_1k_epoch0425.pt` along a ray through a
reconstructed object (S = 0.5 m):

| true SDF | raw predicted |
|--:|--:|
| −15 … −5 cm (deep inside) | **+12 … +19 cm — interior reads as FREE SPACE** |
| −4 … +0.5 cm | −0.7 … +2.4 cm (accurate band) |
| +1 … +13 cm (outside) | +19 … +24 cm (saturated) |

Cause: TSDF clamp δ=0.1 + surface-weighted loss leave everything outside the
band unconstrained, including the interior sign. Marching cubes still works
(the band has zero crossings), so **reconstruction metrics hide this
completely**. Consequence: planners must consume a propagated ESDF (grid eval →
occupancy → hole-fill → signed EDT), the standard TSDF→ESDF move (cf. Voxblox).
This is a paper-worthy sentence: *reconstruction quality ≠ field quality*.

### Finding B — the GT/label pipeline has a sign artifact near concave edges

`train_sdf_head.compute_sdf` signs distances by dotting with the nearest face's
normal. Verified failure: a point with `mesh.contains = False` (provably
outside, neighbors +6 cm) got **−11.9 cm** — nearest-feature-is-a-concave-edge
case, on a watertight, winding-consistent mesh (`fix_normals` does NOT fix it).
The demo's metric now uses `gt_sdf_signed()` = |compute_sdf| × sign from
ray-parity `contains`. **The training labels still use the raw heuristic** —
worth quantifying how often it mislabels near concave/sharp features, since
those are exactly where v0.6x reconstruction quality is weakest. (For
non-watertight meshes `contains` is also unreliable — the demo's default object
is watertight; the training pipeline's repaired meshes should be too.)

---

## 5. Known imperfections / next steps

1. **ESDF interior still not fully filled** for the default object even with
   the ground slab (occupancy fraction printed at build: 0.244; slices show a
   faint positive pocket deep inside). Harmless here (the pocket is fenced by a
   ≥5 cm-thick believed-negative shell + emergency retreat), but for arbitrary
   objects consider: raise `tau_norm`, more closing iterations, or take
   occupancy = voxels enclosed by the *marching-cubes mesh* (watertight by
   construction) instead of thresholding.
2. **Arm coverage is 4 spheres** (3 hand + forearm link5). Elbow (link3/4) is
   unmodeled. Adding links is cheap: extend the sphere list in `run_episode`
   (`sim.link_pos(i)`); lookups are µs.
3. **The baseline video ends wedged** rather than dramatically plowing through
   (static concave object + position control = stall). Fine for contrast;
   could cap baseline force for a more visible "crash".
4. **Grasp demo (journal demo #1)** not built. Reuses the oracle directly:
   gripper width = ESDF(p_left) + ESDF(p_right) at the zero crossing; the
   pre-positioning is just IK to a believed-surface offset.
5. **Multiple objects**: one triplane + one ESDF each, `min()` compose. The
   fun talking point: SDFs compose by min for free.
6. **Better planner**: CHOMP-style trajectory optimization using ESDF
   gradients (analytic from the grid) — natural next step, cite-able.
7. **Real-robot path**: needs pose+scale registration (depth/fiducial) and
   segmentation/background removal for the input image (stock TripoSR path).
   Both flagged in ROBOT_DEMO.md; neither attempted.
8. **Label-noise audit** from Finding B: sample training points near concave
   edges, compare heuristic sign vs `contains` sign, report the mislabel rate.
9. If a **new checkpoint** changes `sdf_clamp`, `n_freqs`, `radius`, or LoRA
   config — nothing to do, the script reads all of it from the checkpoint.
   If the *training render convention* changes (fov/elevation/size/greyscale),
   update the render constants at the top of the script to match.

## 6. Misc operational notes

- To analyze a run offline (host is fine):
  `TripoSR/.venv/bin/python` + `json.load(open('robot_demo_output/run_log.json'))`;
  video frames via `imageio.v3.imread('demo_sdf_avoidance.mp4')`.
- The `robot_demo_output_v*/` archives can be deleted whenever; v1/v5/v6 are
  the interesting failure videos (wall-trap, infeasible-scene wedge, interior
  tunneling).
- Objaverse downloads cache under `<output-dir>/objaverse_cache/`; other UIDs
  already cached under `infer_output/objaverse_cache/` (pass via `--mesh` to
  skip re-download). Preview contact sheet of those UIDs was generated with
  `load_and_normalize_mesh` + `render_mesh_to_image` — 11 objects; `97a038cd…`
  (blob, watertight) chosen; `86fbd2cf…` and `91ceac9f…` (chairs) are
  watertight alternates but thin-limbed.
- `sdf_robot_demo.py` deliberately imports everything geometry-critical from
  `train_sdf_head.py` / `infer_sdf_mesh.py` (query_triplane_features,
  fourier_encode, render_mesh_to_image, load_R_world_from_recon_json_strict,
  load_sdf_mlp_from_checkpoint, apply_finetuned_lora, …) so the demo cannot
  drift from the training/inference conventions. Keep it that way.
- `journal.md` has NOT been updated with any of this session's work — worth a
  new entry summarizing §1/§4 if the journal is the paper trail.
