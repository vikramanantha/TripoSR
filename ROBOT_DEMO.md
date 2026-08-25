# Robot SDF-Avoidance Demo — System Design

> **Status: working.** Final run (defaults, Objaverse `97a038cd…` @ 0.35 m):
> goal reached in 6.0 s, **zero contacts**, min believed clearance 6.3 cm vs
> min true clearance 5.9 cm (belief tracks truth to ~4 mm), triplane 0.95 s +
> ESDF 0.65 s once, then ~9 ms/tick sensing (incl. the raw-MLP reference
> query; the ESDF lookup itself is µs). Baseline without SDF checking collides.
> Outputs in `robot_demo_output/`; earlier failed iterations are archived as
> `robot_demo_output_v*/` — v1 (wall object traps arm), v5 (kinematically
> infeasible scene), v6 (tunnel through unfilled ESDF interior) are useful
> failure-mode illustrations for a talk.

Goal: a video of a robot arm in simulation moving its end effector from one side of an
object to the other. A single image of the object is fed through the fine-tuned
TripoSR + SDF head **once**; during motion the planner queries the learned SDF at
every control tick and steers around the obstacle keeping a safety buffer. This
demonstrates (a) the SDF is accurate enough to navigate by, and (b) queries are fast
enough to sit inside a real-time control loop.

Script: [`sdf_robot_demo.py`](sdf_robot_demo.py)

---

## Answers to the open questions

### 1. What simulator?

**PyBullet.** Reasons:

- It is already installed in `TripoSR/.venv` (pybullet 3.2.7) and already used by
  `pybullet_to_triposr.py`, so nothing new enters the docker image.
- Headless (`DIRECT` mode) with easy offscreen camera capture → video export via
  `imageio` + `imageio-ffmpeg` (both already in the venv).
- Ships a Franka Panda URDF (`pybullet_data/franka_panda/panda.urdf`) with built-in
  IK (`calculateInverseKinematics`), so the arm side is ~30 lines.
- Alternatives considered: MuJoCo (nicer renders, but a new dependency + license
  files + no asset in-tree), Isaac Sim (way too heavy for one demo), Drake
  (overkill). PyBullet is the pragmatic choice; if the presentation later needs
  prettier frames, the same trajectory log can be replayed in anything.

### 2. What path-planning algorithm?

**A potential-field / sliding local planner** (Khatib 1986 style), the simplest
algorithm that *continuously consumes* the SDF — which is exactly the point of the
demo:

Every control tick (~30 Hz), in one batched ESDF lookup:

1. Sense clearance for two control spheres (gripper tip r = 3 cm; wrist bulk
   r = 5.5 cm, 11 cm up the hand): `clearance = ESDF(center) − r`, min over
   spheres, plus a central-difference gradient `n` at the closest sphere. The
   ESDF is propagated once from the raw model field — see the narrow-band
   finding below for why the planner cannot consume the raw field directly.
2. Nominal velocity points straight at the goal.
3. If clearance `d < d_safe` (buffer), blend in:
   - a **sliding** direction — the goal direction projected onto the plane tangent
     to the SDF level set (`dir − (dir·n)n`), which walks the EE *around* the
     obstacle,
   - a **repulsive** term along the SDF gradient `n`, growing as `d → d_stop`, and
   - a small **upward bias**, so a perfectly head-on approach (where the slide
     direction is degenerate) resolves as "go over the top". Avoidance is fully
     3-D; the EE height is only clamped to `[0.05, 0.80]` m.
4. Speed is scaled down near the obstacle; below `d_stop` motion is pure retreat.
5. The target is tracked by the Panda via IK + position control (the planner reads
   back the *actual* EE position each tick, so it reacts to real robot state).

**FINDING (measured, not assumed): the raw learned field is a narrow-band
surface detector, not a metric SDF.** Probing the v0.64_1k checkpoint along a
ray through a reconstructed object (true SDF −15 cm → +13 cm at S = 0.5 m):

| true SDF | raw predicted |
|--:|--:|
| −15 … −5 cm (deep inside) | **+12 … +19 cm** (reads as free space!) |
| −4 … +0.5 cm (surface band) | −0.7 … +2.4 cm (accurate) |
| +1 … +13 cm (outside) | +19 … +24 cm (saturated) |

Only a ~±3 cm shell around the surface is metric. This is a direct consequence
of the v0.64 training recipe — TSDF clamping (δ = 0.1) plus surface-weighted
loss means nothing constrains the field away from the surface, *including the
interior sign*. Marching cubes still works (the shell has zero crossings, so
reconstruction metrics look fine), but **raw queries are unusable for
planning**: a controller stepping 0.7 cm/tick tunnels through the thin shell
into an interior the model calls free. The first two demo iterations failed
exactly this way. This is worth a sentence in the paper: reconstruction-metric
quality does not imply field-metric quality.

**The fix, and it is the standard one: ESDF propagation.** Narrow-band TSDFs
are the normal output of mapping pipelines, and planners never consume them
raw — they propagate a Euclidean SDF from the band (cf. Voxblox). The demo
does the same, once, right after the triplane pass (~1 s for a 128³ grid):

1. Evaluate the raw field on a grid over the triplane cube.
2. Occupancy = region enclosed by the predicted shell (`raw < τ`, τ = 0.02
   normalized ≈ 1 cm, then morphological hole-filling — the τ-inflation is a
   small conservative margin).
3. Signed Euclidean distance transform → ESDF grid; queries are trilinear
   lookups (CPU, ~50 µs for a 26-point batch).

Everything the planner sees is still derived from the single input image; the
buffer no longer has a horizon cap (the ESDF extends everywhere); clearance is
back to the exact `ESDF(center) − radius` form. The clearance plot shows all
three curves — ESDF belief, raw model value, GT — so the narrow-band story is
visible in one figure.

Why not RRT/PRM/CHOMP: sampling planners query the SDF only at plan time (one
batch), which undersells the "runs fast inside the loop" story; CHOMP is the
natural next step (it needs SDF *gradients*, which we get for free) but is more
code than a first demo warrants. The potential-field loop is ~40 lines,
transparent in a talk, and every frame of the video is literally "the robot is
reading the model right now".

### 3. What object?

Any Objaverse UID or local mesh — default is the UID already used by
`infer_sdf_mesh.py` (`85739db9b70e47c28be0340e2d1907b7`). Practical guidance:

- Pick something the v0.64 checkpoint reconstructs well (check in
  `infer_sdf_mesh.py` first — F-score / Chamfer are printed). A chunky convex-ish
  object (couch, toilet, box-like) makes the avoidance arc read clearly on video.
- The demo also drops the *predicted* SDF surface into the scene as a translucent
  orange "ghost" over the grey ground-truth object, so the audience sees the robot
  is navigating with respect to the belief, not the GT.

### 4. How does the model understand scale?

**It doesn't — and can't, from one image (monocular scale ambiguity). The demo
hard-codes the sim→model transform, and that is stated honestly.**

The trained pipeline lives in a *normalized* frame: mesh centered at its centroid,
longest bounding-box edge scaled to 1, SDF valid inside the `[−0.87, 0.87]³`
triplane cube, SDF units = normalized units. The demo constructs the world so the
transform is *exact by construction*:

- The normalized mesh itself is exported and loaded into PyBullet with uniform
  scale `S` (metres per normalized unit, default 0.35 m) at a known position
  `c` (resting on the ground plane).
- World → model: `p_norm = (p_world − c)/S`, then the training-time rotation
  `p_trip = p_norm @ Rᵀ` (from `camera_extrinsics.json`, written by
  `render_mesh_to_image`), then triplane + MLP.
- Metric distance: `SDF_norm × S`.

For a **real** robot later, `c`, `S`, and the camera rotation must be estimated —
e.g. from a depth camera / known fiducial / known camera extrinsics + object
segmentation. That is a genuine open piece of the system and is flagged as such;
it is *not* solved by this demo.

**Outside the triplane cube** (`±0.87·S ≈ ±0.30 m` around the object) the model is
undefined. The oracle clamps the query into the cube, evaluates there, and adds
the Euclidean distance to the clamp point. Exact when the nearest surface lies
along the clamp direction (true for this scene geometry); far from the object it
just needs to say "plenty of room", which it does. The demo path deliberately
starts/ends outside the cube so this fallback is exercised.

---

## Pipeline (what the script does)

```
             ┌─ ONCE, BEFORE MOTION ──────────────────────────────────────────┐
 mesh (UID) → load_and_normalize_mesh → greyscale                             │
             ├→ render_mesh_to_image (pyrender, grey bg, az/el/fov like       │
             │   training) → input_render.png + camera_extrinsics.json (R)    │
             ├→ TripoSR (LoRA fine-tuned from checkpoint) → TRIPLANE  (~1 s)  │
             ├→ raw field on 128³ grid → hole-fill → distance transform       │
             │   → ESDF grid (~1 s; see narrow-band finding)                  │
             ├→ reconstruct_mesh_from_triplane → ghost mesh for the scene     │
             └────────────────────────────────────────────────────────────────┘

             ┌─ EVERY CONTROL TICK (~30 Hz) ──────────────────────────────────┐
 EE pos ───→ SDFOracle.sense: batched trilinear ESDF lookups (~µs) →          │
             clearance d (min over control spheres), gradient n →             │
             potential-field velocity → IK → position control →               │
             step physics → capture frame                                     │
             └────────────────────────────────────────────────────────────────┘

 outputs:  demo video (avoidance ON) + baseline video (straight line, collides),
           top-down SDF heatmap + executed path, predicted-vs-GT clearance plot,
           metrics.json (min clearance, query latency, collision flags)
```

Key detail for honesty of the demo: **the input image is a pyrender render of the
object alone** (same renderer / framing / grey background as training data, no
robot arm in frame) — exactly matching the training domain, and exactly what the
prompt asked for ("in a manner similar to the training data, ideally without the
robot arm"). Using a PyBullet screenshot instead would be a gratuitous domain
shift; on a real robot this step becomes segmentation + background removal (the
stock TripoSR input path).

**Ground-truth verification built in:** every tick the script also computes the
*exact* clearance with `compute_sdf` on the GT mesh (the same exact-KDTree code
used in training). The clearance plot shows predicted vs. true distance on the
same axes — this is the single most convincing figure for "the model is accurate
enough to navigate by".

---

## Scene defaults (all overridable by flags)

| Thing | Default | Why |
|---|---|---|
| Robot | Franka Panda, fixed base at origin | ships with pybullet_data; reach 0.855 m |
| Object | Objaverse `97a038cd…` (chunky watertight blob) | `infer_sdf_mesh`'s default `85739db9…` is a figure glued to a huge flat panel that walls off the whole workspace and traps the arm — bad demo scene |
| Object scale `S` | 0.35 m longest edge | see the kinematic-feasibility note below — 0.5 m leaves the Panda no legal detour |
| Object position | (0.45, 0.00), resting on plane | dead-center on the EE path |
| EE path | (0.45, −0.40, z_mid) → (0.45, +0.40, z_mid) | straight line passes *through* the object; z_mid = object centroid height |
| Safety buffer `d_safe` | 0.07 m | must exceed surface-band model error (mm-scale) + ESDF voxel (~0.5 cm) + tracking error |
| Hard stop `d_stop` | 0.025 m | |
| Control spheres | hand envelope r = 0.03/0.06/0.065 m at +0/6/12 cm, plus the forearm (panda_link5, r = 0.07 m) tracked live | v4–v5 showed a single thin wrist sphere lets the hand body and forearm graze |

**Kinematic feasibility is a hard scene constraint** (learned the slow way, runs
3–5): a 0.5 m object centered 0.5 m out gives the 0.855 m-reach Panda *no*
collision-free detour — the outer route needs the EE at ~0.85 m, over-the-top
puts the wrist at ~0.90 m, and even the *start pose* drapes the forearm across
the object. The arm wedges against the obstacle and no local planner can fix
impossible kinematics. Size the scene so `‖object_center‖ + object_half_width +
buffer + hand_size < reach` before blaming the planner or the model.

Note the original prompt said "move from (0,0) to (2,0)" — **flagged below**; a 2 m
translation is impossible for a fixed-base Panda (0.855 m reach), so the demo uses
a 0.9 m sweep within the workspace. `--mode floating` swaps the arm for a floating
gripper sphere if an arm-free, arbitrary-length path is ever wanted.

---

## How to run

Same environment as training/inference: start the container, activate the venv.

```bash
cd TripoSR/docker && ./run.sh          # enter the container
cd ~/TripoSR && source .venv/bin/activate

# everything default (v0.64_1k checkpoint, default UID):
python sdf_robot_demo.py

# choose object / checkpoint / geometry:
python sdf_robot_demo.py \
    --uid 85739db9b70e47c28be0340e2d1907b7 \
    --checkpoint sdf_checkpoints/sdf_head_v0.64_1k_epoch0425.pt \
    --object-scale 0.35 --buffer 0.10

# floating gripper instead of the Panda; skip the baseline run:
python sdf_robot_demo.py --mode floating --no-baseline
```

Outputs land in `robot_demo_output/`:
`demo_sdf_avoidance.mp4`, `demo_baseline_straight.mp4`, `sdf_slices.png`
(top-down + side SDF slices with the executed path), `clearance.png`
(predicted vs true clearance over time), `input_render.png`,
`sdf_mlp_mesh.obj`, `metrics.json`.

---

## FLAGS — things in the original plan that don't hold up as stated

1. **"(0,0) to (2,0)"** — outside a fixed-base Panda's 0.855 m reach. Shrunk to a
   0.9 m sweep through the workspace (same story, feasible kinematics). If the
   2 m figure matters, use `--mode floating` or a mobile base later.
2. **Only the end effector is collision-checked** (as a sphere). The elbow/forearm
   could still clip the object in principle; whole-arm checking is just more
   sphere queries (batched, still cheap) but needs per-link sphere decompositions
   — deferred. The scene geometry keeps the arm above/beside the object so this
   doesn't bite in practice, and `metrics.json` reports GT clearance so a
   violation would be visible.
3. **Scale is hard-coded, and so is the object's pose.** The model predicts SDF in
   normalized units; metres come entirely from the known sim-side scale `S` and
   placement `c`. Fine for the demo, but on real hardware pose+scale registration
   (depth, fiducials, or calibrated extrinsics) is unsolved in this pipeline —
   don't let a reviewer discover that instead of the paper saying it.
4. **The triplane cube is small**: valid queries only within ±0.87·S (≈ ±30 cm)
   of the object center. The out-of-cube fallback is an approximation (exact only
   along the clamp direction). Any claim "the robot always knows its distance to
   the obstacle" needs this caveat, or multiple triplanes for multiple objects.
   Related: the **raw field is a narrow band** (~±δ·S around the surface, and
   the interior sign is wrong) — the planner runs on the propagated ESDF, never
   on raw queries. Both limits shrink with the object.
5. **"Processing is instant"** — be precise in the talk: one TripoSR pass (~1 s)
   plus one ESDF propagation (~1 s), once per scene change; after that, queries
   are trilinear lookups (µs). The script measures and reports all of it; use
   the measured numbers, not "instant". Also be upfront that the planner
   consumes the propagated ESDF, not raw per-point MLP calls — the narrow-band
   finding above is *why*, and it's a finding worth presenting, not hiding.
6. **The demo object should be cherry-picked** for reconstruction quality (v0.64
   generalizes but detail varies per object). That's fine for a demo — but say
   "representative object" not "arbitrary object" unless you also show a failure
   case. The buffer (0.10 m ≫ typical error) is what actually guarantees safety.
7. **The input image is a clean pyrender render**, not a sim camera screenshot —
   deliberate (matches training domain, no arm in frame), but it means the demo
   does not exercise segmentation/background-removal. Real deployment needs that
   front-end.

8. **The GT label pipeline has a sign artifact** (found while debugging the
   clearance metric): `compute_sdf` derives the sign from the nearest face's
   normal, which misfires near concave edges — verified isolated readings of
   −11.9 cm at points that are provably outside (`mesh.contains` = False,
   neighboring ticks +6 cm). The demo's metric now uses magnitude from
   `compute_sdf` + sign from ray-parity containment (`gt_sdf_signed`). **The
   training labels in `train_sdf_head.py` use the raw heuristic** — worth
   checking how often it mislabels near-concave-edge samples, since those are
   exactly the sharp-feature regions v0.6x struggles to reconstruct.

## Future extensions (not in this script)

- Demo 1 from the journal (grasp pre-positioning: EE pose + gripper width from the
  SDF zero crossing) reuses the same oracle — `width = d(p_left) + d(p_right)`.
- Multiple objects → one triplane each, `SDF_scene = min_i SDF_i` (SDFs compose
  by min for free — a genuinely good talking point).
- CHOMP-style trajectory optimization using the analytic SDF gradients.
- Whole-arm collision spheres (batch all spheres in one oracle call).
