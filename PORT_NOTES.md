# Simulator fidelity port — `port-to-dino`

Semantic port of the useful parts of two older research branches (`refactor`,
`GenesisWorld`) onto current `upstream/dino_integration`. Not a merge: the
upstream code has moved on, so each change was re-implemented against the
current architecture and **re-measured on Genesis 1.3.3** rather than trusted
from the old branch.

Detailed measurements, methodology and the reasoning behind each value live in
[`scaling_to_200_objects.md`](scaling_to_200_objects.md). Read the
[corrections](#corrections-to-the-scaling-guide) section below first — two of
that document's claims do not survive the move to the current tray model.

Every number in this file was measured on this branch, not carried over.

---

## 1. Bug fixes that change recorded physics

These four alter the dynamics of **every transition**. Data collected before
and after them is not comparable.

| # | Was | Now | Measured effect |
|---|---|---|---|
| 1 | Sweep loop's per-step `set_dofs_position` zeroed the plate's velocity. `RigidEntity.set_dofs_position` overrides its base signature to default `zero_velocity=True`, and `zero_all_dofs_velocity` zeroes `slice(0, n_dofs)` — it ignores `dofs_idx_local`. So a call meant to pin z/roll/pitch/yaw zeroed **x and y velocity at 250 Hz**, inside the loop whose job is to move the plate along x/y. | `zero_velocity=False` | Sweep tracking error **5.00 → 0.33 mm** mean; cruise **121.6 → 125.0 mm/s** against 125 commanded. |
| 2 | Particles are built with no explicit material, so `rho` is `None` and their mass comes from Genesis' `RHO_OBJECT = 600`. `_set_particle_density_value` skipped `set_mass` on the first call (guarded on `old_density is not None`), then rescaled from 600 as if it were the configured value. | Seed `old_density` from `RHO_OBJECT`, imported from Genesis rather than hardcoded | Implied particle density **600 → 750**, i.e. every particle mass was **0.8× its recorded density** in every dataset collected before this. |
| 3 | Plate friction never set → Genesis default **1.0**, and contacts combine as `max(µa, µb)`. | Explicit `plate.friction: 0.3` | Sampled particle friction had **zero effect at the tool interface** — the one interface the action acts through. |
| 4 | `enable_torsional_friction` off (Genesis default) | `True` | A cube resting on the tray had no resistance to twisting in place. The tool is a thin blade that strikes most particles off their centre of mass, so induced spin is a large part of what a push does. |

Two latent fixes with no behavioural change today: `dt`/`substeps` fallbacks
were `4e3` (4000 seconds) and `1`, now `4e-3` and `5`; and `safety_margin` was
declared in `basic.yaml` but hardcoded at `0.02` in the code and never read —
the code now reads it and the config states the value that was in force.

## 2. The plate is modelled as a gantry axis

The tool is a 2.4 g box — the *lightest* dynamic object in the scene. What
previously kept it on course was that four of its DOFs were hard-set every step
and its x/y velocity was zeroed every step. The dominant tracking error was
never granular reaction; it was the control law, which handed the tool its
**endpoint** as a PD position target and therefore ran it at a speed
proportional to distance remaining.

Now, all config-driven under `plate:`:

- `set_dofs_armature(moving_mass)` on x/y/z — the drivetrain's reflected
  inertia, added to the mass-matrix diagonal the constraint solver already
  uses. The correct knob rather than a denser plate, which would also change
  the tool's weight and contact response.
- gains from mass and bandwidth: `kp = mω²`, `kv = 2ζmω` at ζ=1 →
  4441.3 N/m and 94.25 N·s/m at the defaults, verified against the built scene.
- `set_dofs_force_range(±max_force)` — previously unbounded; a real stepper
  loses steps rather than applying unlimited force to a jam.
- a **trapezoidal** position + velocity reference replacing the endpoint target.

| | measured |
|---|---|
| tracking error vs commanded path | mean **0.33 mm**, max 0.97 mm |
| cruise speed | **125.0 mm/s** vs 125 commanded |
| final tracking error | **0.010 mm** |
| goal reached, sampled actions | **12/12** |
| sweep step count, 90 mm | **208** vs 306 under the old law |

The step count falls because it now comes from the trapezoid's real duration
rather than a `1.7×` fudge that was compensating for the speed error, and the
per-step `.nonzero()`/`.item()` GPU syncs are gone.

`plate.approach_mode: servo` (new default) drives the descent with the actuator
instead of teleporting the pose each step, so particles can resist it.
`hold_mode: servo` was built, measured and **rejected** — 3–6× higher particle
penetration — and is kept only as a documented option.

## 3. Settling is convergence-based

`update_material_state` ran a fixed 200 steps with no check that anything had
stopped. `settle_steps` is now a **cap** with a velocity-convergence exit and a
loud warning if the cap is hit.

Three things were needed to make the criterion work:

1. **A quantile, not a max.** Testing `max` over every particle in every env
   makes the criterion harder the more envs are batched, so the settle always
   ran to its cap. `settle_rest_quantile: 0.995` tolerates ~1 straggler per
   200-cube env.
2. **An angular threshold derived from the linear one.** A bare rad/s number is
   not comparable to a m/s number: 0.1 rad/s on a 5 mm cube is a corner speed
   of 0.35 mm/s, three times *stricter* than the 1 mm/s linear threshold, so it
   silently became the binding criterion. It is now derived through the
   particle's half body diagonal — 0.2309 rad/s at 5 mm.
3. **Not resetting the solver every step.** Holding the plate with a per-step
   `set_dofs_position` calls `collider.reset()` and
   `constraint_solver.reset()`, discarding the constraint solver's warm start
   with only 10 iterations to rebuild it. The plate is lifted clear during a
   settle and its PD holds it, so the teleport bought nothing. The control
   target is now set once.

True steps-to-rest, measured with the check interval set to 1:

| | n=50 | n=200 |
|---|---|---|
| fresh spawn | 34 | 34 |
| after a push | 1 | 1 |

Flat in `n`, and **~6× cheaper than the fixed 200 on a reset, ~20× per
transition**. Post-push is nearly free because the pile has already relaxed
during `execute_action`'s 40-step lift — it is at rest before the loop starts.

The criterion is genuine rather than lenient: at exit the *peak* particle speed
(not just the quantile) is 0.017 mm/s against a 1 mm/s threshold, and holding
the pile 200 further steps moves it **0.001 mm**, with no particle drifting
more than 1 mm.

`settle_check_every: 10` also acts as a minimum settle. That floor is
deliberate — it guards the one failure this criterion cannot see, a cube at the
top of its tipping arc, whose speed passes through a *minimum*. It costs ~2% of
a batch.

## 4. Contact budget is checked, not assumed

Genesis reports contact-pair overflow through an error bit that
`Simulator.step` inspects periodically. That mechanism **cannot fire here**:
`set_dofs_position` clears the bit as a side effect and the sweep loop calls it
every step, so the bit is always wiped before the next check reads it. Overflow
would be completely silent — contacts dropped, wrong physics recorded, no
exception. It has been observed as a CUDA illegal memory access rather than a
clean failure.

`contact_budget_usage()` therefore reads the collider's counters directly at
the end of every sweep, where usage peaks, comparing broad-phase pairs and
contact *points* against **their own separate caps** — two limits that differ
by more than an order of magnitude. The point cap is read from
`ColliderInfo.max_contacts` rather than recomputed as
`max_collision_pairs × n_contacts_per_pair`: since Genesis 1.2.x the buffer is
sized per contact regime and then reduced by link-pair pruning, so the old
product would *overstate* the cap and hide exactly the overflow this check
exists to catch.

`max_collision_pairs` now defaults to `max(150, n_particles // 2)`. This is not
cosmetic — the constraint Jacobian is
`O(max_collision_pairs × contacts_per_pair × n_dofs × n_envs)` while raw step
time is independent of the cap, so an oversized value converts directly into
lost parallelism.

`escaped_particle_count()` reports particles that have left the tray, which can
only happen if the contact solver failed for them.

## 5. Performance, with no behavioural change

**Batched particle pose writes.** `RigidEntity.set_pos`/`set_quat` each run a
forward-kinematics pass over the *whole scene*, so the per-particle loop in
`_set_particle_positions` cost 2N kernel launches and 2N full-scene FK passes —
400 of each at n=200, on every reset and every state restore. Replaced by two
solver-level calls taking a link-index array plus a single FK pass.

**Bit-identical to the old loop (max diff 0.0) and 48.5× faster** at 100
particles × 4 envs.

**`set_n_active(n)`** places only the active prefix and parks the rest outside
the tray *on a grid*. Particle count is otherwise a rebuild-only parameter
(particles are created before `scene.build()`, and `performance_mode` makes
every distinct scene shape pay a full kernel recompile). Parking inactive
particles at one shared point — the obvious implementation — piles them into a
single permanent contact cluster that costs solver time on every step of every
env, which matters more than it sounds given Newton's dense per-island Hessian.

## 6. New collection features

All opt-in. Omitting the flags reproduces the previous behaviour exactly.

### Settled-state library — `Genesis/state_library.py`

`--state-library N` settles N piles once per build, expands each by the
container's symmetry group, saves `settled_states.pt` beside the data, and
resets by restoring instead of re-settling. `shuffle_particles()` runs zero
simulation steps, so all of a reset's cost is the settle that follows it.

The symmetry expansion is what makes a handful of settles worth it: a settled
arrangement rotated or mirrored into another orientation of the tray is still a
valid settled arrangement, and a different configuration to sample from. A
square tray admits the full dihedral group **D4 — 8 variants per settle**.
Mirroring is applied to orientations too: a reflection `M` is improper, but
`M R M` is a proper rotation, which is legitimate because cubes and spheres are
achiral. Verified against explicit rotation matrices including `det = +1`.

Measured: 3 settles × 4 envs × 8 symmetries = **96 states**; restore is
**306× faster** than shuffle + settle and the restored pile is **at rest with
no settle at all** (peak motion 0.0000 mm/s).

All envs in a batch share one initial state, drawn without replacement. A batch
of identical piles is cheaper to simulate, and within a batch the sampled
action parameters already vary the dynamics substantially, so the variance
given up is small. Diversity is preserved *across* batches.
`StateLibrary.apply_per_env` exists where per-env states are genuinely needed.

### Touchdown pose sampling — `Genesis/placement_sampling.py`

`--start-sampling` selects between four samplers. The two mechanisms are
complementary rather than rival: density-weighted answers *where is there
material worth pushing* (a property of the pile), free-space answers *where can
the tool actually come down* (a property of the tool).

| mode | touchdown overlap | mean start radius |
|---|---|---|
| `uniform` | 89 % | 23.8 mm |
| `density` (upstream's, current default via `auto`) | 95 % | 34.1 mm |
| `free` | **16 %** | 25.1 mm |
| `composed` | **28 %** | **34.2 mm** |

*Overlap* = the blade's footprint at touchdown contains a particle centre. The
plate descending **into** a particle is resolved by the solver ejecting it — an
artifact recorded as though it were a push.

`density` deliberately *raises* overlap, because it aims at material. `free`
avoids particles but drifts toward empty tray, where a push moves nothing.
`composed` lets density choose the neighbourhood and then moves the pose the
shortest distance that makes it legal: it keeps density's spatial distribution
(34.2 vs 34.1 mm) while cutting overlap to 28 %.

`free` and `composed` fall back to the underlying draw per sample wherever the
free set is empty. They are refinements, not guarantees — as the tray fills the
free set shrinks and eventually empties.

### Shared travel distance — `Genesis/action_sampling.py`

`--shared-travel-distance` gives every env in a batch the same push *length*
for a given sample, while each keeps its own start point, direction and blade
yaw. A batching artefact fix, not a modelling choice: `sweep_steps` is derived
from the *longest* travel in the batch, so one long push makes every env run
for its duration. What is given up is the within-batch spread of one of five
action dimensions; a push that cannot reach the shared distance without leaving
its sampling box is truncated at the boundary.

### Reproducibility and audit trail

`data_collection.py` had no seed control: `np.random.default_rng()` with no
argument, and every torch draw (spawn poses, orientations, actions) unseeded. A
run could not be repeated, which also meant a run that produced something odd
could not be replayed to look at it.

`--seed` now seeds **both** generators and is recorded in each batch's config.
Verified: two independent runs with the same seed are **bit-identical**,
including `states_`.

Each saved batch also records `unchanged_transitions` (a run that "succeeds"
while nothing moves is the silent failure worth guarding against),
`escaped_particles`, and peak `contact_budget` usage — so a finished dataset
carries the evidence that it is trustworthy instead of requiring the run to be
repeated to find out.

---

## Corrections to the scaling guide

[`scaling_to_200_objects.md`](scaling_to_200_objects.md) was written against the
older simulator. Two of its claims do not transfer, and the guide should be read
with these in mind until it is revised:

**§1.1 fix 7 and all of §2 (capacity) describe a tray that no longer exists.**
Upstream now *derives* the box height rather than reading it from config:

```python
self._box_params["vol"][2] = self._wall_thickness + max_particle_height(...)
```

`max_particle_height`'s docstring states its purpose — "so a resting monolayer
never sticks out above the walls" — and `random_sequential_addition` raises if
the height is any less. The tray is therefore **a monolayer by construction**,
which a top-down camera feeding the DINO pipeline also depends on. Measured on
this branch: n=200 at 5 mm places and settles fine in a single flat layer.
So the guide's "150 and 200 could not be placed at any size" was a limitation
of the *old* placer, which upstream's rejection sampler with grid fallback has
already solved. The layered spawn is **not** part of this port; it will live in
a separate `_layered` script for taller-tray experiments, where the guide's §2
capacity table and its ~1460-step respawn settle *do* apply.

**§1.3's settle step counts are two-layer figures.** The ~250 post-push and
~1460 fresh-respawn numbers were measured on a spawn where particles fell from
a second layer and had to collapse. On the monolayer, true steps-to-rest is
34 fresh and 1 post-push, flat from n=50 to n=200 (§3 above). The cap here is
500, not the guide's 2500.

The rest of the guide — the plate model, the solver-knob measurements (§1.4),
the contact-island cost law (§8.7), the solver-equivalence study (§8.8) and the
end-to-end verification (§8) — is unaffected.

---

## Not ported

| | Why |
|---|---|
| Deletion of `GranularDynamics2/*`, `train_unet_genesis.py`, `Genesis/training/dataset_cop.py` | Those deletions were a reorganisation on the old branch. Upstream uses these files actively alongside `dino_wm`. |
| `Genesis/training/dataset.py` refactor | Removes upstream capability (`run`, `split=None`, `include_sweep_removed`) and couples the file to packages outside `Genesis/`. Upstream has independently rewritten it and it now feeds the DINO and LeWM exporters. Deferred pending a proper side-by-side. |
| Layered spawn | Superseded here; separate `_layered` script. See corrections above. |
| `Genesis/transition_buffer.py`, `push_and_record` | An incremental recording path built for oracle MPC. Upstream has its own (`_save_rollout`, `_render_all_envs`, `export_dino_wm_dataset`); two overlapping mechanisms would be worse than either. |
| The MPC / model research stack | Large, separable, and belongs in its own branch under `MPC/`. Verified separable: the new `Genesis/*` modules import nothing outside the `Genesis` package. |
| Data files, run outputs, rendered videos | ~2500 generated files on the old branches. |

## Pre-existing issues found but deliberately left alone

- **Eight of the nine files in `Genesis/configs/` no longer load.**
  `basic_example.yaml`, the six `chick*.yaml` and `param_optim.yaml` use an
  older nested schema (`sandbox: {box:, material:, safety_margin:}` with
  `properties:` sub-dicts) while `__init__` reads `box`, `material` and `plate`
  at the top level, so they raise `KeyError`. All predate this work. Migrating
  configs that cannot be tested here is out of scope for a physics change; they
  may also be kept deliberately as references for the older pipeline.
- Unknown `rigid_options` keys now **warn** rather than raise. Previously they
  were silently dropped, which is how `enable_torsional_friction: True` could
  have been written in `basic.yaml` and done nothing — the same failure mode
  `safety_margin` had.
- Saved configs contain `!!python/tuple` tags (from `particle_sizes`), so they
  need `yaml.full_load`, not `safe_load`. Upstream's readers already use
  `full_load`; noted only because a new consumer would trip on it.

## Still to do

- `Genesis/run_collection.py` — one subprocess per pile size, with preflight
  (placement feasibility, free VRAM) and postflight validation. The CLI flags
  and audit fields it depends on are in; the driver is not.
- `tests/scaling_investigation/` — `verify_fixes`, `verify_new_features` and
  the probe scripts. The measurements quoted here were run as throwaway
  scripts and should become the committed regression suite.
- The `_layered` spawn script.
- Genesis-dependent tests. The 52 tests below are all Genesis-free.

## Testing

```bash
python -m pytest tests/ -q          # 52 passing, pure torch, no Genesis needed
```

Collection, run from inside `Genesis/` (upstream's convention — this package
uses flat sibling imports, not relative ones):

```bash
cd Genesis
python data_collection.py \
    --num-particles 50 --particle-sizes 0.005 \
    --n-envs 4 --samples-per-env 5 \
    --seed 0 --state-library 8 \
    --start-sampling composed --shared-travel-distance
```

## New config keys

All have defaults matching the values documented in `Genesis/configs/basic.yaml`,
where each carries its measured justification.

| section | keys |
|---|---|
| `simulation` | `settle_steps`, `settle_check_every`, `settle_velocity_threshold`, `settle_rest_quantile`, `settle_angular_velocity_threshold` (derived if absent), `pos_ctrl_steps`, `sweep_settle_steps` |
| `rigid_options` | `constraint_solver`, `enable_torsional_friction`, `max_collision_pairs` — plus any other `RigidOptions` field, now forwarded rather than silently dropped |
| `plate` | `friction`, `moving_mass`, `acceleration`, `control_bandwidth_hz`, `max_force`, `hold_mode`, `approach_mode`, `arrival_steps`, `orientation_inertia`, `orientation_bandwidth_hz`, `max_torque` |
| top level | `safety_margin` (now actually read) |

## New files

| file | lines | what it is |
|---|---|---|
| `Genesis/state_library.py` | 410 | settled-state bank, symmetry augmentation |
| `Genesis/placement_sampling.py` | 296 | occupancy grid, C-space free set, `nearest_free_placement` |
| `Genesis/action_sampling.py` | 93 | shared batch travel distance |
| `tests/test_state_library.py` | 235 | 27 tests |
| `tests/test_placement_sampling.py` | 289 | 19 tests |
| `tests/test_action_sampling.py` | 103 | 6 tests |

`requirements.txt` pins `genesis-world==1.3.3` (was unpinned) and adds `scipy`,
previously an undeclared transitive dependency.
