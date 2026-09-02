# Layered spawn

An alternative spawn path for particle counts that **will not fit in a single
layer**. Everything here is self-contained; the monolayer path in `Genesis/` is
untouched by it.

Read this before using it — the headline is that you probably don't need it.

## When you need it, and when you don't

The tray in `Genesis/` is a monolayer by construction:
`sandbox_manipulation.py` derives the box height as `wall_thickness +
max_particle_height(...)`, sized so a resting single layer never pokes above
the walls, and `random_sequential_addition` raises if the height is any less.

The ceiling is set by two things that are easy to get wrong, so it was
measured against the simulator rather than estimated:

- **The reshuffle binds, not creation.** Particles are placed twice with
  *different* clearances. `random_sequential_addition` (creation, pre-build)
  clears `size/2`; `_sample_nonoverlapping_particle_positions` (every reset)
  clears `size/2 · √2`, the footprint a cube sweeps at free yaw. That is 1.41×
  per axis, so ~2× the area per particle — a count that *creates* fine can
  still fail on the first reset, and the reset runs every batch.
- **The grid fallback sets the ceiling, not rejection sampling.** Rejection
  sampling gives up well before the tray is full (132 of a possible 225 at
  5 mm) and then falls through to `_grid_particle_positions`, silently. So the
  effective capacity is the regular grid's.

Verified at the boundary: 8.5 mm cubes shuffle at n=81 and **fail at n=82**;
5 mm shuffles at n=225. In the stock 128 mm tray, cubes:

| particle size | max in one layer | historical 50 / 70 / 100 / 150 / 200 sweep |
|---|---|---|
| 5.00 mm | **225** | all five fit in one layer |
| 6.75 mm | **144** | up to 100 |
| 8.50 mm | **81** | up to 70 |
| 10.25 mm | **64** | 50 only |
| 12.00 mm | **49** | none — even 50 needs 2 layers |

So **at 5 mm nothing in the historical sweep needs this path**, but 14 of its
25 (size, count) cells do — every count at 12 mm included.

Note that *fitting* is a weaker condition than being *usable*: the simulator
warns from 70 % of the ceiling, because a nearly-full tray leaves the tool
almost nowhere to touch down. Adding a layer is one way to get back under that
threshold — 150 of 8.5 mm is 185 % of one layer's capacity but 93 % across two,
which still warns; three layers would put it at 62 %. If your
configuration is inside the ceiling above, use `Genesis/data_collection.py` —
it is the maintained path and gives fully-observable, single-layer piles.

`materials_layered.plan_layers()` computes the same answer for any
configuration, analytically (the binding capacity is a closed form, so there is
nothing to trial-pack):

```python
plan_layers("cube", 0.0085, 150, (0.128, 0.128, 0.05))   # -> 2
single_layer_capacity("cube", 0.0085, 150, 0.128)        # -> 81
```

## Two costs you are accepting

**Partial observability.** The top-down camera (`_CLOUDGRIPPER_CAMERA_MAIN`,
feeding the DINO and LeWM exporters) sees the top of the pile only. A second
layer occludes the first, so an image observation no longer determines the
state. Rendering is deliberately left enabled — an image-based model trained on
this data simply has to cope with partial information — but that is a modelling
decision you are making by using this path, not something the code hides.

**Material the tool cannot reach.** The blade rides half a particle above the
floor and is only `plate.size[2]` tall (10 mm by default). A deep enough pile
rises above its top edge, and that material is never pushed. The code warns at
construction when the *spawn stack* exceeds the blade, and reports by how much.
Settling shrinks the problem but does not necessarily remove it:

> measured, 150 cubes of 8.5 mm, 2 layers. The spawn stack reaches 30.0 mm; the
> settled pile's top *surface* is at 27.0 mm, against a blade top edge of
> 24.25 mm. So 2.74 mm of pile is still above the blade after settling — the
> warning is correct, just smaller than at spawn.
>
> The blade cuts a 10 mm band out of a 17 mm pile: **58 % of the pile height is
> engaged**, 2.74 mm escapes above and 4.27 mm below. All 150 particles are
> overlapped to some degree (mean 4.94 of 8.5 mm), so none is wholly out of
> reach, but none is wholly engaged either.

Note the lower gap is **not** a layering artefact. The blade's bottom edge rides
half a particle above the floor by construction
(`_operation_height = wall_thickness/2 + particle_size/2 + plate_size[2]/2`), so
the bottom half of the resting layer passes underneath it in the monolayer case
too — this is the scaling guide's open decision 3. Layering adds a gap at the
top; it does not create the one at the bottom.

Raising `plate.size[2]` closes the top gap, and lowering `_operation_height`
closes the bottom one, but each changes the dynamics of every transition — so
both are modelling decisions, not fixes.

## Files

Copies, not subclasses. The two things that must change to stack particles —
the box height and the creation-time placement — both happen inside `__init__`
before `scene.build()`, and one of them lives in a module-level function that
cannot be overridden at all. So a subclass would have had to duplicate
`__init__` anyway, which is the method most likely to change upstream.

| file | copied from | changed lines |
|---|---|---|
| `materials_layered.py` | `Genesis/utilities/materials.py` | 98 |
| `sandbox_manipulation_layered.py` | `Genesis/sandbox_manipulation.py` | 238 of ~2500 |
| `data_collection_layered.py` | `Genesis/data_collection.py` | 53 |
| `configs/basic_layered.yaml` | `Genesis/configs/basic.yaml` | 37 |

Each file opens with a docstring listing exactly what differs, and the changes
were kept textually minimal so `diff` remains the re-sync tool. **When upstream
changes, diff the pair and re-apply** — the listed regions are the only ones
expected to conflict.

### What actually differs

`materials_layered.py`
- `stack_height()` — new; the interior height a stack of `n_layers` needs.
- `random_sequential_addition()` — takes `n_layers`, packs each layer
  independently, and checks the box against the stack rather than one layer.
- `single_layer_capacity()` / `plan_layers()` — new; the exact single-layer
  ceiling (the grid fallback's, computed from the reshuffle's √2 clearance) and
  `ceil(n / capacity)`. Appended at the end of the file, away from the copied
  body.

`sandbox_manipulation_layered.py`
- imports `materials_layered`, and puts `Genesis/` on `sys.path` so the shared
  modules (`state_library`, `placement_sampling`, `action_sampling`,
  `utilities/`) resolve from a subdirectory.
- `__init__` resolves `material.n_layers`, sizes the box with `stack_height()`,
  scales `_clearance_height` with the stack, and warns about blade reach.
- `_add_entities` forwards `n_layers`.
- `_sample_nonoverlapping_particle_positions` packs per layer — the one
  substantially rewritten method. `placed` resets per layer, because overlap is
  only a constraint *within* a layer, and that is precisely what lets the total
  exceed the single-layer capacity.
- `_grid_particle_positions` (the fallback) sizes its grid for one layer's load
  and stacks the rest on it.

`configs/basic_layered.yaml`
- `settle_steps: 3000` instead of 500. Layers are *dropped*, so unlike a
  monolayer the pile has real potential energy to shed. Still a cap with an
  early exit — the 8.5 mm / 150 / 2-layer case above settled in 60 steps.
- `material.n_layers: auto`.

## Running

From inside this directory:

```bash
cd Genesis/layered
python data_collection_layered.py \
    --num-particles 150 --particle-sizes 0.0085 \
    --n-envs 4 --samples-per-env 5 --seed 0
```

`--n-layers` defaults to `auto` (ask the planner) and accepts an integer to
force a count. Every other flag behaves as in `Genesis/data_collection.py`,
including `--state-library`, `--start-sampling` and `--seed`.

Output goes under `data/corl_layered/<shape>/n<N>/size<S>/**layers<L>**/`. The
`layers<L>` component is deliberate: a layered dataset has different dynamics
from a monolayer one at the same (shape, count, size), and the two must not
land in the same directory and be loaded as one distribution.

## Tests

```bash
python -m pytest tests/test_layered_spawn.py -q      # 34 tests
python -m pytest tests/test_packing_capacity.py -q   # 17 tests (shared model)
```

These cover the planning and geometry logic, and — most importantly —
`test_one_layer_matches_the_single_layer_original`, which guards the copy
against quietly diverging from upstream in the `n_layers=1` case. The placement
itself needs a built scene and is verified by running a collection; the numbers
quoted above come from such a run.
