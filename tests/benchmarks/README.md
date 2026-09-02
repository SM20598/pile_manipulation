# Benchmarks and probes

**Every performance number in this repo's docs is machine-specific and is
deliberately not written down as a promise.** Run these to get figures for your
own machine.

They also exist because the simulator has several non-obvious settings whose
justification is a measurement, not a preference — if you change one of those
defaults, the relevant probe here is how you find out what it cost.

## Check your backend first

Genesis falls back to CPU **silently** when a GPU is unavailable, printing only

```
[Genesis] [WARNING] Backend gs.gpu not available on this machine. Falling back to CPU.
```

CPU and GPU figures differ by more than an order of magnitude, and a CPU run
looks like a successful GPU run unless you read that line. Every script here
prints the backend it actually used, in its header. If you are comparing
numbers, compare the headers first.

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

## Scripts

Run from the repo root.

| script | what it measures | backend-sensitive? |
|---|---|---|
| `bench_performance.py` | batched vs per-particle pose writes; state-library restore vs shuffle+settle | **yes — timings** |
| `probe_physics.py` | plate tracking error, cruise speed, sweep step count, settle steps, post-settle drift | no — step counts and distances |
| `probe_piles.py` | whether a shape retains a stacked spawn, and whether a poured heap holds | no — geometry |

`bench_performance.py` is the only one whose output belongs in a performance
discussion. The other two produce numbers that are properties of the physics
and should reproduce anywhere, so a disagreement there is a real regression
rather than a hardware difference.

## Reading `probe_piles.py`

Two separate questions, easy to conflate:

- **Retention** — spawn a stack, settle, push. Does structure survive?
- **Repose** — pour a random tower, settle. Does a heap form at all?

A *lattice* start answers neither: a hexagonal sphere packing is a stable
crystal, each sphere resting in the dimple of three below, so it stands up
without friction doing any work. It will report a slope far above any physical
angle of repose (25–40° for spheres), which is the tell that the initial
condition, not the physics, is holding it up. `probe_piles.py --pour`
rejection-samples random positions per level for this reason.
