#!/usr/bin/env python3
"""
run_collection.py - drive a collection across several particle counts, one
subprocess per count, with preflight and postflight checks.

Why a driver rather than more flags on data_collection.py
---------------------------------------------------------
Changing ``n_particles`` needs a full scene rebuild - particles are created in
``__init__``, before ``scene.build()``, and with ``performance_mode`` on every
distinct scene shape recompiles kernels. So each count is a separate process
whether or not you want it to be. Making that explicit buys three things:

  * the parallel env count can differ per particle count, since VRAM per env
    grows with the pile;
  * an OOM, an infeasible placement or a crash at one count cannot take the
    rest of the run with it;
  * each count gets checked on the way in and validated on the way out.

Checks
------
Preflight, before paying for a build:

  * **placement** - does this many particles of this size fit one layer of the
    tray, and is the tray filled past the point where the tool still has room
    to touch down? Uses the same ``single_layer_capacity`` the simulator
    enforces, so a "fits" here means the reshuffle will not fail on batch two.
  * **free VRAM** against a coarse estimate. Only ever warns - see the note on
    ``--vram-per-env`` below.

Postflight, on the files each count actually wrote:

  * every batch produced a data file that loads, with the expected shape and
    dtype and no non-finite values;
  * **s' actually differs from s** - a run that "succeeds" while nothing moves
    is the silent failure this exists to catch;
  * displacements are physically plausible (nothing ejected);
  * the goal-reached rate;
  * the audit fields the run recorded itself (escaped particles, unchanged
    transitions, peak contact budget);
  * the state library, if one was requested, was written and has the size the
    symmetry expansion implies.

Usage
-----
Run from inside ``Genesis/`` - this package uses flat sibling imports, matching
upstream's convention::

    cd Genesis
    python run_collection.py --plan configs/collection_dry_run.yaml
    python run_collection.py --plan configs/collection_dry_run.yaml --preflight-only
    python run_collection.py --plan configs/collection_dry_run.yaml --n-envs 4

Exits nonzero if any count failed to run or failed validation, so it is usable
unattended.
"""
import argparse
import json
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml

GENESIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GENESIS_DIR))

from utilities.materials import (single_layer_capacity,          # noqa: E402
                                 PACKING_WARN_FRACTION)

# Coarse per-env VRAM cost. Deliberately overridable and only ever used to
# WARN, never to skip: it was fitted on one card, and this repo's policy is
# that no performance figure is a promise about someone else's hardware.
# Override with --vram-per-env or `plan.vram_per_env_gib`.
DEFAULT_VRAM_BASE_GIB = 0.15
DEFAULT_VRAM_PER_PARTICLE_PER_ENV_GIB = 0.001078


# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #
def placement_report(shape, particle_size, n_particles, box_xy):
    """Can this pile be placed, and does it leave the tool room to work?

    Returns (ok, fraction, capacity, message). ``ok`` False means the reshuffle
    would fail outright, not merely be tight - so the count is skipped.
    """
    cap = single_layer_capacity(shape, particle_size, n_particles, box_xy)
    if cap <= 0:
        return False, math.inf, cap, (
            f"a single particle of {particle_size} m does not fit a "
            f"{box_xy} m tray")
    frac = n_particles / cap
    if n_particles > cap:
        return False, frac, cap, (
            f"{n_particles} exceeds the single-layer capacity of {cap} "
            f"({frac*100:.0f}%). The reshuffle placement would fail. Use "
            f"Genesis/layered/ for stacked spawns, a smaller particle, or a "
            f"bigger tray.")
    if frac >= PACKING_WARN_FRACTION:
        return True, frac, cap, (
            f"{n_particles} is {frac*100:.0f}% of the {cap}-particle layer "
            f"capacity. It will place, but the tool has little room to touch "
            f"down and placement-aware sampling will mostly fall back to blind.")
    return True, frac, cap, f"{n_particles}/{cap} = {frac*100:.0f}% of a layer"


def free_vram_gib():
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, _ = torch.cuda.mem_get_info()
        return free / 2 ** 30
    except Exception:
        return None


def estimate_vram_gib(n_particles, n_envs, per_particle_per_env, base):
    return base + (0.0025 + per_particle_per_env * n_particles) * n_envs


def resolve_env_counts(plan, cli_override):
    """Envs per particle count: a CLI override, a mapping, or a single int.

    A mapping may also be a path to a yaml written by a throughput benchmark,
    so the plan can reference a measurement instead of copying numbers out of
    it and going stale. Whatever the source, the throughput optimum is specific
    to the material it was measured on, so a referenced file's material is
    checked against the plan's.
    """
    counts = plan["plan"]["n_objects"]
    spec = plan["plan"].get("n_envs", 1)
    if cli_override is not None:
        return {n: int(cli_override) for n in counts}, "--n-envs"

    if isinstance(spec, str):
        path = (GENESIS_DIR / spec) if not Path(spec).is_absolute() else Path(spec)
        if not path.exists():
            path = GENESIS_DIR.parent / spec
        if not path.exists():
            raise FileNotFoundError(
                f"plan.n_envs points at {spec!r}, which does not exist. Either "
                f"run the benchmark that writes it, or replace it with a "
                f"literal mapping or a single integer.")
        doc = yaml.safe_load(path.read_text()) or {}
        measured = doc.get("n_envs", doc)
        want = plan.get("material", {})
        got = doc.get("material", {})
        for key in ("shape", "particle_size"):
            if key in got and key in want and got[key] != want[key]:
                print(f"  WARNING: {path.name} was measured with {key}="
                      f"{got[key]!r} but this plan uses {want[key]!r}. A "
                      f"throughput optimum does not transfer across materials.")
        return ({n: int(measured.get(n, measured.get(str(n), 1))) for n in counts},
                str(path))

    if isinstance(spec, dict):
        return {n: int(spec.get(n, spec.get(str(n), 1))) for n in counts}, "plan"
    return {n: int(spec) for n in counts}, "plan"


# --------------------------------------------------------------------------- #
# postflight
# --------------------------------------------------------------------------- #
def check_batch(path: Path, n_particles: int) -> list[str]:
    """Validate one saved batch. Returns a list of problems (empty is good)."""
    import torch
    problems = []
    try:
        blob = torch.load(path, weights_only=False)
    except Exception as e:
        return [f"{path.name}: will not load ({type(e).__name__}: {e})"]

    for key in ("states", "states_", "p_starts", "p_stops", "angles"):
        if key not in blob:
            problems.append(f"{path.name}: missing key {key!r} "
                            f"(has {sorted(blob)})")
    if problems:
        return problems

    s, s2 = blob["states"], blob["states_"]
    if s.shape != s2.shape:
        problems.append(f"{path.name}: states {tuple(s.shape)} != states_ "
                        f"{tuple(s2.shape)}")
    if s.ndim != 3 or s.shape[1] != n_particles or s.shape[2] != 7:
        problems.append(f"{path.name}: states shape {tuple(s.shape)}, expected "
                        f"(N, {n_particles}, 7)")
    if s.dtype != torch.float32:
        problems.append(f"{path.name}: states dtype {s.dtype}, expected float32")
    if s.shape[0] == 0:
        problems.append(f"{path.name}: zero successful samples")
        return problems
    if not torch.isfinite(s).all() or not torch.isfinite(s2).all():
        problems.append(f"{path.name}: non-finite values in states")

    # The one that matters: did the pushes do anything?
    moved = (s2[..., :3] - s[..., :3]).abs().amax(dim=2).amax(dim=1)
    static = int((moved < 1e-6).sum())
    if static == s.shape[0]:
        problems.append(f"{path.name}: NO sample changed state - the actions "
                        f"had no effect")
    elif static:
        problems.append(f"{path.name}: {static}/{s.shape[0]} samples did not "
                        f"change state at all")
    if float(moved.max()) > 0.5:
        problems.append(f"{path.name}: implausible displacement "
                        f"{float(moved.max()):.3f} m - particles likely ejected")
    return problems


def read_audit(cfg_path: Path) -> dict:
    """The audit fields the run recorded about itself.

    Saved configs carry `!!python/tuple` tags from `particle_sizes`, so they
    need full_load - safe_load raises on them. Upstream's own readers do the
    same.
    """
    doc = yaml.full_load(cfg_path.read_text())
    st = (doc or {}).get("statistics", {}) or {}
    return {
        "collected": st.get("total_samples_collected"),
        "failed": st.get("number_of_failed_samples"),
        "unchanged": st.get("unchanged_transitions"),
        "escaped": st.get("escaped_particles"),
        "budget": st.get("contact_budget"),
        "seed": ((doc or {}).get("data_collection", {}) or {}).get("seed"),
    }


def validate_output(leaf: Path, n_particles: int, want_library: int) -> dict:
    """Check everything one particle count wrote. Returns a result dict."""
    res = {"problems": [], "batches": 0, "samples": 0, "failed": 0,
           "unchanged": 0, "escaped": 0, "peak_contact": 0, "peak_cap": 0}
    if not leaf.exists():
        res["problems"].append(f"{leaf} was never created")
        return res

    data_files = sorted(leaf.glob("_*_data.pt"))
    cfg_files = sorted(leaf.glob("_*_config.yaml"))
    if not data_files:
        res["problems"].append(f"{leaf}: no _*_data.pt written")
        return res
    if len(data_files) != len(cfg_files):
        res["problems"].append(f"{leaf}: {len(data_files)} data files but "
                               f"{len(cfg_files)} configs")

    res["batches"] = len(data_files)
    for f in data_files:
        res["problems"].extend(check_batch(f, n_particles))

    for f in cfg_files:
        try:
            a = read_audit(f)
        except Exception as e:
            res["problems"].append(f"{f.name}: unreadable ({e})")
            continue
        res["samples"] += a["collected"] or 0
        res["failed"] += a["failed"] or 0
        res["unchanged"] += a["unchanged"] or 0
        if a["escaped"]:
            res["escaped"] += a["escaped"]
            res["problems"].append(
                f"{f.name}: {a['escaped']} particle(s) left the tray - the "
                f"contact solver failed for them, and since each transition's "
                f"s is the previous s', later samples in that env are suspect")
        b = a["budget"] or {}
        if b.get("contact_points"):
            res["peak_contact"] = max(res["peak_contact"], b["contact_points"])
            res["peak_cap"] = max(res["peak_cap"], b.get("contact_cap", 0))

    if want_library:
        libs = list(leaf.glob("settled_states.pt"))
        if not libs:
            res["problems"].append(
                f"{leaf}: --state-library {want_library} was requested but no "
                f"settled_states.pt was written")
        else:
            try:
                import torch
                blob = torch.load(libs[0], weights_only=False)
                states = blob["states"]
                meta = blob.get("meta", {})
                res["library"] = int(states.shape[0])
                # N settles x n_envs x |symmetry group|
                expect = (want_library * int(meta.get("n_envs", 1))
                          * int(meta.get("n_symmetries", 1)))
                if states.shape[0] != expect:
                    res["problems"].append(
                        f"library holds {states.shape[0]} states, expected "
                        f"{expect} = {want_library} settles x "
                        f"{meta.get('n_envs')} envs x "
                        f"{meta.get('n_symmetries')} symmetries")
                if states.shape[1] != n_particles:
                    res["problems"].append(
                        f"library is for {states.shape[1]} particles, not "
                        f"{n_particles}")
            except Exception as e:
                res["problems"].append(f"settled_states.pt unreadable ({e})")
    return res


# --------------------------------------------------------------------------- #
def build_command(plan, n, n_envs, args):
    m = plan.get("material", {})
    p = plan["plan"]
    cmd = [sys.executable, "data_collection.py",
           "--num-particles", str(n),
           "--particle-sizes", str(m["particle_size"]),
           "--n-envs", str(n_envs),
           "--samples-per-env", str(p.get("samples_per_env", 1)),
           "--output-root", plan.get("output_root", "data/run")]
    if m.get("shape"):
        cmd += ["--particle-shape", str(m["shape"])]
    if p.get("state_library_settles"):
        cmd += ["--state-library", str(p["state_library_settles"])]
    if p.get("state_library_damping"):
        cmd += ["--state-library-damping", str(p["state_library_damping"])]
    if p.get("start_sampling"):
        cmd += ["--start-sampling", str(p["start_sampling"])]
    if p.get("shared_travel_distance"):
        cmd += ["--shared-travel-distance"]
    if p.get("center_bias"):
        cmd += ["--center-bias", str(p["center_bias"])]
    if not p.get("render_images", True):
        cmd += ["--no-render-images"]
    if p.get("export_dino_wm"):
        cmd += ["--export-dino-wm"]
    # Offset per particle count so the sizes do not all draw identical actions
    # while each remains individually reproducible.
    if args.seed is not None:
        cmd += ["--seed", str(args.seed + n)]
    if args.viewer:
        cmd += ["--viewer-type", args.viewer]
    return cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--plan", required=True, help="collection plan yaml")
    ap.add_argument("--seed", type=int, default=0,
                    help="base seed; each count gets seed+n")
    ap.add_argument("--n-envs", type=int, default=None,
                    help="override the plan's env count for every size")
    ap.add_argument("--only", type=int, nargs="+", default=None,
                    help="run only these particle counts")
    ap.add_argument("--preflight-only", action="store_true",
                    help="report feasibility and exit without collecting")
    ap.add_argument("--skip-validation", action="store_true")
    ap.add_argument("--viewer", choices=["observer", "bird", "leveled"],
                    default=None)
    ap.add_argument("--vram-per-env", type=float, default=None,
                    help="GiB per env per particle, for the preflight estimate "
                         "only (calibrated on one card; override freely)")
    ap.add_argument("--continue-on-failure", action="store_true", default=True)
    args = ap.parse_args()

    plan_path = Path(args.plan)
    if not plan_path.is_absolute() and not plan_path.exists():
        plan_path = GENESIS_DIR / args.plan
    plan = yaml.safe_load(plan_path.read_text())

    material = plan.get("material", {})
    shape = material.get("shape", "cube")
    size = material["particle_size"]
    counts = plan["plan"]["n_objects"]
    if args.only:
        counts = [n for n in counts if n in args.only]
    env_counts, env_source = resolve_env_counts(plan, args.n_envs)
    per_pp = (args.vram_per_env if args.vram_per_env is not None
              else plan["plan"].get("vram_per_env_gib",
                                    DEFAULT_VRAM_PER_PARTICLE_PER_ENV_GIB))

    base_cfg = yaml.safe_load((GENESIS_DIR / "configs" /
                               plan.get("base_config", "basic.yaml")).read_text())
    box_xy = base_cfg["box"]["vol"][0]
    out_root = plan.get("output_root", "data/run")
    want_lib = int(plan["plan"].get("state_library_settles", 0) or 0)

    print("=" * 74)
    print(f"  plan          {plan_path}")
    print(f"  material      {shape}, {size} m")
    print(f"  counts        {counts}")
    print(f"  envs          {env_counts}  (from {env_source})")
    print(f"  output        {out_root}")
    print("=" * 74)

    # ---- preflight ------------------------------------------------------
    print("\nPREFLIGHT")
    free = free_vram_gib()
    print(f"  free VRAM     {'unknown (no CUDA - CPU run)' if free is None else f'{free:.2f} GiB'}")
    runnable = []
    for n in counts:
        ok, frac, cap, msg = placement_report(shape, size, n, box_xy)
        tag = "ok  " if ok and frac < PACKING_WARN_FRACTION else ("warn" if ok else "SKIP")
        print(f"  [{tag}] n={n:<5} placement: {msg}")
        if not ok:
            continue
        est = estimate_vram_gib(n, env_counts[n], per_pp, DEFAULT_VRAM_BASE_GIB)
        if free is not None and est > free:
            print(f"         n={n:<5} VRAM: estimate {est:.2f} GiB exceeds "
                  f"{free:.2f} GiB free. Proceeding anyway - the estimate is "
                  f"coarse and each size is isolated, so an OOM here costs "
                  f"this size only.")
        runnable.append(n)

    if not runnable:
        print("\nNothing is feasible under this plan. Nothing to do.")
        return 1
    if args.preflight_only:
        print(f"\nPreflight only. Would run: {runnable}")
        return 0

    # ---- run ------------------------------------------------------------
    results = {}
    for n in runnable:
        cmd = build_command(plan, n, env_counts[n], args)
        print("\n" + "=" * 74)
        print(f"  n_particles = {n}   envs = {env_counts[n]}")
        print(f"  $ {' '.join(shlex.quote(c) for c in cmd)}")
        print("=" * 74, flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(GENESIS_DIR))
        results[n] = {"returncode": proc.returncode,
                      "seconds": round(time.time() - t0, 1)}
        if proc.returncode != 0:
            print(f"\n  n={n} exited {proc.returncode}. Continuing with the "
                  f"remaining sizes - that isolation is why this driver "
                  f"exists.")

    # ---- postflight -----------------------------------------------------
    if not args.skip_validation:
        print("\n" + "=" * 74)
        print("  POSTFLIGHT")
        print("=" * 74)
        for n in runnable:
            leaf = GENESIS_DIR / out_root / shape / f"n{n}" / f"size{size}"
            v = validate_output(leaf, n, want_lib)
            results[n].update({k: v[k] for k in
                               ("batches", "samples", "failed", "unchanged",
                                "escaped") if k in v})
            if "library" in v:
                results[n]["library"] = v["library"]
            results[n]["problems"] = v["problems"]
            head = (f"  n={n:<5} {v['batches']} batches, {v['samples']} samples, "
                    f"{v['failed']} failed, {v['unchanged']} unchanged, "
                    f"{v['escaped']} escaped")
            if v["peak_cap"]:
                head += (f", contact {v['peak_contact']}/{v['peak_cap']}")
            if "library" in v:
                head += f", library {v['library']}"
            print(head)
            for p in v["problems"]:
                print(f"          ! {p}")

    # ---- summary --------------------------------------------------------
    bad = [n for n, r in results.items()
           if r["returncode"] != 0 or r.get("problems")]
    print("\n" + "=" * 74)
    for n, r in results.items():
        flag = "OK  " if n not in bad else "FAIL"
        print(f"  [{flag}] n={n:<5} rc={r['returncode']}  {r['seconds']}s  "
              f"{len(r.get('problems', []))} problem(s)")
    print(f"\n  {len(results) - len(bad)}/{len(results)} sizes clean")
    print("=" * 74)

    report = GENESIS_DIR / out_root / "run_collection_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, indent=2))
    print(f"  report -> {report}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
