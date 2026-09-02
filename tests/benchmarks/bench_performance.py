"""The two performance claims in PORT_NOTES.md, measured on THIS machine.

  1. Batched particle pose writes vs the per-particle loop they replaced.
     RigidEntity.set_pos/set_quat each run a whole-scene forward-kinematics
     pass, so the loop costs 2N kernel launches and 2N full FK passes; the
     solver-level setters take a link-index array and do it in two launches
     plus one FK. This also asserts the two are BIT-IDENTICAL, which is the
     part that must hold regardless of hardware.

  2. Restoring a settled state from the library vs shuffling and settling.
     shuffle_particles() runs zero simulation steps, so a reset's whole cost
     is the settle that follows it.

Neither speedup is quoted in the docs, because both depend entirely on the
backend and the particle count. Run this instead.

    python tests/benchmarks/bench_performance.py [--n 100] [--envs 4]
"""
import argparse
import time

from _common import add_paths, banner, make_sim
add_paths()

import torch                                             # noqa: E402
import genesis as gs                                      # noqa: E402


def bench(fn, repeats):
    fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1e3     # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="particles")
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--size", type=float, default=0.005)
    ap.add_argument("--repeats", type=int, default=30)
    a = ap.parse_args()

    sim = make_sim(size=a.size, n=a.n, n_envs=a.envs)
    banner()
    print(f"  {a.n} particles of {a.size*1000:.2f} mm x {a.envs} envs\n")

    sim.shuffle_particles()
    sim.update_material_state()
    target = sim._particle_state.clone()

    # ---- 1. pose writes -------------------------------------------------
    envs_idx = torch.arange(sim._n_envs, device=gs.device)

    def loop_write():
        for i, p in enumerate(sim.material):
            p.set_pos(target[:, i, 0:3].contiguous(), envs_idx=envs_idx)
            p.set_quat(target[:, i, 3:7].contiguous(), envs_idx=envs_idx)

    def batched_write():
        sim._set_particle_positions(target[..., 0:3], target[..., 3:7])

    sim.shuffle_particles(); loop_write()
    ref = (sim._get_particle_positions().clone(), sim._get_particle_quats().clone())
    sim.shuffle_particles(); batched_write()
    new = (sim._get_particle_positions().clone(), sim._get_particle_quats().clone())
    dp = float((ref[0] - new[0]).abs().max())
    dq = float((ref[1] - new[1]).abs().max())

    t_loop = bench(loop_write, a.repeats)
    t_batch = bench(batched_write, a.repeats)
    print("1. PARTICLE POSE WRITE")
    print(f"   per-particle loop   {t_loop:9.3f} ms")
    print(f"   batched             {t_batch:9.3f} ms   ->  {t_loop/t_batch:.1f}x")
    print(f"   equivalence         pos {dp:.3e}   quat {dq:.3e}"
          f"   {'OK (bit-identical)' if dp == 0 and dq == 0 else 'MISMATCH'}")

    # ---- 2. reset path --------------------------------------------------
    from state_library import build_state_library
    print("\n2. RESET: shuffle+settle vs library restore")
    t0 = time.perf_counter(); sim.shuffle_particles(); sim.update_material_state()
    t_settle = time.perf_counter() - t0
    lib = build_state_library(sim, n_settles=2, verbose=False)
    t0 = time.perf_counter(); lib.apply(sim)
    t_restore = time.perf_counter() - t0
    print(f"   shuffle + settle    {t_settle*1e3:9.1f} ms")
    print(f"   library restore     {t_restore*1e3:9.1f} ms   ->  "
          f"{t_settle/max(t_restore,1e-9):.0f}x")
    lin, ang = sim._pile_motion()
    print(f"   restored pile at rest without settling: {sim._pile_is_at_rest()} "
          f"(peak {lin*1e3:.4f} mm/s)")
    print(f"   library size        {len(lib)} states from 2 settles x {a.envs} envs "
          f"x {lib.meta['n_symmetries']} symmetries")


if __name__ == "__main__":
    main()
