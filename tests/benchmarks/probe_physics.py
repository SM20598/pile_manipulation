"""Physics properties that should reproduce on ANY machine.

Unlike bench_performance.py, nothing here is a timing: these are step counts,
distances and angles, so a disagreement is a real regression rather than a
hardware difference. That is the point of separating them.

Checks:
  plate     sweep step count, tracking error vs the commanded path, cruise
            speed against the commanded speed, PD gains against m*w^2 / 2*m*w
  settling  true steps-to-rest (check interval forced to 1), peak particle
            speed at exit, and net drift over a further hold - the last is
            what shows the rest criterion is not exiting early
  action    goal-reached rate and final tracking error over sampled actions

    python tests/benchmarks/probe_physics.py [--n 50] [--size 0.005]
"""
import argparse
import math
import statistics as st

from _common import add_paths, banner, make_sim
add_paths()

import torch                                              # noqa: E402
import genesis as gs                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--size", type=float, default=0.005)
    ap.add_argument("--envs", type=int, default=2)
    ap.add_argument("--hold", type=int, default=200,
                    help="steps to hold past the rest criterion, for drift")
    a = ap.parse_args()

    sim = make_sim(size=a.size, n=a.n, n_envs=a.envs)
    banner()
    print(f"  {a.n} cubes of {a.size*1000:.2f} mm x {a.envs} envs\n")

    # ---- plate actuator -------------------------------------------------
    m, bw = sim._plate_moving_mass, sim._plate_bandwidth
    w = 2 * math.pi * bw
    kp = sim.plate.get_dofs_kp()[:3].tolist()
    kv = sim.plate.get_dofs_kv()[:3].tolist()
    print("PLATE ACTUATOR")
    print(f"   kp expected {m*w**2:9.2f}   actual {kp[0]:9.2f}")
    print(f"   kv expected {2*m*w:9.2f}   actual {kv[0]:9.2f}")
    print(f"   armature    {sim.plate.get_dofs_armature()[:3].tolist()[0]:9.2f}"
          f"   (= plate.moving_mass)")

    sim.shuffle_particles(); sim.update_material_state()

    D, h = gs.device, sim._operation_height
    # Broadside (yaw = pi/2) and 80 mm, both deliberate. Edge-on, the blade
    # spans its 40 mm axis ALONG the travel, so a +/-45 mm sweep puts the
    # leading edge at 65 mm against a 64 mm tray half-width and drives into the
    # wall - measured, that gave 7 N of "contact" and a 1 mm terminal
    # deflection ON AN EMPTY TRAY, which looks exactly like a tracking failure
    # and is not one. 80 mm broadside is inside generate_action_samples' own
    # bound for this yaw (85 mm), so it is also a realistic action.
    dist, yaw = 0.08, math.pi / 2
    p0 = torch.tensor([[-dist/2, 0.0, h]], device=D).expand(a.envs, -1).contiguous()
    p1 = torch.tensor([[+dist/2, 0.0, h]], device=D).expand(a.envs, -1).contiguous()
    sim._vertical_dof_fix[:, 0] = p0[:, 0]
    sim._vertical_dof_fix[:, 1] = p0[:, 1]
    sim._vertical_dof_fix[:, 4] = yaw
    lower = p0 + sim._clearance_offset
    sim.plate.set_pos(lower, zero_velocity=True)
    sim._reaction_reset()
    # Descend the way execute_action does. Sweeping straight from the parked
    # pose teleports the blade into the pile and the sweep spends its length
    # recovering, which is a property of the probe, not the controller.
    sim.plate_position_translation(lower, p0, sim._clearance_ctrl_steps, phase="lower")
    tr = []
    rg, fp = sim.plate_velocity_translation(
        p0, p1, torch.full((a.envs,), yaw, device=D),
        on_step=lambda s, pr, vr: tr.append((sim.plate.get_pos()[0].clone(),
                                             pr[0].clone())))
    err = [float((x[:2] - y[:2]).norm()) * 1e3 for x, y in tr]
    dt = sim._scene.dt
    sp = [float((tr[i][0][:2] - tr[i-1][0][:2]).norm()) / dt * 1e3
          for i in range(1, len(tr))]
    mid = sp[len(sp)//3: 2*len(sp)//3]
    old_law = math.ceil(dist / (sim._plate_params["speed"] * dt) * 1.7)
    print(f"\nSWEEP ({dist*1000:.0f} mm, blade broadside at yaw={yaw:.3f})")
    print(f"   steps            {len(tr):6d}   (endpoint-target law would use {old_law})")
    print(f"   tracking error   mean {st.mean(err):.2f} mm   max {max(err):.2f} mm")
    print(f"   cruise speed     {st.mean(mid):6.1f} mm/s   commanded "
          f"{sim._plate_params['speed']*1000:.1f}")
    print(f"   final error      {float((fp[:,:2]-p1[:,:2]).norm(dim=1).max())*1e3:.3f} mm")

    # ---- settling -------------------------------------------------------
    sim._settle_check_every = 1          # expose the TRUE convergence step
    sim.shuffle_particles()
    c = [0]; sim.update_material_state(on_step=lambda s: c.__setitem__(0, s+1))
    fresh = c[0]
    lin, ang = sim._pile_motion()
    st_, sp_, an_ = sim.generate_action_samples(1)
    sim.execute_action(st_[:, 0], sp_[:, 0], an_[:, 0])
    c = [0]; sim.update_material_state(on_step=lambda s: c.__setitem__(0, s+1))
    push = c[0]
    lin2, ang2 = sim._pile_motion()
    before = sim._get_particle_positions().clone()
    frozen = sim.plate.get_dofs_position()
    sim.plate.control_dofs_position_velocity(frozen, torch.zeros_like(frozen),
                                             dofs_idx_local=[0, 1, 2, 3, 4, 5])
    for _ in range(a.hold):
        sim._step_scene()
    drift = (sim._get_particle_positions() - before).norm(dim=-1) * 1e3
    print(f"\nSETTLING (check interval forced to 1, so these are TRUE step counts)")
    print(f"   fresh spawn      {fresh:6d} steps   peak {lin*1e3:.3f} mm/s "
          f"{ang:.3f} rad/s")
    print(f"   after a push     {push:6d} steps   peak {lin2*1e3:.3f} mm/s "
          f"{ang2:.3f} rad/s")
    print(f"   thresholds       {sim._settle_vel_threshold*1e3:.1f} mm/s   "
          f"{sim._settle_angvel_threshold:.4f} rad/s (derived)")
    print(f"   drift over {a.hold} further steps: max {float(drift.max()):.3f} mm, "
          f"{int((drift>1).sum())} particles > 1 mm")

    # ---- sampled actions ------------------------------------------------
    torch.manual_seed(0)
    starts, stops, angles = sim.generate_action_samples(3)
    ok = tot = 0
    worst = 0.0
    for k in range(3):
        s0 = sim._particle_state.clone()
        rg, fp = sim.execute_action(starts[:, k], stops[:, k], angles[:, k])
        sim.update_material_state()
        d = (sim._particle_state[..., :3] - s0[..., :3]).norm(dim=-1) * 1e3
        e = float((fp[:, :2] - stops[:, k, :2]).norm(dim=1).max()) * 1e3
        worst = max(worst, e)
        ok += int(rg.sum()); tot += a.envs
    print(f"\nSAMPLED ACTIONS")
    print(f"   goal reached     {ok}/{tot}   worst final error {worst:.3f} mm")
    print(f"   escaped          {sim.escaped_particle_count()}")
    print(f"   contact budget   {sim.contact_budget_usage()}")


if __name__ == "__main__":
    main()
