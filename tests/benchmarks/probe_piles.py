"""Which particle shapes can hold a 3D pile, and can pushes build one?

Three separate questions. Conflating them produced two wrong conclusions
during this port, so they are separated explicitly:

  --retain   spawn a compact multi-layer stack, settle, then push. Does the
             structure survive? (a shape can retain without being able to
             build)
  --pour     drop a random tower over the tray centre, well clear of every
             wall, and settle. Does a heap form at all, held by friction
             alone? The slope reported is an angle of repose; a physical one
             for spheres is 25-40 degrees, so a much larger value means the
             INITIAL CONDITION is holding it up, not the physics. Positions
             are rejection-sampled per level for exactly that reason - a
             hexagonal lattice is a stable crystal and will happily stand at
             70 degrees while telling you nothing.
  --build    start from a settled monolayer and push repeatedly, with the
             stop biased toward the tray centre. Does material ever end up
             on top of other material?

"stacked" is measured against the particle's own Z EXTENT, not its nominal
size: a `rectangle` rod of nominal 8.5 mm is 8.5 x 4.25 x 4.25, so testing
against the nominal would miss a genuine second layer entirely.

    python tests/benchmarks/probe_piles.py --retain --shape cube
    python tests/benchmarks/probe_piles.py --pour   --shape sphere
    python tests/benchmarks/probe_piles.py --pour   --shape sphere --size 0.005,0.012
    python tests/benchmarks/probe_piles.py --build  --shape cube
"""
import argparse
import math

from _common import add_paths, banner, make_sim
add_paths(layered=True)

import torch                                              # noqa: E402
import genesis as gs                                       # noqa: E402


def parse_size(spec):
    return [float(x) for x in spec.split(",")] if "," in spec else float(spec)


def reporter(sim, zext, floor, n):
    def report(tag):
        p = sim._get_particle_positions()[0, :n]
        z, r = p[:, 2] - floor, p[:, :2].norm(dim=1)
        top = float(z.max()) + zext / 2
        r90 = float(torch.quantile(r, 0.90))
        slope = math.degrees(math.atan2(top, max(r90, 1e-9)))
        return (f"  {tag:22s} top {top*1e3:7.2f} mm = {top/zext:5.2f} extents | "
                f"r90 {r90*1e3:5.1f} mm | slope {slope:5.1f} deg | "
                f"stacked {int((z > zext).sum()):4d}/{n}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="cube",
                    choices=["cube", "sphere", "cylinder", "rectangle"])
    ap.add_argument("--size", default="0.0085",
                    help="scalar, or min,max for a polydisperse spread")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--friction", type=float, default=0.9)
    ap.add_argument("--pushes", type=int, default=5)
    ap.add_argument("--radius", type=float, default=0.025, help="--pour column radius")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--retain", action="store_true")
    g.add_argument("--pour", action="store_true")
    g.add_argument("--build", action="store_true")
    a = ap.parse_args()

    size = parse_size(a.size)
    dmax = max(size) if isinstance(size, list) else size
    layers = 14 if a.pour else (1 if a.build else 2)

    sim = make_sim(shape=a.shape, size=size, n=a.n, friction=a.friction,
                   layered=True, n_layers=layers,
                   config_overrides={"plate": {"scale_height_with_layers": not a.pour}})
    banner()
    zext = max(s[2] for s in sim._sampled_params["particle_sizes"])
    floor = sim._wall_thickness / 2
    print(f"  {a.shape} {a.size} n={a.n} friction={a.friction} layers={layers}")
    print(f"  extents (mm) {[round(v*1e3,2) for v in sim._sampled_params['particle_sizes'][0]]}"
          f"   z extent {zext*1e3:.2f} mm\n")
    report = reporter(sim, zext, floor, a.n)

    if a.pour:
        gen = torch.Generator().manual_seed(0)
        pos = torch.zeros((a.n, 7)); pos[:, 3] = 1.0
        placed, in_level, z = 0, [], floor + dmax / 2
        while placed < a.n:
            for _ in range(400):
                th = float(torch.rand(1, generator=gen)) * 2 * math.pi
                rr = a.radius * math.sqrt(float(torch.rand(1, generator=gen)))
                x, y = rr * math.cos(th), rr * math.sin(th)
                if all(math.hypot(x-px, y-py) >= dmax*1.02 for px, py in in_level):
                    in_level.append((x, y))
                    pos[placed, 0:3] = torch.tensor([x, y, z]); placed += 1
                    break
            else:
                in_level = []; z += dmax * 1.02
        sim.set_particle_state(pos)
        print(report("poured"))
    else:
        sim.shuffle_particles()
        print(report("spawned"))

    c = [0]
    sim.update_material_state(on_step=lambda s: c.__setitem__(0, s+1))
    cap = " (HIT CAP - did not converge)" if c[0] >= sim._settle_steps else ""
    print(report(f"settled ({c[0]} st)") + cap)

    for k in range(a.pushes):
        kw = dict(particle_xy=sim._particle_state[:, :, 0:2])
        if a.build:
            kw["center_bias"] = 0.7
        st, sp, an = sim.generate_action_samples(1, **kw)
        sim.execute_action(st[:, 0], sp[:, 0], an[:, 0])
        sim.update_material_state()
        print(report(f"after push {k+1}"))
    print(f"  escaped {sim.escaped_particle_count()}")
    if a.pour:
        print("  NB a physical angle of repose is 25-40 deg. Far above that means")
        print("     the initial arrangement is load-bearing, not friction.")


if __name__ == "__main__":
    main()
