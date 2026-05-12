"""
MPM to Rigid Link Attachment

Demonstrates attaching MPM particles to rigid links using soft constraints.
"""

import argparse
import os

import torch

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=20),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-1.0, -1.0, 0.0),
            upper_bound=(1.0, 1.0, 1.5),
            grid_density=64,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0.0, 0.8),
            camera_lookat=(0.0, 0.0, 0.4),
        ),
        show_viewer=args.vis,
    )

    gripper = scene.add_entity(
        morph=gs.morphs.URDF(file="gripper.urdf", fixed=True),
        material=gs.materials.Hybrid(
            material_rigid=gs.materials.Rigid(gravity_compensation=1.0),
            material_soft=gs.materials.MPM.Muscle(E=1e4, nu=0.45),
            thickness=0.02,
            damping=100.0,
        ),
    )

    # Add object to grasp
    ball = scene.add_entity(
        morph=gs.morphs.Sphere(pos=(0.5, 0.5, 0.1), radius=0.05),
    )

    scene.build()

    # Close gripper
    for step in range(500):
        gripper.control_dofs_position([0.5] * gripper.n_dofs)
        scene.step()


if __name__ == "__main__":
    main()