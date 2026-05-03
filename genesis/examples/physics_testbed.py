import argparse
import os

import torch
import numpy as np
import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(precision="32", logging_level="warning")

    ########################## create a scene ##########################

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            substeps=10,
            gravity=(0, 0, -9.81)
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2, 0., 1),
            camera_lookat=(0.0, 0.0, 1),
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    plane = scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    
    slope = scene.add_entity(
        material=gs.materials.Rigid(
            friction=0.01
        ),
        morph=gs.morphs.Box(
            fixed=True,
            pos=(0, 0, 3),
            size=(5, 5, 0.5),
            euler=(0, 45, 0)
            
        )
    )

    obj1 = scene.add_entity(
        material=gs.materials.Rigid(
            friction=0.01
        ),
        morph=gs.morphs.Box(
            pos=(-0.8, -0.2, 4.5),
            size=(0.2, 0.2, 0.2),
            euler=(0, 45, 0)
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0, 0, 1.0),
        ),
        vis_mode="collision"
    )

    obj2 = scene.add_entity(
        material=gs.materials.Rigid(
            friction=0.01
        ),
        morph=gs.morphs.Box(
            pos=(-0.8, 0.2, 4.5),
            size=(0.2, 0.2, 0.2),
            euler=(0, 45, 0)
        ),
        surface=gs.surfaces.Default(
            color=(0.0, 1.0, 0, 1.0),
        ),
        vis_mode="collision"
    )


    ########################## build ##########################
    scene.build()
    obj1.set_mass(0.001)
    obj2.set_mass(1)
    obj1.morph.size = (0.001, 0.001, 0.001)
    for _ in range(1000):
        scene.step()


if __name__ == "__main__":
    main()