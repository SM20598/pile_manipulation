import genesis as gs
import numpy as np

from utilities.sandbox import add_sandbox

########################## init ##########################
dt = 4e-3
substeps = 10
debug_viewer = True
horizon = 1000

gs.init(
    backend=gs.gpu,
    precision='32', # increase to 64 if f64 is needed
    performance_mode=False # set to True for policy training
)

########################## Scene ##########################
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt       = dt,
        substeps = substeps,
    ),
    mpm_options=gs.options.MPMOptions(
        lower_bound = (-1, -1, 0),
        upper_bound = (1, 1, 1),
        particle_size = 0.01
    ),
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (0.5, 0.0, 0.5),
        camera_lookat = (0.5, 0.0, 0.1),
        camera_fov    = 30,
        max_FPS       = 60,
    ),
    show_viewer=debug_viewer,
)
plane = scene.add_entity(
    gs.morphs.Plane()
)
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)
add_sandbox(scene)

scene.build()

for i in range(horizon):
    scene.step()