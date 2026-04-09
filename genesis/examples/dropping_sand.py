import genesis as gs

########################## init ##########################
gs.init()
dt = 4e-3
substeps = 10
debug_viewer = True
horizon = 250
########################## create a scene ##########################

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt       = dt,
        substeps = substeps,
    ),
    mpm_options=gs.options.MPMOptions(
        lower_bound   = (-0.5, -0.5, 0),
        upper_bound   = (0.5, 0.5, 0.5),
        particle_size = 0.01,
    ),
    vis_options=gs.options.VisOptions(
        visualize_mpm_boundary = True,
    ),
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (3, -1, 0.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 30,
    ),
    show_viewer = debug_viewer,
)

########################## entities ##########################
plane = scene.add_entity(
    morph=gs.morphs.Plane(),
)

liquid = scene.add_entity(
    material=gs.materials.MPM.Sand(
        E=1e6,
        nu=0.2,
        rho=1000,
        sampler='random',
        friction_angle=80
    ),
    morph=gs.morphs.Box(
        pos  = (0.0, 0.0, 0.2),
        size = (0.3, 0.3, 0.3),
    ),
    surface=gs.surfaces.Default(
        color    = (0.3, 0.3, 1.0),
        vis_mode = 'particle',
    ),
)

cam = scene.add_camera(
    res    = (640, 480),
    pos    = (3.5, 0.0, 0.5),
    lookat = (0, 0, 0.5),
    fov    = 30,
    GUI    = False,
)

########################## build ##########################
scene.build()

cam.start_recording()
for i in range(horizon):
    scene.step()
    cam.render()
cam.stop_recording(save_to_filename='video.mp4', fps=1/dt)