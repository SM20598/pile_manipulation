import genesis as gs

########################## init ##########################
gs.init()
dt = 4e-3
substeps = 40
debug_viewer = True
horizon = 500
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
    pbd_options=gs.options.PBDOptions(
        lower_bound   = (-0.5, -0.5, 0),
        upper_bound   = (0.5, 0.5, 0.5),
        particle_size = 0.01,
    ),
    sph_options=gs.options.SPHOptions(
        lower_bound   = (-0.5, -0.5, 0),
        upper_bound   = (0.5, 0.5, 0.5),
        particle_size = 0.01,
    ),
    vis_options=gs.options.VisOptions(
        visualize_mpm_boundary = True,
    ),
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (0.8, 0.0, 0.2),
        camera_lookat = (0.0, 0.0, 0.1),
        camera_fov    = 30,
    ),
    show_viewer = debug_viewer,
)

########################## entities ##########################
plane = scene.add_entity(
    morph=gs.morphs.Plane(),
)

cube = scene.add_entity(
    morph=gs.morphs.Box(
        pos=(0.0, 0.0, 0.1),
        size= (0.1, 0.1, 0.1),
        fixed=True,
    )
)

liquid = scene.add_entity(
    material=gs.materials.PBD.Liquid(
        rho = 1000.0,
        sampler = 'pbs',
        density_relaxation = 0.2,
        viscosity_relaxation = 0.01
    ),
    morph=gs.morphs.Box(
        pos  = (0.0, 0.0, 0.25),
        size = (0.1, 0.1, 0.1),
    ),
    surface=gs.surfaces.Default(
        color    = (0.3, 0.3, 1.0),
        vis_mode = 'particle',
    ),
)

# emitter = scene.add_emitter(
#     material=gs.materials.PBD.Liquid(),
#     max_particles=100000,
#     surface=gs.surfaces.Glass(color=(0.7, 0.85, 1.0, 0.7)),
# )

cam = scene.add_camera(
    res    = (640, 480),
    pos    = (3.5, 0.0, 0.5),
    lookat = (0, 0, 0.5),
    fov    = 30,
    GUI    = False,
)

########################## build ##########################
scene.build()

# cam.start_recording()
# for i in range(horizon):
#     scene.visualizer.update()
#     # cam.render()
for i in range(horizon):
    emitter.emit(
        pos=(0.0, 0.0, 0.3),
        direction=(0.0, 0.0, -1.0),
        speed=1.0,
        droplet_shape="circle",
        droplet_size=0.04,
    )
    scene.step()

# cam.stop_recording(save_to_filename='video.mp4', fps=1/dt)