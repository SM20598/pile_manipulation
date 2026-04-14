import genesis as gs
import numpy as np

from utilities.sandbox import spawn_sandbox
from utilities.robots import spawn_robot

########################## init ##########################

# Settings
dt = 4e-3
substeps = 1
debug_viewer = True
horizon = 1000
backend = gs.gpu
precision = '32' # increase to 64 if f64 is needed
performance_mode = False # set to True for policy training

# Visualization
camera_type = "observer" # look at robot from front
# camera_type = "bird" # top down view on robot
# camera_type = "leveled" # look at robot from the side, with camera height leveled with box center

# Granular sandbox settings
box_pos = (0.5, 0.0, 0.5) # center of the box
box_vol = (0.3, 0.5, 0.1)
wall_thickness = 0.02 # thickness of the box wall
granular_vol = (0.27, 0.47, 0.1)
material_type = "granular" # granular, sand, or liquid. Default is granular (Random Sequential Addition)
material_properties = dict()
particle_size = 0.01
omit_box = False # whether to omit the box and only spawn granular material


# Robot settings
robot_name = 'franka'
robot_pos = (0.0, 0.0, 0.0) # base position of the robot
dofs_kp = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100])
dofs_kv = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10])
dofs_force_range = (
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
)


##################################### #####################
b_x, b_y, b_z = box_pos
v_x, v_y, v_z = granular_vol

gs.init(
    backend=backend,
    precision=precision,
    performance_mode=performance_mode,
)

if camera_type == "observer":
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (1.5, 0.0, 1),
        camera_lookat = (0.5, 0.0, 0.2),
        camera_fov    = 30,
        max_FPS       = 60,
    )
elif camera_type == "bird":
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (0.5, 0.0, 1.2),
        camera_lookat = (0.5, 0.0, 0),
        camera_fov    = 30,
        max_FPS       = 60,
    )
elif camera_type == "leveled":
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (b_x+1.5, b_y, b_z),
        camera_lookat = box_pos,
        camera_fov    = 30,
        max_FPS       = 60,
    )

########################## Scene ##########################
l_bound = (b_x-2*v_x, b_y-2*v_y, b_z-2*v_z)
u_bound = (b_x+2*v_x, b_y+2*v_y, b_z+2*v_z+wall_thickness)


scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt       = dt,
        substeps = substeps,
    ),
    mpm_options=gs.options.MPMOptions(
        lower_bound = l_bound,
        upper_bound = u_bound,
        particle_size = particle_size,
    ),
    sph_options=gs.options.SPHOptions(
        lower_bound = l_bound,
        upper_bound = u_bound,
        particle_size = particle_size,
    ),
    pbd_options=gs.options.PBDOptions(
        lower_bound = l_bound,
        upper_bound = u_bound,
        particle_size = particle_size,
    ),
    viewer_options = viewer_options,
    show_viewer=debug_viewer,
)

# Entities
plane = scene.add_entity(
    gs.morphs.Plane()
)
robot = spawn_robot(
    scene=scene,
    robot_name=robot_name,
    pos=robot_pos
)

granular_material = spawn_sandbox(
    scene=scene,
    material_type=material_type,
    material_properties=material_properties,
    box_pos=box_pos,
    box_vol=box_vol,
    granular_vol=granular_vol,
    wall_thickness=wall_thickness,
    omit_box=omit_box
)

scene.build()

# Tune robot control gains
robot.set_dofs_kp(dofs_kp)
robot.set_dofs_kv(dofs_kv)
robot.set_dofs_force_range(*dofs_force_range)

# initial_pose
# end_effector = robot.get_link("hand")
# qpos = robot.inverse_kinematics(
#     link = end_effector,
#     pos  = np.array([0.5, 0.0, 0.3]),  # target place pose
#     quat = np.array([0, 1, 0, 0]),
# )
# qpos[7:] = 0.0
# robot.set_qpos(qpos)


# qpos = robot.inverse_kinematics(
#     link = end_effector,
#     pos  = np.array([0.5, 0.1, 0.3]),  # target place pose
#     quat = np.array([0, 1, 0, 0]),
# )
# qpos[7:] = 0.0
# path = robot.plan_path( 
#     qpos_goal     = qpos,
#     num_waypoints = 200, # 2s duration
# )
# for waypoint in path:
#     robot.control_dofs_position(waypoint)
#     scene.step()    

for _ in range(250):
    scene.visualizer.update()

for _ in range(horizon):
    scene.step()
    
# q_pregrasp = robot.inverse_kinematics(
#     link = end_effector,
#     pos  = np.array([0.65, 0.0, 0.13]),  # just above the cube
#     quat = np.array([0, 1, 0, 0]),       # down-facing orientation
# )
# robot.control_dofs_position(q_pregrasp[:-2], np.arange(7))  # arm joints only
# for _ in range(horizon):
#     scene.step()

# qpos = robot.inverse_kinematics(
#     link = end_effector,
#     pos  = np.array([0.75, 0.0, 0.15]),  # target place pose
#     quat = np.array([0, 1, 0, 0]),
# )
# qpos[7:]=0.0
# path = robot.plan_path(
#     qpos_goal     = qpos,
#     num_waypoints = 200, # 2s duration
# )

# for waypoint in path:
#     robot.control_dofs_position(waypoint)
#     scene.step()
