import genesis as gs
import numpy as np

gs.init(
    backend=gs.gpu,
    precision='32', # increase to 64 if f64 is needed
    performance_mode=False # set to True for policy training
)

########################### Build Scene ##########################

scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'))
cube = scene.add_entity(gs.morphs.Box(size=(0.08, 0.08, 0.08), pos=(0.65, 0.0, 0.13)))
scene.build()


# Retrieve some commonly used handles
rigid        = scene.sim.rigid_solver   # low-level rigid body solver
end_effector = franka.get_link("hand")  # Franka gripper frame
cube_link    = cube.base_link           # the link we want to pick

################ Reach pre-grasp pose ################
q_pregrasp = franka.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.65, 0.0, 0.13]),  # just above the cube
    quat = np.array([0, 1, 0, 0]),       # down-facing orientation
)
franka.control_dofs_position(q_pregrasp[:-2], np.arange(7))  # arm joints only
for _ in range(50):
    scene.step()

################ Attach (activate suction) ################
link_cube   = np.array(cube_link.idx,    dtype=gs.np_int)
link_franka = np.array(end_effector.idx, dtype=gs.np_int)
print(link_cube.ndim, link_franka.ndim)
rigid.add_weld_constraint(link_cube, link_franka)

################ Lift and transport ################
q_lift = franka.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.65, 0.0, 0.28]),  # lift up
    quat = np.array([0, 1, 0, 0]),
)
franka.control_dofs_position(q_lift[:-2], np.arange(7))
for _ in range(50):
    scene.step()

q_place = franka.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.4, 0.2, 0.18]),  # target place pose
    quat = np.array([0, 1, 0, 0]),
)
franka.control_dofs_position(q_place[:-2], np.arange(7))
for _ in range(100):
    scene.step()

################ Detach (release suction) ################
rigid.delete_weld_constraint(link_cube, link_franka)
for _ in range(400):
    scene.step()