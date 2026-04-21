import genesis as gs
import genesis.utils.geom as gu 
import numpy as np
import yaml
from utilities.materials import *
from utilities.helper_functions import quaternion_multiply, get_horizontal_path, get_vertical_path 
import pathlib
import quaternion as qu


class SandboxManipulation:

    def __init__(self, config,):
        base_dir = pathlib.Path(__file__).parent
        full_path = base_dir / config
        with open(full_path) as stream:
            try:
                self._config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        
        # Initialize Genesis Environment
        gs.init(
            backend=getattr(gs, self._config["simulation"].get('backend', 'gpu')),
            precision=self._config["simulation"].get('precision', '32'),
            performance_mode=self._config["simulation"].get('performance_mode', False),
        )

        # PARAMETERS FOR TRAINING
        self._box_pos = self._config["sandbox"]["box"].get('pos', [0.5, 0.0, 0.0])
        self._box_vol = self._config["sandbox"]["box"].get('vol', [0.3, 0.3, 0.1])
        self._wall_thickness = self._config["sandbox"]["box"].get('wall_thickness', 0.02)
        self._particle_size = self._config["sandbox"]["material"].get('particle_size', 0.01)
        self._granular_vol = self._config["sandbox"]["material"].get('vol', [0.27, 0.27, 0.1])
        

        self._init_scene()
        self._add_entities()

    def _init_scene(self):
        viewer_settings = self._config["simulation"].get('viewer_options', dict())
        viz_settings = self._config["simulation"].get('viz_options', dict())
        c_fov = viewer_settings.get('camera_fov', 30)
        max_fps = viewer_settings.get('max_FPS', 60)
        resolution = viewer_settings.get('resolution', [1280, 1280])

        b_x, b_y, b_z = self._box_pos   
        v_x, v_y, v_z = self._box_vol
        l_bound = (b_x-2*v_x, b_y-2*v_y, b_z-2*v_z)
        u_bound = (b_x+2*v_x, b_y+2*v_y, b_z+2*v_z+self._wall_thickness)

        viewer_type = viewer_settings.get('viewer_type', None)
        
        if viewer_type == "observer":
            self._viewer_options = gs.options.ViewerOptions(
                camera_pos    = viewer_settings.get('camera_pos', [1.5, 0.0, 1.3]),
                camera_lookat = viewer_settings.get('camera_lookat', [0.5, 0.0, 0.2]),
                camera_fov    = c_fov,
                max_FPS       = max_fps,
                res           = resolution,
            )
        elif viewer_type == "bird":
            self._viewer_options = gs.options.ViewerOptions(
                camera_pos    = viewer_settings.get('camera_pos', [0.5, 0.0, 1.2]),
                camera_lookat = viewer_settings.get('camera_lookat', [0.5, 0.0, 0.0]),
                camera_fov    = c_fov,
                max_FPS       = max_fps,
                res           = resolution,
            )
        elif viewer_type == "leveled":
            self._viewer_options = gs.options.ViewerOptions(
                camera_pos    = viewer_settings.get('camera_pos', [b_x+1.5, b_y, b_z]),
                camera_lookat = viewer_settings.get('camera_lookat', [0.5, 0.0, 0.2]),
                camera_fov    = c_fov,
                max_FPS       = max_fps,
                res           = resolution,
            )
        else:
            # No viewer --> Training mode
            self._viewer_options = None

        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt       = self._config["simulation"].get('dt', 4e3),
                substeps = self._config["simulation"].get('substeps', 1),
            ),
            mpm_options=gs.options.MPMOptions(
                lower_bound = l_bound,
                upper_bound = u_bound,
                particle_size = self._particle_size,
            ),
            sph_options=gs.options.SPHOptions(
                lower_bound = l_bound,
                upper_bound = u_bound,
                particle_size = self._particle_size,
            ),
            pbd_options=gs.options.PBDOptions(
                lower_bound = l_bound,
                upper_bound = u_bound,
                particle_size = self._particle_size,
            ),
            viewer_options = self._viewer_options,
            vis_options=gs.options.VisOptions(
                show_link_frame=viz_settings.get('show_link_frame', False),
            ),
            show_viewer=viewer_settings.get('show_viewer', False)
        )
        self._scene.profiling_options.show_FPS = viz_settings.get('show_FPS', False)
    
    def _add_entities(self):

        self.plane = self._scene.add_entity(
            gs.morphs.Plane()
        )

        base_dir = pathlib.Path(__file__).parent
        full_path = base_dir / self._config["robot"].get('file', "utilities/xml/franka_emika_panda_with_tool/panda_robola_compile.xml")
        
        self.robot = self._scene.add_entity(
            gs.morphs.MJCF(
                file=full_path,
                pos=tuple(self._config["robot"].get('pos', [0, 0, 0]))   
                ),
        )

        if not self._config["sandbox"]["box"].get('omit', False):
            self._add_box()
        
        self._add_material()

    def _add_box(self):
        x, y, z = self._box_pos
        width, depth, height = self._box_vol
        box_color = self._config["sandbox"]["box"].get('color', [0.0, 0.0, 0.0])

        self.box_dimensions = {}
        self.box_dimensions["ground_plate"] = self._scene.add_entity(
            gs.morphs.Box(
                pos=self._box_pos,
                size=(width, depth, self._wall_thickness),
                fixed=True
            ),     
            surface=gs.surfaces.Default(
                color = box_color,
            ),
        )
        
        self.box_dimensions["front_wall"] = self._scene.add_entity(
            gs.morphs.Box(
                pos=(x-(width+self._wall_thickness)/2, y, z+(height-self._wall_thickness)/2),
                size=(self._wall_thickness, depth, height),
                fixed=True
            ),
            surface=gs.surfaces.Default(
                color = box_color,
            ),
        )
        
        self.box_dimensions["back_wall"] = self._scene.add_entity(
            gs.morphs.Box(
                pos=(x+(width+self._wall_thickness)/2, y, z+(height-self._wall_thickness)/2),
                size=(self._wall_thickness, depth, height),
                fixed=True
            ),
            surface=gs.surfaces.Default(
                color = box_color,
            ),
        )
        
        self.box_dimensions["left_wall"] = self._scene.add_entity(
            gs.morphs.Box(
                pos=(x, y+(depth+self._wall_thickness)/2, z+(height-self._wall_thickness)/2),
                size=(width, self._wall_thickness, height),
                fixed=True
            ),
            surface=gs.surfaces.Default(
                color = box_color,
            ),
        )
        
        self.box_dimensions["right_wall"] = self._scene.add_entity(
            gs.morphs.Box(
                pos=(x, y-(depth+self._wall_thickness)/2, z+(height-self._wall_thickness)/2),
                size=(width, self._wall_thickness, height),
                fixed=True
            ),
            surface=gs.surfaces.Default(
                color = box_color,
            ),
        )

    def _add_material(self):
        material_type = self._config["sandbox"]["material"].get('type', 'rsa')
        material_properties = self._config["sandbox"]["material"].get('type_config', {})
        granular_color = self._config["sandbox"]["material"].get('color', [1.0, 1.0, 0.0])
        self._safety_margin = self._config["sandbox"].get('safety_margin', 0.02)


        if (self._granular_vol[0] > self._box_vol[0]-self._safety_margin or self._granular_vol[1] > self._box_vol[1]-self._safety_margin):
            raise ValueError(
                f"Safety margin of {self._safety_margin} exceeded. Box volume is x={self._box_vol[0]}, y={self._box_vol[1]}, but granular volume is x={self._granular_vol[0]}, y={self._granular_vol[1]}.")

        
        if material_type == "rsa":
            self.material = random_sequential_addition(
                scene=self._scene,
                box_pos=self._box_pos,
                granular_vol=self._granular_vol,
                particle_size=self._particle_size,
                material_properties=material_properties,
                wall_thickness=self._wall_thickness,
                color=granular_color
            )
        
        elif material_type == "sand":
            self.material = add_sand(
                scene=self._scene,
                box_pos=self._box_pos,
                granular_vol=self._granular_vol,
                material_properties=material_properties,
                wall_thickness=self._wall_thickness,
                sand_color=granular_color
            )
        elif material_type == "liquid":
            self.material = add_liquid(
                scene=self._scene,
                box_pos=self._box_pos,
                granular_vol=self._granular_vol,
                material_properties=material_properties,
                wall_thickness=self._wall_thickness,
                color=granular_color,
            )
        else:
            raise ValueError(f"Unsupported material type {material_type}. Supported types are 'granular', 'sand', and 'liquid'.")
        
    def _robot_track_path(self, path, quaternion):
        for pos in path:
            q = self.robot.inverse_kinematics(
                link=self.plate_frame,
                pos=pos,
                quat=quaternion,
            )
            self.robot.control_dofs_position(q)
        # wait for robot to reach last position
        for _ in range(150):
            self._scene.step()

    def _robot_jump_to(self, position, quaternion):
        q_init = self.robot.inverse_kinematics(
            link = self.plate_frame,
            pos  = position,
            quat = quaternion,
        )
        self.robot.set_qpos(q_init)
                     
    def build(self):
        self._scene.build()

        # Default control settings refer to panda robot with fixed fingers.
        self.robot.set_dofs_kp(self._config["robot"].get('dofs_kp', [4500, 4500, 3500, 3500, 2000, 2000, 2000]))
        self.robot.set_dofs_kv(self._config["robot"].get('dofs_kv', [450, 450, 350, 350, 200, 200, 200]))
        self.robot.set_dofs_force_range(*self._config["robot"].get('dofs_force_range', [
            [-87, -87, -87, -87, -12, -12, -12],
            [ 87,  87,  87,  87,  12,  12,  12]
        ]))

        self.plate_frame = self.robot.get_link("plate")

    def view(self, horizon=1000):
        for _ in range(horizon):
            self._scene.visualizer.update()
    
    def simulate(self, horizon=1000):
        for _ in range(horizon):
            self._scene.step()

    def collect_data_samples(self, n_samples=200, operation_height=None, robot_safety_margin=0.005, step_size=200):
        box_x, box_y, box_z = self._box_pos

        # timesteps for path interpolation
        ts = np.linspace(0, 1, step_size)


        if operation_height is None:
            operation_height = box_z + (self._wall_thickness + self._granular_vol[2])/2

        # move end effector to box center (simplifies calculation of tool dimensions)

        self._robot_jump_to(
            position=(box_x, box_y, operation_height*2),
            quaternion=(0, 1, 0, 0)
        )
        # plate_frame = self.robot.get_link("plate")
        # q_init = self.robot.inverse_kinematics(
        #     link = plate_frame,
        #     pos  = np.array([box_x, box_y, operation_height*2]),
        #     quat = np.array([0, 1, 0, 0]),
        # )
        # self.robot.set_qpos(q_init)
        
        # Approximate tool dimensions from bounding box
        # Note that the bounding box is aligned with the frame orientation, so tool dimensions can be derived directly
        bbox_l, bbox_u = self.robot.geoms[-1].get_AABB()
        tool_length = np.round(np.array(abs(bbox_l[0] - bbox_u[0]).cpu()), 3)
        tool_width = np.round(np.array(abs(bbox_l[1] - bbox_u[1]).cpu()), 3)
        tool_height = np.round(np.array(abs(bbox_l[2] - bbox_u[2]).cpu()), 3)

        print(f">> tool dimensions : {tool_length}x{tool_width}x{tool_height}")

        thetas = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=n_samples)        
        sample_space_x = self._granular_vol[0]/2 - abs(np.cos(thetas) * tool_length/2 + np.sin(thetas) * tool_width/2 + robot_safety_margin)
        sample_space_y = self._granular_vol[1]/2 - abs(np.sin(thetas) * tool_length/2 + np.cos(thetas) * tool_width/2 + robot_safety_margin)

        # Min and max coordinates of action sample areas
        low = np.stack([box_x - sample_space_x, box_y - sample_space_y], axis=1)
        high = np.stack([box_x + sample_space_x, box_y + sample_space_y], axis=1)

        print(f">> Largest sample x+ : {max(sample_space_x)}")
        print(f">> Largest sample x- : {min(sample_space_x)}")
        print(f">> Largest sample y+ : {max(sample_space_y)}")
        print(f">> Largest sample y- : {min(sample_space_y)}")

        # Sampling n_samples start and end positions of action  
        action_starts = np.random.uniform(low=low, high=high, size=(n_samples, 2))
        action_stops = np.random.uniform(low=low, high=high, size=(n_samples, 2))


        # Broadcasting
        action_starts = action_starts[:, None, :]
        action_stops = action_stops[:, None, :]
        ts = ts[None, :, None]

        # action paths: (start_pos) --------> (end_pos) trajectories for all 'n_samples' actions
        action_xys = (1 - ts) * action_starts + ts * action_stops
        action_paths = np.concatenate((
            action_xys,
            np.repeat(np.ones_like(ts)*operation_height, n_samples, axis=0)
        ), axis=2)

        # lowering paths: (above start_pos) ---------> (start_pos) trajectories for all 'n_samples' actions
        decreasing_height = (1 - ts) * operation_height*2 + ts * operation_height
        lowering_paths = np.concatenate((
            np.repeat(action_starts, step_size, axis=1),
            np.repeat(decreasing_height, n_samples, axis=0)
        ), axis=2)

        # increasing paths: (end_pos) ---------> (above end_pos) trajectories for all 'n_samples' actions
        increasing_height = (1 - ts) * operation_height + ts * operation_height*2
        lifting_paths = np.concatenate((
            np.repeat(action_stops, step_size, axis=1),
            np.repeat(increasing_height, n_samples, axis=0)
        ), axis=2)

        # xyzs = np.pad(thetas[:, None], ((0,0),(0,2)), 'constant', constant_values=(0))
        # quats = qu.from_euler_angles(xyzs)
        # quats = gu.xyz_to_quat(xyzs)
        # print(quats.shape)
        # target_quat = quaternion_multiply([0, 1, 0, 0], quat)
        

        for i in range(n_samples):

            lowering = lowering_paths[i, :, :]
            action = action_paths[i, :, :]
            lifting = lifting_paths[i, :, :]
            angle = thetas[i]

            if (
                (action[0][0] - box_x > 0.135) or
                (action[-1][0] - box_x > 0.135) or
                (abs(action[0][1]) > 0.135) or
                (abs(action[-1][1]) > 0.135)
            ):
                print(f">> Theta : {angle}")
                print(f">> Start : ( {action[0][0] - box_x} , {action[0][1]} )")
                print(f">> End   : ( {action[-1][0] - box_x} , {action[-1][1]} )")

                print(sample_space_x[i])
                print(sample_space_y[i])

            quat = gu.xyz_to_quat(np.array([0, 0, angle]))
            target_quat = quaternion_multiply([0, 1, 0, 0], quat)

            
            # position robot above action starting position
            self._robot_jump_to(
                position=lowering[0],
                quaternion=target_quat
            )

            # go down to starting position
            self._robot_track_path(
                path=lowering,
                quaternion=target_quat
            )
            
            # execute action
            self._robot_track_path(
                path=action,
                quaternion=target_quat
            )

            # go up after reaching the end
            self._robot_track_path(
                path=lifting,
                quaternion=target_quat
            )
            





        return
        for p_start, p_stop, angle in zip(xy_start, xy_stop, thetas):
            x_start, y_start = p_start
            x_stop, y_stop = p_stop
            quat = gu.xyz_to_quat(np.array([0, 0, angle]))
            target_quat = quaternion_multiply([0, 1, 0, 0], quat)
            
            # jump to start x-y coordinates
            action_init_pos = (x_start, y_start, operation_height*2)
            action_init_q = self.robot.inverse_kinematics(
                link=plate_frame,
                pos=action_init_pos,
                quat=target_quat,
            )
            self.robot.set_qpos(action_init_q)
            

            # move to start pose of action
            path = get_vertical_path(x_start, y_start, operation_height*2, operation_height)
            for wp in path:
                q = self.robot.inverse_kinematics(
                    link=plate_frame,
                    pos=wp,
                    quat=target_quat,
                )
                self.robot.control_dofs_position(q)
            for _ in range(100):
                self._scene.step()
            

            # execute action
            path = get_horizontal_path(p_start, p_stop, operation_height, n_steps=100)
            for wp in path:
                q = self.robot.inverse_kinematics(
                        link=plate_frame,
                        pos=wp,
                        quat=target_quat,
                    )
                self.robot.control_dofs_position(q)
                self._scene.step()
            for _ in range(100):
                self._scene.step()

            # move up to end pose of action
            path = get_vertical_path(x_stop, y_stop, operation_height, operation_height*2)
            for wp in path:
                q = self.robot.inverse_kinematics(
                    link=plate_frame,
                    pos=wp,
                    quat=target_quat,
                )
                self.robot.control_dofs_position(q)
            for _ in range(100):
                self._scene.step()
            
