from doctest import debug
import genesis as gs
import genesis.utils.geom as gu 
from matplotlib.pyplot import box
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
                rigid_options=gs.options.RigidOptions(
                dt=0.01,
            ),
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

        x, y, z = self._box_pos
        _, _, box_height = self._box_vol
        self._plate_size = self._config["plate"].get("size", [0.1, 0.005, 0.06])
        self.plate = self._scene.add_entity(
            material=gs.materials.Rigid(
            ),
            morph=gs.morphs.Box(
                    pos=(x, y, z + (self._wall_thickness + self._granular_vol[2])/2 + box_height),
                    size=self._plate_size, 
                ),    
            surface=gs.surfaces.Default(
                color = self._config["plate"].get("color", [0.0, 1.0, 0.0]),
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
        self.material_type = self._config["sandbox"]["material"].get('type', 'rsa')
        material_properties = self._config["sandbox"]["material"].get('type_config', {})
        granular_color = self._config["sandbox"]["material"].get('color', [1.0, 1.0, 0.0])
        self._safety_margin = self._config["sandbox"].get('safety_margin', 0.02)


        if (self._granular_vol[0] > self._box_vol[0]-self._safety_margin or self._granular_vol[1] > self._box_vol[1]-self._safety_margin):
            raise ValueError(
                f"Safety margin of {self._safety_margin} exceeded. Box volume is x={self._box_vol[0]}, y={self._box_vol[1]}, but granular volume is x={self._granular_vol[0]}, y={self._granular_vol[1]}.")

        granular_touch_height = self._granular_vol[2]/2
        if self.material_type == "rsa":
            self.material = random_sequential_addition(
                scene=self._scene,
                box_pos=self._box_pos,
                granular_vol=self._granular_vol,
                particle_size=self._particle_size,
                material_properties=material_properties,
                wall_thickness=self._wall_thickness,
                color=granular_color
            )
            granular_touch_height = self._particle_size/2
        
        elif self.material_type == "sand":
            self.material = add_sand(
                scene=self._scene,
                box_pos=self._box_pos,
                granular_vol=self._granular_vol,
                material_properties=material_properties,
                wall_thickness=self._wall_thickness,
                sand_color=granular_color
            )
        elif self.material_type == "liquid":
            self.material = add_liquid(
                scene=self._scene,
                box_pos=self._box_pos,
                granular_vol=self._granular_vol,
                material_properties=material_properties,
                wall_thickness=self._wall_thickness,
                color=granular_color,
            )
        else:
            raise ValueError(f"Unsupported material type {self.material_type}. Supported types are 'granular', 'sand', and 'liquid'.")

        self._operation_height = self._box_pos[2] + granular_touch_height + self._wall_thickness/2
    
    
    
    def _plate_velocity_translation(self, p_start, p_end, speed, fix_pose, fix_dofs, debug=True):
        if debug:
            self._scene.clear_debug_objects()
            T_start = gu.trans_to_T(p_start)
            T_end = gu.trans_to_T(p_end)
            self._scene.draw_debug_frame(T_start, axis_length=0.05, origin_size=0.001, axis_radius=0.001)
            self._scene.draw_debug_frame(T_end, axis_length=0.05, origin_size=0.001, axis_radius=0.001)
        
        # direction of movement
        delta = p_end - p_start
        dist = np.linalg.norm(delta)
        
        # speed
        direction = delta / dist
        v = direction * speed
        print("direction", direction)
        # move plate
        self.plate.set_pos(p_start)
        # self.plate.set_dofs_velocity(v, dofs_idx_local=[0, 1, 2])
        self.plate.control_dofs_position_velocity(p_end, v, dofs_idx_local=[0, 1, 2])
        # self.plate.control_dofs_velocity(v, dofs_idx_local=[0, 1, 2])
        
        
        
        # number of steps to reach target position
        n_required = int(np.ceil(dist/(speed * self._scene.dt)))
        n_current = 0
        reached_goal, abort = False, False
        
        min_dist = dist
        while not reached_goal and not abort:
            n_current += 1
            self.plate.set_dofs_position(fix_pose, dofs_idx_local=fix_dofs)
            self._scene.step()
            # print(f"Distance to goal{np.linalg.norm(np.array(self.plate.get_pos().cpu())-p_end)}")
            cur_dist = np.linalg.norm(np.array(self.plate.get_pos().cpu())-p_end)
            if cur_dist < 0.001:
                reached_goal = True
            abort = (n_current > n_required)
        
        
        if abort:
            print("================ Abort =====================")
            print(">> distance", dist)
            print(">> velocity", speed)
            print(">> n_required", n_required)
            print("min dist", min_dist)
        print("Distance at end", np.linalg.norm(np.array(self.plate.get_pos().cpu())-p_end))
    
        return reached_goal
    
    def _plate_position_translation(self, p_start, p_end, n_steps, fix_pose, fix_dofs, debug=True):
    
        t = np.linspace(0, 1, n_steps)
        path = (1 - t[:, None]) * p_start[None, :] + t[:, None] * p_end[None, :]
                
        self.plate.set_pos(path[0])
        for p in path:
            self.plate.set_pos(pos=p)
            self.plate.set_dofs_position(fix_pose, dofs_idx_local=fix_dofs)
            self._scene.step()
        
    def _save_sample(self, start_state, action, end_state):
        # TODO: SAVE
        pass
          
    def build(self):
        self._scene.build()
        
        dofs_idx = [0, 1, 2, 3, 4, 5]
        self.plate.set_dofs_kp((0.3,) * 6, dofs_idx)
        self.plate.set_dofs_kv((1.0,) * 6, dofs_idx)
        # self.plate.set_mass(0.1)

    def view(self, horizon=1000):
        for _ in range(horizon):
            self._scene.visualizer.update()
    
    def simulate(self, horizon=1000):
        for _ in range(horizon):
            self._scene.step()

    def collect_data_samples(self, n_samples=200, operation_height=None, robot_safety_margin=0.005, step_size=100, v_min=0.02, v_max=0.05, v_lift=0.01):
        box_x, box_y, _ = self._box_pos
        _, _, box_height = self._box_vol
        tool_length, tool_width, tool_height = self._plate_size
        
        
        self._operation_height += tool_height/2
        if operation_height is not None:
            self._operation_height = operation_height

        ###################
        # Action sampling #
        ###################
        
        thetas = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=n_samples)   
        velocities = np.random.uniform(v_min, v_max, size=n_samples)     
        
        # sampling dimensions in x and y from box center
        sample_space_x = self._granular_vol[0]/2 - abs(np.cos(thetas) * tool_length/2 + np.sin(thetas) * tool_width/2 + robot_safety_margin)
        sample_space_y = self._granular_vol[1]/2 - abs(np.sin(thetas) * tool_length/2 + np.cos(thetas) * tool_width/2 + robot_safety_margin)

        # Min and max coordinates of action sample areas
        low = np.stack([box_x - sample_space_x, box_y - sample_space_y], axis=1)
        high = np.stack([box_x + sample_space_x, box_y + sample_space_y], axis=1)
        
        
        # Sampling n_samples start and end positions of action  
        start_samples = np.random.uniform(low=low, high=high, size=(n_samples, 2))
        stop_samples = np.random.uniform(low=low, high=high, size=(n_samples, 2))
        _z = np.ones((n_samples, 1)) * self._operation_height
        action_starts = np.concatenate((start_samples, _z), axis=1)
        action_stops = np.concatenate((stop_samples, _z), axis=1)

        
        for n in range(n_samples):
            p_start = action_starts[n, :]
            p_stop = action_stops[n, :]
            speed = velocities[n]
            angle = thetas[n]
            
            state = []
            if self.material_type == "rsa":
                for e in self.material:
                    state.append(np.array(e.get_pos().cpu()))
                    
            
            # Lowering
            success = self._plate_velocity_translation(
                p_start + np.array([0, 0, box_height]),
                p_start,
                speed,
                [p_start[0], p_start[1], 0, 0, angle],
                [0, 1, 3, 4, 5],
            )
            
            if not success:
                print(f"ACTION {n}: Failed to reach target. Skipping.")
                continue
            
            # Execute Sweeping
            success = self._plate_velocity_translation(
                p_start,
                p_stop,
                speed,
                [self._operation_height, 0, 0, angle],
                [2, 3, 4, 5],
            )
            
            if not success:
                print(f"ACTION {n}: Failed to reach target. Skipping.")
                continue
                
            # Lifting
            self._plate_position_translation(
                p_stop,
                p_stop + np.array([0, 0, box_height]),
                100,
                [p_stop[0], p_stop[1], 0, 0, angle],
                [0, 1, 3, 4, 5],
            )
            
            if self.material_type == "rsa":
                moving = 0
                while moving > 0:
                    moving = 0
                    for e in self.material:
                            v = np.linalg.norm(
                                np.array(e.get_vel().cpu())
                            ) 
                            if v > 0.01:
                                moving +=1
                    self._scene.step()
                    print(f"Action {n}: Number of moving particles: {moving}")
                print(f"All particles {n} stopped.")
            
            state_ = []
            if self.material_type == "rsa":
                for e in self.material:
                    state_.append(np.array(e.get_pos().cpu()))
        
            self._save_sample(state, (p_start, p_stop, speed, angle), state_)                    
