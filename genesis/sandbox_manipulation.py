import genesis as gs
import genesis.utils.geom as gu 
import numpy as np
import yaml
from utilities.materials import *
import quaternion as qu
from pathlib import Path
import pickle
import os
import torch


class SandboxManipulation:

    def __init__(self, config, n_envs: int = 1):
        """
        Initialize sandbox manipulation with multi-environment support.
        
        Args:
            config: Configuration dict or path to YAML file
            n_envs: Number of parallel environments within a single scene (default: 1)
        """
        if isinstance(config, dict):
            self._config = config
        elif isinstance(config, (str, Path)):
            base_dir = Path(__file__).parent
            full_path = base_dir / config
            with open(full_path) as stream:
                try:
                    self._config = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
        else:
            raise TypeError("config must be dict or a path to a YAML file")
        
        # Initialize Genesis Environment
        gs.init(
            backend=getattr(gs, self._config["simulation"].get('backend', 'gpu')),
            precision=self._config["simulation"].get('precision', '32'),
            performance_mode=self._config["simulation"].get('performance_mode', True),  # Enable for multi-env
        )

        # PARAMETERS FOR TRAINING
        self._box_pos = self._config["sandbox"]["box"].get('pos', [0.0, 0.0, 0.0])
        self._box_vol = self._config["sandbox"]["box"].get('vol', [0.3, 0.3, 0.1])
        self._wall_thickness = self._config["sandbox"]["box"].get('wall_thickness', 0.02)
        self._particle_size = self._config["sandbox"]["material"]["properties"].get('particle_size', 0.01)
        self._granular_vol = self._config["sandbox"]["material"].get('vol', [0.27, 0.27, 0.1])
        self._material_type = self._config["sandbox"]["material"].get('type', 'rsa')
        
        # Multi-environment settings
        self._n_envs = n_envs

        self._init_scene()
        self._add_entities()

        self._n_aborted_down = torch.zeros(n_envs, device=gs.device)
        self._n_aborted_action = torch.zeros(n_envs, device=gs.device)

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
                camera_pos    = viewer_settings.get('camera_pos', [3 * v_x, 0.0, 10*v_z]),
                camera_lookat = viewer_settings.get('camera_lookat', [0.0, 0.0, v_z/2]),
                camera_fov    = c_fov,
                max_FPS       = max_fps,
                res           = resolution,
            )
        elif viewer_type == "bird":
            self._viewer_options = gs.options.ViewerOptions(
                camera_pos    = viewer_settings.get('camera_pos', [b_x, b_y, 10*v_z]),
                camera_lookat = viewer_settings.get('camera_lookat', [0.0, 0.0, 0.0]),
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
            rigid_options=gs.options.RigidOptions(
            ),
            mpm_options=gs.options.MPMOptions(
                lower_bound = l_bound,
                upper_bound = u_bound,
                particle_size = self._particle_size,
            ) if self._material_type in ("sand", "liquid") else None,
            sph_options=gs.options.SPHOptions(
                lower_bound = l_bound,
                upper_bound = u_bound,
                particle_size = self._particle_size,
            ) if self._material_type == "liquid" else None,
            pbd_options=gs.options.PBDOptions(
                lower_bound = l_bound,
                upper_bound = u_bound,
                particle_size = self._particle_size,
            ) if self._material_type == "liquid" else None,
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
                rho=3000,
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
        friction = self._config["sandbox"]["box"]["properties"].get('friction', 1)

        self.box_parts = {}
        self.box_parts["ground_plate"] = self._scene.add_entity(

            material=gs.materials.Rigid(
                friction=friction,
            ),    
            morph=gs.morphs.Box(
                pos=self._box_pos,
                size=(width, depth, self._wall_thickness),
                fixed=True
            ),     
            surface=gs.surfaces.Default(
                color = box_color,
            ),
        )
        
        self.box_parts["front_wall"] = self._scene.add_entity(
            material=gs.materials.Rigid(
                friction=friction,
            ),    
            morph=gs.morphs.Box(
                pos=(x-(width+self._wall_thickness)/2, y, z+(height-self._wall_thickness)/2),
                size=(self._wall_thickness, depth, height),
                fixed=True
            ),
            surface=gs.surfaces.Default(
                color = box_color,
            ),
        )
        
        self.box_parts["back_wall"] = self._scene.add_entity(
            material=gs.materials.Rigid(
                friction=friction,
            ),    
            morph=gs.morphs.Box(
                pos=(x+(width+self._wall_thickness)/2, y, z+(height-self._wall_thickness)/2),
                size=(self._wall_thickness, depth, height),
                fixed=True
            ),
            surface=gs.surfaces.Default(
                color = box_color,
            ),
        )
        
        self.box_parts["left_wall"] = self._scene.add_entity(
            material=gs.materials.Rigid(
                friction=friction,
            ),    
            morph=gs.morphs.Box(
                pos=(x, y+(depth+self._wall_thickness)/2, z+(height-self._wall_thickness)/2),
                size=(width, self._wall_thickness, height),
                fixed=True
            ),
            surface=gs.surfaces.Default(
                color = box_color,
            ),
        )
        
        self.box_parts["right_wall"] = self._scene.add_entity(
            material=gs.materials.Rigid(
                friction=friction,
            ),    
            morph=gs.morphs.Box(
                pos=(x, y-(depth+self._wall_thickness)/2, z+(height-self._wall_thickness)/2),
                size=(width, self._wall_thickness, height),
                fixed=True
            ),
            surface=gs.surfaces.Default(
                color = box_color,
            ),
        )

    def _add_material(self):
        material_properties = self._config["sandbox"]["material"].get('properties', {})
        granular_color = self._config["sandbox"]["material"].get('color', [1.0, 1.0, 0.0])
        self._safety_margin = self._config["sandbox"].get('safety_margin', 0.02)


        if (self._granular_vol[0] > self._box_vol[0]-self._safety_margin or self._granular_vol[1] > self._box_vol[1]-self._safety_margin):
            raise ValueError(
                f"Safety margin of {self._safety_margin} exceeded. Box volume is x={self._box_vol[0]}, y={self._box_vol[1]}, but granular volume is x={self._granular_vol[0]}, y={self._granular_vol[1]}.")

        granular_touch_height = self._granular_vol[2]/2
        if self._material_type == "rsa":
            self.material = random_sequential_addition(
                scene=self._scene,
                box_pos=self._box_pos,
                granular_vol=self._granular_vol,
                material_properties=material_properties,
                wall_thickness=self._wall_thickness,
                color=granular_color
            )                
            granular_touch_height = self._particle_size/2 if isinstance(self._particle_size, float) else min(self._particle_size)/4
        
        elif self._material_type == "sand":
            self.material = add_sand(
                scene=self._scene,
                box_pos=self._box_pos,
                granular_vol=self._granular_vol,
                material_properties=material_properties,
                wall_thickness=self._wall_thickness,
                sand_color=granular_color
            )
        elif self._material_type == "liquid":
            self.material = add_liquid(
                scene=self._scene,
                box_pos=self._box_pos,
                granular_vol=self._granular_vol,
                material_properties=material_properties,
                wall_thickness=self._wall_thickness,
                color=granular_color,
            )
        else:
            raise ValueError(f"Unsupported material type {self._material_type}. Supported types are 'granular', 'sand', and 'liquid'.")

        self._operation_height = self._box_pos[2] + granular_touch_height + self._wall_thickness/2       

    def _save_data(
            self,
            path : str | Path,
    ):
        
        states = self.valid_states.cpu()
        states_ = self.valid_states_.cpu()
        p_starts = self.valid_p_starts.cpu()
        p_stops = self.valid_p_stops.cpu()
        angles = self.valid_angles.cpu()

        data = [
            {
                "state" : states[i],
                "state_" : states_[i],
                "action" : (p_starts[i], p_stops[i], angles[i])
            } for i in range(states.shape[0])
        ]
        with open(path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _save_config(
            self,
            path : str | Path
        ):

        with open(path, 'w') as outfile:
            try: 
                yaml.dump(self._config, outfile, default_flow_style=False)
            except yaml.YAMLError as exc:
                print(exc)
       
    def build(self):
        """Build the scene with multiple environments"""
        self._scene.build(n_envs=self._n_envs, env_spacing=(self._box_vol[0] *2 , self._box_vol[1] *2 ))  # Adjust env_spacing as needed
        
        dofs_idx = [0, 1, 2, 3, 4, 5]
        self.plate.set_dofs_kp((0.8,) * 6, dofs_idx)
        self.plate.set_dofs_kv((1.0,) * 6, dofs_idx)

    def destroy(self):
        """Destroying environment"""
        gs.destroy()

    def view(self, horizon=1000):
        for _ in range(horizon):
            self._scene.visualizer.update()
    
    def simulate(self, horizon=1000):
        """Simulate all environments (vectorized)"""
        for _ in range(horizon):
            self._scene.step()

    def set_material_state(self, positions: torch.Tensor):
        """
        Set position of particles across all environments.
        
        Args:
            positions: Tensor of shape [n_envs, n_particles, 3]
        """
        if self._material_type != "rsa":
            raise NotImplementedError("Method not implemented for materials other than RSA")

        if positions.shape[0] != self._n_envs:
            raise ValueError(
                f"Expected {self._n_envs} environments, got {positions.shape[0]}"
            )
        
        # Set positions for all particles in all environments
        for env_idx in range(self._n_envs):
            for particle_idx, particle in enumerate(self.material):
                particle.set_pos(positions[env_idx, particle_idx])

    def get_material_state(self):
        """
        Returns particle state (positions and sizes) for all environments.
        
        Returns:
            Tensor of shape [n_envs, n_particles, 4] with (x, y, z, size)
        """
        if self._material_type != "rsa":
            raise NotImplementedError("Method not implemented for materials other than RSA")
        
        # Wait for particles to settle
        n_p = len(self.material)
        moving = True

        # Hold plate still
        self.plate.set_pos(self.plate.get_pos())
        self.plate.control_dofs_position_velocity(self.plate.get_pos(), torch.zeros((self._n_envs, 3)), dofs_idx_local=[0, 1, 2])
        
        while moving:
            v = torch.zeros((self._n_envs, len(self.material)), device=gs.device)
            for i, particle in enumerate(self.material):
                vel = particle.get_vel()  # Shape: [n_envs, 3]
                v[:, i] = torch.linalg.norm(vel, axis=1)
            
            # Check if all particles in all envs are settled (velocity < threshold)
            if (v < 0.01).all():
                moving = False
            
            # Freeze plate
            self.plate.set_dofs_position(self.plate.get_dofs_position())            
            self._scene.step()
        
        # Collect state from all environments
        state = torch.zeros((self._n_envs, n_p, 4), device=gs.device)
        for i, particle in enumerate(self.material):
            pos = particle.get_pos()  # Shape: [n_envs, 3]
            size = particle.morph.size[0]  # Single value for all envs
            state[:, i, 0:3] = pos
            state[:, i, 3] = size
        
        return state

    def get_collected_samples(self):
        """
        Return previously collected samples
        
        Each samples consists of state(i), state(i+1), start_position, end_position, angle, velocity
        """
        return self._data_samples.values()
    
    def plate_velocity_translation(self, p_start, p_end, speed, angle, debug=True):
        """
        Move plates with velocity control across all environments.
        
        Args:
            p_start: Starting positions [n_envs, 3] or [3]
            p_end: Ending positions [n_envs, 3] or [3]
            speed: Movement speed (scalar)
            angle: Rotation angle (scalar)
        Returns:
            reached_goal : Mask of environments that reached the goal
        """
        
        if debug:
            self._scene.clear_debug_objects()
            T_start = gu.trans_to_T(p_start[0])
            T_end = gu.trans_to_T(p_end[0])
            self._scene.draw_debug_frame(T_start, axis_length=0.05, origin_size=0.001, axis_radius=0.001)
            self._scene.draw_debug_frame(T_end, axis_length=0.05, origin_size=0.001, axis_radius=0.001)
        
        # Horizontal movement
        fix_z_and_rot = torch.stack([
            # x is free dof
            # y is free dof
            torch.full((self._n_envs,), self._operation_height, device=gs.device),
            torch.zeros(self._n_envs, device=gs.device),
            torch.zeros(self._n_envs, device=gs.device),
            angle
        ], dim=1)

        # Lifting
        fix_x_y_and_rot = torch.stack([
            p_end[:, 0],
            p_end[:, 1],
            # z is free dof
            torch.zeros(self._n_envs, device=gs.device),
            torch.zeros(self._n_envs, device=gs.device),
            angle
        ], dim=1)


        # Calculate velocity vector for each environment
        delta = p_end - p_start  # [n_envs, 3]
        dist = torch.linalg.norm(delta, axis=1, keepdim=True)  # [n_envs, 1]  
        direction = delta / (dist + 1e-8)
        v = direction * speed  # [n_envs, 3]

        # Set initial position, velocity and goal for all plates in all environments
        self.plate.set_pos(p_start)
        self.plate.control_dofs_position_velocity(p_end, v, dofs_idx_local=[0, 1, 2])
        
        # For tracking env statis: reached goal, aborted, or still active
        to_sweep = torch.ones(self._n_envs, dtype=torch.bool, device=gs.device)
        to_lift = torch.zeros(self._n_envs, dtype=torch.bool, device=gs.device)
        to_stop = torch.ones(self._n_envs, dtype=torch.bool, device=gs.device)
        reached_goal = torch.zeros(self._n_envs, dtype=torch.bool, device=gs.device)
        abort = torch.zeros(self._n_envs, dtype=torch.bool, device=gs.device)
        done = torch.zeros(self._n_envs, dtype=torch.bool, device=gs.device)

        lift_progress = torch.zeros(self._n_envs, device=gs.device)  # how many lift steps taken
        lift_z_per_step = self._operation_height / 100  # scalar, same for all envs
        
        # Calculate approximate steps based on distance and speed
        n_required = torch.ceil(dist / (speed * self._scene.dt)).int().squeeze(dim=1)
        n_current = 0
        while not done.all():
            n_current += 1
            
            # lift plates that reached goal or aborted
            if to_lift.any():
                lift_env_idxs = to_lift.nonzero().squeeze(dim=1)

                # save the final pose only for plates that reached the final position in last iteration
                # this is done to prevent that plates keep moving after finishing (because plate.vel!= 0)
                to_save_final_pos = to_lift & to_stop
                if to_save_final_pos.any():
                    save_pos_env_idxs = to_save_final_pos.nonzero().squeeze(dim=1)
                    p_stop_final = self.plate.get_pos(save_pos_env_idxs)
                    fix_x_y_and_rot[to_save_final_pos, 0:2] = p_stop_final[:, 0:2]
        
                # remember for which plates the final position was obtained already
                to_stop = to_stop & ~to_lift

                # get current position of plates to lift
                lifted_z = (p_end[to_lift, 2] + lift_progress[to_lift] * lift_z_per_step) # [n_lift_envs]
                self.plate.set_pos(
                    pos=torch.stack([
                        fix_x_y_and_rot[to_lift, 0],
                        fix_x_y_and_rot[to_lift, 1],
                        lifted_z
                    ],dim=1),
                    envs_idx=lift_env_idxs
                )
                
                self.plate.set_dofs_position(
                    fix_x_y_and_rot[to_lift],
                    dofs_idx_local=[0, 1, 3, 4, 5],
                    envs_idx=lift_env_idxs
                )
                lift_progress[to_lift & ~ done] += 1

                done = lift_progress >= 100  # envs that finished

            # Sweep all envs that are not done
            if to_sweep.any(): 
                sweep_env_idx = to_sweep.nonzero().squeeze(dim=1)
                self.plate.set_dofs_position(
                    fix_z_and_rot[to_sweep, :],
                    dofs_idx_local=[2, 3, 4, 5],
                    envs_idx=sweep_env_idx
                )
            self._scene.step()
            
            # Check distance for each environment
            cur_pos = self.plate.get_pos()  # [n_envs, 3]
            cur_dist = torch.linalg.norm(cur_pos[:, :2] - p_end[:, :2], axis=1) # [n_envs]

            reached_goal = (cur_dist < 0.002) # envs that reached goal
            abort = (n_current > n_required * 1.7) & ~reached_goal # envs that exceeded time limit
            to_lift = reached_goal | abort # lift envs that reached goal or aborted
            to_sweep = ~to_lift # remaining envs to sweep
            
        return reached_goal
    
    def plate_position_translation(self, p_start, p_end, n_steps, fix_pose, fix_dofs, debug=False):
        """
        Move plates with position control across all environments.
        
        Args:
            p_start: Starting positions [n_envs, 3] or [3]
            p_end: Ending positions [n_envs, 3] or [3]
            n_steps: Number of steps for interpolation
            fix_pose: Pose values for locked DOFs [n_envs, n_dofs] or [n_dofs]
            fix_dofs: Indices of DOFs to lock
        """
        # Ensure batch dimensions
        if p_start.shape[0] != self._n_envs:
            p_start = p_start.unsqueeze(0).expand(self._n_envs, -1)
        if p_end.shape[0] != self._n_envs:
            p_end = p_end.unsqueeze(0).expand(self._n_envs, -1)
        if len(fix_pose.shape) == 1:
            fix_pose = fix_pose.unsqueeze(0).expand(self._n_envs, -1)
        
        t = torch.linspace(0, 1, n_steps, device=gs.device)
        # Create interpolated path for all environments
        # path shape: [n_steps, n_envs, 3]
        path = (1 - t[:, None, None]) * p_start[None, :, :] + t[:, None, None] * p_end[None, :, :]
        
        self.plate.set_pos(path[0])
        for i in range(n_steps):
            self.plate.set_pos(pos=path[i])
            self.plate.set_dofs_position(fix_pose, dofs_idx_local=fix_dofs)
            self._scene.step()

    def generate_action_samples(
            self,
            n_samples: int,
        ):
        """
        Generate random action samples for all environments.
        
        Returns:
            Tuple of (action_starts, action_stops, angles) each of shape [n_envs * n_samples, 3/1]
        """
        box_x, box_y, _ = self._box_pos
        tool_length, tool_width, tool_height = self._plate_size
        self._operation_height = self._operation_height + tool_height / 2

        # Generate samples for each environment
        n_total = self._n_envs * n_samples
        angles = (-torch.pi/2) + torch.rand(n_total, device=gs.device) * torch.pi
        
        # Sampling dimensions in x and y from box center
        sample_space_x = self._granular_vol[0]/2 - (torch.cos(angles) * tool_length/2 + abs(torch.sin(angles)) * tool_width/2 + self._safety_margin)
        sample_space_y = self._granular_vol[1]/2 - (abs(torch.sin(angles)) * tool_length/2 + torch.cos(angles) * tool_width/2 + self._safety_margin)

        # Min and max coordinates
        low = torch.stack([box_x - sample_space_x, box_y - sample_space_y], axis=1)
        high = torch.stack([box_x + sample_space_x, box_y + sample_space_y], axis=1)
        
        # Sample start and end positions
        start_samples = (high - low) * torch.rand((n_total, 2), device=gs.device) + low
        stop_samples = (high - low) * torch.rand((n_total, 2), device=gs.device) + low
        _z = torch.ones((n_total, 1), device=gs.device) * self._operation_height
        
        action_starts = torch.concatenate((start_samples, _z), axis=1)
        action_stops = torch.concatenate((stop_samples, _z), axis=1)
        
        # Reshape to [n_envs, n_samples, ...]
        action_starts = action_starts.reshape(self._n_envs, n_samples, 3)
        action_stops = action_stops.reshape(self._n_envs, n_samples, 3)
        angles = angles.reshape(self._n_envs, n_samples)

        return action_starts, action_stops, angles

    def execute_action(self, p_start, p_stop, angle, speed, lift_height):
        """
        Execute action (lower, sweep, lift) for all environments.
        
        Args:
            p_start: Starting positions [n_envs, 3]
            p_stop: Stopping positions [n_envs, 3]
            angle: Angles [n_envs]
            speed: Movement speed (scalar)
            lift_height: Lift height [n_envs, 3]
        
        Returns:
            Tensor of shape [n_envs] with success status
        """

        # Lowering
        fix_pose_lower = torch.stack([
            p_start[:, 0],
            p_start[:, 1],
            # z is free dof 
            torch.zeros(self._n_envs, device=gs.device),
            torch.zeros(self._n_envs, device=gs.device),
            angle
        ], dim=1)
        
        self.plate_position_translation(
            p_start + lift_height,
            p_start,
            100,
            fix_pose_lower,
            [0, 1, 3, 4, 5],
        )
        
        reached_goal = self.plate_velocity_translation(
            p_start,
            p_stop,
            speed,
            angle,
        )

        return reached_goal

    def collect_data_samples(
            self,
            n_samples: int = 200,
            speed: float = 0.125,
            path : str | Path = "training"
        ):
        """
        Collect data samples from all environments efficiently.
        
        Args:
            n_samples: Number of samples to collect per environment
            operation_height: Height at which to operate
            speed: Plate movement speed
            lift_height: Height to lift plate
        """
        # Setup lift height


        lift_height = self._box_vol[2]
        lift_height_tensor = torch.tensor([0, 0, lift_height], device=gs.device)
        lift_height_tensor = lift_height_tensor.unsqueeze(0).expand(self._n_envs, -1)
        
        # Generate action samples for all environments
        action_starts, action_stops, angles = self.generate_action_samples(
            n_samples,
        )
        max_samples = n_samples * self._n_envs
        self.valid_states = torch.empty((max_samples, len(self.material), 4), device=gs.device)
        self.valid_states_ = torch.empty((max_samples, len(self.material), 4), device=gs.device)
        self.valid_p_starts = torch.empty((max_samples, 3), device=gs.device)
        self.valid_p_stops = torch.empty((max_samples, 3), device=gs.device)
        self.valid_angles = torch.empty((max_samples), device=gs.device)

        write_ptr = 0
        for sample_idx in range(n_samples):

            # Collect sample in all environments
            state = self.get_material_state()  # [n_envs, n_particles, 4]
            
            p_start = action_starts[:, sample_idx, :]  # [n_envs, 3]
            p_stop = action_stops[:, sample_idx, :]    # [n_envs, 3]
            angle = angles[:, sample_idx]              # [n_envs]
            
            reached_goal = self.execute_action(
                p_start,
                p_stop,
                angle,
                speed,
                lift_height_tensor,
            )
            num_valid = int(reached_goal.sum())
            if num_valid == 0:
                continue

            idx = slice(write_ptr, write_ptr + num_valid)
            
            state_ = self.get_material_state()  # [n_envs, n_particles, 4]

            # Save samples for all environments that reached the goal
            self.valid_states[idx] = state[reached_goal]
            self.valid_states_[idx] = state_[reached_goal]
            self.valid_p_starts[idx] = p_start[reached_goal]
            self.valid_p_stops[idx] = self.plate.get_pos(reached_goal.nonzero().squeeze(dim=1))  # [n_envs, 3]
            self.valid_angles[idx] = angle[reached_goal]
        
            write_ptr += num_valid
        

        # Trim unused space
        self.valid_states = self.valid_states[:write_ptr]
        self.valid_states_ = self.valid_states_[:write_ptr]
        self.valid_p_starts = self.valid_p_starts[:write_ptr]
        self.valid_p_stops = self.valid_p_stops[:write_ptr]
        self.valid_angles = self.valid_angles[:write_ptr]

        # Print statistics
        print("\nStatistics (Multi-Environment Collection)")
        print("=" * 50)
        print(f">> Number of environments   : {self._n_envs}")
        print(f">> Samples per environment  : {n_samples}")
        print(f">> Total samples collected  : {write_ptr}")
        print(f">> Number of failed samples : {max_samples - write_ptr}")

        self._config["statistics"] = {
            "Number of environments"   : self._n_envs,
            "Samples per environment"  : n_samples,
            "Total samples collected"  : write_ptr,
            "Number of failed samples" : max_samples - write_ptr,
        }

        base_dir = Path(__file__).parent
        full_path = base_dir / path
        Path.mkdir(full_path, parents=True, exist_ok=True)

        n_runs = int(len([name for name in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, name))])/2)
        self._save_config(full_path / (str(n_runs) + "_config.yaml"))
        self._save_data(full_path / (str(n_runs) + "_data.pkl"))