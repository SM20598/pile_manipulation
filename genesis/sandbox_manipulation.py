import genesis as gs
import genesis.utils.geom as gu 
import numpy as np
import yaml
from utilities.materials import *
import quaternion as qu
from pathlib import Path
import pickle
import os
import math
import time
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
        collection_cfg = self._config.get("data_collection", {})
        self._settle_steps = collection_cfg.get("settle_steps", 100)
        self._settle_check_interval = collection_cfg.get("settle_check_interval", 20)
        self._settle_threshold = collection_cfg.get("settle_threshold", 0.01)
        self._lower_steps = collection_cfg.get("lower_steps", 100)
        self._lift_steps = collection_cfg.get("lift_steps", 100)
        self._goal_threshold = collection_cfg.get("goal_threshold", 0.002)
        self._progress = collection_cfg.get("progress", True)
        self._sample_progress_interval = max(1, collection_cfg.get("sample_progress_interval", 1))
        self._phase_progress_interval = max(1, collection_cfg.get("phase_progress_interval", 100))
        self._trace_scene_steps = collection_cfg.get("trace_scene_steps", False)
        self._update_visualizer = collection_cfg.get("update_visualizer", False)
        self._settle_stabilization = collection_cfg.get("settle_stabilization", True)
        self._settle_angular_damping = collection_cfg.get("settle_angular_damping", 0.2)
        self._settle_linear_damping = collection_cfg.get("settle_linear_damping", 1.0)
        self._settle_sleep_threshold = collection_cfg.get("settle_sleep_threshold", 0.01)
        
        # Multi-environment settings
        self._n_envs = n_envs

        self._init_scene()
        self._add_entities()

        self._n_aborted_down = torch.zeros(n_envs, device=gs.device)
        self._n_aborted_action = torch.zeros(n_envs, device=gs.device)
        self._particle_sizes = None
        self._particle_links_idx = None
        self._particle_dofs_idx = None
        self._particle_linear_dofs_idx = None
        self._particle_angular_dofs_idx = None

    def _log(self, message: str):
        if self._progress:
            print(message, flush=True)

    def _log_step_progress(self, label: str | None, step: int, total_steps: int):
        if not label or total_steps <= 0:
            return
        if step == 1 or step == total_steps or step % self._phase_progress_interval == 0:
            self._log(f"  {label}: step {step}/{total_steps}")

    def _step_scene(self, label: str | None = None, step: int | None = None, total_steps: int | None = None):
        if self._trace_scene_steps and label and step is not None and total_steps is not None:
            self._log(f"  {label}: before scene.step {step}/{total_steps}")
            start_time = time.monotonic()
            self._scene.step(
                update_visualizer=self._update_visualizer,
                refresh_visualizer=self._update_visualizer,
            )
            self._log(f"  {label}: after scene.step {step}/{total_steps} ({time.monotonic() - start_time:.3f}s)")
        else:
            self._scene.step(
                update_visualizer=self._update_visualizer,
                refresh_visualizer=self._update_visualizer,
            )

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

        rigid_cfg = self._config.get("rigid_options", {})

        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt       = self._config["simulation"].get('dt', 4e3),
                substeps = self._config["simulation"].get('substeps', 1),
            ),
            rigid_options=gs.options.RigidOptions(
                iterations=rigid_cfg.get("iterations", 50),
                ls_iterations=rigid_cfg.get("ls_iterations", 50),
                tolerance=rigid_cfg.get("tolerance", 1e-6),
                ls_tolerance=rigid_cfg.get("ls_tolerance", 0.01),
                box_box_detection=rigid_cfg.get("box_box_detection", False),
                use_contact_island=rigid_cfg.get("use_contact_island", False),
                use_hibernation=rigid_cfg.get("use_hibernation", False),
                max_collision_pairs=rigid_cfg.get("max_collision_pairs", 150),
                enable_multi_contact=rigid_cfg.get("enable_multi_contact", True),
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
        """
        Save data efficiently by moving to CPU only when needed.
        Uses non-blocking transfers for better GPU utilization.
        """
        # Move to CPU only for pickle serialization
        # Use non_blocking=True for async transfer if CUDA is available
        use_non_blocking = gs.device.type == 'cuda'

        states = self.valid_states.to('cpu', non_blocking=use_non_blocking)
        states_ = self.valid_states_.to('cpu', non_blocking=use_non_blocking)
        p_starts = self.valid_p_starts.to('cpu', non_blocking=use_non_blocking)
        p_stops = self.valid_p_stops.to('cpu', non_blocking=use_non_blocking)
        angles = self.valid_angles.to('cpu', non_blocking=use_non_blocking)

        # Ensure transfers are complete before pickling
        if use_non_blocking:
            torch.cuda.synchronize()

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
        self._cache_particle_sizes()

    def _cache_particle_sizes(self):
        if self._material_type != "rsa":
            self._particle_sizes = None
            return

        sizes = []
        links_idx = []
        dofs_idx = []
        linear_dofs_idx = []
        angular_dofs_idx = []
        for particle in self.material:
            size = particle.morph.size[0] if hasattr(particle.morph, "size") else particle.morph.radius * 2
            sizes.append(float(size))
            links_idx.append(particle.link_start)
            if particle.n_dofs == 6:
                particle_dofs = list(range(particle.dof_start, particle.dof_end))
                dofs_idx.extend(particle_dofs)
                linear_dofs_idx.extend(particle_dofs[:3])
                angular_dofs_idx.extend(particle_dofs[3:])
        self._particle_sizes = torch.tensor(sizes, device=gs.device).view(1, -1)
        self._particle_links_idx = torch.tensor(links_idx, dtype=gs.tc_int, device=gs.device)
        self._particle_dofs_idx = torch.tensor(dofs_idx, dtype=gs.tc_int, device=gs.device)
        self._particle_linear_dofs_idx = torch.tensor(linear_dofs_idx, dtype=gs.tc_int, device=gs.device)
        self._particle_angular_dofs_idx = torch.tensor(angular_dofs_idx, dtype=gs.tc_int, device=gs.device)

    def destroy(self):
        """Destroying environment"""
        gs.destroy()

    def view(self, horizon=1000):
        for _ in range(horizon):
            self._scene.visualizer.update()
    
    def simulate(self, horizon=1000):
        """Simulate all environments (vectorized)"""
        for _ in range(horizon):
            self._step_scene()

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
        
        positions = positions.to(device=gs.device)
        if positions.shape[1] != len(self.material):
            raise ValueError(
                f"Expected {len(self.material)} particles, got {positions.shape[1]}"
            )

        envs_idx = torch.arange(self._n_envs, device=gs.device)
        for particle_idx, particle in enumerate(self.material):
            particle.set_pos(positions[:, particle_idx, :], envs_idx=envs_idx)

    def _get_particle_positions(self):
        if self._particle_links_idx is not None:
            return self._scene.rigid_solver.get_links_pos(links_idx=self._particle_links_idx)

        positions = torch.empty((self._n_envs, len(self.material), 3), device=gs.device)
        for particle_idx, particle in enumerate(self.material):
            positions[:, particle_idx, :] = particle.get_pos()
        return positions

    def _get_particle_velocities(self):
        if self._particle_links_idx is not None:
            return self._scene.rigid_solver.get_links_vel(links_idx=self._particle_links_idx)

        velocities = torch.empty((self._n_envs, len(self.material), 3), device=gs.device)
        for particle_idx, particle in enumerate(self.material):
            velocities[:, particle_idx, :] = particle.get_vel()
        return velocities

    def _stabilize_settled_particles(self, progress_label: str | None = None):
        if (
            not self._settle_stabilization
            or self._particle_dofs_idx is None
            or self._particle_dofs_idx.numel() == 0
        ):
            return

        start_time = time.monotonic()
        vel = self._scene.rigid_solver.get_dofs_velocity(dofs_idx=self._particle_dofs_idx)
        vel = vel.reshape(self._n_envs, len(self.material), 6)

        if self._settle_linear_damping != 1.0:
            vel[:, :, :3] *= self._settle_linear_damping
        if self._settle_angular_damping != 1.0:
            vel[:, :, 3:] *= self._settle_angular_damping

        if self._settle_sleep_threshold > 0:
            linear_speed = torch.linalg.norm(vel[:, :, :3], dim=2, keepdim=True)
            angular_speed = torch.linalg.norm(vel[:, :, 3:], dim=2, keepdim=True)
            vel[:, :, :3] = torch.where(linear_speed < self._settle_sleep_threshold, 0.0, vel[:, :, :3])
            vel[:, :, 3:] = torch.where(angular_speed < self._settle_sleep_threshold, 0.0, vel[:, :, 3:])

        self._scene.rigid_solver.set_dofs_velocity(
            vel.reshape(self._n_envs, -1),
            dofs_idx=self._particle_dofs_idx,
            skip_forward=True,
        )
        if progress_label:
            self._log(
                f"  {progress_label}: stabilized particle velocities "
                f"in {time.monotonic() - start_time:.3f}s"
            )

    def get_material_state(self, settle_steps: int | None = None, progress_label: str | None = None):
        """
        Returns particle state (positions and sizes) for all environments.
        Optimized for GPU processing.

        Returns:
            Tensor of shape [n_envs, n_particles, 4] with (x, y, z, size)
        """
        if self._material_type != "rsa":
            raise NotImplementedError("Method not implemented for materials other than RSA")

        n_p = len(self.material)
        if settle_steps is None:
            settle_steps = self._settle_steps
        start_time = time.monotonic()
        if progress_label:
            self._log(f"  {progress_label}: settling {settle_steps} steps")

        # Hold plate still
        self.plate.set_pos(self.plate.get_pos(), skip_forward=True)
        self.plate.control_dofs_position_velocity(
            self.plate.get_pos(),
            torch.zeros((self._n_envs, 3), device=gs.device),
            dofs_idx_local=[0, 1, 2]
        )

        frozen_plate_dofs = self.plate.get_dofs_position()
        for step in range(settle_steps):
            self.plate.set_dofs_position(frozen_plate_dofs)
            self._step_scene(progress_label, step + 1, settle_steps)
            self._log_step_progress(progress_label, step + 1, settle_steps)
        self._stabilize_settled_particles(progress_label)

        state = torch.empty((self._n_envs, n_p, 4), device=gs.device)
        if progress_label:
            self._log(f"  {progress_label}: reading particle positions")
            read_start = time.monotonic()
        state[:, :, 0:3] = self._get_particle_positions()
        if progress_label:
            self._log(f"  {progress_label}: particle positions read in {time.monotonic() - read_start:.1f}s")

        if self._particle_sizes is None:
            self._cache_particle_sizes()
        state[:, :, 3] = self._particle_sizes.expand(self._n_envs, -1)

        if progress_label:
            self._log(f"  {progress_label}: done in {time.monotonic() - start_time:.1f}s")
        return state

    def get_collected_samples(self):
        """
        Return previously collected samples
        
        Each samples consists of state(i), state(i+1), start_position, end_position, angle, velocity
        """
        if not hasattr(self, "valid_states"):
            return []

        return [
            {
                "state": self.valid_states[i],
                "state_": self.valid_states_[i],
                "action": (self.valid_p_starts[i], self.valid_p_stops[i], self.valid_angles[i]),
            }
            for i in range(self.valid_states.shape[0])
        ]
    
    def plate_velocity_translation(
            self,
            p_start,
            p_end,
            speed,
            angle,
            sweep_steps: int | None = None,
            debug=False,
            progress_label: str | None = None,
        ):
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
        
        operation_height = getattr(self, "_action_operation_height", self._operation_height)

        # Horizontal movement
        fix_z_and_rot = torch.stack([
            # x is free dof
            # y is free dof
            torch.full((self._n_envs,), operation_height, device=gs.device),
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
        self.plate.set_pos(p_start, skip_forward=True)
        self.plate.control_dofs_position_velocity(p_end, v, dofs_idx_local=[0, 1, 2])
        
        if sweep_steps is None:
            max_sweep_distance = float(dist.max().item())
            sweep_steps = max(1, math.ceil(max_sweep_distance / (speed * self._scene.dt) * 1.7))
            if progress_label:
                self._log(
                    f"  {progress_label}: max action distance {max_sweep_distance:.4f}m "
                    f"-> {sweep_steps} sweep steps"
                )

        start_time = time.monotonic()
        if progress_label:
            self._log(f"  {progress_label}: sweeping {sweep_steps} steps")

        for step in range(sweep_steps):
            self.plate.set_dofs_position(
                fix_z_and_rot,
                dofs_idx_local=[2, 3, 4, 5],
            )
            self._step_scene(progress_label, step + 1, sweep_steps)
            self._log_step_progress(progress_label, step + 1, sweep_steps)

        final_pos = self.plate.get_pos()
        cur_dist = torch.linalg.norm(final_pos[:, :2] - p_end[:, :2], axis=1)
        reached_goal = cur_dist < self._goal_threshold

        print("Current Distance:", cur_dist.cpu().numpy())

        if progress_label:
            self._log(f"  {progress_label}: done in {time.monotonic() - start_time:.1f}s")
        return reached_goal, final_pos
    
    def plate_position_translation(
            self,
            p_start,
            p_end,
            n_steps,
            fix_pose,
            fix_dofs,
            debug=False,
            progress_label: str | None = None,
        ):
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
        
        start_time = time.monotonic()
        if progress_label:
            self._log(f"  {progress_label}: moving {n_steps} steps")

        self.plate.set_pos(path[0], skip_forward=True)
        for i in range(n_steps):
            self.plate.set_pos(pos=path[i], skip_forward=True)
            self.plate.set_dofs_position(fix_pose, dofs_idx_local=fix_dofs)
            self._step_scene(progress_label, i + 1, n_steps)
            self._log_step_progress(progress_label, i + 1, n_steps)

        if progress_label:
            self._log(f"  {progress_label}: done in {time.monotonic() - start_time:.1f}s")

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
        self._action_operation_height = self._operation_height + tool_height / 2

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
        _z = torch.ones((n_total, 1), device=gs.device) * self._action_operation_height
        
        action_starts = torch.concatenate((start_samples, _z), axis=1)
        action_stops = torch.concatenate((stop_samples, _z), axis=1)
        
        # Reshape to [n_envs, n_samples, ...]
        action_starts = action_starts.reshape(self._n_envs, n_samples, 3)
        action_stops = action_stops.reshape(self._n_envs, n_samples, 3)
        angles = angles.reshape(self._n_envs, n_samples)

        return action_starts, action_stops, angles

    def execute_action(
            self,
            p_start,
            p_stop,
            angle,
            speed,
            lift_height,
            sweep_steps: int | None = None,
            progress_label: str | None = None,
        ):
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
            self._lower_steps,
            fix_pose_lower,
            [0, 1, 3, 4, 5],
            progress_label=f"{progress_label} lower" if progress_label else None,
        )
        
        reached_goal, final_pos = self.plate_velocity_translation(
            p_start,
            p_stop,
            speed,
            angle,
            sweep_steps=sweep_steps,
            progress_label=f"{progress_label} sweep" if progress_label else None,
        )

        fix_pose_lift = torch.stack([
            final_pos[:, 0],
            final_pos[:, 1],
            torch.zeros(self._n_envs, device=gs.device),
            torch.zeros(self._n_envs, device=gs.device),
            angle
        ], dim=1)

        self.plate_position_translation(
            final_pos,
            final_pos + lift_height,
            self._lift_steps,
            fix_pose_lift,
            [0, 1, 3, 4, 5],
            progress_label=f"{progress_label} lift" if progress_label else None,
        )

        return reached_goal, final_pos

    def collect_data_samples(
            self,
            n_samples: int = 200,
            speed: float = 0.125,
            path : str | Path = "training",
            settle_steps: int | None = None,
            sweep_steps: int | None = None,
        ):
        """
        Collect data samples from all environments efficiently.
        Optimized for GPU processing and memory efficiency.

        Args:
            n_samples: Number of samples to collect per environment
            speed: Plate movement speed
            path: Output path for data
        """
        collection_start = time.monotonic()
        self._log(
            f"Preparing collection: n_envs={self._n_envs}, samples_per_env={n_samples}, "
            f"particles={len(self.material)}, speed={speed}"
        )

        # Setup lift height
        lift_height = self._box_vol[2]
        lift_height_tensor = torch.tensor([0, 0, lift_height], device=gs.device)
        lift_height_tensor = lift_height_tensor.unsqueeze(0).expand(self._n_envs, -1)

        # Generate action samples for all environments
        action_start = time.monotonic()
        self._log("Generating action samples...")
        action_starts, action_stops, angles = self.generate_action_samples(n_samples)
        self._log(f"Generated action samples in {time.monotonic() - action_start:.1f}s")

        max_samples = n_samples * self._n_envs

        alloc_start = time.monotonic()
        self._log(
            f"Allocating GPU buffers for up to {max_samples} samples "
            f"({n_samples} batches x {self._n_envs} envs)"
        )
        states = torch.empty((n_samples, self._n_envs, len(self.material), 4), device=gs.device)
        states_ = torch.empty_like(states)
        p_starts = torch.empty((n_samples, self._n_envs, 3), device=gs.device)
        p_stops = torch.empty((n_samples, self._n_envs, 3), device=gs.device)
        sample_angles = torch.empty((n_samples, self._n_envs), device=gs.device)
        success_mask = torch.empty((n_samples, self._n_envs), dtype=torch.bool, device=gs.device)
        self._log(f"Allocated buffers in {time.monotonic() - alloc_start:.1f}s")

        for sample_idx in range(n_samples):
            should_log_batch = (
                sample_idx == 0
                or sample_idx == n_samples - 1
                or sample_idx % self._sample_progress_interval == 0
            )
            batch_label = f"batch {sample_idx + 1}/{n_samples}"
            batch_start = time.monotonic()
            if should_log_batch:
                self._log(f"Collecting action {batch_label}")

            state = self.get_material_state(
                settle_steps=settle_steps,
                progress_label=f"{batch_label} pre-state" if should_log_batch else None,
            )

            p_start = action_starts[:, sample_idx, :]  # [n_envs, 3]
            p_stop = action_stops[:, sample_idx, :]    # [n_envs, 3]
            angle = angles[:, sample_idx]              # [n_envs]

            reached_goal, final_pos = self.execute_action(
                p_start,
                p_stop,
                angle,
                speed,
                lift_height_tensor,
                sweep_steps=sweep_steps,
                progress_label=batch_label if should_log_batch else None,
            )

            states[sample_idx] = state
            states_[sample_idx] = self.get_material_state(
                settle_steps=settle_steps,
                progress_label=f"{batch_label} post-state" if should_log_batch else None,
            )
            p_starts[sample_idx] = p_start
            p_stops[sample_idx] = final_pos
            sample_angles[sample_idx] = angle
            success_mask[sample_idx] = reached_goal
            if should_log_batch:
                self._log(f"Finished action {batch_label} in {time.monotonic() - batch_start:.1f}s")

        self._log("Compacting successful samples...")
        flat_success_mask = success_mask.reshape(max_samples)
        self.valid_states = states.reshape(max_samples, len(self.material), 4)[flat_success_mask]
        self.valid_states_ = states_.reshape(max_samples, len(self.material), 4)[flat_success_mask]
        self.valid_p_starts = p_starts.reshape(max_samples, 3)[flat_success_mask]
        self.valid_p_stops = p_stops.reshape(max_samples, 3)[flat_success_mask]
        self.valid_angles = sample_angles.reshape(max_samples)[flat_success_mask]
        write_ptr = int(flat_success_mask.sum().item())

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
        self._log(f"Saving run {n_runs} to {full_path}...")
        self._save_config(full_path / (str(n_runs) + "_config.yaml"))
        self._save_data(full_path / (str(n_runs) + "_data.pkl"))
        self._log(f"Collection complete in {time.monotonic() - collection_start:.1f}s")

        # Clean up GPU memory
        self._cleanup_gpu_memory()
    
    def _cleanup_gpu_memory(self):
        """Clean up GPU memory after data collection"""
        # Delete large tensors
        if hasattr(self, 'valid_states'):
            del self.valid_states
        if hasattr(self, 'valid_states_'):
            del self.valid_states_
        if hasattr(self, 'valid_p_starts'):
            del self.valid_p_starts
        if hasattr(self, 'valid_p_stops'):
            del self.valid_p_stops
        if hasattr(self, 'valid_angles'):
            del self.valid_angles

        # Force garbage collection
        import gc
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
