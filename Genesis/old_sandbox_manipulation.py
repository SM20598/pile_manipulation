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
import torch


class SandboxManipulation:

    def __init__(
        self,
        config: dict | str | Path,
        n_envs: int = 1,
        debug : bool = False,
    ):
        """
        Initialize sandbox manipulation with multi-environment support.
        
        Args:
            config: Configuration dict or path to YAML file
            n_envs: Number of parallel environments within a single scene (default: 1)
        """
        if isinstance(config, dict):
            self._config = config
        elif isinstance(config, (str, Path)):
            full_path = Path(__file__).parent / config
            with open(full_path) as stream:
                self._config = yaml.safe_load(stream)
        else:
            raise TypeError("config must be dict or a path to a YAML file")
    
        # extract subdicts from config
        self._sim_params = self._config["simulation"]
        self._box_params = self._config["box"]
        self._plate_params = self._config["plate"] 
        self._material_params = self._config["material"]
        
        self._rigid_options = self._config.get("rigid_options", {})
        
        # Init simulation
        gs.init(
            backend=getattr(gs, self._sim_params.get('backend', 'gpu')),
            precision=self._sim_params.get('precision', '32'),
            performance_mode=self._sim_params.get('performance_mode', True),  # Enable for multi-env
        )

        # PARAMETERS FOR TRAINING
        self._wall_thickness = self._box_params.get('wall_thickness', 0.02)
        self._particle_size = self._material_params["properties"].get('particle_size', 0.01)
        self._granular_vol = self._material_params.get('vol', [0.27, 0.27, 0.1])
        self._material_type = self._material_params.get('type', 'rsa')
        
        collection_cfg = self._config.get("data_collection", {})
        self._settle_steps = collection_cfg.get("settle_steps", 100)
        self._lower_steps = collection_cfg.get("lower_steps", 100)
        self._lift_steps = collection_cfg.get("lift_steps", 100)
        self._goal_threshold = collection_cfg.get("goal_threshold", 0.001)
        
        self._update_visualizer = collection_cfg.get("update_visualizer", False)
        self._debug = debug
        
        # Multi-environment settings
        self._n_envs = n_envs

        self._init_scene()
        self._add_entities()

        self._particle_sizes = None
        self._particle_links_idx = None
        self._particle_dofs_idx = None
        self._particle_linear_dofs_idx = None
        self._particle_angular_dofs_idx = None
        
        
         
        
        ###########
        # HELPERS #
        ###########
        
        # operation height
        p_height = self._particle_size/2 if isinstance(self._particle_size, float) else min(self._particle_size)/4
        self._operation_height = self._wall_thickness/2 + p_height + self._plate_params["size"][2]/2
        
        # helpers to fix all dofs except x, y during sweeping
        self._vertical_dofs_local = [2, 3, 4, 5] 
        self._horizontal_dof_fix = torch.zeros((self._n_envs, 4), device=gs.device)
        self._horizontal_dof_fix[:, 0] = self._operation_height
        
        # lift height for plate
        lift_height = self._box_params["vol"][2]
        self._lift_height_tensor = torch.tensor([0, 0, lift_height], device=gs.device).expand(self._n_envs, -1)
        
        # used to create path for position control
        self._steps_0to1 = torch.linspace(0, 1, 100, device=gs.device)
        
        # helpers to fix all dofs except z during lowering and lifting
        self._vertical_dofs_local = [0, 1, 3, 4, 5] 
        self._vertical_dof_fix = torch.zeros((self._n_envs, 5), device=gs.device)

    def _log(self, message: str):
        print(message, flush=True)

    def _step_scene(self, label: str | None = None, step: int | None = None, total_steps: int | None = None):
                    
        self._scene.step(
            update_visualizer=self._update_visualizer,
            refresh_visualizer=self._update_visualizer,
        )

    def _init_scene(self):
        viewer_settings = self._config["simulation"].get('viewer_options', dict())
        viz_settings = self._config["simulation"].get('viz_options', dict())
        resolution = (1280, 1280)

        v_x, v_y, v_z = self._box_params["vol"]
        l_bound = (-2*v_x, -2*v_y, -2*v_z)
        u_bound = (2*v_x, 2*v_y, 2*v_z+self._wall_thickness)

        viewer_type = viewer_settings.get('viewer_type', None)
        
        if viewer_type == "observer":
            viewer_options = gs.options.ViewerOptions(
                camera_pos    = viewer_settings.get('camera_pos', [3 * v_x, 0.0, 10*v_z]),
                camera_lookat = viewer_settings.get('camera_lookat', [0.0, 0.0, v_z/2]),
                res           = resolution,
            )
        elif viewer_type == "bird":
            viewer_options = gs.options.ViewerOptions(
                camera_pos    = viewer_settings.get('camera_pos', [0, 0, 10*v_z]),
                camera_lookat = viewer_settings.get('camera_lookat', [0.0, 0.0, 0.0]),
                res           = resolution,
            )
        elif viewer_type == "leveled":
            viewer_options = gs.options.ViewerOptions(
                camera_pos    = viewer_settings.get('camera_pos', [1.5, 0, b_z]),
                camera_lookat = viewer_settings.get('camera_lookat', [0.5, 0.0, 0.2]),
                res           = resolution,
            )
        else:
            # No viewer --> Training mode
            viewer_options = None

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
            viewer_options = viewer_options,
            vis_options=gs.options.VisOptions(
                show_link_frame=viz_settings.get('show_link_frame', False),
            ),
            show_viewer=viewer_settings.get('show_viewer', False)
        )
        self._scene.profiling_options.show_FPS = viz_settings.get('show_FPS', False)
    
    def _add_entities(self):
        width, depth, height = self._box_params["vol"]

        def add_box_entity(pos, size, rho=1000, color=[0, 0, 0]):
            material = gs.materials.Rigid(rho=rho)
            box = gs.morphs.Box(pos=pos, size=size, fixed=True)
            surface = gs.surfaces.Default(color=color)
            return self._scene.add_entity(material=material, morph=box, surface=surface)
        
        # floor        
        self.plane = self._scene.add_entity(gs.morphs.Plane())

        # add container
        self.box_parts = {
            "ground_plate": add_box_entity(
                pos=(0, 0, 0),
                size=(width, depth, self._wall_thickness),
            ),
            "front_wall" : add_box_entity(
                pos=(-(width+self._wall_thickness)/2, 0, (height-self._wall_thickness)/2),
                size=(self._wall_thickness, depth, height),
            ),
            "back_wall" : add_box_entity(
                pos=((width+self._wall_thickness)/2, 0, (height-self._wall_thickness)/2),
                size=(self._wall_thickness, depth, height),
            ),
            "left_wall" : add_box_entity(
                pos=(0, (depth+self._wall_thickness)/2, (height-self._wall_thickness)/2),
                size=(width, self._wall_thickness, height),
            ),
            "right_wall" : add_box_entity(
                pos=(0, -(depth+self._wall_thickness)/2, (height-self._wall_thickness)/2),
                size=(width, self._wall_thickness, height),
            ),
        }
        
        # add tool
        self.plate = add_box_entity(
            pos=(0, 0, height * 2),
            size=self._plate_params["size"], 
            rho=self._plate_params.get("rho", 3000),
            color=[0, 1, 0]
        )
        
        # add granular
        material_properties = self._material_params.get('properties', {})
        self._safety_margin = 0.02

        self.material = random_sequential_addition(
            scene=self._scene,
            box_pos=(0, 0, 0),
            granular_vol=self._granular_vol,
            material_properties=material_properties,
            wall_thickness=self._wall_thickness,
            color=[1.0, 1.0, 0.0]
        )                    

    def _save_data(self, path : str | Path, flat_success_mask : torch.Tensor):
        """
        Save data in the legacy list-of-dicts pickle format.

        Each row is cloned before pickling. Indexing a large tensor produces a
        view, and pickling many views can serialize much more backing storage
        than the row itself needs.
        """
        
        valid_states = self._states.reshape(max_samples, len(self.material), state_dim)[flat_success_mask]
        valid_states_ = self._states_.reshape(max_samples, len(self.material), state_dim)[flat_success_mask]
        valid_p_starts = self._p_starts.reshape(max_samples, 3)[flat_success_mask]
        valid_p_stops = self._p_stops.reshape(max_samples, 3)[flat_success_mask]
        valid_angles = self._sample_angles.reshape(max_samples)[flat_success_mask]
        
        path = Path(path)
        use_non_blocking = any(
            tensor.is_cuda
            for tensor in (
                valid_states,
                valid_states_,
                valid_p_starts,
                valid_p_stops,
                valid_angles,
            )
        )

        states = valid_states.detach().to('cpu', non_blocking=use_non_blocking).contiguous()
        states_ = valid_states_.detach().to('cpu', non_blocking=use_non_blocking).contiguous()
        p_starts = valid_p_starts.detach().to('cpu', non_blocking=use_non_blocking).contiguous()
        p_stops = valid_p_stops.detach().to('cpu', non_blocking=use_non_blocking).contiguous()
        angles = valid_angles.detach().to('cpu', non_blocking=use_non_blocking).contiguous()

        # Ensure transfers are complete before pickling
        if use_non_blocking:
            torch.cuda.synchronize()

        data = [
            {
                "state" : states[i].clone(),
                "state_" : states_[i].clone(),
                "action" : (p_starts[i].clone(), p_stops[i].clone(), angles[i].clone())
            } for i in range(states.shape[0])
        ]
        with open(path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _save_config(
            self,
            path : str | Path
        ):
        with open(path, 'w') as outfile:
            yaml.dump(self._config, outfile, default_flow_style=False)
       
    def build(self):
        """Build the scene with multiple environments"""
        self._scene.build(
            n_envs=self._n_envs,
            env_spacing=(self._box_params["vol"][0]*2 , self._box_params["vol"][1]*2)
            )  # Adjust env_spacing as needed
        
        dofs_idx = [0, 1, 2, 3, 4, 5]
        self.plate.set_dofs_kp((0.8,) * 6, dofs_idx)
        self.plate.set_dofs_kv((1.0,) * 6, dofs_idx)
        self._cache_particle_sizes()

    def _cache_particle_sizes(self):
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

    def _sample_particle_property(self, value, *, min_value: float | None = None):
        n_particles = len(self.material)
        if isinstance(value, (int, float)):
            values = np.full(n_particles, float(value), dtype=np.float32)
        else:
            if len(value) == 2:
                values = np.random.uniform(float(value[0]), float(value[1]), n_particles).astype(np.float32)
            elif len(value) == n_particles:
                values = np.asarray(value, dtype=np.float32)
            else:
                raise ValueError(
                    "Particle property must be a scalar, [min, max], "
                    "or one value per particle."
                )
        if min_value is not None:
            values = np.maximum(values, min_value)
        return values

    def _set_particle_density_value(self, particle, density: float):
        old_density = getattr(particle.material, "rho", None)
        particle.material.rho = float(density)
        if getattr(self._scene, "is_built", False) and old_density is not None and old_density > 0:
            particle.set_mass(particle.get_mass() * (float(density) / float(old_density)))

    def _allocate_collection_buffers(self, n_samples: int):
        """Allocate persistent GPU buffers for repeated data collection."""
        state_dim = 7
        max_samples = n_samples * self._n_envs
        
        # self._collection_buffers = {
        #     'states': torch.empty((max_samples, len(self.material), state_dim), device=gs.device),
        #     'states_': torch.empty((max_samples, len(self.material), state_dim), device=gs.device),
        #     'p_starts': torch.empty((max_samples, 3), device=gs.device),
        #     'p_stops': torch.empty((max_samples, 3), device=gs.device),
        #     'angles': torch.empty((max_samples,), device=gs.device),
        #     'env_indices': torch.empty((max_samples,), dtype=torch.long, device=gs.device),
        #     'success_mask': torch.empty((max_samples,), dtype=torch.bool, device=gs.device),
        # }
        
        self._collection_buffers = {
            "states" : torch.empty((n_samples, self._n_envs, len(self.material), state_dim), device=gs.device),
            "states_" : torch.empty((n_samples, self._n_envs, len(self.material), state_dim), device=gs.device),
            "p_starts" : torch.empty((n_samples, self._n_envs, 3), device=gs.device),
            "p_stops" : torch.empty((n_samples, self._n_envs, 3), device=gs.device),
            "sample_angles" : torch.empty((n_samples, self._n_envs), device=gs.device),
            "success_mask" : torch.empty((n_samples, self._n_envs), dtype=torch.bool, device=gs.device),
        }

    def set_material_properties(self, setting):
        """
        Set one material configuration shared by all parallel environments.

        This keeps Genesis on the fast shared link-info path. Density changes
        are applied as scalar entity mass updates, not per-environment masses.
        """
        if self._material_type != "rsa":
            raise NotImplementedError("Shared material property updates are only implemented for RSA particles.")

        particle_frictions = self._sample_particle_property(setting["particle_friction"], min_value=1e-2)
        particle_densities = self._sample_particle_property(setting["particle_density"], min_value=gs.EPS)
        table_friction = max(float(setting["table_friction"]), 1e-2)

        for particle_idx, particle in enumerate(self.material):
            particle.set_friction(float(particle_frictions[particle_idx]))
            self._set_particle_density_value(particle, float(particle_densities[particle_idx]))

        if hasattr(self, "box_parts"):
            for part in self.box_parts.values():
                part.set_friction(table_friction)
        if hasattr(self, "plane") and getattr(self.plane, "material", None) is not None:
            self.plane.set_friction(table_friction)

        material_properties = self._config["material"].setdefault("properties", {})
        material_properties["sampled_friction"] = particle_frictions.tolist()
        material_properties["sampled_density"] = particle_densities.tolist()
        self._config["box"].setdefault("properties", {})["friction"] = table_friction

        metadata = dict(setting.get("metadata", {}))
        metadata.update(
            {
                "particle_friction": material_properties["sampled_friction"],
                "particle_density": material_properties["sampled_density"],
                "table_friction": table_friction,
            }
        )
        self._env_property_metadata = [dict(metadata) for _ in range(self._n_envs)]

        self._config["data_collection_property_sweep"] = {
            "mode": "shared_batch",
            "n_property_envs": self._n_envs,
            "env_settings": self._env_property_metadata,
        }
        return {
            "particle_friction": particle_frictions,
            "particle_density": particle_densities,
            "table_friction": table_friction,
        }

    def _get_particle_positions(self):
        if self._particle_links_idx is not None:
            return self._scene.rigid_solver.get_links_pos(links_idx=self._particle_links_idx)

        positions = torch.empty((self._n_envs, len(self.material), 3), device=gs.device)
        for particle_idx, particle in enumerate(self.material):
            positions[:, particle_idx, :] = particle.get_pos()
        return positions

    def _get_particle_quats(self):
        if self._particle_links_idx is not None:
            return self._scene.rigid_solver.get_links_quat(links_idx=self._particle_links_idx)
        else:
            print("DANKSDN:KS")

    def get_material_state(self, settle_steps: int | None = None):
        """
        Returns particle state (positions and sizes) for all environments.
        Optimized for GPU processing.

        Returns:
            Tensor of shape [n_envs, n_particles, 4] with (x, y, z, size)
        """

        n_p = len(self.material)
        if settle_steps is None:
            settle_steps = self._settle_steps

        # Hold plate still
        self.plate.set_pos(self.plate.get_pos())
        self.plate.control_dofs_position_velocity(
            self.plate.get_pos(),
            torch.zeros((self._n_envs, 3), device=gs.device),
            dofs_idx_local=[0, 1, 2]
        )

        frozen_plate_dofs = self.plate.get_dofs_position()
        for step in range(settle_steps):
            self.plate.set_dofs_position(frozen_plate_dofs)
            self._step_scene()

        state = torch.empty((self._n_envs, n_p, 7), device=gs.device)
        state[:, :, 0:3] = self._get_particle_positions()

        if self._particle_sizes is None:
            self._cache_particle_sizes()
        state[:, :, 3:] = self._get_particle_quats()

        return state
    
    def plate_velocity_translation(
            self,
            p_start,
            p_end,
            angle,
            sweep_steps: int | None = None,
            debug=False,
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
        speed = 0.125
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
        self.plate.set_pos(p_start)
        self.plate.control_dofs_position_velocity(p_end, v, dofs_idx_local=[0, 1, 2])
        
        if sweep_steps is None:
            max_sweep_distance = float(dist.max().item())
            sweep_steps = max(1, math.ceil(max_sweep_distance / (speed * self._scene.dt) * 1.7))
            

        reached_goal = torch.zeros(self._n_envs, dtype=torch.bool, device=gs.device)
        best_dist = torch.full((self._n_envs,), torch.inf, device=gs.device)
        frozen_pos = self.plate.get_pos()

        for step in range(sweep_steps):
            if self._freeze_reached_envs and reached_goal.any():
                reached_envs_idx = reached_goal.nonzero().squeeze(dim=1)
                n_reached = reached_envs_idx.shape[0]
                self.plate.set_pos(
                    frozen_pos[reached_goal],
                    envs_idx=reached_envs_idx,
                    zero_velocity=True,
                )
                self.plate.set_dofs_position(
                    torch.stack([
                        frozen_pos[reached_goal, 0],
                        frozen_pos[reached_goal, 1],
                        torch.full((n_reached,), operation_height, device=gs.device),
                        torch.zeros(n_reached, device=gs.device),
                        torch.zeros(n_reached, device=gs.device),
                        angle[reached_goal],
                    ], dim=1),
                    dofs_idx_local=[0, 1, 2, 3, 4, 5],
                    envs_idx=reached_envs_idx,
                    zero_velocity=True,
                )

            self.plate.set_dofs_position(
                fix_z_and_rot,
                dofs_idx_local=[2, 3, 4, 5],
            )
            self._step_scene()

            cur_pos = self.plate.get_pos()
            cur_dist = torch.linalg.norm(cur_pos[:, :2] - p_end[:, :2], axis=1)
            improved = cur_dist < best_dist
            best_dist = torch.where(improved, cur_dist, best_dist)
            newly_reached = (cur_dist < self._goal_threshold) & ~reached_goal
            frozen_pos = torch.where(newly_reached[:, None], cur_pos, frozen_pos)
            reached_goal |= newly_reached
            if self._freeze_reached_envs and reached_goal.all():
                break

        if self._freeze_reached_envs:
            final_pos = torch.where(reached_goal[:, None], frozen_pos, self.plate.get_pos())
        else:
            final_pos = torch.where(reached_goal[:, None], p_end, self.plate.get_pos())


        return reached_goal, final_pos, sweep_steps
    
    def _plate_velocity_translation(
            self,
            p_start,
            p_end,
            angle,
            sweep_steps: int | None = None,
            debug=False,
        ):
        """
        Move plates with velocity control across all environments.
        
        Args:
            p_start: Starting positions [n_envs, 3] or [3]
            p_end: Ending positions [n_envs, 3] or [3]
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
        self._horizontal_dof_fix[:, -1] = angle  # Set rotation for all envs
        # fix_z_and_rot = torch.stack([
        #     # x is free dof
        #     # y is free dof
        #     torch.full((self._n_envs,), self._operation_height, device=gs.device),
        #     torch.zeros(self._n_envs, device=gs.device),
        #     torch.zeros(self._n_envs, device=gs.device),
        #     angle
        # ], dim=1)


        # Calculate velocity vector for each environment
        delta = p_end - p_start  # [n_envs, 3]
        dist = torch.linalg.norm(delta, axis=1, keepdim=True)  # [n_envs, 1]  
        direction = delta / (dist + 1e-8)
        v = direction * self._plate_params["speed"]  # [n_envs, 3]

        # Set initial position, velocity and goal for all plates in all environments
        self.plate.set_pos(p_start)
        self.plate.control_dofs_position_velocity(p_end, v, dofs_idx_local=[0, 1, 2])
        
        if sweep_steps is None:
            max_sweep_distance = float(dist.max().item())
            sweep_steps = max(1, math.ceil(max_sweep_distance / (self._plate_params["speed"] * self._scene.dt) * 1.7))
        
        reached_goal = torch.zeros(self._n_envs, dtype=torch.bool, device=gs.device)
        best_dist = torch.full((self._n_envs,), torch.inf, device=gs.device)
        frozen_pos = self.plate.get_pos()
        
        for step in range(sweep_steps):
            if reached_goal.any():
                reached_envs_idx = reached_goal.nonzero().squeeze(dim=1)
                n_reached = reached_envs_idx.shape[0]
                self.plate.set_pos(
                    frozen_pos[reached_goal],
                    envs_idx=reached_envs_idx,
                    zero_velocity=True,
                )
                self.plate.set_dofs_position(
                    torch.stack([
                        frozen_pos[reached_goal, 0],
                        frozen_pos[reached_goal, 1],
                        torch.full((n_reached,), operation_height, device=gs.device),
                        torch.zeros(n_reached, device=gs.device),
                        torch.zeros(n_reached, device=gs.device),
                        angle[reached_goal],
                    ], dim=1),
                    dofs_idx_local=[0, 1, 2, 3, 4, 5],
                    envs_idx=reached_envs_idx,
                    zero_velocity=True,
                )

            self.plate.set_dofs_position(
                self._horizontal_dof_fix,
                dofs_idx_local=[2, 3, 4, 5],
            )
            self._step_scene()

            cur_pos = self.plate.get_pos()
            cur_dist = torch.linalg.norm(cur_pos[:, :2] - p_end[:, :2], axis=1)
            improved = cur_dist < best_dist
            best_dist = torch.where(improved, cur_dist, best_dist)
            newly_reached = (cur_dist < self._goal_threshold) & ~reached_goal
            frozen_pos = torch.where(newly_reached[:, None], cur_pos, frozen_pos)
            reached_goal |= newly_reached
            if reached_goal.all():
                if self._debug:
                    print(f"All environments reached target at step {step + 1}")
                break

        final_pos = torch.where(reached_goal[:, None], frozen_pos, self.plate.get_pos())

        print(
            f" > Goal reached : {int(reached_goal.sum().item())}/{self._n_envs}; "
            f" > Best distance range {float(best_dist.min().item()):.4f}-"
            f"{float(best_dist.max().item()):.4f}m"
        )

        return reached_goal, final_pos
    
    def plate_position_translation(self, p_start, p_end, n_steps):
        """
        Move plates with position control across all environments.
        
        Args:
            p_start: Starting positions [n_envs, 3] or [3]
            p_end: Ending positions [n_envs, 3] or [3]
            n_steps: Number of steps for interpolation
        """        
        path = (1 - self._steps_0to1[:, None, None]) * p_start[None, :, :] + self._steps_0to1[:, None, None] * p_end[None, :, :]

        self.plate.set_pos(p_start)
        for i in range(n_steps):
            self.plate.set_pos(pos=path[i])
            self.plate.set_dofs_position(
                position=self._vertical_dof_fix,
                dofs_idx_local=self._vertical_dofs_local
            )
            self._step_scene()

    def generate_action_samples(
            self,
            n_samples: int,
        ):
        """
        Generate random action samples for all environments.
        
        Returns:
            Tuple of (action_starts, action_stops, angles) each of shape [n_envs * n_samples, 3/1]
        """
        tool_length, tool_width, _ = self._plate_params["size"]

        # Generate samples for each environment
        n_total = self._n_envs * n_samples
        angles = (-torch.pi/2) + torch.rand(n_total, device=gs.device) * torch.pi
        
        # Sampling dimensions in x and y from box center
        sample_space_x = self._granular_vol[0]/2 - (torch.cos(angles) * tool_length/2 + abs(torch.sin(angles)) * tool_width/2 + self._safety_margin)
        sample_space_y = self._granular_vol[1]/2 - (abs(torch.sin(angles)) * tool_length/2 + torch.cos(angles) * tool_width/2 + self._safety_margin)

        # Min and max coordinates
        low = torch.stack([-sample_space_x, -sample_space_y], axis=1)
        high = torch.stack([sample_space_x, sample_space_y], axis=1)
        
        # Sample start and end positions
        start_samples = (high - low) * torch.rand((n_total, 2), device=gs.device) + low
        stop_samples = (high - low) * torch.rand((n_total, 2), device=gs.device) + low
        _z = torch.ones((n_total, 1), device=gs.device) * self._operation_height
        
        action_starts = torch.cat((start_samples, _z), axis=1)
        action_stops = torch.cat((stop_samples, _z), axis=1)
        
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
            sweep_steps: int | None = None,
        ):
        """
        Execute action (lower, sweep, lift) for all environments.
        
        Args:
            p_start: Starting positions [n_envs, 3]
            p_stop: Stopping positions [n_envs, 3]
            angle: Angles [n_envs]
            lift_height: Lift height [n_envs, 3]
        
        Returns:
            Tensor of shape [n_envs] with success status
        """

        # Lowering
        # fix_pose_lower = torch.stack([
        #     p_start[:, 0],
        #     p_start[:, 1],
        #     # z is free dof 
        #     torch.zeros(self._n_envs, device=gs.device),
        #     torch.zeros(self._n_envs, device=gs.device),
        #     angle
        # ], dim=1)
        self._vertical_dof_fix[:, 0] = p_start[:, 0]
        self._vertical_dof_fix[:, 1] = p_start[:, 1]
        self._vertical_dof_fix[:, 3] = angle
        
        self.plate_position_translation(
            p_start + self._lift_height_tensor,
            p_start,
            self._lower_steps,
        )
        
        # Sweeping
        reached_goal, final_pos = self.plate_velocity_translation(
            p_start,
            p_stop,
            angle,
            sweep_steps=sweep_steps,
        )

        # fix_pose_lift = torch.stack([
        #     final_pos[:, 0],
        #     final_pos[:, 1],
        #     torch.zeros(self._n_envs, device=gs.device),
        #     torch.zeros(self._n_envs, device=gs.device),
        #     angle
        # ], dim=1)
        self._vertical_dof_fix[:, 0] = final_pos[:, 0]
        self._vertical_dof_fix[:, 1] = final_pos[:, 1]

        self.plate_position_translation(
            final_pos,
            final_pos + self._lift_height_tensor,
            self._lift_steps,
        )

        return reached_goal, final_pos

    def collect_data_samples(
            self,
            n_samples: int = 200,
            path : str | Path = "training",
            settle_steps: int | None = None,
            sweep_steps: int | None = None,
        ):
        """
        Collect data samples from all environments efficiently.
        Optimized for GPU processing and memory efficiency.

        Args:
            n_samples: Number of samples to collect per environment
            path: Output path for data
        """
        max_samples = n_samples * self._n_envs
        
        effective_settle_steps = self._settle_steps if settle_steps is None else settle_steps
        effective_sweep_steps = sweep_steps

        self._config.setdefault("data_collection", {})
        self._config["data_collection"].update({
            "n_envs": self._n_envs,
            "samples_per_env": n_samples,
            "settle_steps": effective_settle_steps,
            "sweep_steps": effective_sweep_steps,
            "sweep_steps_mode": "auto" if effective_sweep_steps is None else "fixed",
            "lower_steps": self._lower_steps,
            "lift_steps": self._lift_steps,
            "goal_threshold": self._goal_threshold,
        })

        # Allocate once or reuse if same size
        if (not hasattr(self, '_collection_buffers') or 
            self._collection_buffers['states'].shape[0] != max_samples):
            self._allocate_collection_buffers(n_samples)
        
        # Clear buffers (much faster than allocating new ones)
        for buf in self._collection_buffers.values():
            buf.zero_()
        
        # Generate action samples
        action_starts, action_stops, angles = self.generate_action_samples(n_samples)

        for sample_idx in range(n_samples):
            print(f" > sample {sample_idx + 1}/{n_samples}")

            state = self.get_material_state(effective_settle_steps)

            p_start = action_starts[:, sample_idx, :]  # [n_envs, 3]
            p_stop = action_stops[:, sample_idx, :]    # [n_envs, 3]
            angle = angles[:, sample_idx]              # [n_envs]

            print("p_start:", p_start)
            print("p_stop:", p_stop)
            print("angle:", angle)

            reached_goal, p_stop = self.execute_action(
                p_start,
                p_stop,
                angle,
                sweep_steps=effective_sweep_steps,
            )
            
            state_ = self.get_material_state(effective_settle_steps)

            self._collection_buffers["states"][sample_idx] = state
            self._collection_buffers["states_"][sample_idx] = state_
            self._collection_buffers["p_starts"][sample_idx] = p_start
            self._collection_buffers["p_stops"][sample_idx] = p_stop
            self._collection_buffers["sample_angles"][sample_idx] = angle
            self._collection_buffers["success_mask"][sample_idx] = reached_goal
            
        # Number of collected samples
        flat_success_mask = self._success_mask.reshape(max_samples)
        num_collected_samples = int(flat_success_mask.sum().item())

        # Print statistics
        print("\nStatistics (Multi-Environment Collection)")
        print("=" * 50)
        print(f">> Number of environments   : {self._n_envs}")
        print(f">> Samples per environment  : {n_samples}")
        print(f">> Total samples collected  : {num_collected_samples}")
        print(f">> Number of failed samples : {max_samples - num_collected_samples}")

        self._config["statistics"] = {
            "Number of environments"   : self._n_envs,
            "Samples per environment"  : n_samples,
            "Total samples collected"  : num_collected_samples,
            "Number of failed samples" : max_samples - num_collected_samples,
        }

        base_dir = Path(__file__).parent
        full_path = base_dir / path
        Path.mkdir(full_path, parents=True, exist_ok=True)

        # look for number of runs in existing dict
        n_runs = int(len([name for name in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, name))])/2)
        self._save_config(full_path / (str(n_runs) + "_config.yaml"))
        self._save_data(full_path / (str(n_runs) + "_data.pkl"), flat_success_mask)
        self._log(f"Material batch finished. Run {n_runs} saved to {full_path}.")

    def destroy(self):
        gs.destroy()

    def view(self, horizon=1000):
        for _ in range(horizon):
            self._scene.visualizer.update()
    
    def simulate(self, horizon=1000):
        for _ in range(horizon):
            self._step_scene()