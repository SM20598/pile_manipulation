import genesis as gs
import genesis.utils.geom as gu 
import numpy as np
import yaml
from utilities.materials import *
from pathlib import Path
import math
import torch

# Pose of "Camera_main" in ../cloudgripper_scene.xml, relative to "Ground_plate"
# (which sits at that file's world origin with identity orientation, so this is
# just Camera_main's own <camera pos=... quat=.../> values). Reused here as the
# camera's pose relative to the center of this box's own ground plate, which is
# likewise placed at (0, 0, 0) - see add_box_entity(pos=(0, 0, 0), ...) below.
_CLOUDGRIPPER_CAMERA_MAIN_POS = (0.14519381523132324, -0.0004741400480270386, 0.12123201787471771)
_CLOUDGRIPPER_CAMERA_MAIN_QUAT_WXYZ = (0.607417, 0.361999, 0.361999, 0.607417)
_CLOUDGRIPPER_CAMERA_MAIN_FOVY = 90  # MuJoCo fovy is vertical FOV in degrees, same convention Genesis's `fov` uses


def _mujoco_camera_to_lookat(pos, quat_wxyz):
    """
    Converts a MuJoCo <camera pos=... quat=.../> pose into Genesis's
    add_camera(pos=..., lookat=..., up=...) convention. MuJoCo cameras look
    down their local -Z axis with local +Y as up.
    """
    w, x, y, z = quat_wxyz
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])
    forward = rot @ np.array([0.0, 0.0, -1.0])
    up = rot @ np.array([0.0, 1.0, 0.0])
    pos = np.array(pos)
    return tuple(pos), tuple(pos + forward), tuple(up)


class SandboxManipulation:

    def __init__(
        self,
        config: dict | str | Path,
        n_envs: int = 1,
        debug : bool = False,
        viewer_type: str | None = None,
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
        self._config.setdefault("data_collection", {})
        self._config["data_collection"].setdefault("sampled", {})
        self._sampled_params = self._config["data_collection"]["sampled"]
        
        self._rigid_options = self._config.get("rigid_options", {})
        
        # Init simulation
        gs.init(
            backend=getattr(gs, self._sim_params.get('backend', 'gpu')),
            precision=self._sim_params.get('precision', '32'),
            performance_mode=self._sim_params.get('performance_mode', True),  # Enable for multi-env
        )

        # PARAMETERS FOR TRAINING
        self._wall_thickness = self._box_params.get('wall_thickness', 0.02)
        self._granular_vol = self._material_params.get('vol', [0.27, 0.27, 0.1])

        # Box height auto-adjusts to the particle size, so a resting monolayer never
        # sticks out above the walls no matter what --particle-sizes is swept over.
        particle_size = self._sampled_params.get(
            "particle_size",
            self._material_params["particle_size"],
        )
        self._box_params["vol"][2] = self._wall_thickness + max_particle_height(
            shape=self._material_params["shape"],
            particle_size=particle_size,
            num_particles=self._material_params["n_particles"],
        )

        self._settle_steps = 200
        self._goal_threshold = 0.001
        
        self._debug = debug
        self._viewer_type = viewer_type

        # Multi-environment settings
        self._n_envs = n_envs

        # Optional per-env camera rendering, saved into "_rollout.pt" files
        self._render_images = bool(self._config["data_collection"].get("render_images", False))
        self._render_resolution = tuple(
            self._config["data_collection"].get("render_resolution", (128, 128))
        )
        self._cameras = []

        self._init_scene()
        self._add_entities()
        
        ###########
        # HELPERS #
        ###########
        
        # operation height
        particle_size = self._material_params["particle_size"]
        p_height = particle_size/2 if isinstance(particle_size, float) else min(particle_size)/4
        self._operation_height = self._wall_thickness/2 + p_height + self._plate_params["size"][2]/2
        
        # lift height for plate
        lift_height = self._box_params["vol"][2]
        self._lift_height_tensor = torch.tensor([0, 0, lift_height], device=gs.device).expand(self._n_envs, -1)
        
        # used to create path for position control
        self._pos_ctrl_steps = 100
        self._steps_0to1 = torch.linspace(0, 1, self._pos_ctrl_steps, device=gs.device)
        
        # helpers to fix all dofs except z during lowering and lifting
        self._vertical_dofs_local = [0, 1, 3, 4, 5] 
        self._vertical_dof_fix = torch.zeros((self._n_envs, 5), device=gs.device)

        # helpers to fix all dofs except x, y during sweeping
        self._horizontal_dofs_local = [2, 3, 4, 5] 
        self._horizontal_dof_fix = torch.zeros((self._n_envs, 4), device=gs.device)
        self._horizontal_dof_fix[:, 0] = self._operation_height

        self._particle_state = torch.empty((self._n_envs, self._material_params["n_particles"], 7), device=gs.device)
        self._particle_state_ = torch.empty((self._n_envs, self._material_params["n_particles"], 7), device=gs.device)
        
        self._zero_n_envsx3 = torch.zeros((self._n_envs, 3), device=gs.device)

        # pre-allocated freeze buffer for reached-goal envs in the sweep loop
        # layout: [x, y, z=operation_height, roll=0, pitch=0, yaw]
        self._freeze_dofs_buf = torch.zeros((self._n_envs, 6), device=gs.device)
        self._freeze_dofs_buf[:, 2] = self._operation_height


    def _log(self, message: str):
        print(message, flush=True)

    def _step_scene(self):
                    
        self._scene.step(
            update_visualizer=self._debug,
            refresh_visualizer=self._debug,
        )

    def _init_scene(self):
        v_x, _, v_z = self._box_params["vol"]
        resolution = (1280, 1280)
        
        if self._viewer_type == "observer":
            viewer_options = gs.options.ViewerOptions(
                camera_pos    = [3 * v_x, 0.0, 3*v_z],
                camera_lookat = [0.0, 0.0, v_z/2],
                res           = resolution,
            )
        elif self._viewer_type == "bird":
            viewer_options = gs.options.ViewerOptions(
                camera_pos    = [0, 0, 10*v_z],
                camera_lookat = [0.0, 0.0, 0.0],
                res           = resolution,
            )
        elif self._viewer_type == "leveled":
            viewer_options = gs.options.ViewerOptions(
                camera_pos    = [1.5, 0, v_z],
                camera_lookat = [0.0, 0.0, v_z],
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
                show_link_frame=self._debug and self._viewer_type == "observer",
            ),
            show_viewer=self._debug
        )
        self._scene.profiling_options.show_FPS=False
    
    def _add_entities(self):
        width, depth, height = self._box_params["vol"]

        def add_box_entity(pos, size):
            box = gs.morphs.Box(pos=pos, size=size, fixed=True)
            surface = gs.surfaces.Default(color=[0, 0, 0])
            return self._scene.add_entity(morph=box, surface=surface)
        
        # floor        
        self.plane = self._scene.add_entity(gs.morphs.Plane())

        # add container
        self.box_parts = {
            "ground_plate": add_box_entity(
                pos=(0, 0, 0),
                size=(width, depth, self._wall_thickness),
            ),
            # front/back walls are extended by 2*wall_thickness in y so they cover the
            # corners too (left/right walls are sized to fit snugly between them) -
            # otherwise each corner has a wall_thickness x wall_thickness hole straight
            # through to the outside, invisible from directly above but obvious at an angle
            "front_wall" : add_box_entity(
                pos=(-(width+self._wall_thickness)/2, 0, (height-self._wall_thickness)/2),
                size=(self._wall_thickness, depth + 2 * self._wall_thickness, height),
            ),
            "back_wall" : add_box_entity(
                pos=((width+self._wall_thickness)/2, 0, (height-self._wall_thickness)/2),
                size=(self._wall_thickness, depth + 2 * self._wall_thickness, height),
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
        self.plate = self._scene.add_entity(
            material=gs.materials.Rigid(
                rho=3000,
            ),
            morph=gs.morphs.Box(
                pos=(0, 0, height * 2),
                size=self._plate_params["size"]
            ),
            surface=gs.surfaces.Default(color=[0, 1, 0])
        )
        
        # add granular
        self._safety_margin = 0.02

        particle_size = self._sampled_params.get(
            "particle_size",
            self._material_params["particle_size"],
        )
        self.material, particle_sizes = random_sequential_addition(
            scene=self._scene,
            granular_vol=self._granular_vol,
            shape=self._material_params["shape"],
            num_particles=self._material_params["n_particles"],
            particle_size=particle_size,
            wall_thickness=self._wall_thickness,
            box_height=height,
        )
        self._config["data_collection"]["sampled"].update({"particle_sizes": particle_sizes})

        # add one camera per env (Genesis renders per env_idx, not batched), posed like
        # cloudgripper_scene.xml's "Camera_main" relative to its ground plate's center
        if self._render_images:
            cam_pos, cam_lookat, cam_up = _mujoco_camera_to_lookat(
                _CLOUDGRIPPER_CAMERA_MAIN_POS, _CLOUDGRIPPER_CAMERA_MAIN_QUAT_WXYZ
            )
            self._cameras = [
                self._scene.add_camera(
                    res=self._render_resolution,
                    pos=cam_pos,
                    lookat=cam_lookat,
                    up=cam_up,
                    fov=_CLOUDGRIPPER_CAMERA_MAIN_FOVY,
                    GUI=False,
                    env_idx=env_idx,
                )
                for env_idx in range(self._n_envs)
            ]

    def _save_data(self, path : str | Path, num : int, flat_success_mask : torch.Tensor, max_samples : int):
        """
        Save data efficiently using torch.save (binary format).
        
        Avoids per-sample cloning and per-element pickling. Supports both
        successful and failed samples. ~2-10x faster than pickle list-of-dicts.
        """
        path = Path(path)
        
        # Split into valid (successful) and failed samples
        valid_states = self._collection_buffers["states"].reshape(max_samples, len(self.material), 7)[flat_success_mask]
        valid_states_ = self._collection_buffers["states_"].reshape(max_samples, len(self.material), 7)[flat_success_mask]
        valid_p_starts = self._collection_buffers["p_starts"].reshape(max_samples, 3)[flat_success_mask]
        valid_p_stops = self._collection_buffers["p_stops"].reshape(max_samples, 3)[flat_success_mask]
        valid_angles = self._collection_buffers["sample_angles"].reshape(max_samples)[flat_success_mask]
        
        failed_states = self._collection_buffers["states"].reshape(max_samples, len(self.material), 7)[~flat_success_mask]
        failed_states_ = self._collection_buffers["states_"].reshape(max_samples, len(self.material), 7)[~flat_success_mask]
        failed_p_starts = self._collection_buffers["p_starts"].reshape(max_samples, 3)[~flat_success_mask]
        failed_p_stops = self._collection_buffers["p_stops"].reshape(max_samples, 3)[~flat_success_mask]
        failed_angles = self._collection_buffers["sample_angles"].reshape(max_samples)[~flat_success_mask]

        # Check if any tensor is on GPU
        use_non_blocking = any(
            tensor.is_cuda
            for tensor in (valid_states, valid_states_, valid_p_starts, valid_p_stops, valid_angles)
        )

        # Transfer all tensors to CPU in bulk (GPU → CPU DMA)
        valid_data = {
            "states": valid_states.detach().to('cpu', non_blocking=use_non_blocking).contiguous(),
            "states_": valid_states_.detach().to('cpu', non_blocking=use_non_blocking).contiguous(),
            "p_starts": valid_p_starts.detach().to('cpu', non_blocking=use_non_blocking).contiguous(),
            "p_stops": valid_p_stops.detach().to('cpu', non_blocking=use_non_blocking).contiguous(),
            "angles": valid_angles.detach().to('cpu', non_blocking=use_non_blocking).contiguous(),
        }
        
        failed_data = {
            "states": failed_states.detach().to('cpu', non_blocking=use_non_blocking).contiguous(),
            "states_": failed_states_.detach().to('cpu', non_blocking=use_non_blocking).contiguous(),
            "p_starts": failed_p_starts.detach().to('cpu', non_blocking=use_non_blocking).contiguous(),
            "p_stops": failed_p_stops.detach().to('cpu', non_blocking=use_non_blocking).contiguous(),
            "angles": failed_angles.detach().to('cpu', non_blocking=use_non_blocking).contiguous(),
        }

        # Ensure GPU→CPU transfers complete before I/O
        if use_non_blocking:
            torch.cuda.synchronize()

        # Save as torch binary format (faster and preserves dtype/shape)
        torch.save(valid_data, str(path / f"_{num}_data.pt"))
        torch.save(failed_data, str(path / f"_{num}_failed.pt"))

    def _save_rollout(self, path: str | Path, num: int):
        """
        Save the un-flattened, per-env rollout: buffers are collected as
        (n_samples, n_envs, ...) with no reshuffling between samples, so each
        env's n_samples steps form one continuous multi-frame trajectory.

        Unlike `_save_data`, samples are NOT dropped based on `success_mask`
        (a "failed" step just means the plate didn't reach its exact target;
        the resulting state is still valid dynamics data) and env/step order
        is preserved so downstream consumers can reconstruct trajectories.
        """
        path = Path(path)

        def to_cpu(t):
            return t.detach().transpose(0, 1).to('cpu').contiguous()

        rollout_data = {
            "states": to_cpu(self._collection_buffers["states"]),
            "states_": to_cpu(self._collection_buffers["states_"]),
            "p_starts": to_cpu(self._collection_buffers["p_starts"]),
            "p_stops": to_cpu(self._collection_buffers["p_stops"]),
            "angles": to_cpu(self._collection_buffers["sample_angles"]),
            "success_mask": to_cpu(self._collection_buffers["success_mask"]),
        }
        if self._render_images:
            rollout_data["frames"] = to_cpu(self._collection_buffers["frames"])
            rollout_data["frames_"] = to_cpu(self._collection_buffers["frames_"])

        torch.save(rollout_data, str(path / f"_{num}_rollout.pt"))

    def _save_config(
            self,
            path : str | Path,
            num : int
        ):
        path = path / (f"_{num}_config.yaml")
        with open(path, 'w') as outfile:
            yaml.dump(self._config, outfile, default_flow_style=False)

    def _allocate_collection_buffers(self, n_samples: int):
        """Allocate persistent GPU buffers for repeated data collection."""
        state_dim = 7
        self._collection_buffers = {
            "states" : torch.empty((n_samples, self._n_envs, len(self.material), state_dim), device=gs.device),
            "states_" : torch.empty((n_samples, self._n_envs, len(self.material), state_dim), device=gs.device),
            "p_starts" : torch.empty((n_samples, self._n_envs, 3), device=gs.device),
            "p_stops" : torch.empty((n_samples, self._n_envs, 3), device=gs.device),
            "sample_angles" : torch.empty((n_samples, self._n_envs), device=gs.device),
            "success_mask" : torch.empty((n_samples, self._n_envs), dtype=torch.bool, device=gs.device),
        }
        if self._render_images:
            h, w = self._render_resolution
            self._collection_buffers["frames"] = torch.empty(
                (n_samples, self._n_envs, h, w, 3), dtype=torch.uint8, device='cpu'
            )
            self._collection_buffers["frames_"] = torch.empty(
                (n_samples, self._n_envs, h, w, 3), dtype=torch.uint8, device='cpu'
            )

    @staticmethod
    def load_data(path: str | Path, split: str = "valid"):
        """
        Load saved data from torch.save format (replaces old pickle loader).
        
        Args:
            path: Can be one of:
                - Full path to .pt file: "/path/to/0_data.pt"
                - Base path without extension: "/path/to/0_data"
                - Run directory with number: "/path/to/training" (looks for "0_data.pt")
            split: "valid" for successful samples, "failed" for failed samples (ignored if path has extension)
        
        Returns:
            Dict with keys: "states", "states_", "p_starts", "p_stops", "angles"
            Each is a CPU-side tensor ready for training.
        
        Example:
            # Full path
            data = SandboxManipulation.load_data("/path/to/0_data.pt")
            
            # Base path with split
            data = SandboxManipulation.load_data("/path/to/0_data", split="valid")
            data = SandboxManipulation.load_data("/path/to/0", split="valid")
        """
        path = Path(path)
        
        # If path has .pt extension, use it directly
        if path.suffix == ".pt":
            file_path = path
        else:
            # Construct filename based on split
            if split == "valid":
                suffix = "_data.pt"
            elif split == "failed":
                suffix = "_failed.pt"
            else:
                raise ValueError("split must be 'valid' or 'failed'")
            
            # Handle case where path ends with _data or _failed already
            path_str = str(path)
            if path_str.endswith("_data"):
                file_path = Path(path_str.replace("_data", suffix))
            elif path_str.endswith("_failed"):
                file_path = Path(path_str.replace("_failed", suffix))
            else:
                file_path = path.parent / (path.name + suffix)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        return torch.load(file_path, weights_only=False)
       
    def build(self):
        """Build the scene with multiple environments"""
        self._scene.build(
            n_envs=self._n_envs,
            env_spacing=(self._box_params["vol"][0]*2 , self._box_params["vol"][1]*2)
            )  # Adjust env_spacing as needed
        
        dofs_idx = [0, 1, 2, 3, 4, 5]
        self.plate.set_dofs_kp((0.8,) * 6, dofs_idx)
        self.plate.set_dofs_kv((1.0,) * 6, dofs_idx)

        self._cache_particle_idx()

    def _cache_particle_idx(self):
        links_idx = []
        dofs_idx = []
        for i, particle in enumerate(self.material):
            links_idx.append(particle.link_start)
            if particle.n_dofs == 6:
                dofs_idx.extend(range(particle.dof_start, particle.dof_end))
                
        self._particle_links_idx = torch.tensor(links_idx, dtype=gs.tc_int, device=gs.device)
        self._particle_dofs_idx = torch.tensor(dofs_idx, dtype=gs.tc_int, device=gs.device)

    def _sample_particle_property(self, value, *, min_value: float | None = None):
        n_particles = len(self.material)
        if isinstance(value, (int, float)):
            values = np.full(n_particles, float(value), dtype=np.float32)
        else:
            if len(value) >= n_particles:
                values = np.asarray(value[:n_particles], dtype=np.float32)
            else:
                raise ValueError(
                    "Particle property must be a scalar or a list with the same length as the number of particles"
                )
        if min_value is not None:
            values = np.maximum(values, min_value)
        return values

    def _set_particle_density_value(self, particle, density: float):
        old_density = getattr(particle.material, "rho", None)
        particle.material.rho = float(density)
        if getattr(self._scene, "is_built", False) and old_density is not None and old_density > 0:
            particle.set_mass(particle.get_mass() * (float(density) / float(old_density)))

    def set_material_properties(self, setting):
        """
        Set one material configuration shared by all parallel environments.

        This keeps Genesis on the fast shared link-info path. Density changes
        are applied as scalar entity mass updates, not per-environment masses.
        """
        particle_friction = (
            setting["sampled_particle_friction"]
            if setting.get("sampled_particle_friction") is not None
            else setting["particle_friction"]
        )
        particle_density = (
            setting["sampled_particle_density"]
            if setting.get("sampled_particle_density") is not None
            else setting["particle_density"]
        )
        particle_frictions = self._sample_particle_property(particle_friction, min_value=1e-2)
        particle_densities = self._sample_particle_property(particle_density, min_value=gs.EPS)
        box_friction = max(float(setting["box_friction"]), 1e-2)

        for particle_idx, particle in enumerate(self.material):
            particle.set_friction(float(particle_frictions[particle_idx]))
            self._set_particle_density_value(particle, float(particle_densities[particle_idx]))

        for part in self.box_parts.values():
            part.set_friction(box_friction)

        # save to config dict
        self._material_params["friction"] = setting["particle_friction"]
        self._material_params["density"] = setting["particle_density"]
        self._box_params["friction"] = setting["box_friction"]
        self._sampled_params.pop("friction", None)
        self._sampled_params.pop("density", None)
        if setting.get("sampled_particle_friction") is not None:
            self._sampled_params["friction"] = particle_frictions.tolist()
        if setting.get("sampled_particle_density") is not None:
            self._sampled_params["density"] = particle_densities.tolist()
        self._sampled_params["box_friction"] = box_friction

    def _particle_shape_extents(self):
        """
        Returns (half_extents, placement_half_extents, collision_half_extents),
        each (n_particles, 3) - shared sizing preamble used by every particle
        placement method (shuffle_particles, arrange_particles_in_area, ...).
        """
        size_values = self._sampled_params.get("particle_sizes", None)
        if size_values is None:
            size_values = [
                particle.morph.size if hasattr(particle.morph, "size")
                else (particle.morph.radius * 2, particle.morph.radius * 2, particle.morph.height)
                if hasattr(particle.morph, "height") and hasattr(particle.morph, "radius")
                else (particle.morph.radius * 2,) * 3
                for particle in self.material
            ]
        sizes = torch.as_tensor(size_values, dtype=torch.float32, device=gs.device)
        half_extents = sizes * 0.5

        # For cubes, a random yaw rotation up to 45° increases the xy footprint by up to sqrt(2).
        # Use conservative collision extents so placed cubes don't overlap after rotation is applied.
        is_cube = torch.tensor(
            [hasattr(p.morph, "size") for p in self.material],
            dtype=torch.float32, device=gs.device,
        )
        xy_scale = 1.0 + (math.sqrt(2) - 1.0) * is_cube  # sqrt(2) for cubes, 1.0 for others
        collision_half_extents = half_extents.clone()
        collision_half_extents[:, :2] = half_extents[:, :2] * xy_scale.unsqueeze(1)
        is_cylinder = torch.tensor(
            [hasattr(p.morph, "height") and hasattr(p.morph, "radius") for p in self.material],
            dtype=torch.bool, device=gs.device,
        )
        placement_half_extents = half_extents.clone()
        if bool(is_cylinder.any().item()):
            cylinder_half_extent = half_extents[is_cylinder].max(dim=1).values
            placement_half_extents[is_cylinder] = cylinder_half_extent.unsqueeze(1).expand(-1, 3)
            collision_half_extents[is_cylinder] = placement_half_extents[is_cylinder]
        return half_extents, placement_half_extents, collision_half_extents

    def _box_inner_bounds(self):
        width, depth, height = self._box_params["vol"]
        wall = float(self._wall_thickness)
        inner_min = torch.tensor([-width / 2, -depth / 2, wall / 2], device=gs.device)
        inner_max = torch.tensor([width / 2, depth / 2, height - wall / 2], device=gs.device)
        return inner_min, inner_max

    def _set_particle_positions(self, positions, quats):
        """positions, quats: (n_envs, n_particles, 3/4). Teleports + zeros velocity."""
        envs_idx = torch.arange(self._n_envs, device=gs.device)
        for particle_idx, particle in enumerate(self.material):
            particle.set_pos(positions[:, particle_idx, :].contiguous(), envs_idx=envs_idx)
            particle.set_quat(quats[:, particle_idx, :].contiguous(), envs_idx=envs_idx)
        if self._particle_dofs_idx.numel() > 0:
            self._scene.rigid_solver.set_dofs_velocity(
                torch.zeros((self._n_envs, self._particle_dofs_idx.numel()), device=gs.device),
                dofs_idx=self._particle_dofs_idx,
                skip_forward=True,
            )

    def shuffle_particles(self):
        n_particles = len(self.material)
        if n_particles == 0:
            return

        max_retries = 10
        for attempt in range(max_retries):
            try:
                half_extents, placement_half_extents, collision_half_extents = self._particle_shape_extents()
                inner_min, inner_max = self._box_inner_bounds()
                positions = self._sample_nonoverlapping_particle_positions(
                    half_extents=half_extents,
                    placement_half_extents=placement_half_extents,
                    collision_half_extents=collision_half_extents,
                    inner_min=inner_min,
                    inner_max=inner_max,
                )

                quats = torch.stack(
                    [self._random_particle_quats(particle, self._n_envs) for particle in self.material],
                    dim=1,
                )
                self._set_particle_positions(positions, quats)
                # Success, break out of retry loop
                break
            except RuntimeError as e:
                if str(e) == "placement_failed":
                    print(f"Placement of particles failed due to overlap, retrying {attempt+1}/{max_retries}...")
                    if attempt == max_retries - 1:
                        raise RuntimeError(
                            f"Could not randomly shuffle particles without overlap after {max_retries} attempts. "
                            "Try a smaller particle size or fewer particles."
                        )
                    # else, try again
                    continue
                else:
                    raise

    def set_particle_state(self, state: torch.Tensor):
        """
        Hard-teleport particles to an explicit state, bypassing physics (same
        set_pos/set_quat mechanism shuffle_particles() uses for random
        placement). Does not settle - call update_material_state() afterwards
        if you want contacts/gravity resolved before reading state or
        rendering.

        Args:
            state: (n_particles, 7) or (n_envs, n_particles, 7) tensor of
                [x, y, z, qw, qx, qy, qz]. A 2D input is broadcast to all envs.
        """
        state = state.to(device=gs.device, dtype=torch.float32)
        if state.ndim == 2:
            state = state.unsqueeze(0).expand(self._n_envs, -1, -1)
        self._set_particle_positions(state[..., 0:3], state[..., 3:7])

    def default_area_radius(self, size_fraction: float = 0.9) -> float:
        """
        The target-area radius arrange_particles_in_area() computes when
        `radius` isn't given directly - exposed so callers that need to know
        the *intended* target region (e.g. a success check) can reference
        the same deterministic, box-geometry-derived value directly, rather
        than re-deriving it from a particular placement's noisy realized
        (settled, possibly overlap-perturbed) particle positions.
        """
        _, _, collision_half_extents = self._particle_shape_extents()
        _, box_inner_max = self._box_inner_bounds()
        max_half = float(collision_half_extents[:, :2].max().item())
        box_half = min(float(box_inner_max[0].item()), float(box_inner_max[1].item())) - max_half
        return max(box_half * size_fraction, 0.0)

    def arrange_particles_in_area(
        self,
        center_xy=(0.0, 0.0),
        radius: float | None = None,
        size_fraction: float = 0.9,
    ) -> float:
        """
        Places all particles inside a circular target area, non-overlapping
        wherever a spot can be found - the goal configuration for a "gather
        material into a target zone" planning task (particles clustered
        within a region, not arranged along its boundary - see the old
        arrange_particles_circle approach, removed because a boundary ring
        made an unnecessarily hard, visually unclean goal). All envs get the
        same arrangement.

        n_particles~30 cube particles at this box/particle size already pack
        close to the *entire* box's capacity (that's what shuffle_particles()
        relies on for the full-box scatter) - a disk has less area than the
        box that circumscribes it (pi/4 of it), so even radius=(the box's own
        half-extent) is packing-tight, and a *smaller* target area (the point
        of this method - a full-box-sized target wouldn't force any real
        gathering, since a random scatter already roughly fills the box) is
        tighter still. Any particle that can't find a non-overlapping spot is
        dropped in anyway (may overlap) - call update_material_state()
        afterwards to let contact resolution settle it, same approximation
        shuffle_particles() would need if you pushed *it* this close to
        capacity. Counterintuitively, a *smaller* size_fraction doesn't
        reliably yield a tighter settled result: more overlap-fallback
        particles means more contact-driven expansion, which can push the
        actual settled spread out beyond the intended radius (verified
        empirically - size_fraction=0.7 settled wider than 0.9 or 1.0 did).
        The default size_fraction=0.9 was picked for a good balance: clearly
        smaller than the box's own corner-to-center reach (which a random
        scatter's own radial spread includes), while still keeping most
        particles' non-overlap-fallback placement intact.

        Args:
            center_xy: target-area center in box-local meters ((0, 0) = box center).
            radius: target-area radius in meters. If None, computed from
                `size_fraction` of the box's usable half-extent.
            size_fraction: fraction of the box's usable half-extent to use
                when `radius` isn't given directly.

        Returns:
            The radius actually used (meters).
        """
        n_particles = len(self.material)
        if n_particles == 0:
            return 0.0

        half_extents, placement_half_extents, collision_half_extents = self._particle_shape_extents()
        box_inner_min, box_inner_max = self._box_inner_bounds()

        if radius is None:
            radius = self.default_area_radius(size_fraction)

        center = torch.tensor(center_xy, dtype=torch.float32, device=gs.device)
        floor_z = box_inner_min[2] + placement_half_extents[:, 2] + 1e-3

        positions = self._sample_positions_in_disk(
            half_extents=half_extents,
            collision_half_extents=collision_half_extents,
            center=center,
            radius=radius,
            floor_z=floor_z,
        )
        quats = torch.stack(
            [self._random_particle_quats(particle, self._n_envs) for particle in self.material],
            dim=1,
        )
        self._set_particle_positions(positions, quats)
        return radius

    def _sample_positions_in_disk(
        self,
        *,
        half_extents: torch.Tensor,
        collision_half_extents: torch.Tensor,
        center: torch.Tensor,
        radius: float,
        floor_z: torch.Tensor,
        min_gap: float = 1e-3,
    ) -> torch.Tensor:
        """
        Like _sample_nonoverlapping_particle_positions, but samples candidates
        uniformly within a disk (uniform-area via sqrt(rand)*radius) instead
        of a rectangle, and never raises: a particle that can't find a
        non-overlapping spot after the retry budget is placed at a random
        (possibly overlapping) spot in the disk rather than failing outright.
        """
        n_particles = half_extents.shape[0]
        positions = torch.empty((self._n_envs, n_particles, 3), device=gs.device)
        placed = torch.zeros(n_particles, dtype=torch.bool, device=gs.device)
        order = torch.argsort(torch.prod(half_extents, dim=1), descending=True)
        candidate_batch = max(1024, min(4096, 64 * n_particles))
        n_overlapping = 0

        for particle_idx_tensor in order:
            particle_idx = int(particle_idx_tensor.item())
            max_particle_half = float(collision_half_extents[particle_idx, :2].max().item())
            placement_radius = max(radius - max_particle_half, 1e-6)
            active = torch.ones(self._n_envs, dtype=torch.bool, device=gs.device)

            for _ in range(256):
                active_idx = torch.nonzero(active, as_tuple=False).squeeze(1)
                if active_idx.numel() == 0:
                    break
                r = placement_radius * torch.sqrt(
                    torch.rand((active_idx.numel(), candidate_batch), device=gs.device)
                )
                theta = torch.rand((active_idx.numel(), candidate_batch), device=gs.device) * (2 * math.pi)
                candidate_xy = center + torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)
                placed_idx = torch.nonzero(placed, as_tuple=False).squeeze(1)
                if placed_idx.numel() == 0:
                    valid = torch.ones((active_idx.numel(), candidate_batch), dtype=torch.bool, device=gs.device)
                else:
                    delta = candidate_xy.unsqueeze(2) - positions[active_idx][:, placed_idx, :2].unsqueeze(1)
                    min_sep = collision_half_extents[particle_idx, :2] + collision_half_extents[placed_idx, :2] + min_gap
                    valid = (torch.abs(delta) >= min_sep.view(1, 1, -1, 2)).any(dim=3).all(dim=2)
                has_valid = valid.any(dim=1)
                if has_valid.any():
                    accepted = active_idx[has_valid]
                    first_valid = valid[has_valid].to(torch.int64).argmax(dim=1)
                    positions[accepted, particle_idx, :2] = candidate_xy[has_valid, first_valid]
                    positions[accepted, particle_idx, 2] = floor_z[particle_idx]
                    active[accepted] = False

            if active.any():
                # no non-overlapping spot found for this particle in some envs -
                # drop it in anyway; update_material_state()'s settle pass will
                # let contact resolution push things apart.
                n_overlapping += int(active.sum().item())
                leftover_idx = torch.nonzero(active, as_tuple=False).squeeze(1)
                r = placement_radius * torch.sqrt(torch.rand(leftover_idx.numel(), device=gs.device))
                theta = torch.rand(leftover_idx.numel(), device=gs.device) * (2 * math.pi)
                xy = center + torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)
                positions[leftover_idx, particle_idx, :2] = xy
                positions[leftover_idx, particle_idx, 2] = floor_z[particle_idx]
            placed[particle_idx] = True

        if n_overlapping > 0:
            print(f"arrange_particles_in_area: {n_overlapping} particle(s) placed with overlap at this radius "
                  f"({radius:.4f}m) - relying on settle physics to resolve it.")
        return positions

    def _sample_nonoverlapping_particle_positions(
        self,
        *,
        half_extents: torch.Tensor,
        placement_half_extents: torch.Tensor,
        collision_half_extents: torch.Tensor,
        inner_min: torch.Tensor,
        inner_max: torch.Tensor,
    ) -> torch.Tensor:
        n_particles = half_extents.shape[0]
        positions = torch.empty((self._n_envs, n_particles, 3), device=gs.device)
        placed = torch.zeros(n_particles, dtype=torch.bool, device=gs.device)
        order = torch.argsort(torch.prod(half_extents, dim=1), descending=True)
        candidate_batch = max(1024, min(4096, 64 * n_particles))
        min_gap = 1e-3

        lower = inner_min + collision_half_extents
        upper = inner_max - collision_half_extents
        fit_eps = 1e-6  # tolerance for float32 rounding when box height is an exact fit
        if (upper[:, 2] < lower[:, 2] - fit_eps).any():
            shortfall = float((lower[:, 2] - upper[:, 2]).max())
            raise ValueError(
                f"Box height is too small for these particles: particles would stick out "
                f"of the box in z. Box height must be at least wall_thickness + particle "
                f"height (short by {shortfall:.4f}m)."
            )
        if (upper[:, :2] < lower[:, :2]).any():
            raise ValueError("At least one particle is too large to fit inside the box in x/y.")

        for particle_idx_tensor in order:
            particle_idx = int(particle_idx_tensor.item())
            active = torch.ones(self._n_envs, dtype=torch.bool, device=gs.device)
            span_xy = upper[particle_idx, :2] - lower[particle_idx, :2]
            z_pos = inner_min[2] + placement_half_extents[particle_idx, 2] + min_gap
            for _ in range(128):
                active_idx = torch.nonzero(active, as_tuple=False).squeeze(1)
                if active_idx.numel() == 0:
                    break
                candidate_xy = (
                    torch.rand((active_idx.numel(), candidate_batch, 2), device=gs.device)
                    * span_xy
                    + lower[particle_idx, :2]
                )
                placed_idx = torch.nonzero(placed, as_tuple=False).squeeze(1)
                if placed_idx.numel() == 0:
                    valid = torch.ones((active_idx.numel(), candidate_batch), dtype=torch.bool, device=gs.device)
                else:
                    delta = candidate_xy.unsqueeze(2) - positions[active_idx][:, placed_idx, :2].unsqueeze(1)
                    min_sep = collision_half_extents[particle_idx, :2] + collision_half_extents[placed_idx, :2] + min_gap
                    valid = (torch.abs(delta) >= min_sep.view(1, 1, -1, 2)).any(dim=3).all(dim=2)
                has_valid = valid.any(dim=1)
                if has_valid.any():
                    accepted = active_idx[has_valid]
                    first_valid = valid[has_valid].to(torch.int64).argmax(dim=1)
                    positions[accepted, particle_idx, :2] = candidate_xy[has_valid, first_valid]
                    positions[accepted, particle_idx, 2] = z_pos
                    active[accepted] = False
            if active.any():
                return self._grid_particle_positions(
                    half_extents=half_extents,
                    placement_half_extents=placement_half_extents,
                    collision_half_extents=collision_half_extents,
                    inner_min=inner_min,
                    inner_max=inner_max,
                    min_gap=min_gap,
                )
            placed[particle_idx] = True

        return positions

    def _grid_particle_positions(
        self,
        *,
        half_extents: torch.Tensor,
        placement_half_extents: torch.Tensor,
        collision_half_extents: torch.Tensor,
        inner_min: torch.Tensor,
        inner_max: torch.Tensor,
        min_gap: float,
    ) -> torch.Tensor:
        n_particles = half_extents.shape[0]
        max_half_xy = collision_half_extents[:, :2].max(dim=0).values
        grid_lower = inner_min[:2] + max_half_xy
        grid_upper = inner_max[:2] - max_half_xy
        grid_span = grid_upper - grid_lower
        min_spacing = 2.0 * max_half_xy + min_gap

        best_dims = None
        best_score = None
        for n_x in range(1, n_particles + 1):
            n_y = math.ceil(n_particles / n_x)
            spacing_x = grid_span[0] / max(n_x - 1, 1)
            spacing_y = grid_span[1] / max(n_y - 1, 1)
            if n_x > 1 and bool((spacing_x < min_spacing[0]).item()):
                continue
            if n_y > 1 and bool((spacing_y < min_spacing[1]).item()):
                continue
            score = abs(float((spacing_x - spacing_y).item())) + 1e-6 * (n_x * n_y - n_particles)
            if best_score is None or score < best_score:
                best_dims = (n_x, n_y)
                best_score = score

        if best_dims is None:
            raise RuntimeError("placement_failed")

        n_x, n_y = best_dims
        xs = torch.linspace(grid_lower[0], grid_upper[0], n_x, device=gs.device)
        ys = torch.linspace(grid_lower[1], grid_upper[1], n_y, device=gs.device)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
        cells = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)

        spacing = torch.stack(
            (
                grid_span[0] / max(n_x - 1, 1),
                grid_span[1] / max(n_y - 1, 1),
            )
        )
        jitter = torch.clamp((spacing - min_spacing) * 0.45, min=0.0)

        positions = torch.empty((self._n_envs, n_particles, 3), device=gs.device)
        for env_idx in range(self._n_envs):
            cell_order = torch.randperm(cells.shape[0], device=gs.device)[:n_particles]
            particle_order = torch.randperm(n_particles, device=gs.device)
            xy = cells[cell_order]
            if bool(torch.any(jitter > 0).item()):
                xy = xy + (torch.rand((n_particles, 2), device=gs.device) * 2.0 - 1.0) * jitter
            positions[env_idx, particle_order, :2] = xy
            positions[env_idx, :, 2] = inner_min[2] + placement_half_extents[:, 2] + min_gap

        return positions

    def _random_particle_quats(self, particle, n_envs: int) -> torch.Tensor:
        if not hasattr(particle.morph, "size") and not hasattr(particle.morph, "height"):
            return torch.tensor((1.0, 0.0, 0.0, 0.0), device=gs.device).repeat(n_envs, 1)

        if hasattr(particle.morph, "height") and hasattr(particle.morph, "radius"):
            lying = torch.rand(n_envs, device=gs.device) < 0.5
            roll = torch.where(
                lying,
                torch.full((n_envs,), math.pi / 2, device=gs.device),
                torch.zeros(n_envs, device=gs.device),
            )
        else:
            roll = torch.zeros(n_envs, device=gs.device)
        pitch = torch.zeros(n_envs, device=gs.device)
        yaw = torch.rand(n_envs, device=gs.device) * math.tau

        cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
        cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
        cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
        return torch.stack(
            (
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ),
            dim=1,
        )

    def _get_particle_positions(self):
        return self._scene.rigid_solver.get_links_pos(links_idx=self._particle_links_idx)

    def _get_particle_quats(self):
        return self._scene.rigid_solver.get_links_quat(links_idx=self._particle_links_idx)

    def _render_all_envs(self) -> torch.Tensor:
        """
        Renders one RGB frame per env from each env's top-down camera.

        Genesis cameras are bound to a single env_idx at creation time and
        cam.render() is not batched across envs, so this issues one render
        call per env.

        Returns:
            uint8 tensor of shape [n_envs, H, W, 3]
        """
        frames = [cam.render()[0] for cam in self._cameras]
        return torch.from_numpy(np.stack(frames, axis=0)).to(torch.uint8)

    def update_material_state(self, store_other=False):
        """
        Returns particle state (positions and sizes) for all environments.
        Optimized for GPU processing.

        Returns:
            Tensor of shape [n_envs, n_particles, 4] with (x, y, z, size)
        """

        # Hold plate still
        self.plate.set_pos(self.plate.get_pos())
        self.plate.control_dofs_position_velocity(
            self.plate.get_pos(),
            self._zero_n_envsx3,
            dofs_idx_local=[0, 1, 2]
        )

        frozen_plate_dofs = self.plate.get_dofs_position()
        for _ in range(self._settle_steps):
            self.plate.set_dofs_position(frozen_plate_dofs)
            self._step_scene()

        self._particle_state[:, :, 0:3] = self._get_particle_positions()
        self._particle_state[:, :, 3:] = self._get_particle_quats()
    
    def plate_velocity_translation(
            self,
            p_start,
            p_end,
            angle,
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
        self._horizontal_dof_fix[:, -1] = angle 

        # Calculate velocity vector for each environment
        delta = p_end - p_start  # [n_envs, 3]
        dist = torch.linalg.norm(delta, axis=1, keepdim=True)  # [n_envs, 1]  
        direction = delta / (dist + 1e-8)
        v = direction * self._plate_params["speed"]  # [n_envs, 3]

        # Set initial position, velocity and goal for all plates in all environments
        self.plate.set_pos(p_start)
        self.plate.control_dofs_position_velocity(p_end, v, dofs_idx_local=[0, 1, 2])
        
        max_sweep_distance = float(dist.max().item())
        sweep_steps = max(1, math.ceil(max_sweep_distance / (self._plate_params["speed"] * self._scene.dt) * 1.7))
        
        reached_goal = torch.zeros(self._n_envs, dtype=torch.bool, device=gs.device)
        best_dist = torch.full((self._n_envs,), torch.inf, device=gs.device)
        frozen_pos = self.plate.get_pos()
        
        n_reached = 0
        for step in range(sweep_steps):
            if n_reached > 0:
                reached_envs_idx = reached_goal.nonzero().squeeze(dim=1)
                self.plate.set_pos(
                    frozen_pos[reached_goal],
                    envs_idx=reached_envs_idx,
                    zero_velocity=True,
                )
                # write only varying columns in-place; z/roll/pitch stay constant
                self._freeze_dofs_buf[reached_envs_idx, 0] = frozen_pos[reached_envs_idx, 0]
                self._freeze_dofs_buf[reached_envs_idx, 1] = frozen_pos[reached_envs_idx, 1]
                self._freeze_dofs_buf[reached_envs_idx, 5] = angle[reached_envs_idx]
                self.plate.set_dofs_position(
                    self._freeze_dofs_buf[reached_envs_idx],
                    dofs_idx_local=[0, 1, 2, 3, 4, 5],
                    envs_idx=reached_envs_idx,
                    zero_velocity=True,
                )

            self.plate.set_dofs_position(
                self._horizontal_dof_fix,
                dofs_idx_local=self._horizontal_dofs_local,
            )
            self._step_scene()

            cur_pos = self.plate.get_pos()
            cur_dist = torch.linalg.norm(cur_pos[:, :2] - p_end[:, :2], axis=1)
            improved = cur_dist < best_dist
            best_dist = torch.where(improved, cur_dist, best_dist)
            newly_reached = (cur_dist < self._goal_threshold) & ~reached_goal
            frozen_pos = torch.where(newly_reached[:, None], cur_pos, frozen_pos)
            reached_goal |= newly_reached
            
            n_reached = int(reached_goal.sum().item())
            if n_reached == self._n_envs:
                if self._debug:
                    print(f"All environments reached target at step {step + 1}")
                break

        final_pos = torch.where(reached_goal[:, None], frozen_pos, self.plate.get_pos())

        if self._debug:
            print(
                f" > Goal reached : {int(reached_goal.sum().item())}/{self._n_envs}; "
                f" > Best distance range {float(best_dist.min().item()):.4f}-"
                f"{float(best_dist.max().item()):.4f}m"
            )

        return reached_goal, final_pos
    
    def plate_position_translation(self, p_start, p_end):
        """
        Move plates with position control across all environments.
        
        Args:
            p_start: Starting positions [n_envs, 3] or [3]
            p_end: Ending positions [n_envs, 3] or [3]
            n_steps: Number of steps for interpolation
        """        
        path = (1 - self._steps_0to1[:, None, None]) * p_start[None, :, :] + self._steps_0to1[:, None, None] * p_end[None, :, :]
        
        self.plate.set_pos(p_start)
        for i in range(self._pos_ctrl_steps):
            self.plate.set_pos(pos=path[i])
            self.plate.set_dofs_position(
                position=self._vertical_dof_fix,
                dofs_idx_local=self._vertical_dofs_local
            )
            self._step_scene()

    def _sample_density_weighted_xy(
            self,
            particle_xy: torch.Tensor,
            n_samples: int,
            grid_res: int,
            density_uniform_mix: float,
        ) -> torch.Tensor:
        """
        particle_xy: (n_envs, n_particles, 2) current particle positions.
        Bins particles into a grid_res x grid_res grid over the box, adds
        density_uniform_mix as a per-cell pseudo-count (so empty cells stay
        reachable - e.g. mix=1.0 means a cell with k particles is k+1x as
        likely as a totally empty one, and an all-empty box samples
        uniformly), then draws a cell per (env, sample) proportional to that
        density and a uniform-random offset within it.

        Returns (n_envs * n_samples, 2) xy samples, flattened env-major to
        match generate_action_samples' n_total convention.
        """
        device = particle_xy.device
        vol_x, vol_y, _ = self._granular_vol
        cell_x, cell_y = vol_x / grid_res, vol_y / grid_res

        col = ((particle_xy[..., 0] + vol_x / 2) / cell_x).long().clamp(0, grid_res - 1)
        row = ((particle_xy[..., 1] + vol_y / 2) / cell_y).long().clamp(0, grid_res - 1)
        cell_idx = row * grid_res + col  # (n_envs, n_particles)

        counts = torch.zeros(self._n_envs, grid_res * grid_res, device=device)
        counts.scatter_add_(1, cell_idx, torch.ones_like(cell_idx, dtype=torch.float32))
        probs = (counts + density_uniform_mix)
        probs = probs / probs.sum(dim=1, keepdim=True)

        chosen = torch.multinomial(probs, n_samples, replacement=True)  # (n_envs, n_samples)
        chosen_row = torch.div(chosen, grid_res, rounding_mode="floor")
        chosen_col = chosen % grid_res

        jitter = torch.rand(self._n_envs, n_samples, 2, device=device)
        x = -vol_x / 2 + (chosen_col.float() + jitter[..., 0]) * cell_x
        y = -vol_y / 2 + (chosen_row.float() + jitter[..., 1]) * cell_y

        return torch.stack([x, y], dim=-1).reshape(self._n_envs * n_samples, 2)

    def generate_action_samples(
            self,
            n_samples: int,
            particle_xy: torch.Tensor | None = None,
            grid_res: int = 8,
            density_uniform_mix: float = 1.0,
        ):
        """
        Generate random action samples for all environments.

        particle_xy: optional (n_envs, n_particles, 2) current particle
            positions. When given, the push START position is sampled with
            probability proportional to local particle density (see
            _sample_density_weighted_xy) instead of uniformly at random - a
            push starting in empty space never contacts any particle, and
            empty space becomes increasingly common as a trajectory
            progresses and particles consolidate (measured on the old
            uniform sampler: ~1% of pushes moved no particle at step 0 of a
            20-step trajectory, vs ~50% by step 19). STOP position and angle
            are still sampled uniformly, so pushes can still redistribute
            material into empty regions rather than only ever shuffling
            already-dense cells. Pass the CURRENT particle state (updated
            after the previous push), not a snapshot from episode start -
            the whole point is tracking density as it evolves.
            When None, falls back to the old fully-uniform behavior.

        Returns:
            Tuple of (action_starts, action_stops, angles) each of shape [n_envs, n_samples, 3/1]
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
        if particle_xy is not None:
            start_samples = self._sample_density_weighted_xy(
                particle_xy, n_samples, grid_res, density_uniform_mix
            )
            # density grid cells can fall just outside the angle-dependent
            # safety margin near the box edge - clamp back into the same
            # valid range uniform sampling was already restricted to.
            start_samples = torch.max(torch.min(start_samples, high), low)
        else:
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
        self._vertical_dof_fix[:, 0] = p_start[:, 0]
        self._vertical_dof_fix[:, 1] = p_start[:, 1]
        self._vertical_dof_fix[:, 4] = angle
        self.plate_position_translation(
            p_start + self._lift_height_tensor,
            p_start,
        )
        
        # Sweeping
        reached_goal, final_pos = self.plate_velocity_translation(
            p_start,
            p_stop,
            angle,
        )

        # Lifting
        self._vertical_dof_fix[:, 0] = final_pos[:, 0]
        self._vertical_dof_fix[:, 1] = final_pos[:, 1]
        self.plate_position_translation(
            final_pos,
            final_pos + self._lift_height_tensor,
        )

        return reached_goal, final_pos

    def collect_data_samples(
            self,
            n_samples: int = 200,
            path : str | Path = "training",
        ):
        """
        Collect data samples from all environments efficiently.
        Optimized for GPU processing and memory efficiency.

        Args:
            n_samples: Number of samples to collect per environment
            path: Output path for data
        """
        max_samples = n_samples * self._n_envs

        self._config["data_collection"].update({
            "n_envs": self._n_envs,
            "samples_per_env": n_samples,
            "goal_threshold": self._goal_threshold,
        })

        # Allocate once or reuse if same size
        if (not hasattr(self, '_collection_buffers') or 
            self._collection_buffers['states'].shape[0] != n_samples or
            self._collection_buffers['states'].shape[1] != self._n_envs):
            self._allocate_collection_buffers(n_samples)
        
        # Clear data buffer
        for buf in self._collection_buffers.values():
            buf.zero_()
        
        self.update_material_state()
        for sample_idx in range(n_samples):
            print(f" > sample {sample_idx + 1}/{n_samples}")


            self._collection_buffers["states"][sample_idx].copy_(self._particle_state)
            if self._render_images:
                if sample_idx == 0:
                    self._collection_buffers["frames"][sample_idx] = self._render_all_envs()
                else:
                    # no reshuffle happens between samples, so this step's "before" frame
                    # is exactly the previous step's "after" frame - no need to re-render
                    self._collection_buffers["frames"][sample_idx] = self._collection_buffers["frames_"][sample_idx - 1]

            # Generate this step's action from the CURRENT particle state, not
            # a snapshot from before the episode started - sampling all
            # n_samples pushes upfront from the initial distribution meant
            # later pushes (once particles had already been consolidated by
            # earlier ones) increasingly swept through empty space.
            action_starts, action_stops, angles = self.generate_action_samples(
                1, particle_xy=self._particle_state[:, :, 0:2]
            )
            p_start = action_starts[:, 0, :]  # [n_envs, 3]
            p_stop = action_stops[:, 0, :]    # [n_envs, 3]
            angle = angles[:, 0]              # [n_envs]

            reached_goal, p_stop = self.execute_action(
                p_start,
                p_stop,
                angle,
            )

            self.update_material_state()

            self._collection_buffers["states_"][sample_idx].copy_(self._particle_state)
            self._collection_buffers["p_starts"][sample_idx] = p_start
            self._collection_buffers["p_stops"][sample_idx] = p_stop
            self._collection_buffers["sample_angles"][sample_idx] = angle
            self._collection_buffers["success_mask"][sample_idx] = reached_goal
            if self._render_images:
                self._collection_buffers["frames_"][sample_idx] = self._render_all_envs()
            if self._debug and torch.equal(self._collection_buffers["states"][sample_idx], self._collection_buffers["states_"][sample_idx]):
                print("State did not change")
            
        # Number of collected samples
        flat_success_mask = self._collection_buffers["success_mask"].reshape(max_samples)
        num_collected_samples = int(flat_success_mask.sum().item())

        # Print statistics
        print("\nStatistics (Multi-Environment Collection)")
        print("=" * 50)
        print(f">> Number of environments   : {self._n_envs}")
        print(f">> Samples per environment  : {n_samples}")
        print(f">> Total samples collected  : {num_collected_samples}")
        print(f">> Number of failed samples : {max_samples - num_collected_samples}")

        self._config["statistics"] = {
            "n_envs"   : self._n_envs,
            "samples_per_env"  : n_samples,
            "total_samples_collected"  : num_collected_samples,
            "number_of_failed_samples" : max_samples - num_collected_samples,
        }

        base_dir = Path(__file__).parent
        full_path = base_dir / path
        Path.mkdir(full_path, parents=True, exist_ok=True)

        # look for number of runs in existing dir: one config file is saved per run regardless
        # of how many other files accompany it, so count those rather than dividing the total
        # file count by a fixed per-run file count (which broke when _rollout.pt was added).
        n_runs = len(list(full_path.glob("_*_config.yaml")))

        self._save_config(full_path, n_runs)
        self._save_data(full_path, n_runs, flat_success_mask, max_samples)
        self._save_rollout(full_path, n_runs)
        self._log(f"Material batch finished. Run {n_runs} saved to {full_path}.")

    def destroy(self):
        gs.destroy()

    def view(self, horizon=1000):
        for _ in range(horizon):
            self._scene.visualizer.update()
    
    def simulate(self, horizon=1000):
        for _ in range(horizon):
            self._step_scene()
