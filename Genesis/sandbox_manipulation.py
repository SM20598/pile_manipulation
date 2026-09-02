import genesis as gs
import genesis.utils.geom as gu 
import numpy as np
import yaml
from utilities.materials import *
from pathlib import Path
import math
import torch

try:
    # The density a rigid link resolves to when its material leaves rho unset.
    # Particles are constructed without an explicit material (see
    # utilities/materials.py), so this - not the configured density - is what
    # their mass is actually built from. Imported rather than hardcoded so an
    # upstream change to the default is a visible break, not a silent 0.8x.
    from genesis.engine.entities.rigid_entity.rigid_link import RHO_OBJECT as _GENESIS_DEFAULT_RHO
except ImportError:  # pragma: no cover - private path, pinned in requirements
    _GENESIS_DEFAULT_RHO = 600.0

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

        # Configurable timing - read from simulation section so they can be
        # tuned without rebuilding the scene.
        # settle_steps is a CAP, not a fixed count: update_material_state stops
        # early once the pile is actually at rest.
        #
        # The win is the early exit, not the cap. Measured on the monolayer
        # tray (see utilities/materials.py::max_particle_height, which sizes
        # the box for exactly one layer), TRUE steps-to-rest is 34 after a
        # fresh spawn and 1 after a push, FLAT from n=50 to n=200 - particles
        # spawn ~1 mm above their resting height, so nothing has to collapse,
        # and after a push the pile has already relaxed during the lift.
        # Against a fixed 200 that is ~6x on a reset and ~20x per transition.
        #
        # The cap is therefore a safety net, not an operating point: 500 is
        # ~15x the worst observed. A denser or multi-layer spawn settles far
        # more slowly (~1460 steps was measured for a two-layer respawn at
        # n=200) and should raise this in its own config rather than inflating
        # the default here. Hitting the cap only warns, and silently records a
        # mid-motion state, which propagates because each transition's s is
        # the previous transition's s'.
        self._settle_steps   = int(self._sim_params.get('settle_steps',   500))
        self._settle_check_every = int(self._sim_params.get('settle_check_every', 10))
        self._settle_vel_threshold = float(
            self._sim_params.get('settle_velocity_threshold', 1e-3))     # m/s
        # Angular rest threshold, derived from the linear one unless set
        # explicitly. A bare rad/s number is not comparable to a m/s number:
        # 0.1 rad/s on a 5 mm cube is a corner speed of 0.35 mm/s, i.e. three
        # times STRICTER than the 1 mm/s linear threshold, so it silently
        # became the binding criterion and kept piles "unsettled" long after
        # their centres had stopped. Converting through the particle's lever
        # arm makes both express the same surface speed.
        _ang_thr = self._sim_params.get('settle_angular_velocity_threshold', None)
        if _ang_thr is None:
            _ps = self._material_params.get("particle_size") or 0.005
            _ps = float(_ps) if isinstance(_ps, (int, float)) else float(max(_ps))
            _lever = max(_ps * math.sqrt(3) / 2, 1e-4)     # half body diagonal
            _ang_thr = self._settle_vel_threshold / _lever
        self._settle_angvel_threshold = float(_ang_thr)  # rad/s
        # Fraction of particles that must be below threshold. A plain max makes
        # the test harder the more envs are batched (32 envs x 200 particles is
        # 6400 chances for one straggler), so the settle never converges and
        # always burns its cap. 0.995 tolerates ~1 straggler per 200-cube env.
        self._settle_rest_quantile = float(
            self._sim_params.get('settle_rest_quantile', 0.995))
        # A max-velocity guard on top of the quantile was tried and deliberately
        # NOT kept. The worry was that the quantile bounds how MANY particles
        # are still moving but not how fast, so s' could be recorded mid-travel.
        # Measured, the correlation runs the other way: the fastest residual
        # particles (73 mm/s at 50x128, 27 mm/s at 100x64) drifted 0.1-1.2 mm
        # over the next 0.2 s -- they vibrate in place -- while the one genuine
        # late movement (14 mm at 70x64) came from a particle moving only
        # 7 mm/s, a cube at the top of its tipping arc where speed passes
        # through a minimum. A max-velocity test would have paid for extra
        # settling on the harmless cases and still missed the real one.
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

        # Active particle count - may be reduced per-experiment via
        # set_n_active() to "park" excess particles outside the camera's field
        # of view. Particles are created in __init__, before scene.build(), so
        # changing n_particles otherwise costs a full rebuild (32-117 s).
        self._n_active = self._material_params["n_particles"]
        # Parking position: far from the box, above the ground plane.
        _bw = self._box_params["vol"][0]
        self._park_pos = [_bw * 15.0, 0.0, self._wall_thickness * 0.5 + 0.005]

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
        
        # used to create path for position control (lower/lift plate)
        self._pos_ctrl_steps = int(self._sim_params.get('pos_ctrl_steps', 100))
        self._steps_0to1 = torch.linspace(0, 1, self._pos_ctrl_steps, device=gs.device)

        # Clearance height for teleport-assisted lower/lift.
        # The plate is teleported to this height above the operating height
        # before simulating only the short final descent (and the first short
        # ascent before teleporting away). 2 x particle_size clears the top of
        # even a two-layer pile; 8 mm is the minimum to avoid spawning inside
        # a cube corner.
        _ps  = particle_size if isinstance(particle_size, float) else max(particle_size)
        _lift_h = self._box_params['vol'][2]          # full lift = box interior height
        self._clearance_height     = max(0.008, 2.0 * _ps)
        self._clearance_ctrl_steps = max(10, int(round(
            self._pos_ctrl_steps * self._clearance_height / _lift_h)))
        self._clearance_offset = torch.zeros((self._n_envs, 3), device=gs.device)
        self._clearance_offset[:, 2] = self._clearance_height

        # helpers to fix all dofs except z during lowering and lifting
        self._vertical_dofs_local = [0, 1, 3, 4, 5] 
        self._vertical_dof_fix = torch.zeros((self._n_envs, 5), device=gs.device)

        # helpers to fix all dofs except x, y during sweeping
        self._horizontal_dofs_local = [2, 3, 4, 5] 
        self._horizontal_dof_fix = torch.zeros((self._n_envs, 4), device=gs.device)
        self._horizontal_dof_fix[:, 0] = self._operation_height

        self._particle_state = torch.empty((self._n_envs, self._material_params["n_particles"], 7), device=gs.device)

        # ---- actuator model (see configs/basic.yaml's `plate:` section) ----
        # The tool is carried by a Cartesian gantry, so it is modelled as a
        # trajectory-tracking servo with the gantry's reflected inertia and a
        # finite force budget, rather than as a free 2.4 g box on a soft
        # spring. _plate_accel shapes the trapezoidal speed reference the servo
        # follows; the gains are derived from moving_mass and bandwidth in
        # build(), where the entity's dofs exist.
        self._plate_moving_mass = float(self._plate_params.get("moving_mass", 0.5))
        self._plate_accel       = float(self._plate_params.get("acceleration", 2.0))
        self._plate_bandwidth   = float(self._plate_params.get("control_bandwidth_hz", 15.0))
        self._plate_max_force   = float(self._plate_params.get("max_force", 30.0))
        # "pinned" overwrites the uncommanded dofs (z/roll/pitch/yaw) every
        # step, which is the original behaviour; "servo" holds them with stiff
        # PD instead. `servo` was built, measured and REJECTED - it fails
        # physics equivalence with 3-6x higher particle penetration - so
        # `pinned` remains the default.
        self._plate_hold_mode = str(self._plate_params.get("hold_mode", "pinned"))
        if self._plate_hold_mode not in ("pinned", "servo"):
            raise ValueError(f"plate.hold_mode must be 'pinned' or 'servo', "
                             f"got {self._plate_hold_mode!r}")
        self._plate_orientation_inertia = float(
            self._plate_params.get("orientation_inertia", 2.0e-4))
        self._plate_orientation_bandwidth = float(
            self._plate_params.get("orientation_bandwidth_hz", 30.0))
        self._plate_max_torque = float(self._plate_params.get("max_torque", 2.0))
        self._plate_arrival_steps = int(
            self._plate_params.get("arrival_steps", 12))
        # How the descent and lift are DRIVEN, independent of how the sweep
        # HOLDS its uncommanded dofs. These were originally one knob, which
        # meant the servo experiment changed two things at once and its failure
        # could not be attributed to either. "teleport" writes the pose each
        # step, so the blade arrives at the pile with zero momentum and each
        # step's penetration is resolved in one solve; "servo" drives it with
        # the actuator, so particles can actually resist it.
        self._plate_approach_mode = str(
            self._plate_params.get("approach_mode", "servo"))
        if self._plate_approach_mode not in ("teleport", "servo"):
            raise ValueError(f"plate.approach_mode must be 'teleport' or "
                             f"'servo', got {self._plate_approach_mode!r}")
        # Extra steps after the reference reaches the goal, letting the servo
        # close its remaining tracking error before the sweep is judged.
        self._sweep_settle_steps = int(self._sim_params.get("sweep_settle_steps", 12))

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

    def _default_max_collision_pairs(self) -> int:
        """Contact-pair budget to preallocate when the config doesn't set one.

        Genesis's own default is a flat 150, independent of how many bodies are
        in the scene. Measured occupancy for a settled-then-pushed pile of 50
        cubes of 5 mm is 51 broad-phase pairs and 211 contact points, i.e. a
        required ``max_collision_pairs`` of only **14** - the pile is mostly
        one floor contact per cube (4 points each under ``box_box_detection``)
        plus a few neighbours. So the flat default is *not* the bottleneck it
        looks like, and scaling it aggressively is actively harmful: the
        dominant GPU allocation is the constraint Jacobian, which is
        ``O(max_collision_pairs x contacts_per_pair x n_dofs x n_envs)``, so an
        oversized cap directly costs parallel environments. Raw step time, by
        contrast, is independent of it (measured flat across 150/800 at both
        settings of ``box_box_detection``).

        Measured requirement is close to ``0.26 * n_particles`` (13 at n=50, 52
        at n=200), so Genesis' flat 150 already carries ~2.8x headroom at 200
        particles and only needs to grow past roughly n=570. Hence the gentle
        ``n/2`` scaling below Genesis' floor. The difference is not academic:
        at n_particles=200 a cap of 200 tops out at 16 parallel envs on an 8 GB
        card, while 150 fits 32 - the cap alone doubles throughput.

        Under-estimating is caught loudly by ``_check_contact_budget`` rather
        than silently corrupting contacts, which is the failure mode that
        matters: on overflow the broadphase sets an error bit and stops adding
        pairs, and that bit never surfaces here (see ``_check_contact_budget``).
        """
        n_particles = int(self._material_params.get("n_particles") or 0)
        return max(150, n_particles // 2)

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

        # Exposed so the constraint solver can be swapped without editing this
        # file. Genesis defaults to Newton, which with use_contact_island builds
        # and factorizes a DENSE Hessian per contact island (island_dofs = 6 x
        # entities in the island); measured cost goes as island_size^2.64, and
        # that is the whole explanation for the cost cliff at 200 objects.
        # Set explicitly rather than left to default because the neighbouring
        # use_contact_island default FLIPPED upstream (False -> True in 1.2.0),
        # and an implicit value that changes under you is how a dataset
        # silently stops being comparable.
        _cs_name = rigid_cfg.get("constraint_solver", "Newton")
        try:
            _constraint_solver = getattr(gs.constraint_solver, _cs_name)
        except AttributeError as e:
            raise ValueError(
                f"rigid_options.constraint_solver={_cs_name!r} is not a Genesis "
                f"solver; expected one of "
                f"{[n for n in dir(gs.constraint_solver) if not n.startswith('_')]}"
            ) from e

        # Defaults this class insists on, then EVERY other key the config sets
        # is forwarded.
        #
        # Previously this was a fixed list of `rigid_cfg.get(...)` calls, so any
        # option not on the list was silently dropped: writing
        # `enable_torsional_friction: True` in basic.yaml would have done
        # exactly nothing, with no error and no warning. That is the same
        # failure mode `safety_margin` had (declared 0.005, hardcoded 0.02,
        # never read), and it is worth removing structurally rather than one
        # key at a time - a config that lies about what the simulation is doing
        # is far worse than one that complains.
        _rigid_kwargs = {
            "iterations": 50, "ls_iterations": 50,
            "tolerance": 1e-6, "ls_tolerance": 0.01,
            "box_box_detection": False, "use_contact_island": False,
            "use_hibernation": False, "enable_multi_contact": True,
            "max_collision_pairs": self._default_max_collision_pairs(),
        }
        _known = set(gs.options.RigidOptions.model_fields)
        _unknown = sorted(set(rigid_cfg) - _known - {"constraint_solver"})
        if _unknown:
            self._log(
                f"WARNING: rigid_options key(s) {_unknown} are not accepted by "
                f"this Genesis version's RigidOptions and will be IGNORED. "
                f"Either the key is misspelled or it was removed upstream "
                f"(e.g. hibernation_thresh_acc and prefer_parallel_linesearch "
                f"were removed in Genesis 1.2.x)."
            )
        _rigid_kwargs.update(
            {k: v for k, v in rigid_cfg.items()
             if k != "constraint_solver" and k in _known})
        _rigid_kwargs["constraint_solver"] = _constraint_solver

        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                # Fallbacks were 4e3 (4000 seconds) and 1. Latent, since
                # basic.yaml sets both, but a landmine for any config that
                # does not.
                dt       = self._config["simulation"].get('dt', 4e-3),
                substeps = self._config["simulation"].get('substeps', 5),
            ),
            rigid_options=gs.options.RigidOptions(**_rigid_kwargs),
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
        #
        # friction must be set explicitly: Genesis defaults an unset geom
        # friction to 1.0 and combines a contact as max(mu_a, mu_b), so
        # leaving it None pins *every* plate-particle contact at 1.0 and makes
        # the sampled particle friction have no effect whatsoever at the tool
        # interface - the one interface the action actually acts through.
        self.plate = self._scene.add_entity(
            material=gs.materials.Rigid(
                rho=3000,
                friction=float(self._plate_params.get("friction", 0.3)),
            ),
            morph=gs.morphs.Box(
                pos=(0, 0, height * 2),
                size=self._plate_params["size"]
            ),
            surface=gs.surfaces.Default(color=[0, 1, 0])
        )
        
        # add granular
        #
        # Extra clearance kept between the tool's footprint and the tray wall
        # when drawing a touchdown pose, on top of the tool's own extent at
        # that yaw. basic.yaml declared `safety_margin: 0.005` while this line
        # hardcoded 0.02 and never read the key, so tuning it did nothing; the
        # config now states the value that was actually in force.
        self._safety_margin = float(self._config.get("safety_margin", 0.02))

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
        self._configure_plate_actuator()

        self._cache_particle_idx()

    def _configure_plate_actuator(self) -> None:
        """Model the pusher as a gantry axis rather than a free light box.

        Three things, all on the translational dofs (rotation is hard-set every
        step by the sweep/descent loops under the default `pinned` hold mode,
        so its gains are irrelevant there):

        armature
            The plate geometry weighs ~2.4 g, which is far lighter than the
            carriage that actually carries it, so granular reaction would move
            it much more than on the real machine. ``set_dofs_armature`` adds
            the drivetrain's reflected inertia to the mass-matrix diagonal -
            the same matrix the constraint solver uses - so contacts see a
            heavy axis while momentum exchange stays exact. This is the right
            knob rather than a denser plate, which would also change the tool's
            weight and its contact response.

        gains
            Chosen from the modelled mass and a target closed-loop bandwidth:
            kp = m*w^2, kv = 2*z*m*w at z = 1 (critically damped). The default
            15 Hz gives w ~ 94 rad/s against a 0.8 ms substep (w*h ~ 0.075),
            comfortably stable, and a disturbance stiffness of kp ~ 4.4e3 N/m -
            a couple of newtons of granular reaction displaces the tool well
            under half a millimetre.

        force range
            Previously unbounded. With stiff gains a particle wedged against a
            wall would draw an arbitrarily large force; a real stepper loses
            steps instead. A finite budget makes a jam degrade gracefully.
        """
        translational = [0, 1, 2]
        m = self._plate_moving_mass
        omega = 2.0 * math.pi * self._plate_bandwidth
        kp, kv = m * omega ** 2, 2.0 * m * omega

        self.plate.set_dofs_armature((m,) * 3, translational)
        self.plate.set_dofs_kp((kp,) * 3, translational)
        self.plate.set_dofs_kv((kv,) * 3, translational)
        self.plate.set_dofs_force_range(
            (-self._plate_max_force,) * 3, (self._plate_max_force,) * 3,
            translational)

        if self._plate_hold_mode == "servo":
            # Orientation gets its own axis model rather than being overwritten.
            # Inertia, not mass: the reflected inertia of the rotary drive, from
            # which gains follow exactly as for the linear axes. Deliberately
            # stiffer -- this axis holds a setpoint, it does not track a profile.
            rotational = [3, 4, 5]
            inertia = self._plate_orientation_inertia
            omega_r = 2.0 * math.pi * self._plate_orientation_bandwidth
            self.plate.set_dofs_armature((inertia,) * 3, rotational)
            self.plate.set_dofs_kp((inertia * omega_r ** 2,) * 3, rotational)
            self.plate.set_dofs_kv((2.0 * inertia * omega_r,) * 3, rotational)
            self.plate.set_dofs_force_range(
                (-self._plate_max_torque,) * 3, (self._plate_max_torque,) * 3,
                rotational)

    def contact_budget_usage(self) -> dict:
        """Peak collider occupancy across envs, against its two real limits.

        Genesis bounds collision work in two independent places, and
        ``max_collision_pairs`` (``mcp``) sets both:

        * broad-phase candidate pairs, capped at ``mcp * 8``
          (``multiplier_collision_broad_phase``)
        * narrow-phase contact *points*, capped by ``max_contacts``

        The pair count and the point count differ by a large factor, so they
        must be compared against their own caps - a settled pile of cubes
        produces roughly four contact points per floor contact.

        ``max_contacts`` is read rather than recomputed as
        ``mcp * n_contacts_per_pair``: from Genesis 1.2.x the buffer is sized
        per regime (convex vs nonconvex pairs have different per-pair caps) and
        then reduced again by link-pair contact pruning, so the old product
        would overstate the cap and hide a real overflow - the one thing this
        check exists to catch.
        """
        collider = self._scene.rigid_solver.collider
        state, info = collider._collider_state, collider._collider_info
        return {
            "broad_pairs": int(torch.as_tensor(state.n_broad_pairs.to_torch()).max()),
            "broad_cap": int(torch.as_tensor(info.max_collision_pairs_broad.to_torch()).max()),
            "contact_points": int(torch.as_tensor(state.n_contacts.to_torch()).max()),
            "contact_cap": int(torch.as_tensor(info.max_contacts.to_torch()).max()),
            "max_collision_pairs": int(torch.as_tensor(info.max_collision_pairs.to_torch()).max()),
        }

    def _check_contact_budget(self) -> None:
        """Warn (once) if the pile is close to exhausting the contact budget.

        Genesis reports contact-pair overflow by setting an error bit that
        ``Simulator.step`` inspects periodically. That mechanism cannot fire
        here: ``RigidSolver.set_dofs_position`` clears the error bit as a side
        effect, and the sweep loop calls it on every step under the default
        `pinned` hold mode, so the bit is always wiped before the next check
        reads it. The failure would therefore be completely silent - contacts
        dropped, wrong physics recorded, no exception - which is exactly the
        kind of thing that must not go unnoticed in collected training data.
        So check the counter directly instead of relying on Genesis to
        complain.
        """
        if getattr(self, "_contact_budget_warned", False):
            return
        try:
            usage = self.contact_budget_usage()
        except Exception:
            self._contact_budget_warned = True   # counters unavailable; don't retry
            return
        self._contact_budget_peak = {
            k: max(v, self._contact_budget_peak.get(k, 0))
            for k, v in usage.items()
        } if getattr(self, "_contact_budget_peak", None) else dict(usage)
        for what, used, cap in (
            ("broad-phase candidate pairs", usage["broad_pairs"], usage["broad_cap"]),
            ("contact points", usage["contact_points"], usage["contact_cap"]),
        ):
            if used >= 0.9 * cap:
                self._contact_budget_warned = True
                self._log(
                    f"WARNING: {used}/{cap} {what} in use. Past the cap Genesis "
                    f"stops adding contacts and only flags it via an error bit "
                    f"that this class's per-step set_dofs_position clears before "
                    f"it can be read - so an overflow here is silent, and the "
                    f"recorded state would come from incomplete contact physics. "
                    f"Raise rigid_options.max_collision_pairs."
                )

    def escaped_particle_count(self) -> int:
        """Particles outside the tray interior, summed over all envs.

        A particle can only leave by being squeezed through a wall, which means
        the contact solver failed for it. It matters more than it sounds: each
        transition's ``s`` is the previous transition's ``s'``, so one escape
        silently corrupts every later sample in that env, and nothing else in
        the pipeline would notice. Recorded per batch alongside the collected
        data so a finished dataset can be audited without re-running it.

        The tolerance is deliberately loose (5 mm laterally, 20 mm above the
        wall) - this is looking for particles that have plainly left, not for
        ones resting slightly proud of the rim.
        """
        n_active = min(getattr(self, "_n_active", len(self.material)), len(self.material))
        pos = self._get_particle_positions()[:, :n_active]
        width, depth, height = self._box_params["vol"]
        half = torch.tensor([width / 2, depth / 2], device=pos.device)
        out_xy = (pos[..., :2].abs() > half + 0.005).any(dim=-1)
        out_z = (pos[..., 2] < -0.005) | (pos[..., 2] > height + 0.02)
        return int((out_xy | out_z).sum())

    def _reaction_reset(self) -> None:
        """Start a fresh reaction record for one action.

        Everything is accumulated as a running max in GPU tensors and read back
        exactly once, by ``reaction_report()``. That matters: the sweep loop
        deliberately has no per-step GPU sync (removing the old
        ``.nonzero()``/``.item()`` calls was part of what made it cheap, see
        docs/scaling_to_200_objects.md section 1.2), and a per-step ``.item()``
        here would put one straight back.

        Split by phase, because the phases are not comparable. The descent and
        lift drive the plate by teleport (``set_pos``) while its PD servo still
        holds an older target, so the servo commands full force against its own
        motion: measured, that pins ``force_N`` to the 30 N limit for ~13 % of
        steps while the actual granular reaction is 0.004 N. Reporting one
        number over all phases would therefore describe a control artifact and
        look like a machine at its structural limit. **The sweep figures are the
        physically meaningful ones.**
        """
        z = lambda: torch.zeros(self._n_envs, device=gs.device)
        def _phase():
            return {
                "force_N": z(),      # |actuator force| on x/y/z, per env
                "torque_Nm": z(),    # |actuator torque| on roll/pitch/yaw
                "contact_N": z(),    # net granular reaction on the blade
                "track_mm": z(),     # deviation from the commanded path
                "tilt_deg": z(),     # blade tilt away from vertical
                "saturated": z(),    # steps at the force limit
                "steps": 0,
            }
        self._reaction = {ph: _phase() for ph in ("lower", "sweep", "lift")}


    def _reaction_update(self, p_ref=None, phase="sweep") -> None:
        """Fold one step into the running maxima for ``phase``. No host sync."""
        all_r = getattr(self, "_reaction", None)
        if all_r is None or phase not in all_r:
            return
        r = all_r[phase]
        f = self.plate.get_dofs_control_force()          # [n_envs, 6]
        lin, rot = f[:, :3].abs().amax(dim=1), f[:, 3:].abs().amax(dim=1)
        r["force_N"] = torch.maximum(r["force_N"], lin)
        r["torque_Nm"] = torch.maximum(r["torque_Nm"], rot)
        # A real stepper loses steps rather than pushing harder, so time spent
        # against the limit is the "would the machine have coped" signal.
        r["saturated"] += (lin >= 0.99 * self._plate_max_force).to(lin.dtype)

        try:
            c = self.plate.get_links_net_contact_force().reshape(self._n_envs, -1, 3)
            r["contact_N"] = torch.maximum(r["contact_N"], c.norm(dim=-1).amax(dim=1))
        except Exception:
            pass

        if p_ref is not None:
            err = (self.plate.get_pos()[:, :2] - p_ref[:, :2]).norm(dim=-1) * 1000.0
            r["track_mm"] = torch.maximum(r["track_mm"], err)

        # Tilt: angle between the blade's own z axis and world z. Roll and pitch
        # are held by a per-step write today, so this is ~0 now; it exists
        # because hold_mode="servo" replaces that write with a finite-stiffness
        # servo, where the blade can in principle tilt. It stays measured because
        # it is cheap and it is the number that would catch it.
        # See docs/plate_model.md section 4.
        q = self.plate.get_quat()
        w, x, y, zq = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        cos_tilt = (1.0 - 2.0 * (x * x + y * y)).clamp(-1.0, 1.0)
        r["tilt_deg"] = torch.maximum(r["tilt_deg"], cos_tilt.arccos() * 180.0 / math.pi)
        r["steps"] += 1


    def reaction_report(self) -> dict:
        """Peak reaction loads on the tool during the last action.

        One host sync, at the end of the action. Intended for setting real-robot
        limits: ``force_N`` and ``torque_Nm`` are what the actuators had to
        supply, ``contact_N`` is the granular reaction they were fighting, and
        ``track_mm`` / ``tilt_deg`` are how far the tool was pushed off its
        commanded pose regardless. ``saturated_frac`` is the fraction of steps
        spent against ``plate.max_force`` — above zero means a real machine
        would have been at its limit and, with steppers, losing position.
        """
        all_r = getattr(self, "_reaction", None)
        if not all_r:
            return {}
        out = {"force_limit_N": float(self._plate_max_force)}
        for phase, r in all_r.items():
            if not r["steps"]:
                continue
            out[phase] = {
                "force_N": float(r["force_N"].max()),
                "torque_Nm": float(r["torque_Nm"].max()),
                "contact_N": float(r["contact_N"].max()),
                "track_mm": float(r["track_mm"].max()),
                "tilt_deg": float(r["tilt_deg"].max()),
                "saturated_frac": float(r["saturated"].max()) / r["steps"],
                "steps": int(r["steps"]),
            }
            if phase == "sweep":
                out[phase]["per_env_force_N"] = r["force_N"].tolist()
                out[phase]["per_env_track_mm"] = r["track_mm"].tolist()
        return out

    def _trapezoid_profile(self, dist: torch.Tensor) -> dict:
        """Pre-compute a trapezoidal speed profile per env for a given travel.

        Matches how the real gantry moves: accelerate at ``plate.acceleration``
        to ``plate.speed``, cruise, then decelerate to rest exactly at the
        target. Short moves that never reach cruise speed degenerate to a
        triangular profile with peak sqrt(a*d), handled by the same expression.
        """
        v_max = float(self._plate_params["speed"])
        a = self._plate_accel
        v_peak = torch.clamp(torch.sqrt(a * dist.clamp(min=0.0)), max=v_max)
        t_acc = v_peak / a
        d_acc = 0.5 * a * t_acc ** 2
        d_flat = torch.clamp(dist - 2.0 * d_acc, min=0.0)
        t_flat = d_flat / v_peak.clamp(min=1e-9)
        return {
            "dist": dist, "a": a, "v_peak": v_peak,
            "t_acc": t_acc, "d_acc": d_acc, "t_flat": t_flat, "d_flat": d_flat,
            "duration": 2.0 * t_acc + t_flat,
        }

    def _trapezoid_at(self, prof: dict, t: float):
        """Distance travelled and speed at time ``t``, per env."""
        a, v_peak = prof["a"], prof["v_peak"]
        t_acc, t_flat, d_acc, d_flat = (
            prof["t_acc"], prof["t_flat"], prof["d_acc"], prof["d_flat"])
        t_cruise_end = t_acc + t_flat
        tc = torch.clamp(torch.full_like(v_peak, float(t)),
                         max=prof["duration"])

        t_dec = torch.clamp(tc - t_cruise_end, min=0.0)
        s_acc = 0.5 * a * torch.minimum(tc, t_acc) ** 2
        s_flat = v_peak * torch.minimum(torch.clamp(tc - t_acc, min=0.0), t_flat)
        s_dec = v_peak * t_dec - 0.5 * a * t_dec ** 2

        s = s_acc + s_flat + s_dec
        v = torch.where(
            tc <= t_acc, a * tc,
            torch.where(tc <= t_cruise_end, v_peak, v_peak - a * t_dec),
        )
        return s, torch.clamp(v, min=0.0)

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
        # Rescale mass from whatever density the particle's mass currently
        # reflects. On the first call material.rho is still None - meaning the
        # built mass came from Genesis' default rho, not from any configured
        # value - so fall back to that default rather than skipping the update.
        # Skipping it (the previous behaviour) left every particle at the
        # default density while the saved config recorded the sampled one, and
        # every subsequent batch then rescaled from the wrong base, leaving all
        # masses at 600/750 = 0.8x their recorded density.
        old_density = getattr(particle.material, "rho", None) or _GENESIS_DEFAULT_RHO
        particle.material.rho = float(density)
        if getattr(self._scene, "is_built", False) and old_density > 0:
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
        """positions, quats: (n_envs, n_particles, 3/4). Teleports + zeros velocity.

        Every particle's pose is written in two batched solver calls instead of
        2N entity calls. ``RigidEntity.set_pos``/``set_quat`` each run a
        forward-kinematics pass over the *whole scene*
        (``skip_forward=False`` by default), so the obvious per-particle loop
        costs 2N kernel launches and 2N full-scene FK passes - 400 of each at
        n_particles=200, on every reset and every state restore (the latter
        being in the oracle-MPC hot path). The solver-level setters take a
        link-index array, so the same work is two launches and a single FK
        pass at the end.
        """
        envs_idx = torch.arange(self._n_envs, device=gs.device)
        solver = self._scene.rigid_solver
        solver.set_base_links_pos(positions.contiguous(),
                                  links_idx=self._particle_links_idx,
                                  envs_idx=envs_idx, skip_forward=True)
        solver.set_base_links_quat(quats.contiguous(),
                                   links_idx=self._particle_links_idx,
                                   envs_idx=envs_idx, skip_forward=False)
        if self._particle_dofs_idx.numel() > 0:
            self._scene.rigid_solver.set_dofs_velocity(
                torch.zeros((self._n_envs, self._particle_dofs_idx.numel()), device=gs.device),
                dofs_idx=self._particle_dofs_idx,
                skip_forward=True,
            )

    def set_n_active(self, n: int) -> None:
        """Set how many particles are active (placed inside the box) on reset.

        Particles with indices ``[n, len(material))`` are moved to a parking
        position outside the box on the next call to ``shuffle_particles()``.
        The change takes effect on the next reset.

        This exists because particles are created in ``__init__``, before
        ``scene.build()``, so ``n_particles`` is otherwise a rebuild-only
        parameter - and with ``performance_mode`` on, every distinct scene
        shape pays a full kernel recompile (32-117 s measured). Parking lets
        one built scene serve a whole sweep over particle counts.
        """
        n_total = len(self.material)
        if not (0 <= n <= n_total):
            raise ValueError(f"n must be in [0, {n_total}], got {n}")
        self._n_active = n

    def shuffle_particles(self):
        n_particles = len(self.material)
        if n_particles == 0:
            return

        max_retries = 10
        for attempt in range(max_retries):
            try:
                half_extents, placement_half_extents, collision_half_extents = self._particle_shape_extents()
                inner_min, inner_max = self._box_inner_bounds()
                # Only the ACTIVE prefix is placed inside the box. Slicing the
                # extents rather than placing all n and overwriting afterwards
                # means the rejection sampler is solving the easier problem it
                # actually faces, which is the point of set_n_active: it lets a
                # sweep over particle counts reuse one built scene instead of
                # paying a 32-117 s rebuild per count.
                n_active = min(getattr(self, "_n_active", n_particles), n_particles)
                positions = self._sample_nonoverlapping_particle_positions(
                    half_extents=half_extents[:n_active],
                    placement_half_extents=placement_half_extents[:n_active],
                    collision_half_extents=collision_half_extents[:n_active],
                    inner_min=inner_min,
                    inner_max=inner_max,
                )

                if n_active < n_particles:
                    # Park the inactive particles outside the box, spread over a
                    # grid rather than heaped on one point: parking them all at
                    # an identical position piles every inactive particle into a
                    # single permanent contact cluster, which consumes the
                    # contact budget (see _default_max_collision_pairs) and
                    # costs solver time on every step of every env, for
                    # particles that are not even part of the experiment. With
                    # Newton's dense per-island Hessian that cluster is also
                    # quadratic-to-cubic in its own size.
                    n_parked = n_particles - n_active
                    park = torch.tensor(self._park_pos, dtype=torch.float32, device=gs.device)
                    pitch = float(2.0 * placement_half_extents[:, :2].max().item()) + 5e-3
                    cols = int(math.ceil(math.sqrt(n_parked)))
                    idx = torch.arange(n_parked, device=gs.device)
                    offsets = torch.zeros((n_parked, 3), device=gs.device)
                    offsets[:, 0] = (idx % cols).to(torch.float32) * pitch
                    offsets[:, 1] = torch.div(idx, cols, rounding_mode="floor").to(torch.float32) * pitch
                    parked = (park.view(1, 3) + offsets).unsqueeze(0).expand(
                        self._n_envs, n_parked, 3)
                    positions = torch.cat([positions, parked], dim=1)

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
        # Keep the cached state in step with what was just written. Restoring an
        # already-settled state is the one path that deliberately SKIPS
        # update_material_state (that is the entire point of the state library),
        # so without this refresh _particle_state would still hold the previous
        # pile and the first recorded s of a batch would be the last pile's s'.
        self._particle_state[:, :, 0:3] = state[..., 0:3]
        self._particle_state[:, :, 3:7] = state[..., 3:7]

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

    def update_material_state(self, store_other=False, on_step=None):
        """
        Returns particle state (positions and sizes) for all environments.
        Optimized for GPU processing.

        Args:
            on_step: optional callback(step), invoked after each settle step.
                Same hook convention as ``execute_action``'s ``on_phase``. The
                settle is the one phase with no other window into it — it exits
                on a convergence test, so its duration is not known in advance
                and nothing outside this loop can observe the pile collapsing.
                Used by ``tests/scaling_investigation/record_simulation_video.py``
                to render the layered spawn, which is the least physically
                natural moment in the pipeline and therefore the one most worth
                watching. A no-op when None.

        Returns:
            Tensor of shape [n_envs, n_particles, 4] with (x, y, z, size)
        """

        # Hold the plate still for the duration of the settle.
        #
        # The plate is lifted clear of the pile here, so the only force acting
        # on it is its own 2.4 g weight — 0.0235 N against a translational
        # kp of 4441 N/m, i.e. a steady-state sag of 5.3 um, about 0.1% of a
        # 5 mm particle. The PD alone therefore pins it, and a per-step
        # set_dofs_position is not just redundant but actively harmful:
        # RigidSolver.set_dofs_position calls collider.reset() AND
        # constraint_solver.reset() — discarding the constraint solver's warm
        # start every step, with only 10 iterations to rebuild it — runs a
        # whole-scene forward-kinematics pass, and clears _errno, which is why
        # a contact-budget overflow could never surface. Setting the control
        # target once is enough: ctrl_pos persists (it is only cleared by
        # control_dofs_velocity, a mode switch) and the actuator reads it every
        # substep.
        frozen_plate_dofs = self.plate.get_dofs_position()
        self.plate.zero_all_dofs_velocity()
        self.plate.control_dofs_position_velocity(
            frozen_plate_dofs,
            torch.zeros_like(frozen_plate_dofs),
            dofs_idx_local=[0, 1, 2, 3, 4, 5],
        )

        settled_at = None
        for step in range(self._settle_steps):
            self._step_scene()
            if on_step is not None:
                on_step(step)
            if (step + 1) % self._settle_check_every == 0 and self._pile_is_at_rest():
                settled_at = step + 1
                break

        if settled_at is None and not getattr(self, "_settle_cap_warned", False):
            lin_max, ang_max = self._pile_motion()
            lin_q, ang_q = self._pile_motion(quantile=self._settle_rest_quantile)
            self._settle_cap_warned = True
            self._log(
                f"WARNING: pile still moving after the full {self._settle_steps}-step "
                f"settle. At the q={self._settle_rest_quantile} rest quantile: "
                f"{lin_q*1000:.2f} mm/s linear, {ang_q:.2f} rad/s angular (thresholds "
                f"{self._settle_vel_threshold*1000:.1f} / "
                f"{self._settle_angvel_threshold:.1f}); worst single particle "
                f"{lin_max*1000:.1f} mm/s, {ang_max:.1f} rad/s. The recorded state is "
                f"mid-motion, and because each transition's s comes from the previous "
                f"s', that error propagates. Raise simulation.settle_steps, or relax "
                f"simulation.settle_rest_quantile if the tail is a few stragglers."
            )
        elif self._debug and settled_at is not None:
            self._log(f"settled after {settled_at}/{self._settle_steps} steps")

        self._check_contact_budget()

        self._particle_state[:, :, 0:3] = self._get_particle_positions()
        self._particle_state[:, :, 3:] = self._get_particle_quats()


    def _pile_motion(self, quantile: float | None = None) -> tuple[float, float]:
        """(linear m/s, angular rad/s) particle speed, peak or at a quantile.

        Linear and angular are kept separate rather than reduced to one number:
        a free joint's dofs are [x, y, z, roll, pitch, yaw], so both live in the
        same tensor but carry different units, and a single max over all six
        conflates metres per second with radians per second — which reads as an
        alarming velocity when it is really a mildly spinning cube.

        ``quantile=None`` gives the peak. A quantile is what the rest test
        actually wants: the peak is taken over *every particle in every env*, so
        its strictness scales with n_envs. At 32 envs x 200 particles a single
        straggler anywhere holds up all 6400, and the settle then always runs to
        its cap — which is exactly what happened before this was quantile-based.
        """
        if self._particle_dofs_idx.numel() == 0:
            return 0.0, 0.0
        vel = self._scene.rigid_solver.get_dofs_velocity(
            dofs_idx=self._particle_dofs_idx).reshape(self._n_envs, -1, 6)
        n_active = getattr(self, "_n_active", vel.shape[1])
        vel = vel[:, :n_active]
        lin = vel[..., :3].norm(dim=-1).flatten()
        ang = vel[..., 3:].norm(dim=-1).flatten()
        if quantile is None:
            return float(lin.max()), float(ang.max())
        q = torch.tensor(quantile, device=lin.device, dtype=lin.dtype)
        return float(torch.quantile(lin, q)), float(torch.quantile(ang, q))


    def _pile_is_at_rest(self) -> bool:
        lin, ang = self._pile_motion(quantile=self._settle_rest_quantile)
        return (lin < self._settle_vel_threshold
                and ang < self._settle_angvel_threshold)

    def plate_velocity_translation(
            self,
            p_start,
            p_end,
            angle,
            debug=False,
            on_step=None,
        ):
        """
        Move plates along a trapezoidal speed profile across all environments.

        Args:
            p_start: Starting positions [n_envs, 3] or [3]
            p_end: Ending positions [n_envs, 3] or [3]
            angle: Rotation angle (scalar)
            on_step: optional callback(step, p_ref, v_ref), invoked after each
                simulation step of the sweep with the reference the servo was
                tracking at that step. A no-op when None. Follows the same
                hook convention as ``execute_action``'s ``on_phase`` (see
                docs/UTILITIES.md) — it exists so diagnostics such as
                ``tests/scaling_investigation/probe_plate_dynamics.py`` can measure the tool's
                realized trajectory against its reference without duplicating
                this control law, which is exactly the kind of drift that
                makes a probe silently stop testing the thing it names.
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

        delta = p_end - p_start                                   # [n_envs, 3]
        dist = torch.linalg.norm(delta, axis=1)                   # [n_envs]
        direction = delta / (dist.unsqueeze(1) + 1e-8)

        prof = self._trapezoid_profile(dist)
        dt = self._scene.dt
        sweep_steps = max(
            1, math.ceil(float(prof["duration"].max().item()) / dt)
        ) + self._sweep_settle_steps

        if self._plate_hold_mode != "servo":
            # In servo mode the descent has already brought the blade here under
            # its own actuator; teleporting would undo that and, if the descent
            # had not converged, insert the blade into the pile in one step.
            self.plate.set_pos(p_start)

        for step in range(sweep_steps):
            # Feed the servo a *moving* reference: where the tool should be and
            # how fast it should be going right now. Commanding the endpoint
            # instead (the previous behaviour) turns the same PD into a
            # position servo whose speed is proportional to distance remaining
            # -- it settles at v = v_cruise + kp*remaining/kv, so it overshoots
            # the commanded speed early in a sweep and undershoots near the
            # goal, and never actually travels at plate.speed.
            s, v_mag = self._trapezoid_at(prof, (step + 1) * dt)
            p_ref = p_start + direction * s.unsqueeze(1)
            v_ref = direction * v_mag.unsqueeze(1)
            self.plate.control_dofs_position_velocity(
                p_ref, v_ref, dofs_idx_local=[0, 1, 2])

            if self._plate_hold_mode == "servo":
                # Hold z and orientation with their own servos. A control target
                # is a request, not a state write: it does not reset the
                # collider or the constraint warm start, does not clear _errno,
                # and does not break hibernation -- unlike set_dofs_position,
                # which does all three (docs/plate_model.md section 5).
                # Set once per sweep, not per step: ctrl targets persist.
                # Orientation only: z is dof 2, already carried by the
                # trapezoid target on [0,1,2] above, so commanding it again here
                # would set two targets for one axis.
                # _horizontal_dof_fix columns are [z, roll, pitch, yaw].
                if step == 0:
                    self.plate.control_dofs_position(
                        self._horizontal_dof_fix[:, 1:], dofs_idx_local=[3, 4, 5])
            else:
                # zero_velocity=False: this call constrains z/roll/pitch/yaw only,
                # but RigidEntity.set_dofs_position defaults zero_velocity=True and
                # zeroes *all six* dofs regardless of dofs_idx_local. Leaving the
                # default on reset the plate's x/y velocity every single step, so
                # the sweep restarted from rest at 250 Hz and the tool carried no
                # momentum into the pile.
                self.plate.set_dofs_position(
                    self._horizontal_dof_fix,
                    dofs_idx_local=self._horizontal_dofs_local,
                    zero_velocity=False,
                )
            self._step_scene()
            self._reaction_update(p_ref, "sweep")
            if on_step is not None:
                on_step(step, p_ref, v_ref)

        # No per-step goal test: the reference itself ends at p_end and holds
        # there, so envs that finish early simply stop, with no freeze
        # bookkeeping. That also removes the two GPU syncs the old loop paid on
        # every step (a .nonzero() and a .item()), which dominated the sweep's
        # per-step cost at small n_envs.
        # Check the contact budget HERE, not only after settling: the pile is
        # most compressed at the end of a sweep, so this is where usage peaks.
        # It also cannot be left to Genesis' own error bit, because the loop
        # above calls set_dofs_position every step and that clears _errno. An
        # unnoticed overflow does not degrade gracefully — with the point cap
        # exceeded it has been observed to corrupt memory outright (CUDA
        # illegal memory access).
        self._check_contact_budget()

        final_pos = self.plate.get_pos()
        final_err = torch.linalg.norm(final_pos[:, :2] - p_end[:, :2], axis=1)
        reached_goal = final_err < self._goal_threshold

        if self._debug:
            print(
                f" > Goal reached : {int(reached_goal.sum().item())}/{self._n_envs}; "
                f" > Final tracking error {float(final_err.min().item()):.4f}-"
                f"{float(final_err.max().item()):.4f}m over {sweep_steps} steps"
            )

        return reached_goal, final_pos

    def plate_position_translation(self, p_start, p_end, n_steps: int | None = None,
                                   on_step=None, phase="lower"):
        """
        Move plates with position control across all environments.

        Args:
            p_start: Starting positions [n_envs, 3] or [3]
            p_end: Ending positions [n_envs, 3] or [3]
            n_steps: Override step count (defaults to self._pos_ctrl_steps)
            on_step: optional callback(step), invoked after each step. This is
                the descent and the lift — the phases where the tool is moved
                by teleport-then-interpolate rather than by the servo, so if it
                ever passes through a particle it happens here. A no-op when
                None.
        """
        n = n_steps if n_steps is not None else self._pos_ctrl_steps
        steps_0to1 = (self._steps_0to1 if n_steps is None
                      else torch.linspace(0, 1, n, device=gs.device))
        path = (1 - steps_0to1[:, None, None]) * p_start[None, :, :] + steps_0to1[:, None, None] * p_end[None, :, :]

        if self._plate_approach_mode == "servo":
            # Drive the move with the servo instead of teleporting. The old path
            # wrote the pose with set_pos every step while the PD servo still
            # held an older target, so the servo fought its own motion: measured,
            # that pinned the actuator at its 30 N limit for 100 % of descent
            # steps while the real granular reaction was 0.02 N. Feeding the
            # interpolated path as a TARGET removes both the fight and the
            # per-step state write.
            # _vertical_dofs_local is [0,1,3,4,5] -- x, y, roll, pitch, yaw --
            # so _vertical_dof_fix columns are [x, y, roll, pitch, yaw] and the
            # orientation part is columns 2: onward. x/y/z all come from the
            # interpolated path below.
            self.plate.control_dofs_position(
                self._vertical_dof_fix[:, 2:], dofs_idx_local=[3, 4, 5])
        else:
            self.plate.set_pos(p_start)
        for i in range(n):
            if self._plate_approach_mode == "servo":
                self.plate.control_dofs_position(path[i], dofs_idx_local=[0, 1, 2])
                if self._plate_hold_mode != "servo":
                    # Orientation is still held the hold_mode way; only the
                    # DRIVING of x/y/z changed here.
                    self.plate.set_dofs_position(
                        position=self._vertical_dof_fix[:, 2:],
                        dofs_idx_local=[3, 4, 5], zero_velocity=False)
            else:
                self.plate.set_pos(pos=path[i])
                # Keep the servo's target in agreement with where the teleport
                # is putting the tool. Without this the target is still whatever
                # update_material_state froze it at -- the PREVIOUS action's
                # parked pose, high above the tray and at a different x/y -- so
                # the servo spends the whole descent driving at its 30 N limit
                # toward a place the tool is not going. That is not only a
                # misleading number in reaction_report(): the actuator force
                # enters the constraint solve, so contacts made late in the
                # descent see it. A target write is cheap and resets nothing.
                self.plate.control_dofs_position_velocity(
                    path[i], torch.zeros_like(path[i]), dofs_idx_local=[0, 1, 2])
                self.plate.set_dofs_position(
                    position=self._vertical_dof_fix,
                    dofs_idx_local=self._vertical_dofs_local
                )
            self._step_scene()
            self._reaction_update(path[i], phase)
            if on_step is not None:
                on_step(i)

        if self._plate_approach_mode == "servo":
            # A PD servo trailing a moving ramp lags by ~v*tau: at 0.5 m/s and
            # tau = 1/omega = 10.6 ms that is ~5 mm, and measured the descent
            # ended 2.1 mm high. The sweep then teleported the blade that 2.1 mm
            # straight down into the pile, which showed up as a 6.5x jump in
            # particle-particle penetration. Holding the final target lets the
            # servo arrive before anything else happens; ~4 time constants takes
            # the residual under 2 % of the lag.
            for _ in range(self._plate_arrival_steps):
                self.plate.control_dofs_position(path[-1], dofs_idx_local=[0, 1, 2])
                self._step_scene()
                self._reaction_update(path[-1], phase)

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

    def _apply_placement_aware_starts(self, action_starts, angles, n_samples, *,
                                      resolution, n_angles, clearance,
                                      clearance_bias, snap_to_nearest=False):
        """Replace blindly-drawn touchdown poses with collision-free ones.

        Only ``p_start`` and the yaw are overridden — the sweep target is left
        alone, since the tool is *supposed* to run into particles once it is
        down; what must be avoided is materializing inside one on the way down.

        Any (env, sample) for which the free set is empty keeps its blind draw,
        so a fully-covered tray degrades to the previous behaviour instead of
        failing.
        """
        from placement_sampling import (build_occupancy, clearance_map,
                                        free_placements, nearest_free_placement,
                                        sample_free_placements)

        tool_length, tool_width, _ = self._plate_params["size"]
        sizes = self._sampled_params.get("particle_sizes", None)
        if sizes is None:
            sizes = [p.morph.size if hasattr(p.morph, "size")
                     else (p.morph.radius * 2,) * 3 for p in self.material]
        half_xy = torch.as_tensor(sizes, dtype=torch.float32,
                                  device=gs.device)[:, :2] * 0.5
        # a cube free to take any yaw sweeps out sqrt(2) of its side
        is_cube = torch.tensor([hasattr(p.morph, "size") for p in self.material],
                               dtype=torch.float32, device=gs.device)
        half_xy = half_xy * (1.0 + (math.sqrt(2) - 1.0) * is_cube).unsqueeze(1)

        try:
            occ, meta = build_occupancy(
                self._particle_state[:, :, 0:3], half_xy,
                (self._box_params["vol"][0], self._box_params["vol"][1]),
                resolution, active=getattr(self, "_n_active", None))
            yaw_bins = (-torch.pi / 2) + torch.arange(
                n_angles, device=gs.device, dtype=torch.float32) * (torch.pi / n_angles)
            free = free_placements(occ, meta, yaw_bins, tool_length, tool_width,
                                   clearance=clearance,
                                   wall_margin=self._safety_margin)
            if snap_to_nearest:
                # Composed mode: keep the neighbourhood the caller's sampler
                # already chose (density-weighted, or uniform) and move the
                # pose the shortest distance that makes it collision-free.
                xy, yaw, ok = nearest_free_placement(
                    free, meta, yaw_bins, action_starts[..., :2])
            else:
                dist = clearance_map(occ, meta) if clearance_bias > 0 else None
                xy, yaw, ok = sample_free_placements(
                    free, meta, yaw_bins, n_samples, clearance=dist,
                    clearance_bias=clearance_bias)
        except Exception as e:                      # never block collection
            self._log(f"placement-aware sampling unavailable ({e}); "
                      f"falling back to blind sampling")
            return action_starts, angles

        n_free = int(ok.sum().item())
        if n_free == 0:
            self._log("placement-aware sampling found no collision-free tool "
                      "placement; falling back to blind sampling")
            return action_starts, angles
        if self._debug and n_free < ok.numel():
            self._log(f"placement-aware: {n_free}/{ok.numel()} samples placed "
                      f"in free space, rest fell back to blind")

        starts = action_starts.clone()
        starts[..., 0] = torch.where(ok, xy[..., 0], starts[..., 0])
        starts[..., 1] = torch.where(ok, xy[..., 1], starts[..., 1])
        return starts, torch.where(ok, yaw, angles)


    def _equalize_batch_travel(self, action_starts, action_stops, low, high):
        """Give every env in a batch the same push length for each sample.

        Envs step in lockstep and ``plate_velocity_translation`` sizes the sweep
        from the LONGEST travel in the batch, so one long push makes every env
        run for its duration. Sharing the distance removes that coupling — worth
        1.54x of a measured 2.64x batching penalty at 8 envs (see
        Genesis/action_sampling.py). Start point, direction and blade yaw stay
        per-env, and the distance still varies from batch to batch, so only the
        within-batch spread is given up.
        """
        from action_sampling import equalize_travel_distance, shared_batch_distance

        starts_xy, stops_xy = action_starts[..., :2], action_stops[..., :2]
        dist = (stops_xy - starts_xy).norm(dim=-1, keepdim=True)
        target = shared_batch_distance(dist).expand_as(dist)
        new_xy, clipped = equalize_travel_distance(
            starts_xy, stops_xy, low, high, target)
        if self._debug and bool(clipped.any()):
            self._log(f"shared travel distance: {int(clipped.sum())}/"
                      f"{clipped.numel()} pushes truncated at the box boundary")
        return torch.cat((new_xy, action_stops[..., 2:]), dim=-1)

    def generate_action_samples(
            self,
            n_samples: int,
            particle_xy: torch.Tensor | None = None,
            grid_res: int = 8,
            density_uniform_mix: float = 1.0,
            center_bias: float = 0.0,
            center_xy: tuple = (0.0, 0.0),
            start_sampling: str = "auto",
            placement_resolution: float = 0.001,
            placement_angles: int = 16,
            placement_clearance: float = 0.0,
            placement_clearance_bias: float = 0.0,
            shared_travel_distance: bool = False,
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
            20-step trajectory, vs ~50% by step 19). Pass the CURRENT
            particle state (updated after the previous push), not a
            snapshot from episode start - the whole point is tracking
            density as it evolves. When None, falls back to the old fully
            uniform START sampling.

        center_bias: when > 0, the STOP position is pulled toward center_xy
            from the (already-sampled) start position by a random fraction
            in [0, center_bias] - e.g. center_bias=0.7 means each push moves
            its contact point somewhere between 0% and 70% of the way to
            center_xy, instead of an independent uniform-random stop. Used
            to generate demonstrations of gathering material toward a point
            (e.g. for the granular-pile "collect into a target area" task),
            which plain uniform/density-weighted stop sampling never
            produces on its own - both of those pick stops independent of
            start, so a typical trajectory has no systematic inward drift.
            When 0 (default), STOP is sampled uniformly as before, and
            center_xy is unused.

        start_sampling: how the touchdown pose is drawn. Four values, and the
            two mechanisms are genuinely complementary rather than rival - one
            is a property of the pile, the other of the tool:

              "auto"    (default, unchanged behaviour) density-weighted when
                        particle_xy is given, uniform otherwise.
              "uniform" force the fully uniform draw even if particle_xy is
                        passed.
              "density" force the density-weighted draw; requires particle_xy.
              "free"    draw p_start and its yaw from the tool's free
                        configuration space, so the plate does not descend INTO
                        a particle - which the solver resolves by ejecting it,
                        an artifact recorded as though it were a push. Measured
                        at n=200: touchdowns overlapping a particle drop from
                        100% to 19%. Ignores density entirely, so it drifts
                        toward empty tray where a push moves nothing.
              "composed" density (or uniform) chooses the neighbourhood, then
                        the pose is moved the shortest distance that makes it
                        collision-free. Keeps both properties; see
                        Genesis/placement_sampling.py::nearest_free_placement.

            "free" and "composed" fall back to the underlying blind draw
            per-sample wherever no collision-free placement exists, which is
            the expected outcome once the pile covers enough of the tray. They
            are refinements, not guarantees.

        placement_resolution: occupancy / C-space grid cell size, metres.
        shared_travel_distance: give every env the same push LENGTH for a given
            sample, keeping its own start, direction and yaw. Envs step in
            lockstep and the sweep is sized from the longest travel in the
            batch, so independent distances make every env run for the longest
            one's duration - measured at 1.54x of a 2.64x batching penalty at
            8 envs, and up to 12x of end-to-end batch time at 150 objects. Off
            by default so single-env and MPC callers are unaffected; collection
            turns it on. Pushes that cannot reach the shared distance without
            leaving the sampling box are truncated at the boundary.

        Returns:
            Tuple of (action_starts, action_stops, angles) each of shape [n_envs, n_samples, 3/1]
        """
        if start_sampling not in ("auto", "uniform", "density", "free", "composed"):
            raise ValueError(
                f"start_sampling must be one of 'auto', 'uniform', 'density', "
                f"'free', 'composed'; got {start_sampling!r}")
        if start_sampling == "density" and particle_xy is None:
            raise ValueError("start_sampling='density' requires particle_xy")
        # "free" replaces the start outright, so the underlying draw only needs
        # to supply a fallback; "composed" needs the density draw to be the
        # thing it snaps, so it keeps whatever "auto" would have done.
        use_density = (
            particle_xy is not None
            and start_sampling in ("auto", "density", "composed", "free")
        )

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
        if use_density:
            start_samples = self._sample_density_weighted_xy(
                particle_xy, n_samples, grid_res, density_uniform_mix
            )
            # density grid cells can fall just outside the angle-dependent
            # safety margin near the box edge - clamp back into the same
            # valid range uniform sampling was already restricted to.
            start_samples = torch.max(torch.min(start_samples, high), low)
        else:
            start_samples = (high - low) * torch.rand((n_total, 2), device=gs.device) + low

        if center_bias > 0:
            center = torch.tensor(center_xy, dtype=start_samples.dtype, device=start_samples.device)
            shrink = center_bias * torch.rand((n_total, 1), device=start_samples.device)
            stop_samples = start_samples + shrink * (center - start_samples)
            stop_samples = torch.max(torch.min(stop_samples, high), low)
        else:
            stop_samples = (high - low) * torch.rand((n_total, 2), device=gs.device) + low
        _z = torch.ones((n_total, 1), device=gs.device) * self._operation_height

        action_starts = torch.cat((start_samples, _z), axis=1)
        action_stops = torch.cat((stop_samples, _z), axis=1)

        # Reshape to [n_envs, n_samples, ...]
        action_starts = action_starts.reshape(self._n_envs, n_samples, 3)
        action_stops = action_stops.reshape(self._n_envs, n_samples, 3)
        angles = angles.reshape(self._n_envs, n_samples)

        if start_sampling in ("free", "composed"):
            action_starts, angles = self._apply_placement_aware_starts(
                action_starts, angles, n_samples,
                resolution=placement_resolution,
                n_angles=placement_angles,
                clearance=placement_clearance,
                clearance_bias=placement_clearance_bias,
                snap_to_nearest=(start_sampling == "composed"),
            )

        if shared_travel_distance and self._n_envs > 1:
            action_stops = self._equalize_batch_travel(
                action_starts, action_stops,
                low.reshape(self._n_envs, n_samples, 2),
                high.reshape(self._n_envs, n_samples, 2))

        return action_starts, action_stops, angles

    def execute_action(
            self,
            p_start,
            p_stop,
            angle,
            on_phase=None,
            on_step=None,
        ):
        """
        Execute action (lower, sweep, lift) for all environments.

        Args:
            p_start: Starting positions [n_envs, 3]
            p_stop: Stopping positions [n_envs, 3]
            angle: Angles [n_envs]
            lift_height: Lift height [n_envs, 3]
            on_phase: optional callback(phase: str), invoked at two points
                inside the push motion:
                    'post_lower' — plate has just reached p_start (about to sweep)
                    'post_sweep' — plate has just reached its stop position
                                   (about to lift)
                Shared across every caller of execute_action (GenesisEnv,
                GenesisOracleEnv, data collection, future MPC/viz code) — e.g.
                to capture an intermediate video frame, log, or debug-plot the
                mid-action state. A no-op when None; callers that don't pass
                it (e.g. batched rollout planning) see no behavior change.
            on_step: optional callback(phase: str, step: int), invoked after
                EVERY simulation step of the push, with phase one of
                'lower' / 'sweep' / 'lift'. Where on_phase gives two snapshots,
                this gives every frame — which is what a video needs, and what
                a per-step diagnostic needs. A no-op when None.

        Returns:
            Tensor of shape [n_envs] with success status
        """
        self._reaction_reset()

        def _phase_step(phase):
            if on_step is None:
                return None
            return lambda step, *_: on_step(phase, step)

        # Lower: teleport to clearance height, then simulate only the short
        # final descent into operating position.  This skips simulating the
        # approach from the full lift height above.
        self._vertical_dof_fix[:, 0] = p_start[:, 0]
        self._vertical_dof_fix[:, 1] = p_start[:, 1]
        self._vertical_dof_fix[:, 4] = angle
        lower_start = p_start + self._clearance_offset
        self.plate.set_pos(lower_start, zero_velocity=True)
        self.plate_position_translation(lower_start, p_start, self._clearance_ctrl_steps,
                                        on_step=_phase_step('lower'), phase="lower")
        if on_phase is not None:
            on_phase('post_lower')

        # Sweep
        reached_goal, final_pos = self.plate_velocity_translation(
            p_start,
            p_stop,
            angle,
            on_step=_phase_step('sweep'),
        )
        if on_phase is not None:
            on_phase('post_sweep')

        # Lift: simulate only the short ascent to clearance height, then
        # teleport the plate out of the way.  Particles are already below
        # clearance height so there is no contact after this point.
        self._vertical_dof_fix[:, 0] = final_pos[:, 0]
        self._vertical_dof_fix[:, 1] = final_pos[:, 1]
        self.plate_position_translation(
            final_pos, final_pos + self._clearance_offset, self._clearance_ctrl_steps,
            on_step=_phase_step('lift'), phase="lift")
        self.plate.set_pos(final_pos + self._lift_height_tensor, zero_velocity=True)

        return reached_goal, final_pos

    def collect_data_samples(
            self,
            n_samples: int = 200,
            path : str | Path = "training",
            center_bias: float = 0.0,
            start_sampling: str = "auto",
            shared_travel_distance: bool = False,
            state_library=None,
        ):
        """
        Collect data samples from all environments efficiently.
        Optimized for GPU processing and memory efficiency.

        Args:
            n_samples: Number of samples to collect per environment
            path: Output path for data
            center_bias: forwarded to generate_action_samples() - see there.
                0 (default) reproduces the existing uniform-stop behavior.
            start_sampling: forwarded to generate_action_samples() - see there.
                "auto" (default) reproduces the existing behaviour.
            shared_travel_distance: forwarded to generate_action_samples().
                False (default) reproduces the existing behaviour.
            state_library: optional StateLibrary. When given, the initial pile
                is RESTORED from an already-settled state instead of being
                shuffled and settled, which is where essentially all of a
                reset's cost lives (measured 54x at n=50 and 6184x at n=200).
                The caller is responsible for having shuffled otherwise.
        """
        max_samples = n_samples * self._n_envs

        self._config["data_collection"].update({
            "n_envs": self._n_envs,
            "samples_per_env": n_samples,
            "goal_threshold": self._goal_threshold,
            "start_sampling": start_sampling,
            "shared_travel_distance": bool(shared_travel_distance),
            "state_library_size": (len(state_library) if state_library is not None else 0),
        })

        # Allocate once or reuse if same size
        if (not hasattr(self, '_collection_buffers') or 
            self._collection_buffers['states'].shape[0] != n_samples or
            self._collection_buffers['states'].shape[1] != self._n_envs):
            self._allocate_collection_buffers(n_samples)
        
        # Clear data buffer
        for buf in self._collection_buffers.values():
            buf.zero_()
        
        # Reset: restore an already-settled state if a library was supplied,
        # otherwise pay for the settle. A restored state is at rest by
        # construction, so no settle follows it.
        if state_library is not None:
            state_library.apply(self)
        else:
            self.update_material_state()
        n_identical = 0
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
                1, particle_xy=self._particle_state[:, :, 0:2],
                center_bias=center_bias,
                start_sampling=start_sampling,
                shared_travel_distance=shared_travel_distance,
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
            # A run that "succeeds" while nothing moves is the silent failure
            # worth guarding against, so count it always rather than only under
            # --debug, and record the total alongside the data.
            if torch.equal(self._collection_buffers["states"][sample_idx],
                           self._collection_buffers["states_"][sample_idx]):
                n_identical += 1
                if self._debug:
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

        # Audit fields: recorded so a finished dataset carries the evidence
        # that it is trustworthy, instead of the run having to be repeated to
        # find out. escaped_particles > 0 means the contact solver failed for a
        # particle and, since each transition's s is the previous s', every
        # later sample in that env is suspect.
        escaped = self.escaped_particle_count()
        budget = getattr(self, "_contact_budget_peak", None)
        if budget is None:
            try:
                budget = self.contact_budget_usage()
            except Exception:
                budget = None
        print(f">> Transitions with s' == s : {n_identical}")
        print(f">> Escaped particles        : {escaped}")
        if budget:
            print(f">> Peak contact points      : {budget['contact_points']}/"
                  f"{budget['contact_cap']}")

        self._config["statistics"] = {
            "n_envs"   : self._n_envs,
            "samples_per_env"  : n_samples,
            "total_samples_collected"  : num_collected_samples,
            "number_of_failed_samples" : max_samples - num_collected_samples,
            "unchanged_transitions" : n_identical,
            "escaped_particles" : escaped,
            "contact_budget" : budget,
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
