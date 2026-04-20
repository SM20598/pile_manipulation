import genesis as gs
import genesis.utils.geom as gu 
import numpy as np
import yaml
from utilities.materials import *
import pathlib
import os

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
        self._tool_l, self._tool_w, self._tool_h = self._config["robot"]['tool_dim']
        

        self._init_scene()
        self._add_entities()

    def _init_scene(self):
        settings = self._config["simulation"].get('viewer_options', dict())
        c_fov = settings.get('camera_fov', 30)
        max_fps = settings.get('max_FPS', 60)

        b_x, b_y, b_z = self._box_pos   
        v_x, v_y, v_z = self._box_vol
        l_bound = (b_x-2*v_x, b_y-2*v_y, b_z-2*v_z)
        u_bound = (b_x+2*v_x, b_y+2*v_y, b_z+2*v_z+self._wall_thickness)

        viewer_type = settings.get('viewer_type', None)
        
        if viewer_type == "observer":
            self._viewer_options = gs.options.ViewerOptions(
                camera_pos    = settings.get('camera_pos', [1.5, 0.0, 1.3]),
                camera_lookat = settings.get('camera_lookat', [0.5, 0.0, 0.2]),
                camera_fov    = c_fov,
                max_FPS       = max_fps,
            )
        elif viewer_type == "bird":
            self._viewer_options = gs.options.ViewerOptions(
                camera_pos    = settings.get('camera_pos', [0.5, 0.0, 1.2]),
                camera_lookat = settings.get('camera_lookat', [0.5, 0.0, 0.0]),
                camera_fov    = c_fov,
                max_FPS       = max_fps,
            )
        elif viewer_type == "leveled":
            self._viewer_options = gs.options.ViewerOptions(
                camera_pos    = settings.get('camera_pos', [b_x+1.5, b_y, b_z]),
                camera_lookat = settings.get('camera_lookat', [0.5, 0.0, 0.2]),
                camera_fov    = c_fov,
                max_FPS       = max_fps,
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
            show_viewer=settings.get('enable_viewer', False)
        )
    
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
                     
    def build(self):
        self._scene.build()

        self.robot.set_dofs_kp(self._config["robot"].get('dofs_kp', [4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
        self.robot.set_dofs_kv(self._config["robot"].get('dofs_kv', [450, 450, 350, 350, 200, 200, 200, 10, 10]))
        self.robot.set_dofs_force_range(*self._config["robot"].get('dofs_force_range', [
            [-87, -87, -87, -87, -12, -12, -12, -100, -100],
            [ 87,  87,  87,  87,  12,  12,  12,  100,  100]
        ]))

    def view(self, horizon=1000):
        for _ in range(horizon):
            self._scene.visualizer.update()
    
    def simulate(self, horizon=1000):
        for _ in range(horizon):
            self._scene.step()

    def collect_data_samples(self, n_samples=100, operation_height=None):
        box_x, box_y, box_z = self._box_pos

        theta = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=n_samples)        
        border_offset_x = (self._granular_vol[0]/2 - np.cos(theta) * self._tool_l/2 - np.sin(theta) * self._tool_w/2)
        border_offset_y = (self._granular_vol[1]/2 - np.sin(theta) * self._tool_l/2 - np.cos(theta) * self._tool_w/2)

        x_max = box_x + border_offset_x
        x_min = box_x - border_offset_x
        y_max = box_y + border_offset_y
        y_min = box_y - border_offset_y

        xy_start = np.random.uniform(low=[x_min, y_min], high=[x_max, y_max], size=(2, n_samples)).T
        xy_end = np.random.uniform(low=[x_min, y_min], high=[x_max, y_max], size=(2, n_samples)).T


        if operation_height is None:
            operation_height = box_z + (self._wall_thickness + self._granular_vol[2])/2
        
        print(f"Sampled {n_samples} action samples")

        # move robot to initial starting position in the middle of the box
        plate_frame = self.robot.get_link("plate")
        q_init = self.robot.inverse_kinematics(
            link = plate_frame,
            pos  = np.array([box_x, box_y, operation_height*2]),  # target place pose
            quat = np.array([0, 1, 0, 0]),
        )
        q_init[7:] = 0.0
        self.robot.set_qpos(q_init)

        for _ in range(100):
            self._scene.visualizer.update()
    
        for p_start, p_end, angle in zip(xy_start, xy_end, theta):
            x, y = p_start
            print(angle)
            quat = gu.xyz_to_quat(np.array([0, 0, angle]))
            
            def quaternion_multiply(quaternion1, quaternion0):
                w0, x0, y0, z0 = quaternion0
                w1, x1, y1, z1 = quaternion1
                return np.array([-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                                    x1*w0 + y1*z0 - z1*y0 + w1*x0,
                                    -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                                    x1*y0 - y1*x0 + z1*w0 + w1*z0]) 

            quat = quaternion_multiply([0, 1, 0, 0], quat)
            print(quat)
            q_start = self.robot.inverse_kinematics(
                link=self.robot.get_link("plate"),
                pos=(x, y, operation_height*2),
                quat=quat
                
            )
            q_start[7:] = 0.0
            self.robot.set_qpos(q_start)
            
            for _ in range(200):
                self._scene.visualizer.update()