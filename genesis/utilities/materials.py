from collections import defaultdict
from typing import Optional, Tuple
import genesis as gs
from genesis import Scene
import numpy as np

_DIMENSIONALITY = 3
_NEIGHBOR_OFFSET = [-1, 0, 1]

def random_sequential_addition(
    scene : Scene,
    box_pos : Tuple[float, float, float],
    granular_vol : Tuple[float, float, float],
    particle_size : float,
    material_properties : dict,
    wall_thickness : float,
    color : Tuple[float, float, float],
):
    """Random Sequential Addition (RSA) of spheres with linked-cell (grid) acceleration."""
    n_p = material_properties.get('n_particles', 1000)
    max_p_size = material_properties.get('max_particle_size', 0.02)
    min_p_size = material_properties.get('min_particle_size', 0.01)
    randomized_size = material_properties.get('randomize_particle_size', False)
    spatial = material_properties.get('spatial', False)
    max_attempts = material_properties.get('max_attempts', 200000)
    x, y, z = box_pos
    width, depth, height = granular_vol
    
    attempts = 0
    positions = []
    accepted_radii = []
    grid = defaultdict(list)    
    
    if randomized_size:
        radii = np.random.uniform(min_p_size, max_p_size, n_p)
    else:
        max_p_size = particle_size/2

    def get_radius(i):
        return radii[i] if randomized_size else max_p_size

    for i in range(n_p):
        r = get_radius(i)
        print(r)
        placed = False
        
        while not placed and attempts < max_attempts:
            attempts += 1
            
            # get a random cell
            if spatial:
                size = (depth - r, width - r, height - r)
                candidate = np.random.uniform(r, size)  
            else:
                size = (depth - r, width -r)
                candidate = np.append(np.random.uniform(r, size), r+wall_thickness/2)

            cell = tuple((candidate // 2*max_p_size).astype(int))
                       
            def particle_overlapping(neighbor_cell : list, depth : int = 0):
                """
                Check if candidate particle overlaps with neighboring particles in grid.
                
                @params:
                    neigbor_cell: empty list that stores x, y, and z indices of neighboring cell. Updated recursively
                    depth: recursion depth, corresponds to dimension
                
                Returns True if there is an overlap
                """
                
                # If x-y-z of neighbor cell is known, check for overlap
                if depth == _DIMENSIONALITY:
                    for j in grid.get(tuple(neighbor_cell), []):
                        dist = np.linalg.norm(candidate - positions[j])
                        if dist < (r + accepted_radii[j]):
                            return True
                    return False      
                 
                for d in _NEIGHBOR_OFFSET:
                    neighbor_cell.append(cell[depth] + d)
                    if particle_overlapping(neighbor_cell, depth+1):
                        return True
                    neighbor_cell.pop()
                return False
            
            if not particle_overlapping(neighbor_cell=[]):
                idx = len(positions)
                positions.append(candidate)
                accepted_radii.append(r)
                grid[cell].append(idx)
                print(candidate)
                scene.add_entity(
                    morph=gs.morphs.Sphere(
                    
                        pos=(x - width/2 + candidate[1], y - depth/2 + candidate[0], z +candidate[2] + wall_thickness/2),
                        radius=r,
                    ),    
                    material=gs.materials.Rigid(
                        # rho=material_properties.get('rho', 600),
                        # friction=material_properties.get('friction', 0.1),
                        # needs_coup=material_properties.get('needs_coup', True),
                        # coup_friction=material_properties.get('coup_friction', 0.1),
                        # coup_softness=material_properties.get('coup_softness', 0.002),
                        # coup_restitution=material_properties.get('coup_restitution', 0.0),
                    ),
                    surface=gs.surfaces.Default(
                        color = color,
                    ),
                )
                
                placed = True
                
        if attempts >= max_attempts:
            print(f"Stopped early at particle {i}")
            break
    print(f"Generated {len(positions)} particles out of {n_p}")

def add_liquid(
    scene : Scene,
    box_pos : Tuple[float, float, float],
    granular_vol : Tuple[float, float, float],
    material_properties : dict,
    wall_thickness : float,
    color : Tuple[float, float, float]
):
    """Add a liquid pile to the scene within a container."""
    x, y, z = box_pos
    g_height = granular_vol[2]
    method = material_properties.get('method', "PBD")
    
    if method == "MPM":
        material=gs.materials.MPM.Liquid(
            E=material_properties.get('E', 1e6),
            nu=material_properties.get('nu', 0.2),
            rho=material_properties.get('rho', 1000),
            viscous=material_properties.get('viscous', False),
        )        
    elif method == "SPH":
        material=gs.materials.SPH.Liquid(
            rho = material_properties.get('rho', 1000.0),
            stiffness = material_properties.get('stiffness', 50000.0),
            exponent = material_properties.get('exponent', 7.0),
            mu = material_properties.get('mu', 0.005),
            gamma = material_properties.get('gamma', 0.01),
            sampler = material_properties.get('sampler', "regular")
        )
    elif method == "PBD":
        
        material=gs.materials.PBD.Liquid(
            rho = material_properties.get('rho', 1000.0),
            sampler = material_properties.get('sampler', 'pbs'),
            density_relaxation = material_properties.get('density_relaxation', 0.2),
            viscosity_relaxation = material_properties.get('viscosity_relaxation', 0.01)
        )
    else:
        raise ValueError(f"Unsupported method {method}. Supported methods are 'MPM', 'SPH', and 'PBD'.")
        
    entity = scene.add_entity(
        material=material,
        morph=gs.morphs.Box(
            pos  = (x, y, z + (g_height + wall_thickness)/2),
            size = granular_vol,
        ),
        surface=gs.surfaces.Default(
            color    = color,
            vis_mode = 'particle',
        ),
    )
    return entity

def add_sand(
    scene : Scene,
    box_pos : Tuple[float, float, float],
    granular_vol : Tuple[float, float, float],
    material_properties : dict,
    wall_thickness : float,
    sand_color : Optional[Tuple[float, float, float]],
):
    """Add a sand pile to the scene within a container."""
    
    x, y, z = box_pos
    g_height = granular_vol[2]
    
    material = scene.add_entity(
        material=gs.materials.MPM.Sand(
            E=material_properties.get('E', 1e6),
            nu=material_properties.get('nu', 0.2),
            rho=material_properties.get('rho', 1000),
            sampler=material_properties.get('sampler', "random"),
            friction_angle=material_properties.get('friction_angle', 45)
        ),
        morph=gs.morphs.Box(
            pos  = (x, y, z + (g_height + wall_thickness)/2),
            size = granular_vol, # safety margin to avoid penetration with the walls
        ),
        surface=gs.surfaces.Default(
            color    = sand_color,
            vis_mode = 'particle',
        ),
    )
    return material

def add_box(
    scene : Scene,
    box_pos : Tuple[float, float, float],
    box_vol : Tuple[float, float, float],
    wall_thickness : float = 0.02,
    box_color : Optional[Tuple[float, float, float]]=(0, 0, 0),
):
    x, y, z = box_pos
    width, depth, height = box_vol
    # ground plate
    scene.add_entity(
        gs.morphs.Box(
            pos=(x, y, z),
            size=(width, depth, wall_thickness),
            fixed=True
        ),     
        surface=gs.surfaces.Default(
            color = box_color,
        ),
    )
    
    # front wall (robot view)
    scene.add_entity(
        gs.morphs.Box(
            pos=(x-(width+wall_thickness)/2, y, z+(height-wall_thickness)/2),
            size=(wall_thickness, depth, height),
            fixed=True
        ),
        surface=gs.surfaces.Default(
            color = box_color,
        ),
    )
    
    # back wall (robot view)
    scene.add_entity(
        gs.morphs.Box(
            pos=(x+(width+wall_thickness)/2, y, z+(height-wall_thickness)/2),
            size=(wall_thickness, depth, height),
            fixed=True
        ),
        surface=gs.surfaces.Default(
            color = box_color,
        ),
    )
    
    # left wall (robot view)
    scene.add_entity(
        gs.morphs.Box(
            pos=(x, y+(depth+wall_thickness)/2, z+(height-wall_thickness)/2),
            size=(width, wall_thickness, height),
            fixed=True
        ),
        surface=gs.surfaces.Default(
            color = box_color,
        ),
    )
     
    # right wall (robot view)
    scene.add_entity(
        gs.morphs.Box(
            pos=(x, y-(depth+wall_thickness)/2, z+(height-wall_thickness)/2),
            size=(width, wall_thickness, height),
            fixed=True
        ),
        surface=gs.surfaces.Default(
            color = box_color,
        ),
    )

def spawn_sandbox(
    scene : Scene,
    material_type : str,
    material_properties : dict,
    box_pos : Tuple[float, float, float]=(0.5, 0.0, 0.0),
    box_vol : Tuple[float, float, float]=(0.5, 0.5, 0.1),
    granular_vol : Tuple[float, float, float]=(0.08, 0.08, 0.08),
    wall_thickness : float = 0.02,
    granular_color : Optional[Tuple[float, float, float]]=(1, 1, 0),
    box_color : Optional[Tuple[float, float, float]]=(0, 0, 0),
    safety_margin : float = 0.02,
    omit_box : bool = False,
    particle_size : float = 0.01
):
    """
    Add a sandbox to the scene with specified material and properties.
    
    @param
        scene: The Genesis scene to which the sandbox will be added.
        material_type: The type of granular material. If not set Random Sequential Addition (RSA) will be used.
        material_properties: A dictionary of properties for the granular material. Look at functions above for specifics.
        box_pos: The position of the center of the box.
        box_vol: The dimensions of the box.
        granular_vol: The dimensions of the granular material. Needs to be smaller than box_vol by 'safety_margin' in x- and y-directions.
        wall_thickness: The thickness of the box walls.
        granular_color: The color of the granular material.
        box_color: The color of the box.
        safety_margin: Ensures that the granular material does not penetrate the walls.
        omit_box: If True, only add the granular material.
    
    returns:
        Material entity added to the scene.
    """
    
    if (granular_vol[0] > box_vol[0]-safety_margin or granular_vol[1] > box_vol[1]-safety_margin):
        raise ValueError(f"granular volume exceeds box volume. Safety margin is set to {safety_margin} per axis.")
    
    if not omit_box:
        add_box(
            scene=scene,
            box_pos=box_pos,
            box_vol=box_vol,
            wall_thickness=wall_thickness,
            box_color=box_color
        )
    
    if material_type == "rsa":
        print (" FOR COMPARISON:")
        print("box_pos: ", box_pos)
        print("granular_vol: ", granular_vol)
        print("particle_size: ", particle_size)
        print("material_properties: ", material_properties)
        print("wall_thickness: ", wall_thickness)
        print("color: ", granular_color)
        material = random_sequential_addition(
            scene=scene,
            box_pos=box_pos,
            granular_vol=granular_vol,
            material_properties=material_properties,
            wall_thickness=wall_thickness,
            color=granular_color,
            particle_size=particle_size
        )
    
    elif material_type == "sand":
        material = add_sand(
            scene=scene,
            box_pos=box_pos,
            granular_vol=granular_vol,
            material_properties=material_properties,
            wall_thickness=wall_thickness,
            sand_color=granular_color
        )
    elif material_type == "liquid":
        material = add_liquid(
            scene=scene,
            box_pos=box_pos,
            granular_vol=granular_vol,
            material_properties=material_properties,
            wall_thickness=wall_thickness,
            color=granular_color,
        )
    else:
        raise ValueError(f"Unsupported material type {material_type}. Supported types are 'granular', 'sand', and 'liquid'.")
    
    return material

