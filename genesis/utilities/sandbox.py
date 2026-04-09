import genesis as gs
from genesis import Scene
import numpy as np
from operator import add

def add_sandbox(scene : Scene, box_pos=(0.5, 0., 0), box_vol=(0.5, 0.5, 0.1), sand_color=(0.3, 0.3, 1.0)):
    x, y, z = box_pos
    width, depth, height = box_vol
    wall_thickness = 0.02
    
    # ground plate
    scene.add_entity(
        gs.morphs.Box(
            pos=(x, y, z),
            size=(width, depth, wall_thickness),
            fixed=True
        ),     
        surface=gs.surfaces.Default(
            color    = (0.3, 0.3, 1.0),
        ),
    )
    
    # front wall (robot view)
    scene.add_entity(gs.morphs.Box(
            pos=(x-(width+wall_thickness)/2, y, z+(height-wall_thickness)/2),
            size=(wall_thickness, depth, height),
            fixed=True
    ))
    
    # back wall (robot view)
    scene.add_entity(gs.morphs.Box(
            pos=(x+(width+wall_thickness)/2, y, z+(height-wall_thickness)/2),
            size=(wall_thickness, depth, height),
            fixed=True
    ))
    
    # left wall (robot view)
    scene.add_entity(gs.morphs.Box(
            pos=(x, y+(depth+wall_thickness)/2, z+(height-wall_thickness)/2),
            size=(width, wall_thickness, height),
            fixed=True
    ))
     
    # right wall (robot view)
    scene.add_entity(gs.morphs.Box(
            pos=(x, y-(depth+wall_thickness)/2, z+(height-wall_thickness)/2),
            size=(width, wall_thickness, height),
            fixed=True
    ))
    
    # Granular material
    # scene.add_entity(
    #     material=gs.materials.MPM.Sand(
    #         E=1e4,
    #         nu=0.4,
    #         rho=2000,
    #         sampler='random',
    #         friction_angle=80
    #     ),
    #     morph=gs.morphs.Box(
    #         pos  = tuple(map(add, box_pos, (0, 0, box_vol[2]/2))),
    #         size = box_vol,
    #     ),
    #     surface=gs.surfaces.Default(
    #         color    = sand_color,
    #         vis_mode = 'particle',
    #     ),
    # )


# def randomize_sandbox(scene : Scene):
    
#     n_particles = sandbox.get_n_particles()
#     new_pos = center + (gs.np.random.rand(n_particles, 3) - 0.5) * size
#     sandbox.set_particle_pos(new_pos)