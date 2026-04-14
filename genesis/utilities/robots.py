import genesis as gs
import numpy as np
from genesis import Scene
from typing import Tuple
import pathlib
import os

def spawn_robot(
    scene : Scene,
    robot_name : str,
    pos : Tuple[float, float, float]=(0.0, 0.0, 0.0)):
    print()
    if robot_name == 'franka':
        robot = scene.add_entity(
            gs.morphs.MJCF(
                file=os.path.join(pathlib.Path(__file__).parent.resolve(), 'xml/franka_emika_panda_with_tool/panda_robola_compile.xml'),
                pos=pos
            ),
        )
    else:
        raise NotImplementedError(f"Robot {robot_name} is not supported.")
    
    return robot
