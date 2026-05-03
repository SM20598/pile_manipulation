import genesis as gs
import numpy as np
from bayes_opt import BayesianOptimization

from sandbox_manipulation import SandboxManipulation
from utilities.helper_functions import chamfer_distance


n_samples = 1
n_particles = 1

sm = SandboxManipulation(config="configs/param_optim.yaml")
sm.build()
pbounds = {
    'friction': (100, 1500),
    'density' : (0.1, 2),
}
sm.set_material_properties(0.1, 1000)

def optimize(friction, density):
        
        sm.set_material_properties(friction, density)

        sm.set_material_state(gt_state)
        
        sm.execute_action(
            gt_p_start,
            gt_p_stop,
            gt_angle,
            gt_speed
        )

        pred_state_ = sm.get_material_state()

        loss = chamfer_distance(gt_state_, pred_state_)

        return (-1) * loss

##### TODO: REPLACE WITH REAL DATA #####
sm.collect_data_samples(
    n_samples=n_samples
)
gt_samples = sm.get_collected_samples()
########################################


optimizer = BayesianOptimization(
    f=optimize,
    pbounds=pbounds,
    random_state=1,
)

for gt_sample in gt_samples:
    gt_state, gt_state_, gt_p_start, gt_p_stop, gt_angle, gt_speed = gt_sample  

    optimizer.maximize(
          init_points=5,
          n_iter=25
    )
    


        



