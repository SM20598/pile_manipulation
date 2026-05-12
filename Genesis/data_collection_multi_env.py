import genesis as gs
import numpy as np

from sandbox_manipulation import SandboxManipulation

# Random variables that need rebuild
# num_particles = [5, 10, 20, 30]
# box_vol = [
#     [0.1, 0.1, 0.1],
#     [0.3, 0.5, 0.1],
#     [0.5, 0.3, 0.1],
# ]
# particle_size = [0.001, 0.002, 0.005, 0.01]

num_particles = [1, 2]
box_vol = [[0.5, 0.5, 0.1]]
particle_size = [0.01]



sm = 0.03 # safety margin of 0.02 - 0.01
config = {
    'simulation': {
        'dt': 4e-3,
        'substeps' : 1,
        'backend' : 'gpu',
        'precision' : '32',
        'performance_mode' : True,
        'viewer_options': {
            'show_viewer' : True,
            'viewer_type': "observer",
        },
        'n_envs' : 3,
    },
    'sandbox' : {
        'box' : {
            'vol' : None, # is set in loop below
            'properties': {
                'friction' : 0.1,
            }
        },
        'material' : {
            'vol' : None, # is set in loop below
            'type' : "rsa",
            'properties' : { 
                'n_particles' : None, # is set loop below
                'particle_size' : None, # is set in loop below
                'rho' : 600, # is set in loop below
                'friction' : 0.1,
            },
        },
    },
    'plate' : {
        'size' : [0.1, 0.005, 0.06],
    }
}


# sm = SandboxManipulation(config)
# sm.build()

for b_vol in box_vol:
    # set box volume and adjust max material volume
    config['sandbox']['box']['vol'] = b_vol
    config['sandbox']['material']['vol'] = [x-sm for x in b_vol]
    
    for n_p in num_particles:
        config['sandbox']['material']['properties']['n_particles'] = n_p

        for p_size in particle_size:
            config['sandbox']['material']['properties']['particle_size'] = p_size
            
            
            sm = SandboxManipulation(config)
            sm.build()
            
            sm.set_material_mass_shift(np.array([0.1, 0.15, 0.2]))
            sm.set_material_friction_ratio(np.array([0.05, 0.1, 0.5]))
            sm.set_box_friction_ratio(np.array([0, 0, 0]))
            sm.collect_data_samples(n_samples=4)
            sm.export_data_samples()

            # sm.random_material_mass()
            # sm.collect_data_samples()
            # sm.export_data_samples
        
            sm.destroy()

    

# sm.build()
# sm.collect_data_samples()

