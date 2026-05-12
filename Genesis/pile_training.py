import genesis as gs
import numpy as np

from sandbox_manipulation import SandboxManipulation

sm = SandboxManipulation(config="configs/basic_example.yaml")

sm.build()
# sm.simulate(1000)
# sm.view()
sm.collect_data_samples()

