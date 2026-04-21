import numpy as np

def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                        x1*w0 + y1*z0 - z1*y0 + w1*x0,
                        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                        x1*y0 - y1*x0 + z1*w0 + w1*z0])

def get_horizontal_path(xy_start, xy_stop, z, n_steps=100):
    
    x_start, y_start = xy_start
    x_stop, y_stop = xy_stop
    path = []
    for i in range(n_steps):
        t = i / (n_steps - 1)

        x = (1 - t) * x_start + t * x_stop
        y = (1 - t) * y_start + t * y_stop
        path.append([x, y, z])
    return path

def get_vertical_path(x, y, z_start, z_end, n_steps=100):
    path = []
    for i in range(n_steps):
        t = i / (n_steps - 1)
        z = (1 - t) * (z_start) + t * z_end
        path.append([x, y, z])
    return path
