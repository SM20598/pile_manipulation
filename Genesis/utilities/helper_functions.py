import numpy as np


def chamfer_distance(p_real, p_sim):

    def min_dist(p, points):
        dists = np.square(np.linalg.norm(p - points, axis=1))
        return np.min(dists)


    loss = 0
    for p_s in p_sim:
        loss += min_dist(p_s, p_real)
    for p_r in p_real:
        loss += min_dist(p_r, p_sim)
    return loss


def earth_movers_distance(p_real, p_sim):
    pass

     
            

