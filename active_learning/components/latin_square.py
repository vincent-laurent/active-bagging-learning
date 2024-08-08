import numpy as np
import scipy.stats as stats
from sklearn.neighbors import BallTree
from smt.sampling_methods import LHS


def check_input(x_input: np.ndarray = None, size: int = 100,
                x_limits: np.ndarray = None, dim: int = None, batch_size=10):
    if x_limits is None and x_input is None and dim is None:
        raise ValueError("You have to specify at least one of the three"
                         "following argument : x_input, x_limits or dim"
                         )
    if batch_size > size:
        raise ValueError("batch_size should be lower than n")
    if dim is None and x_input is not None:
        dim = x_input.shape[1]

    if x_limits is None:
        x_limits = np.array([[0, 1]] * dim)
    return x_limits


def one_d_iterative_sampler(x_input: np.ndarray = None, size: int = 100,
                            x_limits: np.ndarray = None,
                            criterion="corr", batch_size=10):
    x_limits = check_input(x_input, size, x_limits, 1, batch_size)

    def sampler(size_):
        return np.random.uniform(
            x_limits[0][0], x_limits[0][1], size=size_)

    if x_input is None:
        return sampler(size)
    else:
        new_points = sampler(max(len(x_input), size))
        dist = BallTree(x_input.reshape(-1, 1))
        distances, query = dist.query(new_points.reshape(-1, 1))
        orders = len(distances) - stats.rankdata(distances, method="ordinal")
        select = orders < batch_size
        x_new = new_points[np.ravel(select)]

        if sum(select) < size:
            return np.concatenate(
                (one_d_iterative_sampler(np.concatenate((x_input, x_new)),
                                         size - batch_size), x_new))
        return x_new


def iterative_sampler(x_input: np.ndarray = None, size: int = 100,
                      x_limits: np.ndarray = None,
                      dim: int = None,
                      criterion="corr", batch_size=10):
    x_limits = check_input(x_input, size, x_limits, dim, batch_size)

    lhs = LHS(xlimits=x_limits, criterion=criterion)

    if x_input is None:
        return lhs(size)
    else:
        new_points = lhs(max(len(x_input), size))
        dist = BallTree(x_input)
        distances, query = dist.query(new_points)
        orders = len(distances) - stats.rankdata(distances, method="ordinal")
        select = orders < batch_size
        x_new = new_points[select, :]

        if sum(select) < size:
            return np.concatenate(
                (iterative_sampler(np.concatenate((x_input, x_new)),
                                   size - batch_size), x_new))
        return x_new


def scipy_lhs_sampler(size: int = 100,
                      x_limits: np.ndarray = None,
                      dim: int = None):
    from scipy.stats import qmc
    if x_limits is None:
        if dim is None:
            dim = 1
        x_limits = np.array([[0, 1]]*dim)

    dim = len(x_limits) if dim is None else dim
    l_bounds = x_limits[:, 0]
    u_bounds = x_limits[:, 1]
    sampler = qmc.LatinHypercube(d=dim)
    sample = sampler.random(size)
    sample = qmc.scale(sample, l_bounds, u_bounds)
    return sample
