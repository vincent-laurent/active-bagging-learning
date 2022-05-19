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


if __name__ == '__main__':
    import matplotlib.pyplot as plot

    c = plot.get_cmap("magma")
    plot.figure(figsize=(6, 6), dpi=200)
    x0 = iterative_sampler(dim=2)
    n_steps = 20
    x = x0
    plot.scatter(x0[:, 0], x0[:, 1], color=c(0), label=f"Iter n°{0}")
    for i in range(n_steps):
        color = c((i + 1) / n_steps)
        x_new = iterative_sampler(x, size=10)
        print(len(x_new))
        if i % 2 == 0:
            plot.scatter(x_new[:, 0], x_new[:, 1], color=color,
                         label=f"Iter n°{i + 1}")
        plot.scatter(x_new[:, 0], x_new[:, 1], color=color)
        x = np.concatenate((x_new, x))
    plot.grid()
    legend = plot.legend(facecolor='white', framealpha=1)

    plot.figure(figsize=(6, 6), dpi=200)
    plot.scatter(x[:, 0], x[:, 1], color="k")
    plot.figure(figsize=(6, 6), dpi=200)

    x_once = iterative_sampler(dim=200, size=len(x))
    plot.scatter(x_once[:, 0], x_once[:, 1], color="k")

    x1d = one_d_iterative_sampler(x_limits=[[0, 1]], size=10)
    x1d_new = one_d_iterative_sampler(x_input=x1d, size=10)

    plot.figure()
    plot.scatter(x1d, np.random.random(size=len(x1d)))
    plot.scatter(x1d_new, np.random.random(size=len(x1d_new)))
