from abc import ABCMeta

from scipy import optimize

from sampling.latin_square import iterative_sampler


class QueryStrategy(ABCMeta):

    def query(self):
        pass


def find_max(X, y, active_function, size=1, batch_size=500,
             bounds=None, **args):
    def fun(*args, **kwargs):
        return - active_function(*args, **kwargs)

    res = optimize.fmin(fun, x0=np.median(X, axis=0),
                        xtol=0.01, maxiter=40, disp=0)
    return res


def reject_on_bounds(X, y, active_function, size=10, batch_size=50,
                     bounds=None, **args):
    from scipy.stats import rankdata
    if bounds is None:
        x_new = iterative_sampler(X, size=batch_size)
    else:
        x_new = iterative_sampler(x_limits=bounds, size=batch_size)
    cov = active_function(x_new)
    order = rankdata(-cov, method="ordinal")
    selector = order <= size
    return x_new[selector], cov[selector]


def reject_on_bounds_ada(X, y, active_function, size=10,
                         bounds=None, alpha=2, **args):
    batch_size = min(int(len(X) * alpha), 5 * size)
    from scipy.stats import rankdata
    if bounds is None:
        x_new = iterative_sampler(X, size=batch_size)
    else:
        x_new = iterative_sampler(x_limits=bounds, size=batch_size)
    cov = active_function(x_new)
    order = rankdata(-cov, method="ordinal")
    selector = order <= size
    return x_new[selector], cov[selector]


if __name__ == '__main__':
    import numpy as np


    def test_function(x): return -np.sum(np.abs(x)) ** 2


    find_max(np.array([[0, 1], [1, 1]]), None, test_function)
