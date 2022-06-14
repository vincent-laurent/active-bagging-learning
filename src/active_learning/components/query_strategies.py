from abc import ABCMeta

import numpy as np
import pandas as pd
from scipy import optimize

from active_learning.components.sampling.latin_square import iterative_sampler


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


##############################################
#  Random with probability density function  #
##############################################


DEFAULT_RNG = np.random.default_rng()

def indices_of_random_sampling_in_finite_set(pdf, candidates, nb_samples, *, rng=DEFAULT_RNG):
    """Pick `nb_samples` items among the candidates according to the probability density function `pdf`."""
    probability = pdf(candidates)
    probability /= np.sum(probability)
    indices = pd.DataFrame(candidates).index
    return rng.choice(indices, size=nb_samples, replace=False, p=probability)

def random_sampling_in_finite_set(pdf, candidates, nb_samples, *, rng=DEFAULT_RNG):
    """Pick `nb_samples` items among the candidates according to the probability density function `pdf`."""
    probability = pdf(candidates)
    probability /= np.sum(probability)
    return rng.choice(candidates, size=nb_samples, replace=False, p=probability, axis=0)

def random_sampling_in_domain(pdf, bounds, nb_samples, *, candidates_per_sample=50, rng=DEFAULT_RNG):
    """Pick `nb_samples` items in the domain delimited by `bounds` according to the probability density function `pdf`."""
    dimension = len(bounds[0])
    candidates = (bounds[1] - bounds[0]) * rng.random((candidates_per_sample * nb_samples, dimension)) + bounds[0]
    return random_sampling_in_finite_set(pdf, candidates, nb_samples, rng=rng)


def random_query(X, y, active_function, size=10, batch_size=10, bounds=None, **args):
    """Wrap the above function with the same API as the other query strategies."""
    return random_sampling_in_domain(active_function, bounds, size, candidates_per_sample=batch_size//size)


################

if __name__ == '__main__':
    import numpy as np


    def test_function(x): return -np.sum(np.abs(x)) ** 2


    find_max(np.array([[0, 1], [1, 1]]), None, test_function)
