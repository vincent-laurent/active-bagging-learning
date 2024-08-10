# Copyright 2024 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import pandas as pd
from scipy.stats import qmc


def get_variance_function(estimator_list):
    def meta_estimator(*args, **kwargs):
        predictions = np.array(
            [est.predict(*args, **kwargs) for est in estimator_list])
        return np.std(predictions, axis=0, ddof=1)

    return meta_estimator


def scipy_lhs_sampler(size: int = 100,
                      x_limits: np.ndarray = None,
                      dim: int = None):
    if (x_limits is None) or x_limits.shape == ():
        raise ValueError("bounds must be specified for LHS sampler")
    dim = len(np.array(x_limits)) if dim is None else dim
    l_bounds = x_limits[:, 0]
    u_bounds = x_limits[:, 1]
    sampler = qmc.LatinHypercube(d=dim)
    sample = sampler.random(size)
    sample = qmc.scale(sample, l_bounds, u_bounds)
    return sample


DEFAULT_RNG = np.random.default_rng()


def indices_of_random_sampling_in_finite_set(pdf, candidates, nb_samples, *,
                                             rng=DEFAULT_RNG):
    """Pick `nb_samples` items among the candidates according
    to the probability density function `pdf`."""
    probability = pdf(candidates)
    probability /= np.sum(probability)
    indices = pd.DataFrame(candidates).index
    return rng.choice(indices, size=nb_samples, replace=False, p=probability)


def random_sampling_in_finite_set(pdf, candidates, nb_samples, *,
                                  rng=DEFAULT_RNG):
    """Pick `nb_samples` items among the candidates according to
     the probability density function `pdf`."""
    probability = pdf(candidates)
    probability /= np.sum(probability)
    return rng.choice(candidates, size=nb_samples, replace=False, p=probability,
                      axis=0)


def random_sampling_in_domain(pdf, bounds, nb_samples, *,
                              candidates_per_sample=50, rng=DEFAULT_RNG):
    """ Pick `nb_samples` items in the domain delimited by `bounds` according to
    the probability density function `pdf`."""
    dimension = len(bounds[0])
    candidates = (bounds[1] - bounds[0]) * rng.random(
        (candidates_per_sample * nb_samples, dimension)) + bounds[0]
    return random_sampling_in_finite_set(pdf, candidates, nb_samples, rng=rng)


def random_query(X, y, active_function, size=10, batch_size=10, bounds=None,
                 **args):
    """ Wrap the above function with the same API as
    the other query strategies."""
    return random_sampling_in_domain(active_function, bounds, size,
                                     candidates_per_sample=batch_size // size)
