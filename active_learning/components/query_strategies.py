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

from abc import ABC, abstractmethod

import numpy as np
from scipy import optimize
from scipy.stats import rankdata
from sklearn.base import BaseEstimator

from active_learning.components import utils


class IQueryStrategy(ABC, BaseEstimator):

    def __init__(self, bounds):
        super().__init__()
        self.__active_function = None
        self.__bounds = bounds

    def set_active_function(self, fun: callable):
        self.__active_function = fun

    def set_bounds(self, bounds):
        self.__bounds = bounds

    @abstractmethod
    def query(self, *args):
        pass

    @property
    def bounds(self):
        return self.__bounds

    @property
    def active_function(self):
        return self.__active_function


class ServiceQueryMax(IQueryStrategy):
    def __init__(self, x0, bounds=None, xtol=0.001, maxiter=40, disp=True):
        super().__init__(bounds)
        self._xtol = xtol
        self._maxiter = maxiter
        self._disp = disp
        self._x0 = x0.reshape(1, -1)

    def query(self, *args_) -> np.ndarray:
        def fun(*args, **kwargs):
            return - self.active_function(*args, **kwargs)

        res = optimize.minimize(
            fun, x0=self._x0.flatten(),
            bounds=self.bounds, options={'gtol': self._xtol, 'disp': self._disp}).x
        return res.reshape(self._x0.shape)


class ServiceQueryVariancePDF(IQueryStrategy):
    def __init__(self, bounds=None, num_eval: int = int(1e5)):
        super().__init__(bounds)
        self.num_eval = num_eval

        self.__rng = np.random.default_rng()

    def query(self, size):
        assert size > 0
        candidates = utils.scipy_lhs_sampler(x_limits=np.array(self.bounds), size=self.num_eval)
        probability = self.active_function(candidates)
        probability = probability + 1e-5
        probability /= np.sum(probability)
        return self.__rng.choice(candidates, size=size, replace=False, p=probability, axis=0)


class ServiceReject(IQueryStrategy):
    def __init__(self, bounds=None, num_eval: int = int(1e5)):
        super().__init__(bounds)

        self.num_eval = num_eval
        self.__rng = np.random.default_rng()

    def query(self, size):
        candidates = utils.scipy_lhs_sampler(x_limits=np.array(self.bounds), size=self.num_eval)
        af = self.active_function(candidates)
        order = rankdata(-af, method="ordinal")
        selector = order <= size
        return candidates[selector]


class ServiceUniform(IQueryStrategy):
    def __init__(self, bounds=None):
        super().__init__(bounds)

    def query(self, size):
        candidates = utils.scipy_lhs_sampler(x_limits=np.array(self.bounds), size=size)
        return candidates