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
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import BaseEnsemble
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import ShuffleSplit

from active_learning.components import utils


class IActiveCriterion(ABC, RegressorMixin):
    def __init__(self, *args):
        super().__init__(*args)

    @abstractmethod
    def __call__(self, X: pd.DataFrame, *args, **kwargs):
        ...

    def criterion(self, X: pd.DataFrame, *args, **kwargs):
        return self(X, *args, **kwargs)

    @abstractmethod
    def fit(self, X, y):
        ...

    @abstractmethod
    def function(self, X):
        ...

    def __add__(self, other):
        # TODO
        ...

    def reshape_x(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        return X


class VarianceCriterion(IActiveCriterion):
    def __init__(self,
                 estimator,
                 splitter: Union[BaseCrossValidator, ShuffleSplit]):
        super().__init__()
        self.models = []
        self.splitter = splitter
        self.estimator = estimator

    def __call__(self, X, *args, **kwargs):
        X = self.reshape_x(X)
        return utils.get_variance_function(self.models)(X)

    def fit(self, X, y):
        self.models = []
        X, y = np.array(X), np.array(y).ravel()
        for train, test in self.splitter.split(X, y):
            model = self.estimator.__sklearn_clone__()
            self.models.append(model.fit(X[train, :], y[train]))

    def function(self, X):
        res = 0
        X = self.reshape_x(X)
        for model_ in self.models:
            res += model_.predict(X)
        return res / len(self.models)


class VarianceEnsembleMethod(IActiveCriterion):
    def __init__(self,
                 estimator: BaseEnsemble,
                 ):
        super().__init__()
        self.estimator = estimator

    def function(self, X):
        X = self.reshape_x(X)
        return self.model_.predict(X)

    def __call__(self, X, *args, **kwargs):
        X = self.reshape_x(X)
        ret = utils.get_variance_function(self.model_.estimators_)(X)

        return np.where(ret < 0, 0, ret)

    def fit(self, X, y):
        X, y = np.array(X), np.array(y).ravel()
        self.model_ = self.estimator.__sklearn_clone__()
        self.model_.fit(X, y)


class GaussianProcessVariance(IActiveCriterion):
    def __init__(self,
                 kernel,
                 ):
        super().__init__()
        self.kernel = kernel
        self.estimator = GaussianProcessRegressor(kernel=kernel)

    def __call__(self, X, *args, **kwargs):
        ret = self.model_.predict(X, return_std=True)[1]
        return np.where(ret < 0, 0, ret)

    def function(self, X):
        X = self.reshape_x(X)
        return self.model_.predict(X)

    def fit(self, X, y):
        X, y = np.array(X), np.array(y).ravel()
        self.model_ = self.estimator.__sklearn_clone__()
        self.model_.fit(X, y)
