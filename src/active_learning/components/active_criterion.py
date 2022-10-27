from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.ensemble import BaseEnsemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import ShuffleSplit


class ActiveCriterion(ABC, BaseEstimator):
    def __init__(self, *args):
        super().__init__(*args)

    @abstractmethod
    def __call__(self, X: pd.DataFrame, *args, **kwargs):
        ...

    @abstractmethod
    def function(self, X: pd.DataFrame):
        ...

    def criterion(self, X: pd.DataFrame, *args, **kwargs):
        return self(X, *args, **kwargs)

    @abstractmethod
    def fit(self, X, y):
        ...


class Variance(ActiveCriterion):
    def __init__(self,
                 estimator: BaseEstimator,
                 splitter: Union[BaseCrossValidator, ShuffleSplit]):
        super().__init__()
        self.models = []
        self.splitter = splitter
        self.estimator = estimator

    def function(self, X):
        res = 0
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        for model_ in self.models:
            res += model_.predict(X)
        return res / len(self.models)

    def __call__(self, X, *args, **kwargs):
        ret = get_variance_function(self.models)(X)
        return np.where(ret < 0, 0, ret)

    def fit(self, X, y):
        self.models = []
        X, y = np.array(X), np.array(y).ravel()
        for train, test in self.splitter.split(X, y):
            model = clone(self.estimator)
            self.models.append(model.fit(X[train, :], y[train]))


class VarianceBis(ActiveCriterion):
    def __init__(self,
                 estimator: BaseEstimator,
                 splitter: Union[BaseCrossValidator, ShuffleSplit]):
        super().__init__()
        self.models = []
        self.splitter = splitter
        self.estimator = estimator

    def __call__(self, X, *args, **kwargs):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        ret = get_variance_function(self.models)(X)
        return np.where(ret < 0, 0, ret)

    def function(self, X):

        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        ret = self.final_estimator.predict(X)
        return ret

    def fit(self, X, y):
        self.models = []
        X, y = np.array(X), np.array(y).ravel()
        for train, test in self.splitter.split(X, y):
            model = clone(self.estimator)
            self.models.append(model.fit(X[train, :], y[train]))
        self.final_estimator = self.estimator.fit(X, y)


class VarianceEnsembleMethod(ActiveCriterion):
    def __init__(self,
                 estimator: BaseEnsemble,
                 ):
        super().__init__()
        self.estimator = estimator

    def function(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        return self.model_.predict(X)

    def __call__(self, X, *args, **kwargs):
        ret = get_variance_function(self.model_.estimators_)(X)
        return np.where(ret < 0, 0, ret)

    def fit(self, X, y):
        X, y = np.array(X), np.array(y).ravel()
        self.model_ = clone(self.estimator)
        self.model_.fit(X, y)


class GaussianProcessVariance(ActiveCriterion):
    def __init__(self,
                 kernel,
                 ):
        super().__init__()
        self.kernel = kernel
        self.estimator = GaussianProcessRegressor(kernel=kernel)

    def function(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        return self.model_.predict(X)

    def __call__(self, X, *args, **kwargs):
        ret = self.model_.predict(X, return_std=True)[1]
        return np.where(ret < 0, 0, ret)

    def fit(self, X, y):
        X, y = np.array(X), np.array(y).ravel()
        self.model_ = clone(self.estimator)
        self.model_.fit(X, y)


# =====================================================================================
#                           TOOLS
# =====================================================================================
def get_variance_function(estimator_list):
    def meta_estimator(*args, **kwargs):
        predictions = np.array([est.predict(*args, **kwargs) for est in estimator_list])
        return np.std(predictions, axis=0)

    return meta_estimator


def estimate_variance(X, y,
                      base_estimator: BaseEstimator = RandomForestRegressor(),
                      splitter: BaseCrossValidator = ShuffleSplit(n_splits=5)):
    list_models = []
    X, y = np.array(X), np.array(y).ravel()
    for train, test in splitter.split(X, y):
        model = clone(base_estimator)
        list_models.append(model.fit(X[train, :], y[train]))

    def mean_predictor(x):
        res = 0
        for model_ in list_models:
            res += model_.predict(x)
        return res / len(list_models)

    return mean_predictor, get_variance_function(list_models)


def gaussian_est(X, y, return_coverage=True):
    from sklearn.gaussian_process import GaussianProcessRegressor
    model = GaussianProcessRegressor().fit(X, y)
    if return_coverage:
        def coverage_function(x):
            return model.predict(x, return_std=True)[1]

        return model.predict, coverage_function
    else:
        return model.predict
