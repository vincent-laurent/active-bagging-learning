from abc import ABCMeta

import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import ShuffleSplit


class ActiveCriterion(ABCMeta):
    pass


def get_variance_function(estimator_list):
    keys = range(len(estimator_list))

    def meta_estimator(*args, **kwargs):
        m = estimator_list[0].predict(*args, **kwargs)
        s = m ** 2
        for key in keys[1:]:
            elt = estimator_list[key].predict(*args, **kwargs)
            m += elt
            s += elt ** 2
        return (s / len(keys)) - (m / len(keys)) ** 2

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
