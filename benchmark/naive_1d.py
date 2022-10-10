#!/usr/bin/env python
# coding: utf-8
"""
ACTIVE LEARNING FOR A SIMPLE 1D FUNCTION

This example is a simple application of active learning.
The 1D function allows for easy vizualisation of the model and its error
estimate.
"""
import matplotlib
import sklearn.model_selection
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

from active_learning.components.active_criterion import VarianceEnsembleMethod, Variance
from active_learning.components.query_strategies import QueryVariancePDF
from active_learning.components.sampling import latin_square
from active_learning.components.test import TestingClass
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
matplotlib.use("Qt5Agg")
SEED = 1234
RNG = np.random.default_rng(seed=SEED)

bounds = [[0, 1]]
domain = np.linspace(0, 1, 100)


def analyse_1d(test: TestingClass):
    plt.figure()
    plt.scatter(test.active_object.x_input, unknown_function(testing.active_object.x_input),
                c=test.active_object.x_input.index, cmap="rainbow")

    plt.figure()
    sns.histplot(test.active_object.x_input, bins=50)

    iter_ = int(test.active_object.x_input.index.max())
    fig, axs = plt.subplots(4, iter_//4, sharey=True, sharex=True)

    for iter, ax in enumerate(axs.ravel()):
        error = test.active_object.active_criterion(domain.reshape(-1, 1))
        prediction = test.active_object.active_criterion.function(domain.reshape(-1, 1))
        ax.plot(domain, prediction, color="C1")
        ax.plot(domain, unknown_function(domain), color="grey", linestyle="--")
        ax.fill_between(domain.ravel(), prediction - error / 2, prediction + error / 2, color="C1", alpha=0.5)
        training_dataset = test.active_object.x_input.loc[:iter]
        new_samples = test.active_object.x_input.loc[iter]
        ax.scatter(training_dataset, unknown_function(training_dataset), color="C0")

        if iter > 0: ax.scatter(new_samples, unknown_function(new_samples), color="C2")
        ax.set_title("iter={}".format(iter))
        ax.set_ylim(-1, 1)


def unknown_function(x):
    return x ** 5 * np.sin(10 * np.pi * x)


def sampler(n):
    x0 = np.random.uniform(*bounds[0], size=n)
    return pd.DataFrame(x0)


xtra_trees = ExtraTreesRegressor(max_features=0.5, bootstrap=True, n_estimators=50, max_samples=0.8)
spline_fitting = make_pipeline(PolynomialFeatures(100, include_bias=True), Ridge(alpha=1e-3))
krg = GaussianProcessRegressor(alpha=0.1)

testing = TestingClass(
    50,
    10,
    unknown_function,
    VarianceEnsembleMethod(base_ensemble=xtra_trees),
    QueryVariancePDF(bounds, num_eval=500),
    sampler, 10, bounds=bounds

)

testing.run()
analyse_1d(testing)

testing = TestingClass(
    1000,
    10,
    unknown_function,
    Variance(base_estimator=spline_fitting,
             splitter=sklearn.model_selection.ShuffleSplit(n_splits=2)),
    QueryVariancePDF(bounds, num_eval=500),
    sampler, 10, bounds=bounds

)

testing.run()
analyse_1d(testing)
