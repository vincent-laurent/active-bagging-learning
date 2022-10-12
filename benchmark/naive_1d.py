#!/usr/bin/env python
# coding: utf-8
"""
ACTIVE LEARNING FOR A SIMPLE 1D FUNCTION

This example is a simple application of active learning.
The 1D function allows for easy vizualisation of the model and its error
estimate.
"""
import matplotlib

try:
    matplotlib.use("qt5agg")
except ImportError:
    pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor

from active_learning.components.active_criterion import GaussianProcessVariance, VarianceBis
from active_learning.components.query_strategies import QueryVariancePDF
from active_learning.components.test import TestingClass, plot_iter

SEED = 1234
RNG = np.random.default_rng(seed=SEED)

bounds = [[0, 1]]


def unknown_function(x):
    return x ** 5 * np.sin(10 * np.pi * x)* np.sin(30 * np.pi * x)


def sampler(n):
    x0 = np.random.uniform(*bounds[0], size=n)
    return pd.DataFrame(x0)


kernel = 1 * RBF(0.01)
xtra_trees = ExtraTreesRegressor(bootstrap=False, n_estimators=50)
xtra_trees_b = ExtraTreesRegressor(bootstrap=True, n_estimators=50, max_samples=0.7)
spline_fitting = make_pipeline(PolynomialFeatures(100, include_bias=True), Ridge(alpha=1e-3))
krg = GaussianProcessRegressor(kernel=kernel)
mlp = MLPRegressor(hidden_layer_sizes=(2, 4), max_iter=3000)
#
# testing = TestingClass(
#     budget=500,
#     budget_0=10,
#     funcion=unknown_function,
#     active_criterion=VarianceEnsembleMethod(base_ensemble=xtra_trees_b),
#     query_strategy=QueryVariancePDF(bounds, num_eval=50000),
#     x_sampler=sampler,
#     n_steps=50,
#     bounds=bounds
#
# )
#
# testing.run()
# plt.figure(figsize=(4, 4), dpi=200)
# testing.plot_error_vs_criterion_pointwise(num_mc=100)
# analyse_1d(testing)

# ======================================================================================
#
#                           Gaussian
# ======================================================================================
n0 = 20
budget = 60
steps = 15

testing_bootstrap = TestingClass(
    budget,
    n0,
    unknown_function,
    VarianceBis(krg, splitter=sklearn.model_selection.ShuffleSplit(n_splits=3, train_size=0.6)),
    QueryVariancePDF(bounds, num_eval=200),
    sampler, steps, bounds=bounds

)

testing_bootstrap.run()

plot_iter(testing_bootstrap)
# plt.figure(figsize=(4, 4), dpi=200)
# testing_bootstrap.plot_error_vs_criterion_pointwise(num_mc=100)
plt.tight_layout()
plt.show()

testing = TestingClass(
    budget,
    n0,
    unknown_function,
    GaussianProcessVariance(kernel=kernel),
    QueryVariancePDF(bounds, num_eval=200),
    sampler, steps, bounds=bounds

)

testing.run()

plot_iter(testing)
# plt.figure(figsize=(4, 4), dpi=200)
# testing.plot_error_vs_criterion_pointwise(num_mc=100)
plt.tight_layout()
plt.show()

err1 = pd.DataFrame(testing_bootstrap.perf_active)
err2 = pd.DataFrame(testing.perf_active)
budgets = pd.DataFrame(testing.active_object.result).loc["budget"].astype(float)

plt.figure()
plt.plot(budgets, err1.values, c="r", label="bootstrap")
plt.plot(budgets, err2.values, label="regular")
plt.legend()


from active_learning.components.test import Experiment
n0 = 20
budget = 60
steps = 15

testing_bootstrap = TestingClass(
    budget, n0, unknown_function,
    VarianceBis(krg, splitter=sklearn.model_selection.ShuffleSplit(n_splits=3, train_size=0.6)),
    QueryVariancePDF(bounds, num_eval=200),
    sampler, steps, bounds=bounds

)
testing = TestingClass(
    budget, n0, unknown_function, GaussianProcessVariance(kernel=kernel),
    QueryVariancePDF(bounds, num_eval=200),
    sampler, steps, bounds=bounds

)

experiment = Experiment(testing_bootstrap, testing)
experiment.run(30)
experiment.plot_performance()

budgets = pd.DataFrame(experiment.results_1[0].active_object.result).loc["budget"].astype(float)
d1 = experiment.cv_result_1_
d2 = experiment.cv_result_2_
plt.figure()
plt.plot(budgets, d1.mean(axis=1), c="r", label="bootstrap")
plt.plot(budgets, d2.mean(axis=1), c="b", label="regular")
plt.fill_between(budgets, d1.mean(axis=1) - d1.std(axis=1), d1.mean(axis=1) + d1.std(axis=1), color="r", alpha=0.2)
plt.fill_between(budgets, d2.mean(axis=1) - d2.std(axis=1), d2.mean(axis=1) + d2.std(axis=1), color="b", alpha=0.2)
plt.legend()


budgets = pd.DataFrame(experiment.results_1[0].active_object.result).loc["budget"].astype(float)
d1 = experiment.cv_result_1_

plt.figure()
plt.plot(budgets, d1.mean(axis=1), c="r", label="bootstrap")
plt.plot(budgets, d2.mean(axis=1), c="b", label="regular")
plt.fill_between(budgets, d1.mean(axis=1) - d1.std(axis=1), d1.mean(axis=1) + d1.std(axis=1), color="r", alpha=0.2)
plt.fill_between(budgets, d2.mean(axis=1) - d2.std(axis=1), d2.mean(axis=1) + d2.std(axis=1), color="b", alpha=0.2)
plt.legend()

