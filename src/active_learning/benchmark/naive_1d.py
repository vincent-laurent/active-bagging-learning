#!/usr/bin/env python
# coding: utf-8
"""
ACTIVE LEARNING FOR A SIMPLE 1D FUNCTION

This example is a simple application of active learning.
The 1D function allows for easy vizualisation of the model and its error
estimate.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from active_learning.components.active_criterion import GaussianProcessVariance, \
    VarianceBis
from active_learning.components.query_strategies import QueryVariancePDF
from active_learning.components.test import TestingClass, plot_iter

SEED = 1234
RNG = np.random.default_rng(seed=SEED)

bounds = [[0, 1]]


def unknown_function(x):
    return x ** 5 * np.sin(10 * np.pi * x)


def sampler(n):
    x0 = np.random.uniform(*bounds[0], size=n)
    return pd.DataFrame(x0)


kernel = 1 * RBF(0.01)
krg = GaussianProcessRegressor(kernel=kernel, )

# ======================================================================================
#
#                           Gaussian
# ======================================================================================
n0 = 10
budget = 30
steps = 8
plt.style.use("bmh")
plt.rcParams["font.family"] = "ubuntu"
plt.rcParams['axes.facecolor'] = "white"

testing_bootstrap = TestingClass(
    budget,
    n0,
    unknown_function,
    VarianceBis(krg, splitter=sklearn.model_selection.ShuffleSplit(n_splits=5,
                                                                   train_size=0.8)),
    QueryVariancePDF(bounds, num_eval=2000),
    sampler, steps, bounds=bounds

)

testing_bootstrap.run()

plot_iter(testing_bootstrap)
plt.tight_layout()
plt.show()

plt.savefig("public//example_krg.png", dpi=100)

testing = TestingClass(
    budget,
    n0,
    unknown_function,
    GaussianProcessVariance(kernel=kernel),
    QueryVariancePDF(bounds, num_eval=2000),
    sampler, steps, bounds=bounds)

testing.run()

plot_iter(testing)
# plt.figure(figsize=(4, 4), dpi=200)
# testing.plot_error_vs_criterion_pointwise(num_mc=100)
plt.tight_layout()
plt.show()

err1 = pd.DataFrame(testing_bootstrap.metric)
err2 = pd.DataFrame(testing.metric)
budgets = pd.DataFrame(testing.learner.result).loc["budget"].astype(float)

plt.figure()
plt.plot(budgets, err1.values, c="r", label="bootstrap")
plt.plot(budgets, err2.values, label="regular")
plt.legend()
