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

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.model_selection
from modAL.models import ActiveLearner
from modAL.acquisition import max_EI
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from active_learning import ActiveSurfaceLearner
from active_learning.benchmark.base import ServiceTestingClassAL, ModuleExperiment, ServiceTestingClassModAL
from active_learning.benchmark.utils import plot_iterations_1d
from active_learning.components.active_criterion import GaussianProcessVariance
from active_learning.components.active_criterion import VarianceCriterion
from active_learning.components.query_strategies import ServiceQueryVariancePDF

RNG = np.random.default_rng(seed=0)

bounds = [[0, 1]]

def GP_regression_std(regressor, X):
    _, std = regressor.predict(X, return_std=True)
    query_idx = np.argmax(std)
    return query_idx, X[query_idx]


def unknown_function(x):
    return x ** 5 * np.sin(10 * np.pi * x)


def sampler(n):
    x0 = np.random.uniform(*bounds[0], size=n)
    return pd.DataFrame(x0)


kernel = 1 * RBF(0.1)
krg = GaussianProcessRegressor(kernel=kernel)
n0 = 10
budget = 20
steps = 10
plt.style.use("bmh")
plt.rcParams['axes.facecolor'] = "white"

# Setup learners
# ==============

learner_bagging = ActiveSurfaceLearner(
    active_criterion=VarianceCriterion(
        krg, splitter=sklearn.model_selection.ShuffleSplit(
            n_splits=5,
            train_size=0.9)),
    query_strategy=ServiceQueryVariancePDF(bounds, num_eval=2000),
    bounds=bounds

)
learner_gaussian = ActiveSurfaceLearner(
    active_criterion=GaussianProcessVariance(kernel=kernel),
    query_strategy=ServiceQueryVariancePDF(bounds, num_eval=2000),
    bounds=bounds)

modal_learner = ActiveLearner(
    estimator=krg,
    query_strategy=GP_regression_std,
)

# Setup testing procedure
# =======================

testing_bootstrap = ServiceTestingClassAL(
    function=unknown_function,
    budget=budget,
    budget_0=n0, learner=learner_bagging,
    x_sampler=sampler, n_steps=steps, bounds=bounds

)
testing = ServiceTestingClassAL(
    function=unknown_function,
    budget=budget,
    budget_0=n0, learner=learner_gaussian,
    x_sampler=sampler, n_steps=steps, bounds=bounds

)

testing_modal = ServiceTestingClassModAL(
    function=unknown_function,
    budget=budget,
    budget_0=n0, learner=modal_learner,
    x_sampler=sampler, n_steps=steps, bounds=bounds
)


def make_1d_example(save=False):
    testing_bootstrap.run()

    plot_iterations_1d(testing_bootstrap)
    plt.tight_layout()
    if save:
        plt.savefig(".public//example_krg.png", dpi=300)

    testing.run()
    plot_iterations_1d(testing)

    testing_modal.run()

    plt.tight_layout()
    if save:
        plt.savefig(".public/example_krg_2")

    err1 = pd.DataFrame(testing_bootstrap.result).T[["budget", "l2"]]
    err2 = pd.DataFrame(testing.result).T[["budget", "l2"]]
    err3 = pd.DataFrame(testing_modal.result).T[["budget", "l2"]]

    if save:
        plt.figure(dpi=300)
        plt.plot(err1["budget"], err1["l2"], c="C0", label="bootstrap")
        plt.plot(err2["budget"], err2["l2"], c="C1", label="regular")
        plt.plot(err3["budget"], err3["l2"], c="C2", label="modAL")
        plt.legend()
        plt.savefig(".public/example_krg_3")


def experiment_1d():
    experiment = ModuleExperiment([
        deepcopy(testing),
        deepcopy(testing_bootstrap),
        deepcopy(testing_modal)

    ], 2)
    experiment.run()


if __name__ == '__main__':
    make_1d_example(save=True)
