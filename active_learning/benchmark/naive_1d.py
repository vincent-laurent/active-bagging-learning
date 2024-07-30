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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from active_learning import ActiveSurfaceLearner
from active_learning.benchmark import utils
from active_learning.benchmark.base import ServiceTestingClassAL, ModuleExperiment, ServiceTestingClassModAL
from active_learning.components.active_criterion import GaussianProcessVariance
from active_learning.components.active_criterion import VarianceCriterion
from active_learning.components.query_strategies import ServiceQueryVariancePDF, ServiceUniform

try:
    plt.style.use("./.matplotlibrc")
except (ValueError, OSError):
    pass
bounds = [[0, 1]]


def gp_regression_std(regressor, X):
    _, std = regressor.predict(X, return_std=True)
    query_idx = np.argmax(std)
    return query_idx, X[query_idx]


def unknown_function(x):
    return x ** 5 * np.sin(10 * np.pi * x)


def sampler(n):
    x0 = np.random.uniform(*bounds[0], size=n)
    return pd.DataFrame(x0)


kernel = 1 * RBF(0.2, length_scale_bounds=(5e-2, 1e1))
krg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
n0 = 10
budget = 25
steps = 15

# Setup learners
# ==============

learner_bagging = ActiveSurfaceLearner(
    active_criterion=VarianceCriterion(
        krg, splitter=sklearn.model_selection.ShuffleSplit(n_splits=3, train_size=0.85)),
    query_strategy=ServiceQueryVariancePDF(bounds, num_eval=2000),
    bounds=bounds

)
learner_gaussian = ActiveSurfaceLearner(
    active_criterion=GaussianProcessVariance(kernel=kernel),
    query_strategy=ServiceQueryVariancePDF(bounds, num_eval=2000),
    bounds=bounds)

learner_uniform = ActiveSurfaceLearner(
    active_criterion=GaussianProcessVariance(kernel=kernel),
    query_strategy=ServiceUniform(bounds),
    bounds=bounds)

modal_learner = ActiveLearner(
    estimator=krg,
    query_strategy=gp_regression_std,
)

# Setup testing procedure
# =======================

testing_bootstrap = ServiceTestingClassAL(
    function=unknown_function,
    budget=budget,
    name="bagging uncertainty",
    budget_0=n0, learner=learner_bagging,
    x_sampler=sampler, n_steps=steps, bounds=bounds

)
testing_gaussian = ServiceTestingClassAL(
    function=unknown_function,
    budget=budget,
    name="gaussian uncertainty",
    budget_0=n0, learner=learner_gaussian,
    x_sampler=sampler, n_steps=steps, bounds=bounds

)

testing_modal = ServiceTestingClassModAL(
    function=unknown_function,
    budget=budget,
    name="gaussian uncertainty (modal)",
    budget_0=n0, learner=modal_learner,
    x_sampler=sampler, n_steps=steps, bounds=bounds
)

testing_uniform = ServiceTestingClassAL(
    function=unknown_function,
    budget=budget,
    name="uniform (passive)",
    budget_0=n0, learner=learner_uniform,
    x_sampler=sampler, n_steps=steps, bounds=bounds
)


def make_1d_example(save=False):
    testing_bootstrap.run()

    utils.plot_iterations_1d(testing_bootstrap, iteration_max=4, color="C0")
    if save:
        plt.savefig(".public/example_krg_bootsrap")

    utils.plot_iterations_1d(testing_bootstrap, iteration_max=6, color="C0")
    if save:
        plt.savefig(".public/example_krg")

    testing_gaussian.run()
    utils.plot_iterations_1d(testing_gaussian, iteration_max=4, color="C1")

    if save:
        plt.savefig(".public/example_krg_gaussian")

    testing_modal.run()
    utils.plot_iterations_1d(testing_modal, iteration_max=4, color="C2")

    if save:
        plt.savefig(".public/example_krg_modal")

    err1 = pd.DataFrame(testing_bootstrap.result).T[["budget", "l2"]]
    err2 = pd.DataFrame(testing_gaussian.result).T[["budget", "l2"]]
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
        deepcopy(testing_gaussian),
        deepcopy(testing_bootstrap),
        deepcopy(testing_modal),
        deepcopy(testing_uniform)

    ], 100)
    experiment.run()
    utils.write_benchmark(data=experiment.cv_result_, path="data/1D_gaussian_vector.csv", update=False)


if __name__ == '__main__':
    import seaborn as sns

    # experiment_1d()
    data = utils.read_benchmark("data/1D_gaussian_vector.csv")
    plt.figure(dpi=300, figsize=(5, 5))
    sns.lineplot(data=data, x="num_sample", hue="name", y="L2-norm", ax=plt.gca())
    plt.xlabel("Sample size")
    plt.ylabel("$L_2$ error")
    plt.tight_layout()
    plt.savefig(".public/example_1D.png")

    make_1d_example(True)
