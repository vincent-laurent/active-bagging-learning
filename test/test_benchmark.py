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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from active_learning import ActiveSurfaceLearner
from active_learning.benchmark import functions, naive_1d
from active_learning.benchmark.base import TestingClass
from active_learning.components.active_criterion import VarianceCriterion
from active_learning.components.query_strategies import ServiceQueryVariancePDF

x2d = np.array([0, 0]).reshape(1, -1)


def test_call_benchmark_function():
    functions.grammacy_lee_2009(x2d)
    functions.marelli_2018(x2d)
    functions.grammacy_lee_2009_rand(x2d)
    functions.annie_sauer_2021(x2d)
    functions.branin(x2d)
    functions.branin_rand(x2d)
    functions.himmelblau(x2d)
    functions.himmelblau_rand(x2d)
    functions.golden_price(x2d)
    functions.golden_price_rand(x2d)
    functions.synthetic_2d_1(x2d)
    functions.synthetic_2d_2(x2d)


def test_plot_benchmark():
    functions.plot_benchamrk_functions()


def test_benchmark_1d():
    naive_1d.make_1d_example()

    bounds = [[0, 1]]

    def unknown_function(x):
        return x ** 5 * np.sin(10 * np.pi * x)

    def sampler(n):
        x0 = np.random.uniform(*bounds[0], size=n)
        return pd.DataFrame(x0)

    kernel = 1 * RBF(0.01)
    krg = GaussianProcessRegressor(kernel=kernel)

    # ======================================================================================
    #
    #                           Gaussian
    # ======================================================================================
    n0 = 10
    budget = 20
    steps = 8
    plt.style.use("bmh")
    plt.rcParams["font.family"] = "ubuntu"
    plt.rcParams['axes.facecolor'] = "white"

    learner_bagging = ActiveSurfaceLearner(
        active_criterion=VarianceCriterion(
            krg, splitter=sklearn.model_selection.ShuffleSplit(
                n_splits=2,
                train_size=0.8)),
        query_strategy=ServiceQueryVariancePDF(bounds, num_eval=2000),
        bounds=bounds

    )

    testing_bootstrap = TestingClass(
        function=unknown_function,
        budget=budget,
        budget_0=n0, learner=learner_bagging,
        x_sampler=sampler, n_steps=steps, bounds=bounds

    )
    testing_bootstrap.run()
    r1 = testing_bootstrap.result[0]["learner"].surface(np.array([0]))
    r2 = testing_bootstrap.result[3]["learner"].surface(np.array([0]))
    assert r1 != r2
    r1 = testing_bootstrap.result[0]["learner"].active_criterion(np.array([0]))
    r2 = testing_bootstrap.result[3]["learner"].active_criterion(np.array([0]))
    assert r1 != r2
