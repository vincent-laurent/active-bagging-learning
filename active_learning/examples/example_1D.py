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
from active_learning.benchmark.base import ServiceTestingClassAL, \
    ModuleExperiment, ServiceTestingClassModAL
from active_learning.components.active_criterion import GaussianProcessVariance
from active_learning.components.active_criterion import VarianceCriterion
from active_learning.components.query_strategies import ServiceQueryVariancePDF, \
    ServiceUniform

bounds = np.array([[0, 1]])


def gp_regression_std(regressor, X):
    _, std = regressor.predict(X, return_std=True)
    query_idx = np.argmax(std)
    return query_idx, X[query_idx]


def unknown_function(x):
    return x ** 5 * np.sin(10 * np.pi * x)




kernel = 1 * RBF(0.1)
krg = GaussianProcessRegressor(kernel=kernel)
n0 = 10
budget = 21
steps = 10

# Setup learners
# ==============

learner_bagging = ActiveSurfaceLearner(
    active_criterion=VarianceCriterion(
        krg, splitter=sklearn.model_selection.ShuffleSplit(
            n_splits=3,
            train_size=0.85)),
    query_strategy=ServiceQueryVariancePDF(bounds),
    bounds=bounds

)
learner_gaussian = ActiveSurfaceLearner(
    active_criterion=GaussianProcessVariance(kernel=kernel),
    query_strategy=ServiceQueryVariancePDF(bounds),
    bounds=bounds)

learner_uniform = ActiveSurfaceLearner(
    active_criterion=GaussianProcessVariance(kernel=kernel),
    query_strategy=ServiceUniform(bounds),
    bounds=bounds)

learner_bagging_uniform = ActiveSurfaceLearner(
    active_criterion=GaussianProcessVariance(kernel=kernel),
    query_strategy=3 * ServiceQueryVariancePDF(bounds) + ServiceUniform(bounds),
    bounds=bounds)

modal_learner = ActiveLearner(
    estimator=krg,
    query_strategy=gp_regression_std,
)

# Setup testing procedure
# =======================
common_args = dict(function=unknown_function,
                   budget=budget,
                   budget_0=n0,
                   x_sampler=utils.Sampler(bounds), 
                   n_steps=steps, 
                   bounds=bounds)

testing_bootstrap = ServiceTestingClassAL(
    name="bagging uncertainty",
    learner=learner_bagging,
    **common_args
)

testing_bootstrap_uniform = ServiceTestingClassAL(
    name="bagging uncertainty + uniform",
    learner=learner_bagging_uniform,
    **common_args

)

testing_gaussian = ServiceTestingClassAL(
    name="gaussian uncertainty",
    learner=learner_gaussian,
    **common_args

)

testing_modal = ServiceTestingClassModAL(
    name="gaussian uncertainty (modal)",
    learner=modal_learner,
    **common_args
)

testing_uniform = ServiceTestingClassAL(
    name="uniform (passive)",
    learner=learner_uniform,
    **common_args
)


def make_1d_example(save=False):
    testing_bootstrap.run()

    utils.plot_iterations_1d(testing_bootstrap, iteration_max=4, color="C0")
    if save:
        plt.savefig(".public/example_krg_bootsrap")

    utils.plot_iterations_1d(testing_bootstrap, iteration_max=6, color="C3")
    if save:
        plt.savefig(".public/example_krg")

    testing_gaussian.run()
    utils.plot_iterations_1d(testing_gaussian, iteration_max=4, color="C1")

    if save:
        plt.savefig(".public/example_krg_gaussian")

    testing_modal.run()

    if save:
        plt.savefig(".public/example_krg_modal")

    utils.plot_iterations_1d(testing_modal, iteration_max=4, color="C5")

    testing_bootstrap_uniform.run()
    if save:
        plt.savefig(".public/example_krg_boot_plus_uniform")

    utils.plot_iterations_1d(testing_bootstrap_uniform, iteration_max=4,
                             color="C2")

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
        deepcopy(testing_uniform),
        deepcopy(testing_bootstrap_uniform)], 100)
    experiment.run()
    utils.write_benchmark(data=experiment.cv_result_,
                          path="data/2024_1D_benchmark.csv", update=False)


if __name__ == '__main__':
    import seaborn as sns
    # experiment_1d()
    
    sns_palette = sns.color_palette(
        ['C0', "C5", "C4", "grey", "C3", "C1", "C5", "C3", "C1"], as_cmap=True)
    
    
    data = utils.read_benchmark("data/2024_1D_benchmark.csv")
    plt.figure(dpi=300, figsize=(5, 5))
    sns.lineplot(data=data, x="num_sample", hue="name", y="L2-norm",
                 ax=plt.gca(), palette=sns_palette)
    plt.xlabel("Sample size")
    plt.ylabel("$L_2$ error")
    plt.tight_layout()
    
    handles, labels = plt.gca().get_legend_handles_labels()

    # Sort labels and handles
    sorted_labels_handles = sorted(zip(labels, handles), key=lambda x: x[0])
    sorted_labels, sorted_handles = zip(*sorted_labels_handles)

    # Pass sorted handles and labels to legend
    plt.legend(sorted_handles, sorted_labels)

    plt.savefig(".public/example_1D.png")

    make_1d_example(True)
