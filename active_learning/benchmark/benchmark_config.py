from typing import List

import matplotlib.pyplot as plt
import numpy as np
from modAL.disagreement import max_std_sampling
from modAL.models import ActiveLearner
from sklearn import ensemble
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from active_learning import ActiveSurfaceLearner
from active_learning import query_strategies as qs
from active_learning.benchmark import functions
from active_learning.benchmark import utils
from active_learning.benchmark.base import (
    ServiceTestingClassAL,
    ServiceTestingClassModAL,
    ModuleExperiment,
)
from active_learning.components import active_criterion
from active_learning.components.active_criterion import VarianceCriterion

N_CANDIDATES = 2000

#
# def max_ucb(opt, x, ):
#     return max_UCB(opt, x, beta=1000)
#
#
# def max_ei(opt, x):
#     return max_EI(opt, x, beta=1000)


est_trees = ensemble.ExtraTreesRegressor(
    bootstrap=True, max_samples=0.9, n_estimators=20
)
est_trees = ensemble.RandomForestRegressor(
    bootstrap=True, max_samples=0.95, n_estimators=30, min_samples_leaf=3,
)
alc_trees = active_criterion.VarianceEnsembleMethod(estimator=est_trees)

est_svc = Pipeline(
    [("scale", StandardScaler()), ("est", SVR(degree=2, C=10, gamma=10))]
)
alc_svc = VarianceCriterion(
    estimator=est_svc, splitter=ShuffleSplit(n_splits=3, train_size=0.8)
)

linear_kernel = KernelRidge()

gp_regressor = GaussianProcessRegressor(kernel=kernels.RBF(length_scale=0.25))
gp_regressor_noisy = GaussianProcessRegressor(kernel=kernels.RBF(length_scale=0.25) + kernels.WhiteKernel())
# gp_ei_optimizer = BayesianOptimizer(estimator=regressor, query_strategy=max_EI)
# gp_ucb_optimizer = BayesianOptimizer(estimator=regressor, query_strategy=max_ucb)
modal_al = ActiveLearner(gp_regressor, max_std_sampling)
modal_al_noisy = ActiveLearner(gp_regressor_noisy, max_std_sampling)

regressor_svc = est_svc
# gp_ei_optimizer_svc = BayesianOptimizer(estimator=regressor_svc, query_strategy=max_EI)
# gp_ucb_optimizer_svc = BayesianOptimizer(estimator=regressor_svc, query_strategy=max_ucb)
modal_al_svc = ActiveLearner(regressor_svc, max_std_sampling)
# ======================================================================================================================
#                           ACTIVE LEARNERS
# ======================================================================================================================
# TREES
learner_trees = ActiveSurfaceLearner(
    active_criterion=alc_trees,
    query_strategy=qs.ServiceQueryVariancePDF(num_eval=N_CANDIDATES),
)

learner_uniform_trees = ActiveSurfaceLearner(
    active_criterion=alc_trees, query_strategy=qs.ServiceUniform(None)
)

learner_bagging_uniform_trees = ActiveSurfaceLearner(
    active_criterion=alc_trees,
    query_strategy=2 * qs.ServiceQueryVariancePDF(bounds=None) + qs.ServiceUniform(bounds=None),
)

# SVC
learner_svc = ActiveSurfaceLearner(
    active_criterion=alc_svc,
    query_strategy=qs.ServiceQueryVariancePDF(num_eval=N_CANDIDATES),
)

learner_uniform_svc = ActiveSurfaceLearner(
    active_criterion=alc_svc, query_strategy=qs.ServiceUniform(None), bounds=None
)

learner_bagging_uniform_svc = ActiveSurfaceLearner(
    active_criterion=alc_svc,
    query_strategy=2 * qs.ServiceQueryVariancePDF(bounds=None) + qs.ServiceUniform(bounds=None),
    bounds=None,
)

# GP
learner_gp = ActiveSurfaceLearner(
    active_criterion=VarianceCriterion(
        estimator=gp_regressor, splitter=ShuffleSplit(n_splits=3, train_size=0.8)),
    query_strategy=qs.ServiceQueryVariancePDF(num_eval=N_CANDIDATES),
)

learner_uniform_gp = ActiveSurfaceLearner(
    active_criterion=VarianceCriterion(
        estimator=gp_regressor,
        splitter=ShuffleSplit(n_splits=3, train_size=0.8)),
    query_strategy=qs.ServiceUniform(None)
)

learner_bagging_uniform_gp = ActiveSurfaceLearner(
    active_criterion=VarianceCriterion(estimator=gp_regressor, splitter=ShuffleSplit(n_splits=3, train_size=0.8)),
    query_strategy=2 * qs.ServiceQueryVariancePDF(bounds=None) + qs.ServiceUniform(bounds=None),
)


# GP noisy
learner_gp_noisy = ActiveSurfaceLearner(
    active_criterion=VarianceCriterion(
        estimator=gp_regressor_noisy, splitter=ShuffleSplit(n_splits=3, train_size=0.8)),
    query_strategy=qs.ServiceQueryVariancePDF(num_eval=N_CANDIDATES),
)

learner_uniform_gp_noisy = ActiveSurfaceLearner(
    active_criterion=VarianceCriterion(
        estimator=gp_regressor_noisy,
        splitter=ShuffleSplit(n_splits=3, train_size=0.8)),
    query_strategy=qs.ServiceUniform(None)
)

learner_bagging_uniform_gp_noisy = ActiveSurfaceLearner(
    active_criterion=VarianceCriterion(estimator=gp_regressor_noisy, splitter=ShuffleSplit(n_splits=3, train_size=0.8)),
    query_strategy=2 * qs.ServiceQueryVariancePDF(bounds=None) + qs.ServiceUniform(bounds=None),
)

# ======================================================================================================================

trees = {
    "passive": learner_uniform_trees,
    "bootstrap": learner_trees,
    "bootstrap + uniform": learner_bagging_uniform_trees,
}
gp = {"modal uncertainty": modal_al,
      "passive": learner_uniform_gp,
      "bootstrap": learner_gp,
      "bootstrap + uniform": learner_bagging_uniform_gp, }

gp_noisy = {"modal uncertainty": modal_al_noisy,
            "passive": learner_uniform_gp_noisy,
            "bootstrap": learner_gp_noisy,
            "bootstrap + uniform": learner_uniform_gp_noisy, }

methods = {
    "marelli_2018": trees,
    "grammacy_lee_2009": gp,
    "grammacy_lee_2009_rand": gp_noisy,
    "branin": gp,
    "himmelblau": gp,
    "branin_rand": gp_noisy,
    "himmelblau_rand": gp_noisy,
    "golden_price": trees,
    "synthetic_2d_1": trees,
    "sum_sine_5pi": trees,
}


def make_testing_classes(name) -> List[ServiceTestingClassAL]:
    n0 = functions.function_parameters[name]["n0"]
    budget = functions.function_parameters[name]["budget"]
    steps = functions.function_parameters[name]["n_step"]

    bounds = np.array(functions.bounds[functions.__dict__[name]])
    __methods = methods[name]
    list_of_classes = []

    sampler = utils.Sampler(bounds)

    for k, v in __methods.items():
        if isinstance(v, ActiveSurfaceLearner):
            list_of_classes.append(
                ServiceTestingClassAL(
                    budget=budget,
                    budget_0=n0,
                    function=functions.__dict__[name],
                    learner=v,
                    x_sampler=sampler,
                    bounds=bounds,
                    n_steps=steps,
                    name=f"{functions.function_parameters[name]['name']}@{k}",
                )
            )
        else:
            list_of_classes.append(
                ServiceTestingClassModAL(
                    budget=budget,
                    budget_0=n0,
                    function=functions.__dict__[name],
                    learner=v,
                    x_sampler=sampler,
                    bounds=bounds,
                    n_steps=steps,
                    name=f"{functions.function_parameters[name]['name']}@{k}",
                )
            )
    return list_of_classes


def create_benchmark_list():
    ret = []
    for k in methods.keys():
        ret += make_testing_classes(k)
    return ret


def profile(function_name, method_name):
    n0 = functions.function_parameters[function_name]["n0"]
    budget = functions.function_parameters[function_name]["budget"]
    steps = functions.function_parameters[function_name]["n_step"]

    bounds = np.array(functions.bounds[functions.__dict__[function_name]])
    method = methods[function_name][method_name]

    steps = 5
    method = learner_gp

    sampler = utils.Sampler(bounds)
    x = sampler(10)
    testing_al = ServiceTestingClassAL(
        budget=budget,
        budget_0=n0,
        function=functions.__dict__[function_name],
        learner=method,
        x_sampler=sampler,
        bounds=bounds,
        n_steps=steps,
        name=f"{functions.function_parameters[function_name]['name']}",
    )
    testing_al.run()
    testing_al.x_input
    test = testing_al
    test.x_input
    test.indexes

    plt.figure()
    for i in np.unique(test.indexes):
        color = plt.get_cmap("RdBu")(i / max(test.indexes))
        color = "k"
        sel = test.indexes == i
        x = test.x_input[sel].iloc[:, :2]

        plt.scatter(x[0], x[1], color=color, s=0.2)
    cmap = utils.cmap

    def as_2d(fun, d=functions.d_periodic_2):
        def function(x):
            xx = np.concatenate((x, np.zeros((len(x), d - x.shape[1]))), axis=1)
            return fun(xx)

        return function

    xx, yy, x, z = utils.eval_surf_2d(fun=as_2d(test.f), bounds=bounds)

    plt.pcolormesh(xx, yy,
                   z.reshape(len(xx), len(yy)),
                   cmap=cmap, vmin=-1, vmax=1, alpha=0.8, zorder=-100)
    plt.savefig("test.png")

    xx, yy, x, z = utils.eval_surf_2d(as_2d(test.result[test.indexes.max()]["learner"].predict), bounds)

    plt.pcolormesh(xx, yy,
                   z.reshape(len(xx), len(yy)),
                   cmap=cmap, vmin=-1, vmax=1)

    plt.savefig("test2.png")

    test.result


def test_in_sum_sine_5pi():
    test = make_testing_classes("sum_sine_5pi")
    me = ModuleExperiment(test, n_experiment=30)
    me.run()
    utils.write_benchmark(me.cv_result_, "data/2024_high_dimension.csv", update=False)
    utils.plot_benchmark(data=me.cv_result_)
    plt.savefig(".public/2024_high_dimension")


def test_function():
    name = "grammacy_lee_2009_rand"
    test = make_testing_classes(name)
    me = ModuleExperiment(test, n_experiment=5)
    me.run()
    utils.write_benchmark(me.cv_result_, f"data/2024_{name}.csv")

    data = utils.read_benchmark(f"data/2024_{name}.csv")
    utils.plot_benchmark(data=data)
    plt.gcf().set_size_inches(5, 5)
    plt.savefig(f".public/2024_{name}")
