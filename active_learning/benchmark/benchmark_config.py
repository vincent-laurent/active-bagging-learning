import matplotlib.pyplot as plt
import numpy as np
from modAL.acquisition import max_EI, max_UCB
from modAL.disagreement import max_std_sampling
from modAL.models import BayesianOptimizer, ActiveLearner
from sklearn import ensemble
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
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


def max_ucb(opt, x, ):
    return max_UCB(opt, x, beta=1000)


def max_ei(opt, x):
    return max_EI(opt, x, beta=1000)


est_trees = ensemble.ExtraTreesRegressor(
    bootstrap=True, max_samples=0.9, n_estimators=100
)
alc_trees = active_criterion.VarianceEnsembleMethod(estimator=est_trees)

est_svc = Pipeline(
    [("scale", StandardScaler()), ("est", SVR(degree=2, C=10, gamma=10))]
)
alc_svc = VarianceCriterion(
    estimator=est_svc, splitter=ShuffleSplit(n_splits=3, train_size=0.8)
)

kernel = Matern(length_scale=1.0)
regressor = GaussianProcessRegressor(kernel=kernel)
gp_ei_optimizer = BayesianOptimizer(estimator=regressor, query_strategy=max_EI)
gp_ucb_optimizer = BayesianOptimizer(estimator=regressor, query_strategy=max_ucb)
modal_al = ActiveLearner(regressor, max_std_sampling)

regressor_svc = est_svc
gp_ei_optimizer_svc = BayesianOptimizer(estimator=regressor_svc, query_strategy=max_EI)
gp_ucb_optimizer_svc = BayesianOptimizer(estimator=regressor_svc, query_strategy=max_ucb)
modal_al_svc = ActiveLearner(regressor_svc, max_std_sampling)
# ======================================================================================================================
#                           ACTIVE LEARNERS
# ======================================================================================================================
# TREES
learner_trees = ActiveSurfaceLearner(
    active_criterion=alc_trees,
    query_strategy=qs.ServiceQueryVariancePDF(num_eval=2000, bounds=None),
    bounds=None,
)

learner_uniform_trees = ActiveSurfaceLearner(
    active_criterion=alc_trees, query_strategy=qs.ServiceUniform(None), bounds=None
)

learner_bagging_uniform_trees = ActiveSurfaceLearner(
    active_criterion=alc_trees,
    query_strategy=2 * qs.ServiceQueryVariancePDF(bounds=None) + qs.ServiceUniform(bounds=None),
    bounds=None,
)

# SVC
learner_svc = ActiveSurfaceLearner(
    active_criterion=alc_svc,
    query_strategy=qs.ServiceQueryVariancePDF(num_eval=2000, bounds=None),
    bounds=None,
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
    active_criterion=VarianceCriterion(estimator=regressor, splitter=ShuffleSplit(n_splits=3, train_size=0.8)),
    query_strategy=qs.ServiceQueryVariancePDF(num_eval=2000, bounds=None),
    bounds=None,
)

learner_uniform_gp = ActiveSurfaceLearner(
    active_criterion=VarianceCriterion(estimator=regressor, splitter=ShuffleSplit(n_splits=3, train_size=0.8)),
    query_strategy=qs.ServiceUniform(None), bounds=None
)

learner_bagging_uniform_gp = ActiveSurfaceLearner(
    active_criterion=VarianceCriterion(estimator=regressor, splitter=ShuffleSplit(n_splits=3, train_size=0.8)),
    query_strategy=2 * qs.ServiceQueryVariancePDF(bounds=None) + qs.ServiceUniform(bounds=None),
    bounds=None,
)

# ======================================================================================================================

trees = {
    "passive": learner_uniform_trees,
    "bootstrap": learner_trees,
    "bootstrap + uniform": learner_bagging_uniform_trees,
}
gp = {"modal EI": gp_ei_optimizer,
      "modal UCB": gp_ucb_optimizer,
      "modal uncertainty": modal_al,
      "passive": learner_uniform_gp,
      "bootstrap": learner_gp,
      "bootstrap + uniform": learner_uniform_gp, }

methods = {
    "marelli_2018": trees,
    "grammacy_lee_2009": gp,
    "grammacy_lee_2009_rand": trees,
    "branin": gp,
    "himmelblau": trees,
    "branin_rand": trees,
    "himmelblau_rand": trees,
    "golden_price": trees,
    "synthetic_2d_1": trees,
    "sum_sine_5pi": trees,
}


def make_testing_classes(name):
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


def test_in_sum_sine_5pi():
    test = make_testing_classes("sum_sine_5pi")
    me = ModuleExperiment(test, n_experiment=10)
    me.run()
    utils.write_benchmark(me.cv_result_, "data/2024_high_dimension.csv")
    utils.plot_benchmark(data=me.cv_result_)
    plt.savefig(".public/2024_high_dimension")


def test_function():
    name = "grammacy_lee_2009"
    name = "branin"
    test = make_testing_classes(name)
    me = ModuleExperiment(test, n_experiment=10)
    me.run()
    utils.write_benchmark(me.cv_result_, f"data/2024_{name}_gaussian_process.csv")

    data = utils.read_benchmark(f"data/2024_{name}_gaussian_process.csv")
    utils.plot_benchmark(data=data)
    plt.gcf().set_size_inches(5, 5)
    plt.savefig(f".public/2024_{name}_gaussian_process")


if __name__ == "__main__":
    t = create_benchmark_list()

    me = ModuleExperiment(t, n_experiment=100)
    me.run()
    utils.write_benchmark(me.cv_result_, "data/benchmark_2024.csv")

    plt.figure()
    data = utils.read_benchmark("data/benchmark_2024.csv")
    data = data[data["date"] > "2024-10-31 18"]
    utils.plot_benchmark(data=data)
    plt.savefig(".public/benchmark_result_2024")
