import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from modAL.acquisition import max_EI
from modAL.models import BayesianOptimizer
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
from active_learning.benchmark.base import ServiceTestingClassAL, \
    ServiceTestingClassModAL, ModuleExperiment
from active_learning.components import active_criterion
from active_learning.components import latin_square
from active_learning.components.active_criterion import VarianceCriterion

est_trees = ensemble.ExtraTreesRegressor(bootstrap=True, max_samples=0.9,
                                         n_estimators=10)
alc_trees = active_criterion.VarianceEnsembleMethod(estimator=est_trees)

est_svc = Pipeline(
    [("scale", StandardScaler()), ("est", SVR(degree=2, C=10, gamma=10))])
alc_svc = VarianceCriterion(estimator=est_svc,
                            splitter=ShuffleSplit(n_splits=3, train_size=0.8))

kernel = Matern(length_scale=1.0)
regressor = GaussianProcessRegressor(kernel=kernel)
optimizer = BayesianOptimizer(
    estimator=regressor,
    query_strategy=max_EI
)

# ======================================================================================================================
#                           ACTIVE LEARNERS
# ======================================================================================================================
# TREES
learner_trees = ActiveSurfaceLearner(
    active_criterion=alc_trees,
    query_strategy=qs.ServiceQueryVariancePDF(num_eval=500, bounds=None),
    bounds=None)

learner_uniform_trees = ActiveSurfaceLearner(
    active_criterion=alc_trees,
    query_strategy=qs.ServiceUniform(None),
    bounds=None)

learner_bagging_uniform_trees = ActiveSurfaceLearner(
    active_criterion=alc_trees,
    query_strategy=2 * qs.ServiceQueryVariancePDF(
        bounds=None) + qs.ServiceUniform(bounds=None),
    bounds=None)

# SVC
learner_svc = ActiveSurfaceLearner(
    active_criterion=alc_svc,
    query_strategy=qs.ServiceQueryVariancePDF(num_eval=500, bounds=None),
    bounds=None)

learner_uniform_svc = ActiveSurfaceLearner(
    active_criterion=alc_svc,
    query_strategy=qs.ServiceUniform(None),
    bounds=None)

learner_bagging_uniform_svc = ActiveSurfaceLearner(
    active_criterion=alc_svc,
    query_strategy=20 * qs.ServiceQueryVariancePDF(
        bounds=None) + qs.ServiceUniform(bounds=None),
    bounds=None)

# ======================================================================================================================


methods = {
    "marelli_2018":
        {"passive": learner_uniform_svc,
         "SVC bootstrap": learner_svc,
         "SVC bootstrap + uniform": learner_uniform_svc},
    "grammacy_lee_2009":
        {"passive": learner_uniform_svc,
         "SVC bootstrap": learner_svc,
         "SVC bootstrap + uniform": learner_uniform_svc,
         },
    "grammacy_lee_2009_rand":
        {"passive": learner_uniform_svc,
         "SVC bootstrap": learner_svc,
         "SVC bootstrap + uniform": learner_uniform_svc,
         "TREES bootstrap": learner_trees,
         },
    "branin":
        {"passive": learner_uniform_svc,
         "SVC bootstrap": learner_svc,
         "SVC bootstrap + uniform": learner_uniform_svc,
         # "TREES bootstrap": learner_trees,
         # "TREES bootstrap + uniform": learner_bagging_uniform_trees,
         },
    "himmelblau":
        {"passive": learner_uniform_trees,
         "TREES bootstrap": learner_trees,
         # "TREES bootstrap + uniform": learner_bagging_uniform_trees,
         },
    "branin_rand":
        {"passive": learner_uniform_trees,
         "TREES bootstrap": learner_trees,
         # "TREES bootstrap + uniform": learner_bagging_uniform_trees,
         },
    "himmelblau_rand":
        {"passive": learner_uniform_trees,
         "TREES bootstrap": learner_trees,
         # "TREES bootstrap + uniform": learner_bagging_uniform_trees,
         },
    "golden_price":
        {"passive": learner_uniform_trees,
         "TREES bootstrap": learner_trees,
         # "TREES bootstrap + uniform": learner_bagging_uniform_trees,
         },
    "synthetic_2d_1":
        {"passive": learner_uniform_trees,
         "TREES bootstrap": learner_trees,
         # "TREES bootstrap + uniform": learner_bagging_uniform_trees,
         },
    "sum_sine_5pi":
        {"passive": learner_uniform_svc,
         "TREES bootstrap": learner_svc,
         # "TREES bootstrap + uniform": learner_bagging_uniform_trees,
         },

}


class Sampler:
    def __init__(self, bounds):
        self.__bounds = bounds

    def __call__(self, size):
        return pd.DataFrame(
            latin_square.scipy_lhs_sampler(size=size, x_limits=self.__bounds))


def make_testing_classes(name):
    n0 = functions.function_parameters[name]["n0"]
    budget = functions.function_parameters[name]["budget"]
    steps = functions.function_parameters[name]["n_step"]

    bounds = np.array(functions.bounds[functions.__dict__[name]])
    __methods = methods[name]
    list_of_classes = []

    sampler = Sampler(bounds)

    for k, v in __methods.items():
        if isinstance(v, ActiveSurfaceLearner):
            list_of_classes.append(ServiceTestingClassAL(
                budget=budget, budget_0=n0, function=functions.__dict__[name],
                learner=v,
                x_sampler=sampler,
                bounds=bounds, n_steps=steps,
                name=f"{functions.function_parameters[name]['name']}@{k}"
            ))
        else:
            list_of_classes.append(ServiceTestingClassModAL(
                budget=budget, budget_0=n0, function=functions.__dict__[name],
                learner=l,
                x_sampler=sampler,
                bounds=bounds, n_steps=steps) for l in __methods.values())
    return list_of_classes


def create_benchmark_list():
    ret = []
    for k in methods.keys():
        ret += make_testing_classes(k)
    return ret


if __name__ == '__main__':
    from active_learning.benchmark import utils

    test = make_testing_classes("sum_sine_5pi")

    t = create_benchmark_list()

    me = ModuleExperiment(t, n_experiment=1)
    me.run()

    plt.figure()
    utils.plot_benchmark(data=me.cv_result_)
    utils.write_benchmark(me.cv_result_, "data/benchmark_2024.csv")
    plt.savefig("test")
