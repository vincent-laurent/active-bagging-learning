import matplotlib.pyplot as plt
import numpy as np

from sklearn import ensemble

from active_learning import ActiveSurfaceLearner
from active_learning import query_strategies as qs
from active_learning.benchmark import functions
from active_learning.benchmark.base import (
    ServiceTestingClassAL,
    ModuleExperiment,
)
from active_learning.benchmark import utils
from active_learning.components import active_criterion
from active_learning.benchmark.utils import Sampler


def main():
    est_trees = ensemble.ExtraTreesRegressor(
        bootstrap=True, max_samples=0.9, n_estimators=100
    )
    alc_trees = active_criterion.VarianceEnsembleMethod(estimator=est_trees)

    # ==========================================================================
    #                           ACTIVE LEARNERS
    # ==========================================================================
    # TREES
    learner_trees = ActiveSurfaceLearner(
        active_criterion=alc_trees,
        query_strategy=qs.ServiceQueryVariancePDF(num_eval=500, bounds=None),
        bounds=None,
    )

    learner_uniform_trees = ActiveSurfaceLearner(
        active_criterion=alc_trees,
        query_strategy=qs.ServiceUniform(None),
        bounds=None)
    
    learner_max_trees = ActiveSurfaceLearner(
        active_criterion=alc_trees,
        query_strategy=qs.ServiceQueryMax(x0=np.array([0, 0]), bounds=None),
        bounds=None,
    )
    
    function = "golden_price"
    bounds = np.array(functions.bounds[functions.__dict__[function]])
    sampler = Sampler(bounds)
    args = dict(
        budget=100,
        budget_0=10,
        function=functions.__dict__[function],
        x_sampler=sampler,
        bounds=bounds,
    )

    s1 = ServiceTestingClassAL(
        **args,
        learner=learner_trees,
        n_steps=18,
        name=f"{functions.function_parameters[function]['name']}@{'5 points pdf'}",
    )
    s2 = ServiceTestingClassAL(
        **args,
        learner=learner_trees,
        n_steps=90,
        name=f"{functions.function_parameters[function]['name']}@{'1 points pdf'}",
    )
    
    s3 = ServiceTestingClassAL(
        **args,
        learner=learner_max_trees,
        n_steps=90,
        name=f"{functions.function_parameters[function]['name']}@{'1 points max'}",
    )
    
    s4 = ServiceTestingClassAL(
        **args,
        learner=learner_uniform_trees,
        n_steps=90,
        name=f"{functions.function_parameters[function]['name']}@{'passive'}",
    )
    
    s5 = ServiceTestingClassAL(
        **args,
        learner=learner_uniform_trees,
        n_steps=5,
        name=f"{functions.function_parameters[function]['name']}@{'18 points pdf'}",
    )

    experiment = ModuleExperiment([s1, s2, s3, s4, s5], n_experiment=100)

    # experiment.run()
    # utils.write_benchmark(experiment.cv_result_, "data/2024_benchmark_sampling.csv")
    data = utils.read_benchmark(path="data/2024_benchmark_sampling.csv")
    data = data.sort_values(by="name")
    utils.plot_benchmark(data=data)
    plt.gcf().set_size_inches(4.3, 5)
    plt.savefig(".public/2024_benchmark_sampling")
    

if __name__ == "__main__":
    main()