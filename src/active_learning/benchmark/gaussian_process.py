import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from active_learning.components.active_criterion import GaussianProcessVariance, \
    VarianceBis
from active_learning.components.query_strategies import QueryVariancePDF, \
    Uniform
from active_learning.components.test import Experiment
from active_learning.components.test import TestingClass, \
    write_benchmark, read_benchmark, plot_benchmark

SEED = 1234
RNG = np.random.default_rng(seed=SEED)

bounds = [[0, 1]]


def unknown_function(x):
    return x ** 5 * np.sin(10 * np.pi * x) * np.sin(30 * np.pi * x)


def sampler(n):
    x0 = np.random.uniform(*bounds[0], size=n)
    return pd.DataFrame(x0)


kernel = 1 * RBF(0.01)
xtra_trees = ExtraTreesRegressor(bootstrap=False, n_estimators=50)
xtra_trees_b = ExtraTreesRegressor(bootstrap=True, n_estimators=50,
                                   max_samples=0.7)
spline_fitting = make_pipeline(PolynomialFeatures(100, include_bias=True),
                               Ridge(alpha=1e-3))
krg = GaussianProcessRegressor(kernel=kernel)
mlp = MLPRegressor(hidden_layer_sizes=(2, 4), max_iter=3000)
n0 = 15
budget = 60
steps = 20

if __name__ == '__main__':
    testing_bootstrap = TestingClass(
        budget, n0, unknown_function,
        VarianceBis(krg,
                    splitter=sklearn.model_selection.ShuffleSplit(n_splits=3,
                                                                  train_size=0.8)),
        QueryVariancePDF(bounds, num_eval=200),
        sampler, steps, bounds=bounds, name="Bootstrap method"

    )
    testing = TestingClass(
        budget, n0, unknown_function, GaussianProcessVariance(kernel=kernel),
        QueryVariancePDF(bounds, num_eval=200),
        sampler, steps, bounds=bounds, name="Gaussian process"

    )

    passive = TestingClass(
        budget, n0, unknown_function, GaussianProcessVariance(kernel=kernel),
        Uniform(bounds),
        sampler, steps, bounds=bounds, name="Gaussian process (passive)"
    )

    experiment = Experiment(test_list=[testing_bootstrap, testing, passive],
                            n_experiment=30)
    experiment.run()
    write_benchmark(experiment.cv_result_)
    data = read_benchmark()

    select_expe = data["test_id"] == data["test_id"].iloc[0]
    data = data[select_expe]

    plot_benchmark(data, cmap="viridis")
