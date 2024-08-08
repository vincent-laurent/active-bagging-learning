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

import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from active_learning.benchmark import functions, base
from active_learning.benchmark import utils
from active_learning.components import latin_square
from active_learning.components.active_criterion import VarianceEnsembleMethod
from active_learning.components.query_strategies import ServiceQueryVariancePDF, ServiceUniform

# try:
#     plt.style.use("./.matplotlibrc")
# except (ValueError, OSError):
#     pass
functions_ = list(functions.bounds.keys())

name = "grammacy_lee_2009_rand"
fun = functions.__dict__[name]

xtra_trees_b = ExtraTreesRegressor(bootstrap=True, n_estimators=50, max_samples=0.7)


def plot_results(path="benchmark/results.csv", n0=100, function=name):
    import seaborn as sns
    df = pd.read_csv(path)
    df_select = df.query(f"n0=={n0} & function==@function")
    sns.lineplot(data=df_select, x="budget", y='error_l2_active', label="Active")
    sns.lineplot(data=df_select, x="budget", y='error_l2_passive', label="Passive")
    plot.ylabel("$||f - f^2||$")
    plot.xlabel("Number of training points")


def add_to_benchmark(data: pd.DataFrame, path="benchmark/results.csv"):
    try:
        data_old = pd.read_csv(path)
        data_new = pd.concat((data, data_old), axis=0)
    except FileNotFoundError:
        data_new = data
    data_new.to_csv(path, index=False)


def plot_all_benchmark_function():
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("mycmap", ['C0', "C2", "C4", "C6", "white", "C7", "C5", "C3", "C1"])

    from active_learning.benchmark.functions import function_parameters
    functions__ = list(function_parameters.keys())
    fig, ax = plot.subplots(ncols=len(functions__) // 2 + len(functions__) % 2,
                            nrows=2, figsize=(len(functions_) * 0.7, 3), dpi=200)
    for i, fun in enumerate(functions__):
        f = function_parameters[fun]["fun"]
        bound = np.array(functions.bounds[f])
        if len(bound) == 2:
            ax_ = ax[i % 2, i // 2]
            xx, yy, x, z = utils.eval_surf_2d(f, bound, num=200)

            ax_.pcolormesh(xx, yy, z.reshape(len(xx), len(yy)),
                           cmap=cmap)

            ax_.axes.yaxis.set_ticklabels([])
            ax_.axes.xaxis.set_ticklabels([])
    if len(functions__) % 2 == 1:
        ax[(i + 1) % 2, (i + 1) // 2].axes.remove()
    plot.savefig(".public/functions.png")


def clear_benchmark_data(path="benchmark/results.csv", function=name):
    df = pd.read_csv(path)
    df_select = df.query(f"function==@function")
    df.drop(df_select.index).to_csv(path)


def run_2d_benchmark():
    from active_learning.benchmark.functions import function_parameters

    estimator = xtra_trees_b

    def get_sampler(bounds):
        def sampler(size):
            return pd.DataFrame(latin_square.scipy_lhs_sampler(size=size, x_limits=np.array(bounds)))

        return sampler

    active_testing_classes = [base.ServiceTestingClassAL(
        function_parameters[name]["budget"],
        function_parameters[name]["n0"],
        function_parameters[name]["fun"],
        VarianceEnsembleMethod(estimator=estimator),
        ServiceQueryVariancePDF(functions.bounds[function_parameters[name]["fun"]], num_eval=200),
        get_sampler(functions.bounds[function_parameters[name]["fun"]]),
        n_steps=function_parameters[name]["n_step"],
        bounds=functions.bounds[function_parameters[name]["fun"]],
        name=name
    ) for name in list(function_parameters.keys())
    ]

    passive_testing_classes = [base.ServiceTestingClassAL(
        function_parameters[name]["budget"],
        function_parameters[name]["n0"],
        function_parameters[name]["fun"],
        VarianceEnsembleMethod(estimator=estimator),
        ServiceUniform(functions.bounds[function_parameters[name]["fun"]]),
        get_sampler(functions.bounds[function_parameters[name]["fun"]]),
        n_steps=function_parameters[name]["n_step"],
        bounds=functions.bounds[function_parameters[name]["fun"]],
        name=name + "_passive"
    ) for name in list(function_parameters.keys())

    ]
    experiment = base.ModuleExperiment([*active_testing_classes, *passive_testing_classes], n_experiment=50)
    experiment.run()
    data = experiment.cv_result_
    base.write_benchmark(data, path="data/benchmark_2d_2024.csv")


def get_benchmark():
    identifier = ["budget", "budget_0", "n_steps", "active_criterion", "query_strategy", "name"]
    data = utils.read_benchmark(path="./examples/2d_benchmark/data/benchmark_2d.csv")

    df_property = data.groupby(identifier)["date"].last().reset_index()
    df_property = df_property.sort_values("date", ascending=False).drop_duplicates(
        "name")
    data_unique = pd.merge(data, df_property[identifier], on=identifier)
    utils.plot_benchmark_whole_analysis(data=data_unique)
    plt.savefig(".public/active_vs_passive.png")


if __name__ == '__main__':
    get_benchmark()
