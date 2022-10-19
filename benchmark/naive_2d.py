import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from active_learning.components.active_criterion import VarianceEnsembleMethod
from active_learning.components.query_strategies import QueryVariancePDF, Uniform
from active_learning.components.sampling import latin_square
from active_learning.components.test import TestingClass
from active_learning.data import functions

functions_ = list(functions.bounds.keys())

name = "grammacy_lee_2009_rand"
fun = functions.__dict__[name]

xtra_trees_b = ExtraTreesRegressor(bootstrap=True, n_estimators=50, max_samples=0.7)


def run(name):
    bounds = functions.bounds[fun]

    def sampler(size): return pd.DataFrame(latin_square.scipy_lhs_sampler(size=size, x_limits=np.array(bounds)))

    testing_bootstrap = TestingClass(
        budget, n0, fun,
        VarianceEnsembleMethod(estimator=estimator),
        QueryVariancePDF(bounds, num_eval=200),
        sampler, n_steps=n_step, bounds=bounds, name=name
    )
    testing_bootstrap.run()
    testing_bootstrap.metric


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
    from active_learning.data.functions import budget_parameters
    functions__ = list(budget_parameters.keys())
    fig, ax = plot.subplots(ncols=len(functions__) // 2 + len(functions__) % 2,
                            nrows=2, figsize=(len(functions_) * 0.7, 3), dpi=200)
    for i, fun in enumerate(functions__):
        f = budget_parameters[fun]["fun"]
        bound = np.array(functions.bounds[f])
        if len(bound) == 2:
            ax_ = ax[i % 2, i // 2]
            xx, yy, x, z = eval_surf_2d(f, bound, num=200)

            ax_.pcolormesh(xx, yy, z.reshape(len(xx), len(yy)),
                           cmap="RdBu")

            ax_.axes.yaxis.set_ticklabels([])
            ax_.axes.xaxis.set_ticklabels([])
    if len(functions__) % 2 == 1:
        ax[(i + 1) % 2, (i + 1) // 2].axes.remove()
    plot.savefig("benchmark/functions.png")


def clear_benchmark_data(path="benchmark/results.csv", function=name):
    df = pd.read_csv(path)
    df_select = df.query(f"function==@function")
    df.drop(df_select.index).to_csv(path)


def run_whole_analysis():
    from active_learning.data.functions import budget_parameters
    functions__ = list(budget_parameters.keys())
    for f in functions__:
        print(f)
        for n in range(10):
            _, _, r = run(**budget_parameters[f], name=f)
            add_to_benchmark(r)


def plot_benchmark_whole_analysis(data: pd.DataFrame) -> None:
    from active_learning.components import test
    from active_learning.data.functions import budget_parameters
    functions__ = data["name"].str.replace("_passive", "").drop_duplicates()
    fig, ax = plot.subplots(ncols=len(functions__) // 2 + len(functions__) % 2,
                            nrows=2, figsize=(len(functions_) * 0.7, 3.5), dpi=200)
    if ax.shape.__len__() == 1:
        ax = ax.reshape(-1, 1)
    for i, f in enumerate(functions__):
        print(f)
        ax_ = ax[i % 2, i // 2]

        plot.sca(ax_)
        data_temp = data[data["name"] == f]
        data_temp_p = data[data["name"] == f + "_passive"]
        test.plot_benchmark(data=pd.concat((data_temp, data_temp_p)), cmap="crest")
        ax_.set_ylabel("$L_2$ error") if i // 2 == 0 else ax_.set_ylabel("")
        plot.yticks(c="w")
        plot.xlabel("")
        ax_.axes.yaxis.set_ticklabels([])
        ax_.get_legend().remove()
    if len(functions__) % 2 == 1:
        ax[(i + 1) % 2, (i + 1) // 2].axes.remove()
    ax_.legend()

    plot.savefig("benchmark/active_passive.png")


if __name__ == '__main__':
    from active_learning.components import test
    from active_learning.data.functions import budget_parameters

    estimator = xtra_trees_b

    def get_sampler(bounds):
        def sampler(size):
            return pd.DataFrame(latin_square.scipy_lhs_sampler(size=size, x_limits=np.array(bounds)))

        return sampler


    active_testing_classes = [test.TestingClass(
        budget_parameters[name]["budget"],
        budget_parameters[name]["n0"],
        budget_parameters[name]["fun"],
        VarianceEnsembleMethod(estimator=estimator),
        QueryVariancePDF(functions.bounds[budget_parameters[name]["fun"]], num_eval=200),
        get_sampler(functions.bounds[budget_parameters[name]["fun"]]),
        n_steps=budget_parameters[name]["n_step"],
        bounds=functions.bounds[budget_parameters[name]["fun"]],
        name=name
    ) for name in list(budget_parameters.keys())
    ]

    passive_testing_classes = [test.TestingClass(
        budget_parameters[name]["budget"],
        budget_parameters[name]["n0"],
        budget_parameters[name]["fun"],
        VarianceEnsembleMethod(estimator=estimator),
        Uniform(functions.bounds[budget_parameters[name]["fun"]]),
        get_sampler(functions.bounds[budget_parameters[name]["fun"]]),
        n_steps=budget_parameters[name]["n_step"],
        bounds=functions.bounds[budget_parameters[name]["fun"]],
        name=name + "_passive"
    ) for name in list(budget_parameters.keys())
    ]

    experiment = test.Experiment([*active_testing_classes, *passive_testing_classes], n_experiment=50)
    experiment.run()
    data = experiment.cv_result_
    test.write_benchmark(data, path="data/benchmark_2d.csv")
    data = test.read_benchmark(path="data/benchmark_2d.csv")

    plot_benchmark_whole_analysis(data)
