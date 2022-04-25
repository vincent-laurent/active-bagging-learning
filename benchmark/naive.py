import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVR

import base
import components.query_strategies as qs
from benchmark.utils import evaluate, eval_surf_2d
from components import active_criterion
from data import functions
from models.smt_api import SurrogateKRG
from components.sampling import latin_square

functions_ = list(functions.bounds.keys())

name = "grammacy_lee_2009_rand"
fun = functions.__dict__[name]

estimator_parameters = {
    0: dict(base_estimator=SVR(kernel="rbf", C=100, gamma="scale",
                               epsilon=0.1)),
    1: dict(base_estimator=GaussianProcessRegressor(kernel=RBF(length_scale=0.1))),
    2: dict(base_estimator=RandomForestRegressor(n_estimators=5,
                                                 min_samples_leaf=3)),
    3: dict(base_estimator=SurrogateKRG(), splitter=ShuffleSplit(n_splits=2))
}
estimator = estimator_parameters[3]


def run(fun, n0=10, budget=100, n_step=5, name=name, estimator=estimator):
    bounds = functions.bounds[fun]
    x0 = latin_square.iterative_sampler(x_limits=np.array(bounds), size=n0,
                                        batch_size=n0 // 2)
    xall = latin_square.iterative_sampler(x_limits=np.array(bounds),
                                          size=budget)
    args = dict(
        bounds=np.array(bounds),
        estimator_parameters=estimator
    )
    active_learner = base.ActiveSRLearner(
        active_criterion.estimate_variance,
        qs.reject_on_bounds,
        pd.DataFrame(x0),
        pd.DataFrame(fun(x0)), **args)

    perf_passive = []
    results = pd.DataFrame(index=range(n_step))
    results["budget"] = n0
    for step in range(n_step):
        b = active_learner.budget
        results.loc[step, "budget"] = b
        n_points = int((budget - n0) / n_step)
        print(active_learner.budget, n_points)
        x_new = active_learner.query(n_points).drop_duplicates()
        y_new = pd.DataFrame(fun(x_new))
        active_learner.add_labels(x_new, y_new)
        active_learner.result[active_learner.iter]["error_l2"] = evaluate(
            fun,
            active_learner.surface,
            bounds, num_mc=10000)

        s = pd.DataFrame(xall).sample(b)
        passive_learner = base.ActiveSRLearner(
            active_learner.estimator,
            active_learner.query_strategy,
            s, fun(s), **args
        )

        passive_learner.query(1)
        perf_passive.append(evaluate(
            fun,
            passive_learner.surface,
            bounds, num_mc=10000))

    results["estimator_param"] = str(estimator)
    results["error_l2_active"] = [
        active_learner.result[i]["error_l2"] for i in
        active_learner.result.keys()]
    results["error_l2_passive"] = perf_passive
    results["function"] = name
    results["n0"] = n0
    results["budget_total"] = budget
    results["estimator"] = str(estimator["base_estimator"]).split("(")[0]
    return active_learner, passive_learner, results


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
    from data.functions import budget_parameters
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
    from data.functions import budget_parameters
    functions__ = list(budget_parameters.keys())
    for f in functions__:
        print(f)
        for n in range(10):
            _, _, r = run(**budget_parameters[f], name=f)
            add_to_benchmark(r)


def plot_benchmark_whole_analysis() -> None:
    from data.functions import budget_parameters
    functions__ = list(budget_parameters.keys())
    fig, ax = plot.subplots(ncols=len(functions__) // 2 + len(functions__) % 2,
                            nrows=2, figsize=(len(functions_) * 0.7, 3.5), dpi=200
                            )
    for i, f in enumerate(functions__):
        print(f)
        ax_ = ax[i % 2, i // 2]

        plot.sca(ax_)
        plot_results(function=f, n0=budget_parameters[f]["n0"])
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
    # run_whole_analysis()
    plot_all_benchmark_function()
    plot_benchmark_whole_analysis()

    # clear_benchmark_data(function="golden_price_rand")
    # clear_benchmark_data(function="branin_rand")
    # clear_benchmark_data(function="himmelblau_rand")
