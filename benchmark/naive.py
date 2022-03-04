import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVR

import base
import components.query_strategies as qs
from benchmark.utils import evaluate, eval_surf_2d
from data import functions
from models.smt_api import SurrogateKRG
from sampling import latin_square

functions_ = list(functions.bounds.keys())

name = "synthetic_2d_2"
fun = functions.__dict__[name]
__n_sim__ = 1
estimator_parameters = {
    0: dict(base_estimator=SVR(kernel="rbf", C=100, gamma="scale",
                               epsilon=0.1)),
    1: dict(base_estimator=GaussianProcessRegressor()),
    2: dict(base_estimator=RandomForestRegressor(n_estimators=5,
                                                 min_samples_leaf=3)),
    3: dict(base_estimator=SurrogateKRG(), splitter=ShuffleSplit(n_splits=2))
}
estimator = estimator_parameters[3]


def run(fun, n0=10, budget=100, n_step=5):
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
        base.estimate_variance,
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
        x_new = active_learner.run(n_points).drop_duplicates()
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

        passive_learner.run(1)
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


def plot_results(path="benchmark/results.csv", n0=30, function=name):
    import seaborn as sns
    df = pd.read_csv(path)
    df_select = df.query(f"n0=={n0} & function==@function")
    sns.lineplot(data=df_select, x="budget", y='error_l2_active')
    sns.lineplot(data=df_select, x="budget", y='error_l2_passive')


def add_to_benchmark(data: pd.DataFrame, path="benchmark/results.csv"):
    try:
        data_old = pd.read_csv(path)
        data_new = pd.concat((data, data_old), axis=0)
    except FileNotFoundError:
        data_new = data
    data_new.to_csv(path, index=False)


def plot_all_benchmark_function():
    from data.functions import __all2D__
    fig, ax = plot.subplots(ncols=len(__all2D__) // 2, nrows=2)
    for i, fun in enumerate(__all2D__):
        bound = np.array(functions.bounds[fun])
        if len(bound) == 2:
            xx, yy, x, z = eval_surf_2d(fun, bound, num=200)

            ax[i // 2, i % 2].pcolormesh(xx, yy, z.reshape(len(xx), len(yy)),
                                         cmap="rainbow")


if __name__ == '__main__':
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    for i in range(__n_sim__):
        a, p, r = run(fun, n0=100, budget=150, n_step=10)
        add_to_benchmark(r)

    bounds = np.array(functions.bounds[fun])
    xx, yy, x, z = eval_surf_2d(fun, bounds, num=200)

    # X = active_learner.result[n_step + 1]["data"]
    X = a.x_input
    prediction = a.surface
    coverage = a.coverage

    zz = prediction(x)
    zzz = p.surface(x)
    std = coverage(x)

    fig, ax = plot.subplots(ncols=2)
    sa = ((zz.ravel() - z).reshape(len(xx), len(yy))) ** 2
    sp = ((zzz.ravel() - z).reshape(len(xx), len(yy))) ** 2

    im = ax[0].pcolormesh(xx, yy, sa,
                          cmap="GnBu", vmin=sa.min(), vmax=sp.max())
    im = ax[1].pcolormesh(xx, yy, sp,
                          cmap="GnBu", vmin=sa.min(), vmax=sp.max())

    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plot.colorbar(im, cax=cax)
    ax[0].set_title("Estimation error $\hat{f}_{active} - f $")
    ax[1].set_title("Estimation error $\hat{f}_{passive} - f $")

    plot.figure()
    plot.pcolormesh(xx, yy, std.reshape(len(xx), len(yy)), cmap="rainbow")
    plot.colorbar()
    plot.title("Estimation error $\sigma(\hat{f})$")

    plot.figure()
    plot.pcolormesh(xx, yy, std.reshape(len(xx), len(yy)), cmap="rainbow")
    x_new = a.unscale_x(a.x_new)
    plot.scatter(x_new[0], x_new[1], c="k")
    plot.scatter(X[0], X[1], c="b")

    plot.figure()
    std_ = std.reshape(len(xx), len(yy))[len(yy) // 2]
    pred = zz.reshape(len(xx), len(yy))[len(yy) // 2]
    f = z.reshape(len(xx), len(yy))[len(yy) // 2]
    fp = zzz.reshape(len(xx), len(yy))[len(yy) // 2]

    plot.plot(yy, pred)
    plot.fill_between(yy, pred, pred + std_, color="b", alpha=0.1)
    plot.fill_between(yy, pred - std_, pred, color="b", alpha=0.1)
    plot.plot(yy, f, label="true function")
    plot.plot(yy, fp, label="passive")
    plot.legend()

    plot.figure()
    plot.contour(xx, yy, z.reshape(len(xx), len(yy)), levels=10, linewidths=0.3,
                 colors='k')
    x_new = a.scaling_method.transform(a.x_new)
    plot.scatter(X[0], X[1], cmap="rainbow", c=X.index)
    # plot.scatter(x0[:, 0], x0[:, 1], c="k")
    # plot.scatter(passive_learner.result[1]["data"][0],
    #              passive_learner.result[1]["data"][1], c="k", marker="+")
    print(
        f"passive performance : {evaluate(fun, p.surface, bounds, num_mc=10000)}"
        "\n"
        f"active  performance : {evaluate(fun, a.surface, bounds, num_mc=10000)}")
