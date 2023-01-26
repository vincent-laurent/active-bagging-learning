import os

import matplotlib.colors as colors
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import ensemble
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from active_learning.benchmark import functions
from active_learning.benchmark.utils import evaluate, eval_surf_2d
from active_learning.components import active_criterion
from active_learning.components import query_strategies
from active_learning.components.active_criterion import VarianceBis
from active_learning.components.sampling import latin_square
from active_learning.components.test import TestingClass

name = "grammacy_lee_2009_rand"
fun = functions.__dict__[name]


def analyse_results(path):
    res = pd.read_csv(path)
    print(res)


def get_method_for_benchmark(name):
    if name == "marelli_2018":
        est = VarianceBis(
            estimator=Pipeline([("scale", StandardScaler()),
                                ("est", SVC(degree=2, C=5, gamma=10))]),
            splitter=ShuffleSplit(n_splits=3, train_size=0.8))
        crit = query_strategies.QueryVariancePDF(num_eval=500)

    elif name == "grammacy_lee_2009":
        est = VarianceBis(
            estimator=Pipeline([("scale", StandardScaler()),
                                ("est", SVR(degree=2, C=10, gamma=10))]),
            splitter=ShuffleSplit(n_splits=3, train_size=0.7))
        crit = query_strategies.QueryVariancePDF(num_eval=1000)

    elif name == "grammacy_lee_2009_rand":
        est = VarianceBis(
            estimator=Pipeline([("scale", StandardScaler()),
                                ("est", SVR(degree=2, C=10, gamma=10))]),
            splitter=ShuffleSplit(n_splits=3, train_size=0.7))
        crit = query_strategies.Reject(num_eval=100)

    elif name == "branin":
        est = active_criterion.VarianceEnsembleMethod(
            estimator=ensemble.ExtraTreesRegressor(bootstrap=True))
        crit = query_strategies.QueryVariancePDF(num_eval=1000)

    elif name == "branin_rand":
        est = active_criterion.VarianceEnsembleMethod(
            estimator=ensemble.ExtraTreesRegressor(bootstrap=True))
        crit = query_strategies.QueryVariancePDF(num_eval=1000)
    elif name == "himmelblau":
        est = active_criterion.VarianceEnsembleMethod(
            estimator=ensemble.ExtraTreesRegressor(bootstrap=True))
        crit = query_strategies.QueryVariancePDF(num_eval=1000)

    elif name == "himmelblau_rand":
        est = active_criterion.VarianceEnsembleMethod(
            estimator=ensemble.ExtraTreesRegressor(bootstrap=True))
        crit = query_strategies.QueryVariancePDF(num_eval=1000)

    elif name == "synthetic_2d_1":
        est = active_criterion.VarianceEnsembleMethod(
            estimator=ensemble.ExtraTreesRegressor(bootstrap=True))
        crit = query_strategies.QueryVariancePDF(num_eval=1000)

    elif name == "synthetic_2d_2":
        est = active_criterion.VarianceEnsembleMethod(
            estimator=ensemble.ExtraTreesRegressor(bootstrap=True,
                                                   max_samples=0.9))
        crit = query_strategies.QueryVariancePDF(num_eval=1000)

    else:
        est = active_criterion.VarianceEnsembleMethod(
            estimator=ensemble.ExtraTreesRegressor(bootstrap=True,
                                                   max_samples=0.9,
                                                   max_features=1))
        crit = query_strategies.QueryVariancePDF(num_eval=1000)
    return est, crit


if __name__ == '__main__':
    path = f"examples/2d_benchmark/figures/{name}"
    if not os.path.exists(path):
        os.makedirs(path)

    n0 = functions.budget_parameters[name]["n0"]
    budget = functions.budget_parameters[name]["budget"]
    steps = functions.budget_parameters[name]["n_step"]

    bounds = np.array(functions.bounds[fun])


    def sampler(size):
        return pd.DataFrame(
            latin_square.scipy_lhs_sampler(size=size, x_limits=bounds))


    active_criterion_, crit_ = get_method_for_benchmark(name)
    a = TestingClass(
        budget, n0, fun,
        active_criterion_,
        crit_,
        sampler, bounds=bounds, n_steps=steps)
    p = TestingClass(
        budget, n0, fun,
        active_criterion_,
        query_strategies.Uniform(),
        sampler, bounds=bounds, n_steps=steps)
    a.run()
    p.run()

    xx, yy, x, z = eval_surf_2d(fun, bounds, num=100)

    # X = active_learner.result[n_step + 1]["data"]
    X = a.learner.x_input
    prediction = a.learner.result[8]["surface"]
    active_criterion = a.active_criterion

    zz = prediction(x)
    zzz = p.learner.result[8]["surface"](x)
    std = active_criterion(x)

    fig, ax = plot.subplots(ncols=2, figsize=(6, 5), dpi=250)
    sa = ((zz.ravel() - z).reshape(len(xx), len(yy))) ** 2
    sp = ((zzz.ravel() - z).reshape(len(xx), len(yy))) ** 2

    cmap = 'rainbow'
    ax[0].pcolormesh(
        xx, yy, sa, cmap=cmap,
        norm=colors.LogNorm(vmin=1e-4, vmax=sa.max()))
    ax[0].contour(xx, yy, z.reshape(len(xx), len(yy)), levels=10,
                  linewidths=0.3,
                  colors='k')

    im = ax[1].pcolormesh(xx, yy, sp, cmap=cmap,
                          norm=colors.LogNorm(vmin=1e-4, vmax=sp.max()))
    ax[1].contour(xx, yy, z.reshape(len(xx), len(yy)), levels=10,
                  linewidths=0.3,
                  colors='k')

    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plot.colorbar(im, cax=cax)
    ax[0].set_title("Estimation error $\\hat{f}_{active} - f $")
    ax[1].set_title("Estimation error $\\hat{f}_{passive} - f $")
    plot.savefig(f"{path}/passive_vs_active.png")

    plot.figure(figsize=(5, 5), dpi=300)
    plot.pcolormesh(xx, yy, std.reshape(len(xx), len(yy)), cmap="rainbow")
    plot.contour(xx, yy, z.reshape(len(xx), len(yy)), levels=10, linewidths=0.3,
                 colors='k')
    plot.title("Estimation variance $\\sigma(\\hat{f})$")
    plot.savefig(f"{path}/sampling_function.png")

    plot.figure(figsize=(5, 5), dpi=300)
    plot.pcolormesh(xx, yy, std.reshape(len(xx), len(yy)), cmap="rainbow")
    x_new = a.learner.x_new
    plot.scatter(x_new[0], x_new[1], c="k", marker='.')
    plot.scatter(X[0], X[1], c="k", marker='.')
    plot.savefig(f"{path}/sampling.png")

    plot.figure(figsize=(5, 5), dpi=300)
    std_ = std.reshape(len(xx), len(yy))[len(yy) // 2]
    pred = zz.reshape(len(xx), len(yy))[len(yy) // 2]
    f = z.reshape(len(xx), len(yy))[len(yy) // 2]
    fp = zzz.reshape(len(xx), len(yy))[len(yy) // 2]

    plot.plot(yy, f, c="r", label="True function")
    plot.plot(yy, fp, c="grey", alpha=1, label="Passive estimation", ls="--")
    plot.plot(yy, pred, c="b", )
    plot.fill_between(yy, pred, pred + std_, color="b", alpha=0.1)
    plot.fill_between(yy, pred - std_, pred, color="b", alpha=0.1)
    plot.legend()
    plot.savefig(f"{path}/1d_cut.png")

    plot.figure(figsize=(4, 4), dpi=200)
    plot.contour(xx, yy, z.reshape(len(xx), len(yy)), levels=30, linewidths=0.3,
                 colors='k')

    plot.scatter(a.learner.x_input.drop(0)[0], a.learner.x_input.drop(0)[1],
                 cmap="rainbow",
                 c=a.learner.x_input.drop(0).index, marker=".")
    plot.scatter(a.learner.x_input.loc[0][0], a.learner.x_input.loc[0][1],
                 c="k", marker="x")

    eval_p = evaluate(fun, p.learner.result[8]["surface"], bounds, num_mc=10000)
    eval_a = evaluate(fun, a.learner.result[8]["surface"], bounds, num_mc=10000)
    print(
        f"passive performance : {eval_p}\n"
        f"active  performance : {eval_a}")
    plot.tight_layout()
    plot.savefig(f"{path}/sampling_example_function.png")
