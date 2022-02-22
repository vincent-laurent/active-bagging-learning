import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVR

import base
from benchmark.utils import evaluate
from data import functions
from models.smt_api import SurrogateKRG
from sampling import latin_square

functions_ = list(functions.bounds.keys())

fun = functions_[1]
n0 = 30
budget = 100
n_step = 70

estimator_parameters = {
    0: dict(base_estimator=SVR(kernel="rbf", C=100, gamma=0.1,
                               epsilon=0.1)),
    1: dict(base_estimator=GaussianProcessRegressor()),
    2: dict(base_estimator=RandomForestRegressor(min_samples_leaf=3)),
    3: dict(base_estimator=SurrogateKRG(), splitter=ShuffleSplit(n_splits=5))
}


def run(fun, n0=10, budget=100, n_step=5):
    bounds = functions.bounds[fun]
    x0 = latin_square.iterative_sampler(x_limits=np.array(bounds), size=n0,
                                        batch_size=n0 // 2)
    xall = latin_square.iterative_sampler(x_limits=np.array(bounds),
                                          size=budget)
    args = dict(
        bounds=np.array(bounds),
        estimator_parameters=estimator_parameters[3],
        query_parameters=dict(batch_size=30)
    )
    active_learner = base.ActiveSRLearner(base.estimate_variance,
                                          base.reject_on_bounds,
                                          pd.DataFrame(x0),
                                          pd.DataFrame(fun(x0)), **args
                                          )

    perf = []
    perf_passive = []
    for step in range(n_step):
        b = active_learner.budget
        print(active_learner.budget)
        x_new = active_learner.run(
            int((budget - n0) / n_step)).drop_duplicates()
        y_new = pd.DataFrame(fun(x_new))
        active_learner.add_labels(x_new, y_new)
        perf.append(evaluate(
            fun,
            active_learner.surface,
            bounds, num_mc=1000))

        s = pd.DataFrame(xall).sample(b)
        passive_learner = base.ActiveSRLearner(
            active_learner.estimator,
            base.reject_on_bounds,
            s, fun(s), **args
        )

        passive_learner.run(1)
        perf_passive.append(evaluate(
            fun,
            passive_learner.surface,
            bounds, num_mc=1000))
    plot.figure()
    plot.plot(perf, c="r")
    plot.plot(perf_passive)

    return active_learner, passive_learner


if __name__ == '__main__':
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    a, p = run(fun, n0=10, budget=40, n_step=6)

    bounds = np.array(functions.bounds[fun])
    xx = np.linspace(bounds[0, 0], bounds[0, 1], num=200)
    yy = np.linspace(bounds[1, 0], bounds[1, 1], num=200)
    x, y = np.meshgrid(xx, yy)
    x = pd.DataFrame(dict(x0=x.ravel(), x1=y.ravel()))
    z = fun(x.values)

    plot.figure()
    plot.pcolormesh(xx, yy, z.reshape(len(xx), len(yy)), cmap="rainbow")

    active_learner = a
    passive_learner = p

    # X = active_learner.result[n_step + 1]["data"]
    X = active_learner.x_input
    prediction = active_learner.surface
    coverage = active_learner.coverage

    zz = prediction(x)
    zzz = passive_learner.surface(x)
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
    x_new = a.scaling_method.transform(a.x_new)
    plot.scatter(a.x_new[0], a.x_new[1], c="k")
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
    plot.contour(xx, yy, z.reshape(len(xx), len(yy)), levels=50, linewidths=0.3,
                 colors='k')
    x_new = a.scaling_method.transform(a.x_new)
    plot.scatter(X[0], X[1], cmap="rainbow", c=X.index)
    # plot.scatter(x0[:, 0], x0[:, 1], c="k")
    # plot.scatter(passive_learner.result[1]["data"][0],
    #              passive_learner.result[1]["data"][1], c="k", marker="+")

    evaluate(
        true_function=fun,
        learned_surface=prediction,
        bounds=list(bounds),
        num_mc=100000, l=1)