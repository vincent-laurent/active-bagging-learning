import matplotlib.pyplot as plot
import numpy as np
import pandas as pd

from benchmark.naive import add_to_benchmark, run, plot_results
from benchmark.utils import evaluate, eval_surf_2d
from data import functions

functions_ = list(functions.bounds.keys())
__n_sim__ = 20
name = "grammacy_lee_2009_rand"
fun = functions.__dict__[name]


def analyse_results(path):
    res = pd.read_csv(path)
    print(res)


if __name__ == '__main__':
    analyse_results("benchmark/results.csv")

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    n0 = 60
    n_step = 10
    budget = n0 + n_step
    a, p, r = None, None, None
    for i in range(__n_sim__):
        a, p, r = run(fun, n0=n0, budget=budget, n_step=n_step)
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

    ax[0].pcolormesh(xx, yy, sa,
                     cmap="GnBu", vmin=sa.min(), vmax=sp.max())
    im = ax[1].pcolormesh(xx, yy, sp,
                          cmap="GnBu", vmin=sa.min(), vmax=sp.max())

    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plot.colorbar(im, cax=cax)
    ax[0].set_title("Estimation error $\\hat{f}_{active} - f $")
    ax[1].set_title("Estimation error $\\hat{f}_{passive} - f $")

    plot.figure()
    plot.pcolormesh(xx, yy, std.reshape(len(xx), len(yy)), cmap="rainbow")
    plot.colorbar()
    plot.title("Estimation error $\\sigma(\\hat{f})$")

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
    eval_p = evaluate(fun, p.surface, bounds, num_mc=10000)
    eval_a = evaluate(fun, a.surface, bounds, num_mc=10000)
    print(
        f"passive performance : {eval_p}\n"
        f"active  performance : {eval_a}")

    plot.figure()
    plot_results(n0=n0, function=name)