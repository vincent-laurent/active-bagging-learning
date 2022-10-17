import matplotlib

try:
    matplotlib.use("qt5agg")
except:
    pass
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVR

from active_learning.components.active_criterion import VarianceBis
from active_learning.components.query_strategies import Uniform, QueryVariancePDF
from sklearn.model_selection import ShuffleSplit
from active_learning.components.sampling import latin_square
from active_learning.components.test import TestingClass
from active_learning.data import functions
from active_learning.models.smt_api import SurrogateKRG
from benchmark.utils import evaluate, eval_surf_2d
from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

functions_ = list(functions.bounds.keys())
__n_sim__ = 1
name = "marelli_2018"
fun = functions.__dict__[name]

estimator_parameters = {
    0: dict(base_estimator=SVR(kernel="rbf", C=100, gamma="scale",
                               epsilon=0.1)),
    1: dict(base_estimator=GaussianProcessRegressor(kernel=RBF(length_scale=0.012))),
    2: dict(base_estimator=RandomForestRegressor(n_estimators=5,
                                                 min_samples_leaf=1)),
    3: dict(base_estimator=SurrogateKRG(), splitter=ShuffleSplit(n_splits=2))
}
estimator = estimator_parameters[3]


def analyse_results(path):
    res = pd.read_csv(path)
    print(res)


est = Pipeline([("est", KNeighborsClassifier())])
est = Pipeline([("est", SVC(C=500))])
est = Pipeline([("poly", PolynomialFeatures(degree=[1, 3])), ("est", LinearRegression())])
est = Pipeline([("est", RandomForestRegressor(n_estimators=10, bootstrap=True, max_samples=0.9))])
est = Pipeline([("est", SVC(C=50, gamma=1))])


if __name__ == '__main__':
    n0 = 40
    budget = 70
    steps = 10

    bounds = np.array(functions.bounds[fun])

    def sampler(size):
        return pd.DataFrame(latin_square.scipy_lhs_sampler(size=size, x_limits=bounds))

    a = TestingClass(
        budget, n0, fun,
        VarianceBis(
            estimator=est, splitter=ShuffleSplit(n_splits=4, train_size=0.9)),
        QueryVariancePDF(
            bounds, num_eval=int(4000)),
        sampler, bounds=bounds, n_steps=steps)
    p = TestingClass(
        budget, n0, fun,
        VarianceBis(
            estimator=est, splitter=ShuffleSplit(n_splits=4, train_size=0.9)),
        Uniform(bounds),

        sampler, bounds=bounds, n_steps=steps)
    a.run()
    p.run()

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    xx, yy, x, z = eval_surf_2d(fun, bounds, num=200)

    # X = active_learner.result[n_step + 1]["data"]
    X = a.learner.x_input
    prediction = a.learner.result[8]["surface"]
    active_criterion = a.active_criterion

    zz = prediction(x)
    zzz = p.learner.result[8]["surface"](x)
    std = active_criterion(x)

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
    x_new = a.learner.x_new
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

    plot.figure(figsize=(4, 4), dpi=200)
    plot.contour(xx, yy, z.reshape(len(xx), len(yy)), levels=10, linewidths=0.3,
                 colors='k')

    plot.scatter(a.learner.x_input[0], a.learner.x_input[1], cmap="rainbow", c=X.index, marker=".")
    # plot.scatter(x0[:, 0], x0[:, 1], c="k")
    # plot.scatter(passive_learner.result[1]["data"][0],
    #              passive_learner.result[1]["data"][1], c="k", marker="+")
    eval_p = evaluate(fun, p.learner.result[8]["surface"], bounds, num_mc=10000)
    eval_a = evaluate(fun, a.learner.result[8]["surface"], bounds, num_mc=10000)
    print(
        f"passive performance : {eval_p}\n"
        f"active  performance : {eval_a}")
    plot.tight_layout()
    plot.savefig(f"benchmark/figures/sampling_example_function_{name}.png")
