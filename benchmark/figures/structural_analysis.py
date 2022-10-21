import matplotlib

try:
    matplotlib.use("qt5agg")
except:
    pass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from active_learning.components.test import TestingClass
from active_learning.components.active_criterion import VarianceBis
from active_learning.components import query_strategies
from active_learning.components.sampling import latin_square
from sklearn.model_selection import ShuffleSplit
from active_learning.data import functions
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import Matern
from benchmark.utils import evaluate, eval_surf_2d
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    name = "marelli_2018"
    fun = functions.__dict__[name]

    name = "marelli_2018"
    fun = functions.__dict__[name]

    est = VarianceBis(
        estimator=Pipeline([("scale", StandardScaler()), ("est", SVC(degree=2, C=5, gamma=10))]),
        splitter=ShuffleSplit(n_splits=3, train_size=0.8))
    crit = query_strategies.QueryVariancePDF(num_eval=5000)

    kernel = 1 * RBF(0.01)
    kernel2 = Matern(nu=1, length_scale=0.02)
    # est = VarianceBis(
    #     estimator=Pipeline([("scale", StandardScaler()),
    #                         # ("feat", PolynomialFeatures(degree=10, include_bias=True)),
    #                         ("est", gaussian_process.GaussianProcessClassifier(kernel=kernel2))]),
    #     splitter=ShuffleSplit(n_splits=2, train_size=0.9))
    # est= VarianceBis(estimator=KNeighborsClassifier(n_neighbors=3),
    #                  splitter=ShuffleSplit(n_splits=2, train_size=0.9)
    #
    # )
    # crit = query_strategies.QueryVariancePDF(num_eval=500)

    n0 = functions.budget_parameters[name]["n0"]*5
    budget = functions.budget_parameters[name]["budget"]*10
    steps = 30
    bounds = np.array(functions.bounds[fun])


    def sampler(size):
        return pd.DataFrame(latin_square.scipy_lhs_sampler(size=size, x_limits=bounds))


    bounds = np.array(functions.bounds[fun])
    a = TestingClass(
        budget, n0, fun,
        est,
        crit,
        sampler, bounds=bounds, n_steps=steps)
    a.run()
    xx, yy, x, z = eval_surf_2d(fun, bounds, num=400)
    # X = active_learner.result[n_step + 1]["data"]
    X = a.learner.x_input


    active_criterion = a.active_criterion

    std = active_criterion(x)

    fig, ax = plot.subplots(figsize=(6, 5), dpi=250)

    ax.contour(xx, yy, z.reshape(len(xx), len(yy)), levels=1, linewidths=0.3,
               colors='k')
    for i in a.learner.result.keys():
        zzz = a.learner.result[i]["surface"](x)
        c = plot.get_cmap("rainbow")(i/(steps - 1))
        ax.contour(xx, yy, zzz.reshape(len(xx), len(yy)), levels=1, linewidths=0.3,
                   colors=[c])

    plot.figure(figsize=(5, 5), dpi=300)
    plot.pcolormesh(xx, yy, std.reshape(len(xx), len(yy)), cmap="rainbow")
    plot.contour(xx, yy, z.reshape(len(xx), len(yy)), levels=10, linewidths=0.3,
                 colors='k')
    plot.title("Estimation variance $\\sigma(\\hat{f})$")

    plot.figure(figsize=(4, 4), dpi=200)
    plot.contour(xx, yy, z.reshape(len(xx), len(yy)), levels=30, linewidths=0.3,
                 colors='k')

    plot.scatter(a.learner.x_input.drop(0)[0], a.learner.x_input.drop(0)[1], cmap="rainbow",
                 c=a.learner.x_input.drop(0).index, marker=".")
    plot.scatter(a.learner.x_input.loc[0][0], a.learner.x_input.loc[0][1],
                 c="k", marker="x")
