import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

from data import functions
from data.functions import bounds
from models.smt_api import SurrogateKRG
from base import ActiveSRLearner
from components.active_criterion import estimate_variance
from components.query_strategies import reject_on_bounds

if __name__ == '__main__':
    fun = functions.grammacy_lee_2009
    # test 2D
    bounds = np.array(bounds[fun])
    xx = np.linspace(bounds[0, 0], bounds[0, 1], num=200)
    yy = np.linspace(bounds[1, 0], bounds[1, 1], num=200)
    x, y = np.meshgrid(xx, yy)
    x = pd.DataFrame(dict(x0=x.ravel(), x1=y.ravel()))
    z = -fun(x.values)

    plot.pcolormesh(xx, yy, z.reshape(len(xx), len(yy)), cmap="rainbow")

    X = x.sample(n=20)
    y = -fun(X)
    active_learner = ActiveSRLearner(estimate_variance, reject_on_bounds, X,
                                     y,
                                     bounds=bounds,
                                     estimator_parameters=dict(
                                         base_estimator=SurrogateKRG(),
                                         splitter=ShuffleSplit(n_splits=5)
                                     ),
                                     query_parameters=dict(
                                         batch_size=40,
                                     ))
    x_new = active_learner.query(2)

    prediction = active_learner.surface
    coverage = active_learner.active_criterion

    zz = prediction(x).ravel()
    std = coverage(x).ravel()

    plot.figure()
    plot.pcolormesh(xx, yy, (zz - z).reshape(len(xx), len(yy)), cmap="rainbow")
    plot.figure()
    plot.pcolormesh(xx, yy, std.reshape(len(xx), len(yy)), cmap="rainbow")

    plot.figure()
    plot.pcolormesh(xx, yy, std.reshape(len(xx), len(yy)), cmap="rainbow")
    plot.scatter(x_new["x0"], x_new["x1"], c="k")
    plot.scatter(X["x0"], X["x1"], c="b")

    plot.figure()
    std_ = std.reshape(len(xx), len(yy))[len(yy) // 2]
    pred = zz.reshape(len(xx), len(yy))[len(yy) // 2]
    f = z.reshape(len(xx), len(yy))[len(yy) // 2]

    plot.plot(yy, pred)
    plot.fill_between(yy, pred, pred + std_, color="b", alpha=0.5)
    plot.fill_between(yy, pred - std_, pred, color="b", alpha=0.5)
    plot.plot(yy, f)
