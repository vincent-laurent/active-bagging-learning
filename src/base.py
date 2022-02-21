import numpy as np
import pandas as pd
from sklearn import clone
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVR

from sampling.latin_square import iterative_sampler


class ActiveSRLearner:
    def __init__(
            self,
            estimator: callable,
            query_strategy: callable,
            X_train: pd.DataFrame,
            y_train: pd.DataFrame,
            scale_input=True,
            bounds=None,
    ):
        self.estimator = estimator
        self.query_strategy = query_strategy
        self.x_input = X_train
        self.y_input = y_train
        self.bounds = bounds
        self.__scale_input = scale_input
        self.result = {}
        self.iter = 0
        self.budget = len(X_train)
        self.x_input.index = 0 * np.ones(len(self.x_input))
        self.scale()

    def scale(self, method=preprocessing.MinMaxScaler):
        if self.__scale_input:
            self.scaling_method = method()

        else:
            self.scaling_method = preprocessing.StandardScaler(with_std=False,
                                                               with_mean=False)
        self.scaling_method.fit(self.bounds.T)
        self.x_input_scaled__ = self.scale_x(self.x_input)

    def scale_x(self, x):
        return pd.DataFrame(self.scaling_method.transform(x),
                            columns=x.columns, index=x.index)

    def unscale_x(self, x):
        return pd.DataFrame(self.scaling_method.inverse_transform(x),
                            columns=x.columns, index=x.index)

    def run(self, size=10):
        self.iter += 1
        self.surface_, self.coverage_ = self.estimator(
            self.x_input_scaled__,
            self.y_input,
            return_coverage=True)
        self.x_new, self.scores_ = self.query_strategy(
            self.x_input_scaled__, self.y_input,
            self.coverage_, size,
            bounds=self.scaling_method.transform(self.bounds.T).T)
        self.x_new = pd.DataFrame(self.x_new, columns=self.x_input.columns)
        self.save()
        return self.unscale_x(self.x_new)

    def add_labels(self, x: pd.DataFrame, y: pd.DataFrame):
        x.index = self.iter * np.ones(len(x))
        y.index = self.iter * np.ones(len(x))
        self.x_input = pd.concat((x, self.x_input), axis=0)
        self.y_input = pd.concat((y, self.y_input), axis=0)
        self.budget = len(self.x_input)

        self.x_input_scaled__ = self.scale_x(self.x_input)

    def surface(self, x):
        return self.surface_(self.scaling_method.transform(x))

    def coverage(self, x):
        return self.coverage_(self.scaling_method.transform(x))

    def estimate_error(self, bounds):
        pass

    def save(self):

        def surf(x): return self.surface(x)

        def coverage_(x): return self.coverage(x)

        self.result[self.iter] = dict(
            surface=surf,
            coverage=coverage_,
            budget=int(self.budget),
            data=self.x_input
        )


def get_pointwise_variance(estimator_list):
    keys = range(len(estimator_list))

    def meta_estimator(*args, **kwargs):
        m = estimator_list[0].predict(*args, **kwargs)
        s = m ** 2
        for key in keys[1:]:
            elt = estimator_list[key].predict(*args, **kwargs)
            m += elt
            s += elt ** 2
        return (s / len(keys)) - (m / len(keys)) ** 2

    return meta_estimator


def gaussian_est_jacknife(X, y, return_coverage=True,
                          base_estimator=GaussianProcessRegressor(
                              kernel=ExpSineSquared(), alpha=10)):
    m = []
    X, y = np.array(X), np.array(y).ravel()
    for train, test in ShuffleSplit(n_splits=40).split(X, y):
        model = clone(base_estimator)
        m.append(model.fit(X[train, :], y[train]))

    def mean_predictor(x):
        res = 0
        for model_ in m:
            res += model_.predict(x)
        return res / len(m)

    if return_coverage:
        return mean_predictor, get_pointwise_variance(m)
    else:
        return mean_predictor


def rf_est_jacknife(X, y, return_coverage=True,
                    base_estimator=RandomForestRegressor(min_samples_leaf=3)):
    m = []
    X, y = np.array(X), np.array(y).ravel()
    for train, test in ShuffleSplit(n_splits=20).split(X, y):
        model = clone(base_estimator)
        m.append(model.fit(X[train, :], y[train]))

    def mean_predictor(x):
        res = 0
        for model_ in m:
            res += model_.predict(x)
        return res / len(m)

    if return_coverage:
        return mean_predictor, get_pointwise_variance(m)
    else:
        return mean_predictor


def dt_est_jacknife(X, y, return_coverage=True,
                    base_estimator=RandomForestRegressor(
                        min_samples_leaf=3, n_estimators=300)):
    m = []
    X, y = np.array(X), np.array(y).ravel()
    model = clone(base_estimator)
    model.fit(X, y)

    if return_coverage:
        return model.predict, get_pointwise_variance(
            model.estimators_)
    else:
        return model.predict


def svr_est_jacknife(X, y, return_coverage=True,
                     base_estimator=SVR(kernel="rbf", C=100, gamma=0.1,
                                        epsilon=0.1)):
    m = []
    X, y = np.array(X), np.array(y).ravel()
    for train, test in ShuffleSplit(n_splits=10).split(X, y):
        model = clone(base_estimator)
        m.append(model.fit(X[train, :], y[train]))

    def mean_predictor(x):
        res = 0
        for model_ in m:
            res += model_.predict(x)
        return res / len(m)

    if return_coverage:
        return mean_predictor, get_pointwise_variance(m)
    else:
        return mean_predictor


def gaussian_est(X, y, return_coverage=True):
    from sklearn.gaussian_process import GaussianProcessRegressor
    model = GaussianProcessRegressor().fit(X, y)
    if return_coverage:
        def coverage_function(x):
            return model.predict(x, return_std=True)[1]

        return model.predict, coverage_function
    else:
        return model.predict


def reject_on_bounds(X, y, coverage_function, size=10, batch_size=50,
                     bounds=None):
    from scipy.stats import rankdata
    if bounds is None:
        x_new = iterative_sampler(X, size=batch_size)
    else:
        x_new = iterative_sampler(x_limits=bounds, size=batch_size)
    cov = coverage_function(x_new)
    order = rankdata(-cov, method="ordinal")
    selector = order <= size
    return x_new[selector], cov[selector]


if __name__ == '__main__':
    from data.functions import golden_price, bounds
    import matplotlib.pyplot as plot

    fun = golden_price
    # test 2D
    bounds = np.array(bounds[fun])
    xx = np.linspace(bounds[0, 0], bounds[0, 1], num=100)
    yy = np.linspace(bounds[1, 0], bounds[1, 1], num=100)
    x, y = np.meshgrid(xx, yy)
    x = pd.DataFrame(dict(x0=x.ravel(), x1=y.ravel()))
    z = -fun(x.values)

    plot.pcolormesh(xx, yy, z.reshape(len(xx), len(yy)), cmap="rainbow")

    X = x.sample(n=12)
    y = -fun(X)
    active_learner = ActiveSRLearner(gaussian_est_jacknife, reject_on_bounds, X,
                                     y,
                                     bounds=bounds)
    x_new = active_learner.run(2)

    prediction = active_learner.surface
    coverage = active_learner.coverage

    zz = prediction(x)
    std = coverage(x)

    plot.figure()
    plot.pcolormesh(xx, yy, (zz - z).reshape(len(xx), len(yy)), cmap="rainbow")
    plot.figure()
    plot.pcolormesh(xx, yy, std.reshape(len(xx), len(yy)), cmap="rainbow")

    plot.figure()
    plot.pcolormesh(xx, yy, std.reshape(len(xx), len(yy)), cmap="rainbow")
    plot.scatter(x_new["x0"], x_new["x1"], c="k")
    plot.scatter(X["x0"], X["x1"], c="b")

    plot.figure()
    std_ = std.reshape(len(xx), len(yy))[len(yy) // 2] * 10000
    pred = zz.reshape(len(xx), len(yy))[len(yy) // 2]
    f = z.reshape(len(xx), len(yy))[len(yy) // 2]

    plot.plot(yy, pred)
    plot.fill_between(yy, pred, pred + std_, color="b", alpha=0.5)
    plot.fill_between(yy, pred - std_, pred, color="b", alpha=0.5)
    plot.plot(yy, f)
