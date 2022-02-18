import numpy as np
import pandas as pd
from sklearn import preprocessing
from sampling.latin_square import iterative_sampler


class ActiveSRLearner:
    def __init__(
            self,
            estimator: callable,
            query_strategy: callable,
            X_train: pd.DataFrame,
            y_train: pd.DataFrame,
            scale_input=True
    ):
        self.estimator = estimator
        self.query_strategy = query_strategy
        self.x_input = X_train
        self.y_input = y_train
        if scale_input:
            self.scale()
        else:
            self.scaling_method = preprocessing.StandardScaler(with_std=False, with_mean=False)
            self.scaling_method.fit(X_train)

    def scale(self, method=preprocessing.MinMaxScaler):
        self.scaling_method = method()
        self.x_input = self.scaling_method.fit_transform(self.x_input)

    def run(self, size=10):
        self.surface_, self.coverage_ = self.estimator(self.x_input, self.y_input, return_coverage=True)
        self.x_new = self.query_strategy(self.x_input, self.y_input, self.coverage_, size)

        return self.scaling_method.inverse_transform(x_new)

    def surface(self, x):
        return self.surface_(self.scaling_method.transform(x))

    def coverage(self, x):
        return self.coverage_(self.scaling_method.transform(x))


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


if __name__ == '__main__':
    from data.functions import himmelblau, branin
    import matplotlib.pyplot as plot

    xx = np.linspace(-5, 5, num=100)
    yy = np.linspace(-5, 5, num=100)
    x, y = np.meshgrid(xx, yy)
    x = pd.DataFrame(dict(x0=x.ravel(), x1=y.ravel()))
    z = -himmelblau(x.values)

    plot.pcolormesh(xx, yy, z.reshape(len(xx), len(yy)), cmap="rainbow")

    def gaussian_est(X, y, return_coverage=True):
        from sklearn.gaussian_process import GaussianProcessRegressor
        model = GaussianProcessRegressor().fit(X, y)
        if return_coverage:
            def coverage_function(x):
                return model.predict(x, return_std=True)[1]

            return model.predict, coverage_function
        else:
            return model.predict


    def reject_on_bounds(X, y, coverage_function, size=10, batch_size=50):
        from scipy.stats import rankdata
        x_new = iterative_sampler(X, size=batch_size)
        cov = coverage_function(x_new)
        order = rankdata(-cov, method="ordinal")
        selector = order < size
        return x_new[selector], cov[selector]


    X = x.sample(n=5)
    active_learner = ActiveSRLearner(gaussian_est, reject_on_bounds, X, -himmelblau(X))
    x_new = active_learner.run(10)

    prediction = active_learner.surface_
    coverage = active_learner.coverage_

    zz = prediction(x)
    std = coverage(x)

    plot.figure()
    plot.pcolormesh(xx, yy, (zz - z).reshape(len(xx), len(yy)), cmap="rainbow")
    plot.figure()
    plot.pcolormesh(xx, yy, std.reshape(len(xx), len(yy)), cmap="rainbow")

    x_new, cov = reject_on_bounds(x.iloc[select], z[select], coverage_function=coverage_function, size=100,
                                  batch_size=100)

    plot.figure()
    plot.pcolormesh(xx, yy, std.reshape(len(xx), len(yy)), cmap="rainbow")
    plot.scatter(scaler.inverse_transform(x_new)[:, 0], scaler.inverse_transform(x_new)[:, 1], c="k")
    plot.scatter(x["x0"].iloc[select], x["x1"].iloc[select], c="b")

    x_new, cov = reject_on_bounds(x.iloc[select], z[select], coverage_function=coverage_function, size=20,
                                  batch_size=50)
    plot.figure()
    plot.pcolormesh(xx, yy, std.reshape(len(xx), len(yy)), cmap="rainbow")
    plot.scatter(scaler.inverse_transform(x_new)[:, 0], scaler.inverse_transform(x_new)[:, 1], c="k")
    plot.scatter(x["x0"].iloc[select], x["x1"].iloc[select], c="b")
