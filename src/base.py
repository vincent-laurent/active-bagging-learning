import numpy as np
import pandas as pd
from sklearn import preprocessing


class ActiveSRLearner:
    def __init__(
            self,
            estimator: callable,
            coverage_method: callable,
            query_strategy: callable,
            X_train: pd.DataFrame,
            y_train: pd.DataFrame,
            scale=True
    ):
        self.estimator = estimator
        self.coverage_method = coverage_method
        self.query_strategy = query_strategy
        self.x_input = X_train
        self.y_input = y_train
        if scale:
            self.scale()

    def scale(self, method=preprocessing.StandardScaler):
        self.scaling_method = method()
        self.x_input = self.scaling_method.fit_transform(self.x_input)

    def __iter__(self):
        pass


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

    def est(X, y):
        from sklearn.gaussian_process import GaussianProcessRegressor
        model = GaussianProcessRegressor().fit(X, y)
        return model.predict
    select = np.random.choice(range(len(x)), size=20)
    prediction = est(x.iloc[select], z[select])
    zz = prediction(x)

    plot.pcolormesh(xx, yy, (zz - z).reshape(len(xx), len(yy)), cmap="rainbow")
