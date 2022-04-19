import numpy as np
import pandas as pd
from sklearn import preprocessing


class ActiveSRLearner:

    def __init__(
            self,
            estimator: callable,
            query_strategy: callable,
            X_train: pd.DataFrame,
            y_train: pd.DataFrame,
            scale_input=True,
            bounds=None,
            estimator_parameters: dict = None,
            query_parameters: dict = None
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
        self.__scale()
        if estimator_parameters is None:
            self.estimator_parameters = {}
        else:
            self.estimator_parameters = estimator_parameters

        if query_parameters is None:
            self.query_parameters = {}
        else:
            self.query_parameters = query_parameters

    # =========================================================================
    #                       Scaling data
    # =========================================================================
    def __scale(self, method=preprocessing.MinMaxScaler):
        if self.__scale_input:
            self.scaling_method = method()

        else:
            self.scaling_method = preprocessing.StandardScaler(with_std=False,
                                                               with_mean=False)
        self.scaling_method.fit(self.bounds.T)
        self.x_input_scaled__ = self.__scale_x(self.x_input)

    def __scale_x(self, x):
        return pd.DataFrame(self.scaling_method.transform(x),
                            columns=x.columns, index=x.index)

    def __unscale_x(self, x):
        return pd.DataFrame(self.scaling_method.inverse_transform(x),
                            columns=x.columns, index=x.index)

    # =========================================================================
    #                       Teach and query
    # =========================================================================
    def teach(self):
        self.iter += 1
        self.surface_, self.active_criterion_ = self.estimator(
            self.x_input_scaled__,
            self.y_input,
            **self.estimator_parameters)

    def query(self, size=10):
        self.iter += 1
        self.surface_, self.active_criterion_ = self.estimator(
            self.x_input_scaled__,
            self.y_input,
            **self.estimator_parameters)
        self.x_new, self.scores_ = self.query_strategy(
            self.x_input_scaled__, self.y_input,
            self.active_criterion_, size,
            bounds=self.scaling_method.transform(self.bounds.T).T,
            **self.query_parameters)
        self.x_new = pd.DataFrame(self.x_new, columns=self.x_input.columns)
        self.save()
        self.x_new = self.__unscale_x(self.x_new)
        return self.x_new

    def add_labels(self, x: pd.DataFrame, y: pd.DataFrame):
        x.index = self.iter * np.ones(len(x))
        y.index = self.iter * np.ones(len(x))
        self.x_input = pd.concat((x, self.x_input), axis=0)
        self.y_input = pd.concat((y, self.y_input), axis=0)
        self.budget = len(self.x_input)

        self.x_input_scaled__ = self.__scale_x(self.x_input)

    def surface(self, x):
        return self.surface_(self.scaling_method.transform(x))

    def active_criterion(self, x):
        return self.active_criterion_(self.scaling_method.transform(x))

    def estimate_error(self, bounds):
        pass

    def save(self):

        def surf(x): return self.surface(x)

        def active(x): return self.active_criterion(x)

        self.result[self.iter] = dict(
            surface=surf,
            active_criterion=active,
            budget=int(self.budget),
            data=self.x_input
        )
