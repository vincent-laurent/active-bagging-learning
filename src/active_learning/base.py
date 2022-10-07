import numpy as np
import pandas as pd

from active_learning.components.active_criterion import ActiveCriterion
from active_learning.components.query_strategies import QueryStrategy


class ActiveSRLearner:

    def __init__(
            self,
            active_criterion: ActiveCriterion,
            query_strategy: QueryStrategy,
            X_train: pd.DataFrame,
            y_train: pd.DataFrame,
            bounds=None,
    ):
        self.active_criterion = active_criterion
        self.query_strategy = query_strategy
        self.x_input = X_train.copy()
        self.y_input = y_train.copy()
        self.bounds = bounds
        self.result = {}
        self.iter = 0
        self.budget = len(X_train)
        self.x_input.index = 0 * np.ones(len(self.x_input))

    def teach(self):
        self.active_criterion.fit(
            self.x_input,
            self.y_input)

    def query(self, *args):
        self.iter += 1
        self.teach()
        self.query_strategy.set_active_function(self.active_criterion)
        self.x_new = self.query_strategy.query(*args)
        self.x_new = pd.DataFrame(self.x_new, columns=self.x_input.columns)
        self.save()

        return self.x_new

    def add_labels(self, x: pd.DataFrame, y: pd.DataFrame):
        x.index = self.iter * np.ones(len(x))
        y.index = self.iter * np.ones(len(x))
        self.x_input = pd.concat((x, self.x_input), axis=0)
        self.y_input = pd.concat((y, self.y_input), axis=0)
        self.budget = len(self.x_input)

    def surface(self, x):
        return self.active_criterion.function(x)

    def save(self):
        def surf(x): return self.surface(x)

        def active(x): return self.active_criterion(x)

        self.result[self.iter] = dict(
            surface=surf,
            active_criterion=active,
            budget=int(self.budget),
            data=self.x_input
        )
