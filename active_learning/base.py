# Copyright 2024 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
from copy import deepcopy

from active_learning.components.active_criterion import IActiveCriterion
from active_learning.components.query_strategies import IQueryStrategy


class ActiveSRLearner:

    def __init__(
            self,
            active_criterion: IActiveCriterion,
            query_strategy: IQueryStrategy,
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
        self.x_new = pd.DataFrame()

    def learn(self):
        self.active_criterion.fit(
            self.x_input,
            self.y_input)

    def query(self, *args):
        self.learn()
        self.query_strategy.set_bounds(self.bounds)
        self.query_strategy.set_active_function(self.active_criterion.__call__)
        self.x_new = pd.DataFrame(self.query_strategy.query(*args), columns=self.x_input.columns)
        self.save()

        return self.x_new

    def add_labels(self, x: pd.DataFrame, y: pd.DataFrame):
        self.iter += 1
        x.index = self.iter * np.ones(len(x))
        y.index = self.iter * np.ones(len(x))
        self.x_input = pd.concat((x, self.x_input), axis=0)
        self.y_input = pd.concat((y, self.y_input), axis=0)
        self.budget = len(self.x_input)

    def save(self):

        self.result[self.iter] = dict(
            surface=deepcopy(self.active_criterion.function),
            active_criterion=deepcopy(self.active_criterion),
            budget=int(self.budget),
            data=self.x_input
        )
