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

import pandas as pd
from typing import Union

from active_learning.components.active_criterion import IActiveCriterion, NoEstimation
from active_learning.components.query_strategies import IQueryStrategy


class ActiveSurfaceLearner:

    def __init__(
            self,
            active_criterion: Union[IActiveCriterion, None],
            query_strategy: IQueryStrategy,
            bounds=None,
    ):
        self.__active_criterion = active_criterion if active_criterion is not None else NoEstimation()
        self.__query_strategy = query_strategy
        self.__bounds = bounds

    def fit(self, X: pd.DataFrame, y):
        self.active_criterion.fit(X, y)
        self.__columns = X.columns

    def query(self, *args) -> pd.DataFrame:
        self.query_strategy.set_bounds(self.__bounds)
        self.query_strategy.set_active_function(self.active_criterion.__call__)
        x_new = pd.DataFrame(self.query_strategy.query(*args), columns=self.__columns)
        return x_new

    @property
    def active_criterion(self) -> IActiveCriterion:
        return self.__active_criterion

    @property
    def query_strategy(self) -> IQueryStrategy:
        return self.__query_strategy

    @property
    def surface(self) -> callable:
        return self.__active_criterion.function

    @property
    def predict(self) -> callable:
        return self.__active_criterion.function

    @property
    def bounds(self) -> iter:
        return self.__bounds

    def set_bounds(self, bounds) -> bounds:
        self.__bounds = bounds
