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

from active_learning.components import query_strategies

bounds = [[0, 1]]
x0 = np.array([0]).reshape(1, -1)


def fun(x):
    return - (x - 1) ** 2


def test_query_max():
    sqm = query_strategies.ServiceQueryMax(x0.flatten(), bounds)
    sqm.set_active_function(fun)
    fun(x0)
    y = sqm.query()
    assert all(y == 1)


def test_reject():
    sqm = query_strategies.ServiceReject(bounds)
    sqm.set_active_function(fun)
    fun(x0)
    y = sqm.query(3)
    assert len(y) == 3


def test_uniform():
    sqm = query_strategies.ServiceUniform(bounds)
    sqm.set_active_function(fun)
    fun(x0)
    y = sqm.query(3000)
    assert np.abs(y.mean() - 1 / 2) < 1e3
    assert np.abs(y.var() - 1 / 12) < 1e3


def test_composition():
    strategy = 0.5 * query_strategies.ServiceUniform(bounds=[[0, 1]]) + 0.5 * query_strategies.ServiceUniform(
        bounds=[[0, 0.2]])
    ret = strategy.query(200)
    assert len(ret) == 200
    assert len(strategy.strategy_list) == 2
    assert len(strategy.strategy_weights) == 2
