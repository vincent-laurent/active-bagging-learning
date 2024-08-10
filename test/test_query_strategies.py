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

from active_learning.components import query_strategies as qs

bounds = [[0, 1]]
x0 = np.array([0]).reshape(1, -1)


def fun(x):
    return - (x - 1) ** 2


def test_query_max():
    sqm = qs.ServiceQueryMax(x0.flatten(), bounds)
    sqm.set_active_function(fun)
    fun(x0)
    y = sqm.query()
    assert all(y == 1)


def test_reject():
    sqm = qs.ServiceReject(bounds)
    sqm.set_active_function(fun)
    fun(x0)
    y = sqm.query(3)
    assert len(y) == 3


def test_uniform():
    sqm = qs.ServiceUniform(bounds)
    sqm.set_active_function(fun)
    fun(x0)
    y = sqm.query(3000)
    assert np.abs(y.mean().mean() - 1 / 2) < 1e3
    assert np.abs(y.var().mean() - 1 / 12) < 1e3


def test_composition():
    strategy = qs.ServiceUniform(bounds=[[0, 1]]) + qs.ServiceUniform(
        bounds=[[0, 0.2]])
    assert len(strategy.strategy_weights) == 2

    strategy.query(1)


def test_composition_and_proportion():
    strategy = 20 * qs.ServiceUniform(bounds=[[0.2, 1]]) + qs.ServiceUniform(
        bounds=[[0, 0.2]])
    x = strategy.query(200)
    n1 = len(x[x < 0.2])
    n2 = len(x[x > 0.2])

    assert n2 > 150
    assert n1 < 80

    assert len(x) == 200


def test_composition_one_point():
    strat = qs.ServiceQueryMax(np.array([0]))
    strat.set_active_function(lambda x: 1)
    strategy = 20 * strat
    x = strategy.query()
    assert len(x) == 1


def test_composition_setting_bounds():
    strat = qs.ServiceUniform(None)
    strategy = 20 * strat
    strategy.set_bounds([[0, 1]])
    x = strategy.query(10)
    assert len(x) == 10
