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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import ShuffleSplit
from active_learning.benchmark import functions
from active_learning.components.active_criterion import VarianceCriterion, NoEstimation

fun = functions.grammacy_lee_2009  # The function we want to learn
bounds = np.array(functions.bounds[fun])  # [x1 bounds, x2 bounds]
n = 100
X_train = pd.DataFrame(
    {'x1': (bounds[0, 0] - bounds[0, 1]) * np.random.rand(n) + bounds[0, 1],
     'x2': (bounds[1, 0] - bounds[1, 1]) * np.random.rand(n) + bounds[1, 1],
     })  # Initiate distribution
y_train = -fun(X_train)



bounds = [[0, 1]]


def unknown_function(x):
    return x ** 5 * np.sin(10 * np.pi * x)


def sampler(n):
    x0 = np.random.uniform(*bounds[0], size=n)
    return pd.DataFrame(x0)


kernel = 1 * RBF(0.1)
krg = GaussianProcessRegressor(kernel=kernel)


def test_active_criterion():
    SEED = 1234
    RNG = np.random.default_rng(seed=SEED)
    svc = VarianceCriterion(krg, ShuffleSplit())
    svc.fit(X_train, y_train)

    pred = svc.function(X_train)

    assert np.sum((pred - y_train)**2) < 1
    assert all(svc(X_train) > 0)
    assert not all(pred > 0)


def test_no_estimation():
    no = NoEstimation()
    no.fit(None, None)
    no([1], None)
    assert all(no.function([1, 2, 3]) == [0, 0, 0])
