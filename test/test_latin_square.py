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

from active_learning.benchmark import functions
from active_learning.components import latin_square


def test_latin_square():
    x = latin_square.scipy_lhs_sampler(100)
    assert len(x) == 100
    assert x.min() >= 0
    assert x.max() <= 1

    x = latin_square.scipy_lhs_sampler(100, dim=2)
    assert len(x) == 100
    assert x.min() >= 0
    assert x.max() <= 1
    assert x.shape == (100, 2)

    x = latin_square.scipy_lhs_sampler(100, x_limits=np.array([[1, 2], [1, 2]]))
    assert len(x) == 100
    assert x.min() >= 1
    assert x.max() <= 2
    assert x.shape == (100, 2)


def test_iterative():
    # FIXME to put in benchmark
    import matplotlib.pyplot as plot
    c = plot.get_cmap("magma")
    plot.figure(figsize=(6, 6), dpi=200)
    x0 = latin_square.iterative_sampler(dim=2)
    n_steps = 20
    x = x0
    plot.scatter(x0[:, 0], x0[:, 1], color=c(0), label=f"Iter n°{0}")
    for i in range(n_steps):
        color = c((i + 1) / n_steps)
        x_new = latin_square.iterative_sampler(x, size=10)
        print(len(x_new))
        if i % 2 == 0:
            plot.scatter(x_new[:, 0], x_new[:, 1], color=color,
                         label=f"Iter n°{i + 1}")
        plot.scatter(x_new[:, 0], x_new[:, 1], color=color)
        x = np.concatenate((x_new, x))
    plot.grid()
    legend = plot.legend(facecolor='white', framealpha=1)

    plot.figure(figsize=(6, 6), dpi=200)
    plot.scatter(x[:, 0], x[:, 1], color="k")
    plot.figure(figsize=(6, 6), dpi=200)

    x_once = latin_square.iterative_sampler(dim=200, size=len(x))
    plot.scatter(x_once[:, 0], x_once[:, 1], color="k")

    x1d = latin_square.one_d_iterative_sampler(x_limits=[[0, 1]], size=10)
    x1d_new = latin_square.one_d_iterative_sampler(x_input=x1d, size=10)

    plot.figure()
    plot.scatter(x1d, np.random.random(size=len(x1d)))
    plot.scatter(x1d_new, np.random.random(size=len(x1d_new)))
