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

import types
import matplotlib.pyplot as plt


def integrate(f: callable, bounds: iter, num_mc=int(1E6)):
    sampling = np.random.uniform(size=num_mc * len(bounds)).reshape(
        -1, len(bounds)
    )
    vol = 1
    for i, (min_, max_) in enumerate(bounds):
        sampling[:, i] = sampling[:, i] * (max_ - min_) + min_
        vol *= (max_ - min_)
    return np.sum(f(sampling)) / len(sampling) * vol


def evaluate(true_function, learned_surface, bounds, num_mc=int(1E6), l=2):
    def f(x):
        return np.abs(np.ravel(true_function(x)) - np.ravel(learned_surface(x))) ** l

    return integrate(f, bounds, num_mc) ** (1 / l)


def eval_surf_2d(fun, bounds, num=200):
    xx = np.linspace(bounds[0, 0], bounds[0, 1], num=num)
    yy = np.linspace(bounds[1, 0], bounds[1, 1], num=num)
    x, y = np.meshgrid(xx, yy)
    x = pd.DataFrame(dict(x0=x.ravel(), x1=y.ravel()))
    z = fun(x.values)
    return xx, yy, x, z


def test_evaluate():
    def true_function(x):
        return np.ones_like(x[:, 0])

    def learned_surface(x):
        return 1.1 * np.ones_like(x[:, 0])

    bounds = [[0, 5], [0, 1]]
    a = evaluate(true_function, learned_surface, bounds, l=1)
    print(abs(a - 0.5) * 1e15)
    assert abs(a - 0.5) < 1e-15


def analyse_1d(test):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure()
    plt.scatter(test.learner.x_input, test.f(test.learner.x_input),
                c=test.learner.x_input.index, cmap="coolwarm", alpha=0.2)

    plt.figure()
    sns.histplot(test.learner.x_input, bins=50)


def plot_iterations_1d(test):
    domain = np.linspace(test.bounds[0][0], test.bounds[0][1], 2000)
    iter_ = int(test.indexes.max())
    n_row = int(np.sqrt(iter_))
    fig, axs = plt.subplots(iter_ // n_row, n_row, sharey=True, sharex=True,
                            figsize=(8, 8), dpi=200)

    for iter, ax in enumerate(axs.ravel()):

        result_iter = test.result[iter]
        learner = result_iter["learner"]
        active_criterion = learner.active_criterion

        error = active_criterion(domain.reshape(-1, 1))
        prediction = learner.surface(domain.reshape(-1, 1))
        ax.plot(domain, prediction, color="b", label="iter={}".format(iter),
                zorder=5)
        ax.plot(domain, test.f(domain), color="grey", linestyle="--", zorder=0)
        ax.fill_between(domain.ravel(), prediction - error*1.96,
                        prediction + error*1.96, color="b", alpha=0.2)
        training_dataset = test.x_input.loc[range(iter + 1)]
        new_samples = test.x_input.loc[iter + 1]

        if hasattr(active_criterion, "models"):
            for m in active_criterion.models:
                y = m.predict(domain.reshape(-1, 1))
                ax.plot(domain, y, color="gray", lw=0.5)
        ax.scatter(training_dataset, test.f(training_dataset), color="k",
                   marker=".", zorder=10)

        ax.scatter(new_samples, test.f(new_samples), color="r", marker=".",
                   zorder=30)
        ax.set_ylim(-0.9, 0.7)
        ax.legend()
        # ax.axis("off")


def plot_active_function(test):
    domain = np.linspace(test.bounds[0][0], test.bounds[0][1], 2000)
    iter_ = int(test.indexes.max())
    n_row = int(np.sqrt(iter_))
    fig, axs = plt.subplots(iter_ // n_row, n_row, sharey=True, sharex=True,
                            figsize=(8, 8), dpi=200)


    for iter, ax in enumerate(axs.ravel()):

        result_iter = test.result[iter]
        learner = result_iter["learner"]
        active_criterion = learner.active_criterion

        error = active_criterion(domain.reshape(-1, 1))
        prediction = learner.surface(domain.reshape(-1, 1))

        ax.plot(domain, prediction, color="b", label="iter={}".format(iter), zorder=5)
        ax.plot(domain, test.f(domain), color="grey", linestyle="--", zorder=0)

        ax.fill_between(domain.ravel(), prediction - error,
                        prediction + error, color="b", alpha=0.2)
        training_dataset = test.x_input.loc[range(iter + 1)]
        new_samples = test.x_input.loc[iter + 1]

        if hasattr(active_criterion, "models"):
            for m in active_criterion.models:
                y = m.predict(domain.reshape(-1, 1))
                ax.plot(domain, y, color="gray", lw=0.5)
        ax.scatter(training_dataset, test.f(training_dataset), color="k",
                   marker=".", zorder=10)

        ax.scatter(new_samples, test.f(new_samples), color="r", marker=".",
                   zorder=30)
        ax.set_ylim(-0.9, 0.7)
        ax.legend()
        # ax.axis("off")

def write_benchmark(
        data: pd.DataFrame,
        path="examples/data/benchmark.csv", update=True):
    import os

    if update:
        if not os.path.exists(path):
            data.to_csv(path, mode='a', index=False)
        else:
            data.to_csv(path, mode='a', header=False, index=False)
    else:
        data.to_csv(path, index=False)


def read_benchmark(path="data/benchmark.csv"):
    return pd.read_csv(path)


def plot_benchmark(data: pd.DataFrame, cmap="rainbow_r"):
    import matplotlib.pyplot as plt
    import seaborn as sns

    names = data["name"].drop_duplicates().values
    for i, n in enumerate(names):
        data_ = data[data["name"] == n]
        color = plt.get_cmap(cmap)(i / (len(names)))
        sns.lineplot(data=data_, x="num_sample", y="L2-norm", label=n,
                     color=color)
