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


def plot_benchmark_whole_analysis(data: pd.DataFrame, n_functions) -> None:
    import matplotlib
    import seaborn as sns
    matplotlib.rcParams.update({'font.size': 6})
    functions__ = data["function_hash"].astype(str).str.replace("_passive", "").drop_duplicates()
    fig, ax = plt.subplots(ncols=len(functions__) // 2 + len(functions__) % 2,
                           nrows=2, figsize=(n_functions * 0.7, 3.5))
    if ax.shape.__len__() == 1:
        ax = ax.reshape(-1, 1)
    for i, f in enumerate(functions__):
        print(f)
        ax_ = ax[i % 2, i // 2]
        plt.sca(ax_)
        data_temp = data[data["name"] == f].copy()
        data_temp_p = data[data["name"] == f + "_passive"].copy()
        data_temp["name"] = "active"
        data_temp_p["name"] = "passive"
        data_plot = pd.concat((data_temp, data_temp_p))
        names = data_plot["name"].drop_duplicates().values

        for j, n in enumerate(names):
            data_ = data_plot[data_plot["name"] == n]
            color = plt.get_cmap("crest")(j / (len(names)))
            if i > 0:
                label = '_nolegend_'
            else:
                label = n
            sns.lineplot(data=data_, x="num_sample", y="L2-norm", label=label, color=color)
        ax_.annotate(f, xy=(1, 0.8), xycoords='axes fraction',
                     xytext=(1, 20), textcoords='offset pixels',
                     horizontalalignment='right',
                     verticalalignment='bottom',
                     bbox=dict(boxstyle="round", fc="white", lw=0.4))
        ax_.legend().set_visible(False)
        ax_.set_ylabel("$L_2$ error") if i // 2 == 0 else ax_.set_ylabel("")
        plt.yticks(c="w")
        plt.xlabel("")
        ax_.axes.yaxis.set_ticklabels([])
        ax_.grid()
        if len(functions__) % 2 == 1:
            ax[(i + 1) % 2, (i + 1) // 2].axes.remove()

    fig.legend(
        bbox_to_anchor=(0.6, 0.98),
        # loc='lower left',
        # mode="expand",
        ncol=2)


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


def plot_iterations_1d(test, iteration_max=None, color="b"):
    domain = np.linspace(test.bounds[0][0], test.bounds[0][1], 2000)
    if iteration_max is None:
        iteration_max = int(test.indexes.max())
    n_row = int(np.sqrt(iteration_max))
    fig, axs = plt.subplots(iteration_max // n_row, n_row, sharey=True, sharex=True,
                            figsize=(6, 6), dpi=200)

    for iteration, ax in enumerate(axs.ravel()):

        result_iter = test.result[iteration + 1]
        learner = result_iter["learner"]

        prediction = learner.predict(domain.reshape(-1, 1))
        ax.plot(domain, test.f(domain), color="grey", linestyle="--", zorder=0)
        training_dataset = test.x_input.loc[test.indexes <= iteration + 1]
        ax.plot(domain, prediction, color=color, label=f"N = {len(training_dataset)}", zorder=5)
        new_samples = test.x_input.loc[test.indexes == iteration + 2]

        if hasattr(learner, 'active_criterion'):
            uncertainty = learner.active_criterion(domain.reshape(-1, 1))
            ax.fill_between(domain.ravel(), prediction - uncertainty,
                            prediction + uncertainty, color=color, alpha=0.2)

            if hasattr(learner.active_criterion, "models"):
                for m in learner.active_criterion.models:
                    y = m.predict(domain.reshape(-1, 1))
                    ax.plot(domain, y, color="gray", lw=0.5)
        ax.scatter(training_dataset, test.f(training_dataset), color="k",
                   marker=".", zorder=10)

        ax.scatter(new_samples, test.f(new_samples), color="r", marker=".",
                   zorder=30)
        ax.set_ylim(-0.9, 0.7)
        ax.legend(loc=2)
        ax.set_axis_off()
        plt.tight_layout()


def plot_active_function(test, color="b"):
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

        ax.plot(domain, prediction, color=color, label="iter={}".format(iter), zorder=5)
        ax.plot(domain, test.f(domain), color="grey", linestyle="--", zorder=0)

        ax.fill_between(domain.ravel(), prediction - error,
                        prediction + error, color=color, alpha=0.2)
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
    path_dir = "/".join(path.split("/")[:-1])
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

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
