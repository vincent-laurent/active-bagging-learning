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
import seaborn as sns
from scipy import interpolate
from matplotlib.colors import LinearSegmentedColormap

from active_learning.benchmark.base import ServiceTestingClassAL
from active_learning.components import latin_square

cmap = LinearSegmentedColormap.from_list(
    "mycmap", ["#1C284E", "#174D7C", "#4C7D8E", "#819898", "white", "#FFC28B",
               "#FF9463", "#FE6D4E", "#FF5342"][::-1])


class Sampler:
    def __init__(self, bounds):
        self.__bounds = bounds

    def __call__(self, size):
        return pd.DataFrame(
            latin_square.scipy_lhs_sampler(size=size, x_limits=self.__bounds)
        )


def plot_benchmark_whole_analysis(data: pd.DataFrame, n_functions=None) -> None:
    import matplotlib
    import seaborn as sns

    if n_functions is None:
        n_functions = len(data["function_hash"].drop_duplicates())
    matplotlib.rcParams.update({"font.size": 6})
    functions__ = data["name"].astype(str).str.replace("_passive", "").drop_duplicates()
    fig, ax = plt.subplots(
        ncols=len(functions__) // 2 + len(functions__) % 2,
        nrows=2,
        figsize=(n_functions * 0.7, 3.5),
    )

    if ax.shape.__len__() == 1:
        ax = ax.reshape(-1, 1)
    for i, f in enumerate(functions__):
        ax_ = ax[i % 2 + i // 2, i // 2]
        print(f)
        data_temp = data[data["name"] == f].copy()
        data_temp_p = data[data["name"] == f + "_passive"].copy()
        data_temp["name"] = "active"
        data_temp_p["name"] = "passive"
        data_plot = pd.concat((data_temp, data_temp_p))
        names = data_plot["name"].drop_duplicates().values

        for j, n in enumerate(names):
            data_ = data_plot[data_plot["name"] == n]
            color = f"C{j}"
            if i > 0:
                label = "_nolegend_"
            else:
                label = n
            sns.lineplot(
                data=data_,
                x="num_sample",
                y="L2-norm",
                label=label,
                color=color,
                ax=ax_,
            )
        ax_.annotate(
            f,
            xy=(1, 0.9),
            xycoords="axes fraction",
            xytext=(1, 20),
            textcoords="offset pixels",
            horizontalalignment="right",
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", fc="white", lw=0.4),
        )
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
        ncol=2,
    )


def plot_benchmark(data: pd.DataFrame, standardise=False) -> None:
    from active_learning.benchmark import functions
    data["name"] = data["name"].str.replace("Golden price", "Goldenstein price")
    function_list = (
        data["name"].astype(str).str.split("@", expand=True)[0].drop_duplicates()
    )
    functions_default = functions.function_parameters.keys()
    functions_default = [
        functions.function_parameters[f]["name"] for f in functions_default
    ]
    function_list = [f for f in function_list]

    function_list = [f for f in functions_default if f in function_list]

    n_functions = len(function_list)
    colors = {
        "c0": '#FF9500',
        "c4": '#1C284E',
        "c2": "#174D7C",
        "c3": '#00B945',
        "c5": '#FF5344',
        "c1": "#FE6D4E",
        "c6": '#9e9e9e',
        "c7": "#845B97"}
    fig, ax = plt.subplots(
        ncols=n_functions // 2 + n_functions % 2,
        nrows=min(n_functions, 2),
        figsize=(n_functions // 2 * 2 + 1, 4),
        dpi=600,
    )
    all_handles = []
    all_labels = []
    all_methods = np.sort(np.unique(
        [__name.split("@")[1] for __name in data["name"].drop_duplicates().values])
    )[::-1]

    y_label = "$L_2$ error" if not standardise else "$L_2/L_2^\\text{passive}$"

    if isinstance(ax, plt.Axes):
        ax = np.array([ax])
    if ax.shape.__len__() == 1:
        ax = ax.reshape(-1, 1)
    for i, f in enumerate(function_list):
        print(f)
        __ax: plt.Axes = ax[i % 2, i // 2]
        ref = data[data["name"] == f"{f}@passive"].groupby("num_sample")["L2-norm"].mean()
        interp = interpolate.interp1d(ref.index, ref.values)

        plt.sca(__ax)
        for j, label in enumerate(all_methods):
            data_ = data[data["name"] == f"{f}@{label}"]
            if standardise:
                data_["L2-norm"] = data_["L2-norm"] / interp(data_["num_sample"])

            color = colors[f"c{j}"]

            if len(data_) > 0:
                sns.lineplot(
                    data=data_,
                    x="num_sample",
                    y="L2-norm",
                    label=label,
                    color=color,
                    ax=__ax, zorder=100 - i
                )
            handles, labels = plt.gca().get_legend_handles_labels()
            all_handles.extend(handles)
            all_labels.extend(labels)

        __ax.annotate(
            f,
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            xytext=(1, 20),
            textcoords="offset pixels",
            horizontalalignment="right",
            verticalalignment="bottom",
            bbox=dict(fc="white", lw=0.7, ec="#bcbcbc", boxstyle="square"), zorder=10000
        )
        __ax.legend().set_visible(False)

        __ax.set_ylabel(y_label) if i // 2 == 0 else __ax.set_ylabel("")
        plt.xlabel("")
        if standardise:
            limits = __ax.get_ylim()
            __ax.set_ylim((max(limits[0], 0), min(limits[1], 2.5)))

    if n_functions % 2 == 1:
        if n_functions > 1:
            ax[(i + 1) % 2, (i + 1) // 2].axes.remove()

    all_labels = pd.Series(all_labels, index=all_handles).sort_values().reset_index()
    all_labels = all_labels.drop_duplicates(keep="first", subset=[0])

    fig.legend(
        all_labels["index"].to_list(),
        all_labels[0].to_list(),
        bbox_to_anchor=(0.5, 0.965),
        # loc='lower left',
        # mode="expand",
        loc="upper center",
        ncol=len(all_methods),
    )


def integrate(f: callable, bounds: iter, num_mc=int(1e6)):
    sampling = np.random.uniform(size=num_mc * len(bounds)).reshape(-1, len(bounds))
    vol = 1
    for i, (min_, max_) in enumerate(bounds):
        sampling[:, i] = sampling[:, i] * (max_ - min_) + min_
        vol *= max_ - min_
    return np.sum(f(sampling)) / len(sampling) * vol


def evaluate(true_function, learned_surface, bounds, num_mc=int(1e6), p=2):
    def f(x):
        return np.abs(np.ravel(true_function(x)) - np.ravel(learned_surface(x))) ** p

    return integrate(f, bounds, num_mc) ** (1 / p)


def eval_surf_2d(fun, bounds, num=200):
    xx = np.linspace(bounds[0, 0], bounds[0, 1], num=num)
    yy = np.linspace(bounds[1, 0], bounds[1, 1], num=num)
    x, y = np.meshgrid(xx, yy)
    x = pd.DataFrame(dict(x0=x.ravel(), x1=y.ravel()))
    z = fun(x=x.values)
    return xx, yy, x, z


def as_2d(fun, d: int):
    def function(x):
        """
        x : 2d input vector
        """
        xx = np.concatenate((x, np.zeros((len(x), d - x.shape[1]))), axis=1)
        return fun(xx)

    return function


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
    plt.figure()
    plt.scatter(
        test.x_input,
        test.f(test.x_input),
        c=test.x_input.index,
        cmap="coolwarm",
        alpha=0.2,
    )

    plt.figure()
    sns.histplot(test.x_input, bins=50)


def plot_iterations_1d(test, iteration_max=None, color="b"):
    domain = np.linspace(test.bounds[0][0], test.bounds[0][1], 2000)
    if iteration_max is None:
        iteration_max = int(test.indexes.max())
    n_row = int(np.sqrt(iteration_max))
    fig, axs = plt.subplots(
        iteration_max // n_row, n_row, sharey=True, sharex=True, figsize=(6, 6), dpi=200
    )

    for iteration, ax in enumerate(axs.ravel()):
        result_iter = test.result[iteration + 1]
        learner = result_iter["learner"]

        prediction = learner.predict(domain.reshape(-1, 1))
        ax.plot(domain, test.f(domain), color="grey", linestyle="--", zorder=0)
        training_dataset = test.x_input.loc[test.indexes <= iteration + 1]
        ax.plot(
            domain,
            prediction,
            color=color,
            label=f"N = {len(training_dataset)}",
            zorder=5,
        )
        new_samples = test.x_input.loc[test.indexes == iteration + 2]

        if hasattr(learner, "active_criterion"):
            uncertainty = learner.active_criterion(domain.reshape(-1, 1))
            ax.fill_between(
                domain.ravel(),
                prediction - uncertainty,
                prediction + uncertainty,
                color=color,
                alpha=0.2,
            )

            if hasattr(learner.active_criterion, "models"):
                for m in learner.active_criterion.models:
                    y = m.predict(domain.reshape(-1, 1))
                    ax.plot(domain, y, color="gray", lw=0.5)
        ax.scatter(
            training_dataset, test.f(training_dataset), color="k", marker=".", zorder=10
        )

        ax.scatter(new_samples, test.f(new_samples), color="r", marker=".", zorder=30)
        ax.set_ylim(-0.9, 0.7)
        ax.legend(loc=2)
        ax.set_axis_off()
        plt.tight_layout()


def plot_active_function(test, color="b"):
    domain = np.linspace(test.bounds[0][0], test.bounds[0][1], 2000)
    iter_ = int(test.indexes.max())
    n_row = int(np.sqrt(iter_))
    fig, axs = plt.subplots(
        iter_ // n_row, n_row, sharey=True, sharex=True, figsize=(8, 8), dpi=200
    )

    for iter, ax in enumerate(axs.ravel()):
        result_iter = test.result[iter]
        learner = result_iter["learner"]
        active_criterion = learner.active_criterion

        error = active_criterion(domain.reshape(-1, 1))
        prediction = learner.surface(domain.reshape(-1, 1))

        ax.plot(domain, prediction, color=color, label="iter={}".format(iter), zorder=5)
        ax.plot(domain, test.f(domain), color="grey", linestyle="--", zorder=0)

        ax.fill_between(
            domain.ravel(),
            prediction - error,
            prediction + error,
            color=color,
            alpha=0.2,
        )
        training_dataset = test.x_input.loc[range(iter + 1)]
        new_samples = test.x_input.loc[iter + 1]

        if hasattr(active_criterion, "models"):
            for m in active_criterion.models:
                y = m.predict(domain.reshape(-1, 1))
                ax.plot(domain, y, color="gray", lw=0.5)
        ax.scatter(
            training_dataset, test.f(training_dataset), color="k", marker=".", zorder=10
        )

        ax.scatter(new_samples, test.f(new_samples), color="r", marker=".", zorder=30)
        ax.set_ylim(-0.9, 0.7)
        ax.legend()
        # ax.axis("off")


def write_benchmark(
        data: pd.DataFrame, path="examples/data/benchmark.csv", update=True
):
    import os

    path_dir = "/".join(path.split("/")[:-1])
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    if update:
        if not os.path.exists(path):
            data.to_csv(path, mode="a", index=False)
        else:
            data.to_csv(path, mode="a", header=False, index=False)
    else:
        data.to_csv(path, index=False)


def plot_iterations_2d(test: ServiceTestingClassAL):
    test.x_input
    test.indexes

    for i in np.unique(test.indexes):
        sel = test.indexes == i
        x = test.x_input[sel].iloc[:, :2]

        plt.scatter(x[0], x[1])


def read_benchmark(path="data/benchmark"):
    data = pd.read_csv(path)
    data = data.loc[~data["name"].str.contains("UCB")]
    data = data.loc[~data["name"].str.contains("EI")]
    data = data.sort_values("date")
    data["__date__"] = data["date"].astype(str).str[:10]
    sel = data[["budget", "budget_0", "n_steps", "name", '__date__']].drop_duplicates(
        keep="last", subset=["budget", "budget_0", "n_steps", "name"]).drop_duplicates(
        keep="last", subset= ["name"]
    )
    ret = pd.merge(sel, data, how="inner", on=["budget", "budget_0", "n_steps", "name", "__date__"])
    ret = ret.drop(columns='__date__')
    return ret
