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

import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# TODO : add time series https://link.springer.com/content/pdf/10.1023/A:1012474916001.pdf


def check_2d(x: np.ndarray):
    if len(x.shape) != 2 or x.shape[1] != 2:
        raise ValueError("fail to understand input data")


def marelli_2018(x: typing.Iterable[typing.Iterable]):
    x_ = np.array(x)
    check_2d(x_)
    x1, x2 = x_[:, 0], x_[:, 1]
    a1 = 3 + 0.1 * (x1 - x2) ** 2 - (x1 + x2) / np.sqrt(2)
    a2 = 3 + 0.1 * (x1 - x2) ** 2 + (x1 + x2) / np.sqrt(2)
    a3 = (x1 - x2) + 3 * np.sqrt(2)
    a4 = (x2 - x1) + 3 * np.sqrt(2)
    g = a1
    for a in [a2, a3, a4]:
        g = np.where(g < a, g, a)

    g = np.where(g <= 0, 0, 1)
    return g


def annie_sauer_2021(x: typing.Iterable[typing.Iterable]):
    """
    f (x) =
    - 1.35 cos(12πx) x ∈ [0, 0.33]
    - 1.35 x ∈ [0.33, 0.66]
    - 1.35 cos(6πx) x ∈ [0.66, 1].
    """
    x_ = np.array(x)
    y = np.where(x_ < 0, np.nan, x_)
    y = np.where(x_ < 0.33, 1.35 * np.cos(12 * np.pi * x_), y)
    y = np.where(x_ < 0.66, 1.35, y)
    y = np.where(x_ <= 1, 1.35 * np.cos(6 * np.pi * x_), y)
    y = np.where(x_ > 1, np.nan, y)
    return y


def grammacy_lee_2009(x: typing.Iterable[typing.Iterable]):
    """ (x1, x2) = 10 x_1 exp (−x^2_1 − x^2_2) for x1, x2 ∈ [−2, 4]"""

    x_ = np.array(x)

    check_2d(x_)

    x_square = x_ ** 2
    return 10 * x_[:, 0] * np.exp(-x_square[:, 0] - x_square[:, 1])


def himmelblau(x: typing.Iterable[typing.Iterable]):
    x_ = np.array(x)

    check_2d(x_)

    return 1 - ((x_[:, 0] ** 2 + x_[:, 1] - 11) ** 2 + (
            x_[:, 0] + x_[:, 1] ** 2 - 7) ** 2) / 75


def branin(x: typing.Iterable[typing.Iterable],
           a=1, b=5.1 / (4 * pow(np.pi, 2)), c=5 / np.pi, r=6,
           s=10, t=1 / (8 * np.pi)):
    x_ = np.array(x)
    check_2d(x_)
    return (a * (x_[:, 1] - b * x_[:, 0] ** 2 + c * x_[:, 0] - r) ** 2 + s * (
            1 - t) * np.cos(x_[:, 0]) + s) / 25


def golden_price(x):
    x_ = np.array(x)
    check_2d(x)
    xx, yy = x_[:, 0], x_[:, 1]
    return ((1 + (xx + yy + 1) ** 2 * (
            19 - 14 * xx + 3 * xx ** 2 - 14 * yy + 6 * xx * yy + 3 * yy ** 2)) * (
                    30 + (2 * xx - 3 * yy) ** 2 * (
                    18 - 32 * xx + 12 * xx ** 2 + 48 * yy - 36 * xx * yy + 27 * yy ** 2))) / 100000 - 1


def synthetic_2d_1(x):
    x_ = np.array(x)
    check_2d(x)
    return (x_[:, 0] * np.sin(x_[:, 0]) * np.cos(x_[:, 1]) - x_[:, 1]) / 10


def synthetic_2d_2(x):
    x_ = np.array(x)
    check_2d(x)
    return synthetic_2d_1(x_) * annie_sauer_2021(x_[:, 1] / 10)


def sum_sine_5pi(x):
    """
    Calculates the function 1/5 * sum(sin(5 * pi * x_i)) for i=1 to 5.

    Parameters:
    x (list or np.ndarray): A 5-dimensional vector with each component in the range [-5, 5].

    Returns:
    float: The calculated value of the function.
    """
    x = np.asarray(x)

    result = (1 / 5) * np.sum(x*np.sin(np.pi * x / 4)**6, axis=1)
    return result


class Randomize:
    def __init__(self, f):
        self.__f = f

    def __call__(self, x, **kwargs):
        return self.__f(x) + 0.5 * np.random.randn(len(np.array(x)[:, 0]))


branin_rand = Randomize(branin)
golden_price_rand = Randomize(golden_price)
himmelblau_rand = Randomize(himmelblau)
grammacy_lee_2009_rand = Randomize(grammacy_lee_2009)

bounds = {
    annie_sauer_2021: [[0, 1]],
    grammacy_lee_2009: [[-4, 4], [-4, 4]],
    grammacy_lee_2009_rand: [[-4, 4], [-4, 4]],
    golden_price: [[-2, 2], [-2, 2]],
    golden_price_rand: [[-2, 2], [-2, 2]],
    branin: [[-5, 10], [0, 15]],
    branin_rand: [[-5, 10], [0, 15]],
    himmelblau: [[-5, 5], [-5, 5]],
    himmelblau_rand: [[-5, 5], [-5, 5]],
    synthetic_2d_1: [[0, 10], [0, 10]],
    synthetic_2d_2: [[0, 10], [0, 10]],
    marelli_2018: [[-6, 6], [-6, 6]],
    sum_sine_5pi: [[-5, 5]] * 5
}

function_parameters = {
    "grammacy_lee_2009": {
        "fun": grammacy_lee_2009,
        'n0': 20,
        "budget": 200,
        "n_step": 10,
        "name": "Grammacy Lee"},
    "grammacy_lee_2009_rand": {
        "fun": grammacy_lee_2009_rand,
        'n0': 60,
        "budget": 100,
        "n_step": 10,
        "name": "Grammacy Lee rand.",
    },
    "branin": {
        "fun": branin,
        'n0': 100,
        "budget": 900,
        "n_step": 10,
        "name": "Branin"},
    "branin_rand": {
        "fun": branin_rand,
        'n0': 100,
        "budget": 1000,
        "n_step": 10,
        "name": "Branin rand."},
    "himmelblau": {
        "fun": himmelblau,
        'n0': 100,
        "budget": 900,
        "n_step": 10,
        "name": "Himmelblau"},
    "himmelblau_rand": {
        "fun": himmelblau_rand,
        'n0': 100,
        "budget": 1000,
        "n_step": 10,
        "name": "Himmelblau rand."},
    "golden_price": {
        "fun": golden_price, 'n0': 50, "budget": 60, "n_step": 10,
        "name": "Golden price"},
    "marelli_2018": {
        "fun": marelli_2018, 'n0': 150, "budget": 600, "n_step": 30,
        "name": "Warts 2000"},
    "synthetic_2d_1": {
        "fun": synthetic_2d_1, 'n0': 100, "budget": 900, "n_step": 10,
        "name": "Periodic 1"},
    "synthetic_2d_2": {
        "fun": synthetic_2d_2,
        'n0': 100,
        "budget": 1000,
        "n_step": 10,
        "name": "Periodic 2"},
    "sum_sine_5pi": {
        "fun": sum_sine_5pi,
        'n0': 100,
        "budget": 1000,
        "n_step": 10,
        "name": "Periodic 2"},
}

__all__ = [
    annie_sauer_2021, branin_rand, himmelblau_rand, golden_price_rand,
]

__all2D__ = [
    "grammacy_lee_2009",
    "grammacy_lee_2009_rand",
    "branin", "branin_rand",
    "himmelblau",
    "himmelblau_rand",
    "golden_price",
    "marelli_2018",
    "synthetic_2d_1",
    "sum_sine_5pi",
]


def plot_benchmark_functions():
    import matplotlib.pyplot as plot
    import pandas as pd
    import matplotlib

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "mycmap", ['C0', "C2", "C4", "C6", "white", "C7", "C5", "C3", "C1"][::-1])

    matplotlib.rcParams.update({'font.size': 6})
    fig, ax = plot.subplots(
        figsize=(10, 4.2),
        ncols=len(__all2D__) // 2 + len(__all2D__) % 2,
        nrows=2, dpi=400)
    for i, fun in enumerate(__all2D__):
        if not fun in function_parameters.keys():
            continue
        bounds_ = np.array(bounds[function_parameters[fun]["fun"]])
        xxx = np.linspace(bounds_[0, 0], bounds_[0, 1], num=200)
        yyy = np.linspace(bounds_[1, 0], bounds_[1, 1], num=200)
        x__, y__ = np.meshgrid(xxx, yyy)
        x__ = pd.DataFrame(dict(x0=x__.ravel(), x1=y__.ravel()))
        if len(bounds_) == 2:
            z = function_parameters[fun]["fun"](x__.values)
        if len(bounds_) > 2:
            for k, b in enumerate(bounds_[2:]):
                x__[f"x{k+2}"] = (b[1] + b[0]) / 2
            z = function_parameters[fun]["fun"](x__.values)
        z = z - np.median(z)

        im = ax[i % 2, i // 2].pcolormesh(xxx, yyy,
                                          z.reshape(len(xxx), len(yyy)),
                                          cmap=cmap, vmin=-1, vmax=1)
        ax_ = ax[i % 2, i // 2]
        ax_.set_xticklabels([])
        ax_.set_yticklabels([])
        ax_.annotate(function_parameters[fun]["name"], xy=(1, 0.9),
                     xycoords='axes fraction',
                     xytext=(1, 20), textcoords='offset pixels',
                     horizontalalignment='right',
                     verticalalignment='bottom',
                     bbox=dict(boxstyle="round", fc="white", lw=0.4))

    # plt.tight_layout()
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.91, 0.11, 0.01, 0.77])
    fig.colorbar(im, cax=cbar_ax, ticks=[-1, 0, 1])
    cbar_ax.set_yticklabels(
        ['low \nvalues', 'medium \nvalue', 'high \nvalues'])  #

    plt.savefig(".public/benchmark_functions")


if __name__ == '__main__':
    from active_learning.benchmark.benchmark_config import Sampler

    pd.Series(golden_price(Sampler(np.array(bounds[golden_price]))(1000))).describe()
    plot_benchmark_functions()
