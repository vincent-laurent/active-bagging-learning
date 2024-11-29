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


import os

import matplotlib.colors as colors
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import ensemble
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from active_learning import ActiveSurfaceLearner
from active_learning.benchmark import functions
from active_learning.benchmark.base import ServiceTestingClassAL
from active_learning.benchmark.utils import evaluate, eval_surf_2d
from active_learning.components import active_criterion
from active_learning.components import latin_square
from active_learning.components import query_strategies
from active_learning.components.active_criterion import VarianceCriterion
from active_learning.benchmark import benchmark_config as bc
name = "marelli_2018"
fun = functions.__dict__[name]


def analyse_results(path):
    res = pd.read_csv(path)
    print(res)


def get_estimator_for_benchmark(name):
    if name == "marelli_2018":
        __est = VarianceCriterion(
            estimator=Pipeline([("scale", StandardScaler()),
                                ("bc", SVC(degree=2, C=5, gamma=10))]),
            splitter=ShuffleSplit(n_splits=3, train_size=0.8))

    elif name == "grammacy_lee_2009":
        __est = bc.alc_svc

    elif name == "grammacy_lee_2009_rand":
        __est = bc.alc_svc

    elif name == "branin":
        __est = bc.alc_trees

    elif name == "branin_rand":
        __est = bc.alc_trees

    elif name == "himmelblau":
        __est = bc.alc_trees

    elif name == "himmelblau_rand":
        __est = bc.alc_trees

    elif name == "synthetic_2d_1":
        __est = bc.alc_trees

    elif name == "synthetic_2d_2":
        __est = bc.alc_trees

    else:
        __est = active_criterion.VarianceEnsembleMethod(
            estimator=ensemble.ServiceExtraTreesRegressor(bootstrap=True,
                                                          max_samples=0.9,
                                                          max_features=1))

    return __est


if __name__ == '__main__':
    path = f"examples/2d_benchmark/figures/{name}"
    if not os.path.exists(path):
        os.makedirs(path)

    n0 = functions.function_parameters[name]["n0"]
    budget = functions.function_parameters[name]["budget"]
    steps = functions.function_parameters[name]["n_step"]

    bounds = np.array(functions.bounds[fun])

    def sampler(size):
        return pd.DataFrame(
            latin_square.scipy_lhs_sampler(size=size, x_limits=bounds))

    active_criterion_ = get_estimator_for_benchmark(name)
    a = ServiceTestingClassAL(
        budget=budget, budget_0=n0, function=fun,
        learner=ActiveSurfaceLearner(
            active_criterion_,
            query_strategy=query_strategies.ServiceQueryVariancePDF(
                num_eval=2000,                                   
                bounds=bounds),
            bounds=bounds),
        x_sampler=sampler,
        bounds=bounds, n_steps=steps)
    p = ServiceTestingClassAL(
        budget, n0, fun,
        learner=ActiveSurfaceLearner(active_criterion_,
                                     query_strategy=query_strategies.ServiceUniform(bounds=bounds),
                                     bounds=bounds),
        x_sampler=sampler, bounds=bounds, n_steps=steps)
    a.run()
    p.run()

    xx, yy, x, z = eval_surf_2d(fun, bounds, num=100)

    # X = active_learner.result[n_step + 1]["data"]
    X = a.x_input
    prediction = a.result[8]["learner"].surface

    zz = prediction(x)
    zzz = p.result[8]["learner"].surface(x)
    std = p.result[8]["learner"].active_criterion(x)

    fig, ax = plot.subplots(ncols=2, figsize=(6, 5), dpi=250)
    sa = ((zz.ravel() - z).reshape(len(xx), len(yy))) ** 2
    sp = ((zzz.ravel() - z).reshape(len(xx), len(yy))) ** 2

    cmap = LinearSegmentedColormap.from_list("mycmap", ['C0', "C2", "C4", "C6", "white", "C7", "C5", "C3", "C1"])
    ax[0].pcolormesh(
        xx, yy, sa, cmap=cmap,
        norm=colors.LogNorm(vmin=1e-4, vmax=sa.max()))
    ax[0].contour(xx, yy, z.reshape(len(xx), len(yy)), levels=10,
                  linewidths=0.3,
                  colors='k')

    im = ax[1].pcolormesh(xx, yy, sp, cmap=cmap,
                          norm=colors.LogNorm(vmin=1e-4, vmax=sp.max()))
    ax[1].contour(xx, yy, z.reshape(len(xx), len(yy)), levels=10,
                  linewidths=0.3,
                  colors='k')

    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plot.colorbar(im, cax=cax)
    ax[0].set_title("Estimation error $\\hat{f}_{active} - f $")
    ax[1].set_title("Estimation error $\\hat{f}_{passive} - f $")
    plot.savefig(f"{path}/passive_vs_active.png")

    plot.figure(figsize=(5, 5), dpi=300)
    plot.pcolormesh(xx, yy, std.reshape(len(xx), len(yy)), cmap=cmap)
    plot.contour(xx, yy, z.reshape(len(xx), len(yy)), levels=10, linewidths=0.1,
                 colors='k')
    plot.title("Estimation variance $\\sigma(\\hat{f})$")
    plot.savefig(f"{path}/sampling_function.png")

    plot.figure(figsize=(5, 5), dpi=300)
    plot.pcolormesh(xx, yy, std.reshape(len(xx), len(yy)), cmap=cmap)
    x_new = a.x_new
    plot.scatter(x_new[0], x_new[1], c="k", marker='.')
    plot.scatter(X[0], X[1], c="k", marker='.')
    plot.savefig(f"{path}/sampling.png")

    plot.figure(figsize=(5, 5), dpi=300)
    std_ = std.reshape(len(xx), len(yy))[len(yy) // 2]
    pred = zz.reshape(len(xx), len(yy))[len(yy) // 2]
    f = z.reshape(len(xx), len(yy))[len(yy) // 2]
    fp = zzz.reshape(len(xx), len(yy))[len(yy) // 2]

    plot.plot(yy, f, c="C2", label="True function")
    plot.plot(yy, fp, c="grey", alpha=1, label="Passive estimation", ls="--")
    plot.plot(yy, pred, c="C1", )
    plot.fill_between(yy, pred, pred + std_, color="C0", alpha=0.1)
    plot.fill_between(yy, pred - std_, pred, color="C0", alpha=0.1)
    plot.legend()
    plot.savefig(f"{path}/1d_cut.png")

    plot.figure(figsize=(4, 4), dpi=200)
    plot.contour(xx, yy, z.reshape(len(xx), len(yy)), levels=15, linewidths=0.05,
                 colors='k')
    plot.contour(xx, yy, z.reshape(len(xx), len(yy)), levels=5, linewidths=0.3,
                 colors='k')

    plot.scatter(a.x_input.drop(0)[0], a.x_input.drop(0)[1],
                 cmap=cmap, c=a.x_input.drop(0).index, marker=".")
    plot.scatter(a.x_input.loc[0][0], a.x_input.loc[0][1], c="k", marker="+")

    eval_p = evaluate(fun, p.result[8]["learner"].surface, bounds, num_mc=10000)
    eval_a = evaluate(fun, a.result[8]["learner"].surface, bounds, num_mc=10000)
    print(
        f"passive performance : {eval_p}\n"
        f"active  performance : {eval_a}")
    plot.tight_layout()
    plot.savefig(f"{path}/sampling_example_function.png")
