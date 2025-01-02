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
from mpl_toolkits.axes_grid1 import make_axes_locatable

from active_learning.benchmark import benchmark_config
from active_learning.benchmark import functions, utils
from active_learning.benchmark.utils import evaluate, eval_surf_2d

name = "grammacy_lee_2009_rand"
fun = functions.__dict__[name]

if __name__ == '__main__':
    path = f"examples/2d_benchmark/figures/{name}"
    if not os.path.exists(path):
        os.makedirs(path)


    def sub_sample(x_input, x_0=None, eps=0.5):
        if x_0 is None:
            x_0 = [0] * x_input.shape[1]
        for __d in range(2, x_input.shape[1]):
            x_input = x_input[np.abs(x_input[__d] - x_0[__d]) < eps]

        return x_input


    bounds = np.array(functions.bounds[fun])

    t = benchmark_config.make_testing_classes(name)
    learner = t[-2]
    learner_ref = t[-3]

    learner.run()
    learner_ref.run()
    d = learner.x_input.shape[1]

    xx, yy, x, z = eval_surf_2d(utils.as_2d(fun, d), bounds, num=100)

    # X = active_learner.result[n_step + 1]["data"]
    X = learner.x_input
    prediction = learner.result[8]["learner"].predict

    zz = utils.as_2d(prediction, d)(x)
    zzz = utils.as_2d(learner_ref.result[8]["learner"].predict, d)(x)
    std = utils.as_2d(learner_ref.result[8]["learner"].active_criterion, d)(x)

    sa = ((zz.ravel() - z).reshape(len(xx), len(yy))) ** 2
    sp = ((zzz.ravel() - z).reshape(len(xx), len(yy))) ** 2

    fig, ax = plot.subplots(ncols=2, figsize=(6, 5), dpi=250)
    ax[0].pcolormesh(
        xx, yy, sa, cmap=utils.cmap,
        norm=colors.LogNorm(vmin=1e-4, vmax=sa.max()))
    ax[0].contour(xx, yy, z.reshape(len(xx), len(yy)), levels=10,
                  linewidths=0.3,
                  colors='k')

    im = ax[1].pcolormesh(xx, yy, sp, cmap=utils.cmap,
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
    plot.pcolormesh(xx, yy, std.reshape(len(xx), len(yy)), cmap=utils.cmap)
    plot.contour(xx, yy, z.reshape(len(xx), len(yy)), levels=10, linewidths=0.1,
                 colors='k')
    plot.title("Estimation variance $\\sigma(\\hat{f})$")
    plot.savefig(f"{path}/sampling_function.png")

    plot.figure(figsize=(5, 5), dpi=300)
    plot.pcolormesh(xx, yy, std.reshape(len(xx), len(yy)), cmap=utils.cmap)
    x_new = sub_sample(learner.x_new, eps=0.5)
    x_plot = sub_sample(X, eps=0.5)
    plot.scatter(x_new[0], x_new[1], c="k", marker='o')
    plot.scatter(x_plot[0], x_plot[1], c="k", marker='.')
    plot.savefig(f"{path}/sampling.png")

    std_ = std.reshape(len(xx), len(yy))[len(yy) // 2]
    pred = zz.reshape(len(xx), len(yy))[len(yy) // 2]
    f = z.reshape(len(xx), len(yy))[len(yy) // 2]
    fp = zzz.reshape(len(xx), len(yy))[len(yy) // 2]

    plot.figure(figsize=(5, 5), dpi=300)
    plot.plot(yy, f, c="k", label="True function")
    plot.plot(yy, fp, c="C2", alpha=1, label=learner_ref.name)
    plot.plot(yy, pred, c="C3", label=learner.name)
    plot.fill_between(yy, pred, pred + std_, color="C3", alpha=0.1)
    plot.fill_between(yy, pred - std_, pred, color="C3", alpha=0.1)
    plot.legend()
    plot.savefig(f"{path}/1d_cut.png")

    plot.figure(figsize=(4, 4), dpi=200)
    plot.contour(xx, yy, z.reshape(len(xx), len(yy)), levels=15, linewidths=0.05,
                 colors='k')
    plot.contour(xx, yy, z.reshape(len(xx), len(yy)), levels=5, linewidths=0.3,
                 colors='k')
    x_plot_n = sub_sample(learner.x_input.drop(0))
    x_plot_0 = sub_sample(learner.x_input.loc[0])

    plot.scatter(x_plot_n[0], x_plot_n[1],
                 cmap=utils.cmap, c=x_plot_n.index, marker=".")
    plot.scatter(x_plot_0[0], x_plot_0[1], c="k", marker="+")

    eval_p = evaluate(fun, learner_ref.result[8]["learner"].predict, bounds,
                      num_mc=100000)
    eval_a = evaluate(fun, learner.result[8]["learner"].predict, bounds,
                      num_mc=100000)

    print(
        f"passive performance : {eval_p}\n"
        f"active  performance : {eval_a}")
    plot.tight_layout()
    plot.savefig(f"{path}/sampling_example_function.png")

    perf = np.array([l["l2"] for l in list(learner.result.values())])
    perf_ref = np.array([l["l2"] for l in list(learner_ref.result.values())])

    size = np.array([l["budget"] for l in list(learner.result.values())])
    size_ref = np.array([l["budget"] for l in list(learner_ref.result.values())])

    plot.figure()
    plot.plot(size, perf, label=learner.name)
    plot.plot(size, perf_ref, label=learner_ref.name)
    plot.legend()
    plot.savefig(f"{path}/perf.png")


def evaluate_convergence(learner):
    list_l2 = []
    list_range = np.linspace(2, 6, num=20)
    for i in list_range:
        list_l2.append(evaluate(fun, learner_ref.result[8]["learner"].predict, bounds,
                                num_mc=int(10 ** i)))

    plot.figure()
    plot.plot(list_range, list_l2, label=learner.name)
    plot.legend()
    plot.savefig(f"{path}/perf_convergence.png")