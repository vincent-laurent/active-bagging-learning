import typing
from copy import deepcopy

import numpy as np
import pandas as pd

from active_learning import base
from active_learning.components.active_criterion import ActiveCriterion
from active_learning.components.query_strategies import QueryStrategy


class TestingClass:
    def __init__(self, budget: int, budget_0: int, funcion: callable,
                 active_criterion: ActiveCriterion,
                 query_strategy: QueryStrategy,
                 x_sampler: callable, n_steps: int,
                 bounds, name=None
                 ):
        self.f = funcion
        self.budget = budget
        self.n_steps = n_steps
        self.budget_0 = budget_0
        self.x_sampler = x_sampler
        self.active_criterion = active_criterion
        self.query_strategy = query_strategy
        self.bounds = bounds
        self.parameters = dict(
            budget=budget,
            budget_0=budget_0,
            n_steps=n_steps,
            function_hash=hash(funcion) + hash(str(bounds)),
            active_criterion=str(active_criterion),
            query_strategy=str(query_strategy),
            x_sampler_hash=hash(x_sampler),
        )
        self.parameters["test_id"] = self.parameters["budget"] + \
                                     self.parameters["function_hash"] + \
                                     self.parameters["n_steps"]
        self.parameters["name"] = hash(np.random.uniform()) if name is None else name

        self.metric = []

    def run(self):
        x_train = self.x_sampler(self.budget_0)
        self.learner = base.ActiveSRLearner(
            self.active_criterion,
            self.query_strategy,
            x_train,
            pd.DataFrame(self.f(x_train)),
            self.bounds)

        nb_sample_cumul = np.linspace(self.budget_0, self.budget, num=self.n_steps, dtype=int)
        nb_sample = np.concatenate(([nb_sample_cumul[0]], np.diff(nb_sample_cumul)))
        print("Distribution of budget :", nb_sample)
        for n_points in nb_sample[1:]:
            i = self.learner.iter
            x_new = self.learner.query(n_points)
            y_new = pd.DataFrame(self.f(x_new))
            self.learner.add_labels(x_new, y_new)

            # EVALUATE STRATEGY
            self.metric.append(evaluate(
                self.f,
                self.learner.result[i]["surface"],
                self.bounds, num_mc=100000))

    def plot_error_vs_criterion(self, n=1000):
        import matplotlib.pyplot as plt
        import seaborn as sns
        x = self.x_large_sample.sample(n)
        for i in self.functions.keys():
            error = np.abs(np.ravel(self.f(x)) - self.functions[i](x)) ** 2
            crit_ = self.learner.active_criterion(x)
            sns.kdeplot(x=error, y=crit_, c=plt.get_cmap("rainbow")(i / len(self.functions)), log_scale=(True, True),
                        linewidth=0.1)
        plt.xlabel("$L_2$ error")
        plt.ylabel("Estimated variance")

    def plot_error_vs_criterion_pointwise(self, num_mc=10000):
        import matplotlib.pyplot as plt
        for i_, i in enumerate(self.learner.result.keys()):
            res = self.metric[i_]
            variance = integrate(self.criterion[i], self.bounds, num_mc)
            plt.loglog([res], [variance], c=plt.get_cmap("rainbow")(i / len(self.functions)), marker="o", lw=0)
        plt.xlabel("$L_2$ error")
        plt.ylabel("Estimated variance")


class Experiment:
    def __init__(self, test_list: typing.List[TestingClass], n_experiment=10, save=False):
        self.test_list = test_list
        self.results = {}
        self.n_experiment = n_experiment
        columns = [*test_list[0].parameters.keys(), "L2-norm"]

        self._cv_result = pd.DataFrame(
            columns=columns,
        )
        self.save = save
        if self.save:
            self.saving_class = {}

    def run(self):

        for test in self.test_list:
            for i in range(self.n_experiment):
                test_ = deepcopy(test)
                test_.run()
                res = pd.DataFrame(columns=self._cv_result.columns)
                res["L2-norm"] = test_.metric
                res["num_sample"] = pd.DataFrame(test_.learner.result).loc["budget"].astype(float).values

                for c in test.parameters.keys():
                    res[c] = test.parameters[c]
                self._cv_result = pd.concat((self._cv_result, res), axis=0)

        self.cv_result_ = self._cv_result


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


def analyse_1d(test: TestingClass):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure()
    plt.scatter(test.learner.x_input, test.f(test.learner.x_input),
                c=test.learner.x_input.index, cmap="coolwarm", alpha=0.2)

    plt.figure()
    sns.histplot(test.learner.x_input, bins=50)


def plot_iter(test: TestingClass):
    import matplotlib.pyplot as plt
    domain = np.linspace(test.bounds[0][0], test.bounds[0][1], 2000)
    iter_ = int(test.learner.x_input.index.max())
    fig, axs = plt.subplots(4, iter_ // 4, sharey=True, sharex=True, figsize=(8, 8), dpi=200)

    test.learner.x_input.index = test.learner.x_input.index.astype(int)
    for iter, ax in enumerate(axs.ravel()):
        res = test.learner.result[iter]
        active_criterion = res["active_criterion"]
        error = active_criterion(domain.reshape(-1, 1))
        prediction = res["surface"](domain.reshape(-1, 1))
        ax.plot(domain, prediction, color="b", label="iter={}".format(iter))
        ax.plot(domain, test.f(domain), color="grey", linestyle="--")
        ax.fill_between(domain.ravel(), prediction - error / 2, prediction + error / 2, color="b", alpha=0.2)
        training_dataset = test.learner.x_input.loc[range(iter + 1)]
        new_samples = test.learner.x_input.loc[iter + 1]

        if hasattr(active_criterion, "models"):
            for m in active_criterion.models:
                y = m.predict(domain.reshape(-1, 1))
                ax.plot(domain, y, color="gray", lw=0.5)
        ax.scatter(training_dataset, test.f(training_dataset), color="k", marker=".")

        ax.scatter(new_samples, test.f(new_samples), color="r", marker=".")
        ax.set_ylim(-0.9, 0.7)
        ax.legend()


def write_benchmark(data: pd.DataFrame, path="data/benchmark.csv", update=True):
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
        sns.lineplot(data=data_, x="num_sample", y="L2-norm", label=n, color=color)

    plt.legend()
