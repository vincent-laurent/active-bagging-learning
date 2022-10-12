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
                 bounds,

                 ):
        self.f = funcion
        self.budget = budget
        x_train = x_sampler(budget_0)
        self.active_object = base.ActiveSRLearner(
            active_criterion,
            query_strategy,
            x_train,
            self.f(x_train),
            bounds)
        self.n_steps = n_steps
        self.budget_0 = budget_0
        self.x_large_sample = pd.DataFrame(x_sampler(budget * 100))
        self.bounds = bounds
        self.perf_passive = []
        self.perf_active = []
        self.functions = {}
        self.criterion = {}

    def run(self):
        results = pd.DataFrame(index=range(self.n_steps))
        for step in range(self.n_steps):
            b = self.active_object.budget
            results.loc[step, "budget"] = b
            n_points = int((self.budget - self.budget_0) / self.n_steps)

            x_new = self.active_object.query(n_points).drop_duplicates()
            y_new = self.f(x_new)
            self.active_object.add_labels(x_new, y_new)
            self.functions[step] = deepcopy(self.active_object.active_criterion.function)
            self.criterion[step] = deepcopy(self.active_object.active_criterion.__call__)

            s = self.x_large_sample.sample(b)

            passive_learner = base.ActiveSRLearner(
                self.active_object.active_criterion,
                self.active_object.query_strategy,
                pd.DataFrame(s),
                pd.DataFrame(self.f(s)),
                bounds=np.array(self.bounds))

            # EVALUATE STRATEGIES
            passive_learner.query(n_points)
            self.perf_active.append(evaluate(
                self.f,
                self.functions[step],
                self.bounds, num_mc=100000))
            self.perf_passive.append(evaluate(
                self.f,
                self.active_object.surface,
                self.bounds, num_mc=100000))

    def plot_error_vs_criterion(self, n=1000):
        import matplotlib.pyplot as plt
        import seaborn as sns
        x = self.x_large_sample.sample(n)
        for i in self.functions.keys():
            error = np.abs(np.ravel(self.f(x)) - self.functions[i](x)) ** 2
            crit_ = self.active_object.active_criterion(x)
            sns.kdeplot(x=error, y=crit_, c=plt.get_cmap("rainbow")(i / len(self.functions)), log_scale=(True, True),
                        linewidth=0.1)
        plt.xlabel("$L_2$ error")
        plt.ylabel("Estimated variance")

    def plot_error_vs_criterion_pointwise(self, num_mc=10000):
        import matplotlib.pyplot as plt
        for i_, i in enumerate(self.functions.keys()):
            res = self.perf_active[i_]
            variance = integrate(self.criterion[i], self.bounds, num_mc)
            plt.loglog([res], [variance], c=plt.get_cmap("rainbow")(i / len(self.functions)), marker="o", lw=0)
        plt.xlabel("$L_2$ error")
        plt.ylabel("Estimated variance")


class Experiment:
    def __init__(self, test1: TestingClass, test2: TestingClass = None):
        self.test1 = test1
        self.test2 = test2
        self.results_1 = {}
        self.results_2 = {}

    def run(self, n_experiment=10):
        self.n_experiment_ = n_experiment
        self.cv_result_1_ = pd.DataFrame(columns=range(n_experiment))
        self.cv_result_2_ = pd.DataFrame(columns=range(n_experiment))
        for i in range(n_experiment):
            test1 = deepcopy(self.test1)
            test2 = deepcopy(self.test2)
            test1.run()
            test2.run()
            self.results_1[i] = test1
            self.results_2[i] = test2
            self.cv_result_1_[i] = test1.perf_active
            self.cv_result_2_[i] = test2.perf_active

    def plot_performance(self):
        import matplotlib.pyplot as plt
        budgets = pd.DataFrame(self.results_1[0].active_object.result).loc["budget"].astype(float)
        d1 = self.cv_result_1_
        d2 = self.cv_result_2_
        plt.figure()
        plt.plot(budgets, d1.mean(axis=1), c="r", label="bootstrap")
        plt.plot(budgets, d2.mean(axis=1), c="b", label="regular")
        plt.fill_between(budgets, d1.mean(axis=1) - d1.std(axis=1), d1.mean(axis=1) + d1.std(axis=1), color="r", alpha=0.2)
        plt.fill_between(budgets, d2.mean(axis=1) - d2.std(axis=1), d2.mean(axis=1) + d2.std(axis=1), color="b", alpha=0.2)
        plt.legend()






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
    plt.scatter(test.active_object.x_input, test.f(test.active_object.x_input),
                c=test.active_object.x_input.index, cmap="coolwarm", alpha=0.2)

    plt.figure()
    sns.histplot(test.active_object.x_input, bins=50)


def plot_iter(test: TestingClass):
    import matplotlib.pyplot as plt
    domain = np.linspace(test.bounds[0][0], test.bounds[0][1], 1000)
    iter_ = int(test.active_object.x_input.index.max())
    fig, axs = plt.subplots(4, iter_ // 4, sharey=True, sharex=True, figsize=(8, 8), dpi=200)

    test.active_object.x_input.index = test.active_object.x_input.index.astype(int)
    for iter, ax in enumerate(axs.ravel()):
        error = test.criterion[iter](domain.reshape(-1, 1))
        prediction = test.functions[iter](domain.reshape(-1, 1))
        ax.plot(domain, prediction, color="b", label="iter={}".format(iter))
        ax.plot(domain, test.f(domain), color="grey", linestyle="--")
        ax.fill_between(domain.ravel(), prediction - error / 2, prediction + error / 2, color="b", alpha=0.2)
        training_dataset = test.active_object.x_input.loc[range(iter + 1)]
        new_samples = test.active_object.x_input.loc[iter + 1]
        ax.scatter(training_dataset, test.f(training_dataset), color="k", marker=".")

        if iter > 0: ax.scatter(new_samples, test.f(new_samples), color="r", marker=".")
        ax.set_ylim(-0.9, 0.7)
        ax.legend()
