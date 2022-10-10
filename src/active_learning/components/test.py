import pandas as pd
import numpy as np
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

    def run(self):
        results = pd.DataFrame(index=range(self.n_steps))
        for step in range(self.n_steps):
            b = self.active_object.budget
            results.loc[step, "budget"] = b
            n_points = int((self.budget - self.budget_0) / self.n_steps)

            x_new = self.active_object.query(n_points).drop_duplicates()
            y_new = self.f(x_new)
            self.active_object.add_labels(x_new, y_new)

            s = self.x_large_sample.sample(b)

            passive_learner = base.ActiveSRLearner(
                self.active_object.active_criterion,
                self.active_object.query_strategy,
                pd.DataFrame(s),
                pd.DataFrame(self.f(s)),
                bounds=np.array(self.bounds))

            # EVALUATE STRATEGIES
            passive_learner.query(n_points)
            self.perf_passive.append(evaluate(
                self.f,
                passive_learner.surface,
                self.bounds, num_mc=10000))
            self.active_object.result[self.active_object.iter]["error_l2"] = evaluate(
                self.f,
                self.active_object.surface,
                self.bounds, num_mc=10000)


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
