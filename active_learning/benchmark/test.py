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
from copy import deepcopy

import numpy as np
import pandas as pd

from active_learning import base
from active_learning.components.active_criterion import IActiveCriterion
from active_learning.components.query_strategies import IQueryStrategy


class TestingClass:
    def __init__(self, budget: int, budget_0: int, function: callable,
                 active_criterion: IActiveCriterion,
                 query_strategy: IQueryStrategy,
                 x_sampler: callable, n_steps: int,
                 bounds, name=None
                 ):
        self.f = function
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
            function_hash=hash(function) + hash(str(bounds)),
            active_criterion=str(active_criterion),
            query_strategy=str(query_strategy),
            x_sampler_hash=hash(x_sampler),
        )
        self.parameters["test_id"] = self.parameters["budget"] + \
                                     self.parameters["function_hash"] + \
                                     self.parameters["n_steps"]
        self.parameters["name"] = hash(
            np.random.uniform()) if name is None else name

        self.metric = []

    def run(self):
        x_train = self.x_sampler(self.budget_0)
        self.learner = base.ActiveSRLearner(
            self.active_criterion,
            self.query_strategy,
            x_train,
            pd.DataFrame(self.f(x_train)),
            self.bounds)

        nb_sample_cumul = np.linspace(self.budget_0, self.budget,
                                      num=self.n_steps, dtype=int)
        nb_sample = np.concatenate(
            ([nb_sample_cumul[0]], np.diff(nb_sample_cumul)))
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
            sns.kdeplot(x=error, y=crit_,
                        c=plt.get_cmap("rainbow")(i / len(self.functions)),
                        log_scale=(True, True),
                        linewidth=0.1)
        plt.xlabel("$L_2$ error")
        plt.ylabel("Estimated variance")

    def plot_error_vs_criterion_pointwise(self, num_mc=10000):
        import matplotlib.pyplot as plt
        for i_, i in enumerate(self.learner.result.keys()):
            res = self.metric[i_]
            variance = integrate(self.criterion[i], self.bounds, num_mc)
            plt.loglog([res], [variance],
                       c=plt.get_cmap("rainbow")(i / len(self.functions)),
                       marker="o", lw=0)
        plt.xlabel("$L_2$ error")
        plt.ylabel("Estimated variance")


class ModuleExperiment:
    def __init__(self, test_list: typing.List[TestingClass], n_experiment=10,
                 save=False):
        self.test_list = test_list
        self.results = {}
        self.n_experiment = n_experiment
        columns = [*test_list[0].parameters.keys(), "L2-norm", "date"]

        self._cv_result = pd.DataFrame(
            columns=columns,
        )
        self.save = save
        if self.save:
            self.saving_class = {}

    def run(self):
        from datetime import datetime

        for test in self.test_list:
            for i in range(self.n_experiment):
                test_ = deepcopy(test)
                test_.run()
                res = pd.DataFrame(columns=self._cv_result.columns)
                res["L2-norm"] = test_.metric
                res["num_sample"] = pd.DataFrame(test_.learner.result).loc[
                    "budget"].astype(float).values
                res["date"] = datetime.today()
                for c in test.parameters.keys():
                    res[c] = test.parameters[c]
                self._cv_result = pd.concat((self._cv_result, res), axis=0)

        self.cv_result_ = self._cv_result
        self.cv_result_["date"] = pd.to_datetime(self.cv_result_["date"])


