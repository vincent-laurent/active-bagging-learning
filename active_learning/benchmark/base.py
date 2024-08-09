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
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd

from active_learning import ActiveSurfaceLearner
from active_learning.benchmark.utils import evaluate


class ITestingClass(ABC):
    def __init__(self, budget: int, budget_0: int, function: callable,
                 x_sampler: callable, learner: ActiveSurfaceLearner,
                 n_steps: int,
                 bounds, name=None,
                 estimator=None):
        self.f = function
        self.budget = budget
        self.n_steps = n_steps
        self.budget_0 = budget_0
        self.x_sampler = x_sampler
        self.bounds = bounds
        self.learner: ActiveSurfaceLearner = learner
        self.iter = 0
        self.estimator = estimator
        self.name = name

        self.metric = []
        self.__check_args()
        self.result = {}
        if hasattr(self.learner, "set_bounds"):
            self.learner.set_bounds(bounds)

    def _start(self):
        self.x_input = self.x_sampler(self.budget_0)
        self.y_input = pd.DataFrame(self.f(self.x_input))

        self.indexes = np.ones(len(self.x_input))
        self._samples = np.linspace(self.budget_0, self.budget,
                                    num=self.n_steps, dtype=int)
        self._samples = np.concatenate(
            ([self._samples[0]], np.diff(self._samples)))

    @abstractmethod
    def run(self):
        ...

    def __check_args(self):
        assert hasattr(self.learner, "query")
        assert callable(self.f)

    def add_labels(self, x: pd.DataFrame, y: pd.DataFrame):
        self.iter += 1
        self.indexes = np.concatenate((self.indexes, self.iter * np.ones(len(x))))
        self.x_input = pd.concat((self.x_input, x), axis=0)
        self.y_input = pd.concat((self.y_input, y), axis=0)
        self.x_new = x
        self.y_new = y

    @property
    def parameters(self):
        ret = dict(
            budget=self.budget,
            budget_0=self.budget_0,
            n_steps=self.n_steps,
            function_hash=hash(self.f) + hash(str(self.bounds)),
            x_sampler_hash=hash(self.x_sampler),
        )
        ret["test_id"] = ret["budget"] + ret["function_hash"] + ret["n_steps"]
        ret["name"] = hash(
            np.random.uniform()) if self.name is None else self.name
        return ret

    def save(self):
        m = evaluate(
            self.f,
            self.learner.predict,
            self.bounds, num_mc=100000)
        self.metric.append(m)
        self.result[self.iter] = dict(
            learner=deepcopy(self.learner),
            budget=int(len(self.x_input)),
            data=deepcopy(self.x_input),
            l2=m,
        )


class ServiceTestingClassAL(ITestingClass):

    def run(self):
        self._start()
        self.learner.fit(
            self.x_input,
            pd.DataFrame(self.y_input))

        self.save()
        for n_points in self._samples[1:]:
            x_new = self.learner.query(n_points)
            y_new = pd.DataFrame(self.f(x_new))

            self.learner = deepcopy(self.learner)
            self.add_labels(x_new, y_new)
            self.learner.fit(self.x_input, self.y_input)
            self.save()


class ServiceTestingClassModAL(ITestingClass):

    def run(self):
        self._start()
        self.learner.teach(self.x_input.values,
                           pd.DataFrame(self.y_input).values)

        self.save()
        for n_points in self._samples[1:]:
            x_new = pd.DataFrame(
                np.array([self.learner.query(self.x_sampler(200).values)[1] for _ in range(n_points)]))
            y_new = pd.DataFrame(self.f(x_new))
            self.learner = deepcopy(self.learner)
            self.add_labels(x_new, y_new)
            self.learner.fit(self.x_input.values, self.y_input.values)
            self.save()


class ModuleExperiment:
    def __init__(self, test_list: typing.List["ITestingClass"], n_experiment=10, save=False):
        self.test_list = test_list
        self.results = {}
        self.n_experiment = n_experiment
        columns = [*test_list[0].parameters.keys(), "L2-norm", "date"]

        self._cv_result = pd.DataFrame(columns=columns)
        self.save = save
        if self.save:
            self.saving_class = {}

    def _run_single_experiment(self, test, i):
        """Helper function to run a single experiment."""
        test_ = deepcopy(test)
        test_.run()
        res = pd.DataFrame(columns=self._cv_result.columns)
        res["L2-norm"] = pd.DataFrame(test_.result).loc["l2"].astype(float).values
        res["num_sample"] = pd.DataFrame(test_.result).loc["budget"].astype(float).values
        res["date"] = datetime.today()
        for c in test.parameters.keys():
            res[c] = test.parameters[c]
        return res

    def run(self):
        with ProcessPoolExecutor(max_workers=20) as executor:
            futures = []
            for test in self.test_list:
                for i in range(self.n_experiment):
                    futures.append(executor.submit(self._run_single_experiment, test, i))

            for future in as_completed(futures):
                res = future.result()
                self._cv_result = pd.concat((self._cv_result, res), axis=0)

        self.cv_result_ = self._cv_result
        self.cv_result_["date"] = pd.to_datetime(self.cv_result_["date"])


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
