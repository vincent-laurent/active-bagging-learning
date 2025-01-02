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

from active_learning.benchmark import benchmark_config
from active_learning.benchmark import utils

from importlib import reload

reload(utils)
reload(benchmark_config)

if __name__ == '__main__':
    benchmark_config.test_in_sum_sine_5pi()
    benchmark_config.test_function()

    # main
    t = benchmark_config.create_benchmark_list()
    me = benchmark_config.ModuleExperiment(t, n_experiment=100)
    me.run()
    utils.write_benchmark(me.cv_result_, "data/benchmark.csv")

    data = utils.read_benchmark("data/benchmark")
    data = data[data["date"] > "2024-10-31 18"]
    utils.plot_benchmark(data=data)
    plt.savefig(".public/benchmark_result_2024")

    data = utils.read_benchmark("data/benchmark")
    data = data[data["date"] > "2024-10-31 18"]
    utils.plot_benchmark(data=data, standardise=True)
    plt.savefig(".public/benchmark_result_2024_standerdized")
