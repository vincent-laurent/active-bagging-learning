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
import numpy as np

from active_learning.benchmark import functions
from active_learning.components import latin_square


def test_latin_square():
    x = latin_square.scipy_lhs_sampler(100)
    assert len(x) == 100
    assert x.min() >= 0
    assert x.max() <= 1

    x = latin_square.scipy_lhs_sampler(100, dim=2)
    assert len(x) == 100
    assert x.min() >= 0
    assert x.max() <= 1
    assert x.shape == (100, 2)

    x = latin_square.scipy_lhs_sampler(100, x_limits=np.array([[1, 2], [1, 2]]))
    assert len(x) == 100
    assert x.min() >= 1
    assert x.max() <= 2
    assert x.shape == (100, 2)
