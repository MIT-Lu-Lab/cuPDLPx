# Copyright 2025 Haihao Lu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import pytest

# set numpy print options for better readability
np.set_printoptions(suppress=True, linewidth=120, precision=6)

# default absolute tolerance for tests
DEFAULT_ATOL = float(os.environ.get("CUPDLPX_TEST_ATOL", "1e-3"))

# put fixture for absolute tolerance
@pytest.fixture(scope="session")
def atol():
    return DEFAULT_ATOL

@pytest.fixture(scope="session")
def base_lp_data():
    """
    Coefficient for a simple LP
    Minimize  x1 + x2
    Subject to
        x1 + 2*x2 == 5
               x2 <= 2
      3*x1 + 2*x2 <= 8
           x1, x2 >= 0
    """
    c = np.array([1.0, 1.0])
    A = np.array([[1.0, 2.0],
                  [0.0, 1.0],
                  [3.0, 2.0]])
    l = np.array([5.0, -np.inf, -np.inf])
    u = np.array([5.0, 2.0, 8.0])
    lb = None
    ub = None
    return c, A, l, u, lb, ub
