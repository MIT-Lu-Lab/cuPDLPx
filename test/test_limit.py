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

import time
import numpy as np
import scipy.sparse as sp
from cupdlpx import Model

SEED = 42

def test_time_limit(atol):
    """
    Test time limit for large sparse LP.
    """
    # setup model
    rng = np.random.default_rng(seed=42)
    m, n = 12000, 10000
    A = sp.rand(m, n, density=0.01, format="csr", random_state=rng)
    c = rng.standard_normal(n)
    l = None
    u = rng.random(m)
    lb = np.zeros(n)
    ub = None
    model = Model(c, A, l, u, lb, ub)
    # turn off output
    model.setParams(OutputFlag=False, Presolve=False)
    # set time limit
    model.setParams(TimeLimit=0.1)
    # optimize
    tick = time.time()
    model.optimize()
    tock = time.time()
    # check status
    assert hasattr(model, "Status"), "Model.Status not exposed."
    assert model.Status == "TIME_LIMIT", f"Unexpected termination status: {model.Status}"
    # check solving time
    solving_time = tock - tick
    assert solving_time < 1, f"Solving time exceeded limit a lot: {solving_time} seconds"
    assert hasattr(model, "Runtime"), "Model.Runtime not exposed."
    assert model.Runtime < 1, f"Internal solving time exceeded limit a lot: {model.Runtime} seconds"


def test_iters_limit(atol):
    """
    Test iteration limit for large sparse LP.
    """
    # setup model
    rng = np.random.default_rng(seed=42)
    m, n = 12000, 10000
    A = sp.rand(m, n, density=0.01, format="csr", random_state=rng)
    c = rng.standard_normal(n)
    l = None
    u = rng.random(m)
    lb = np.zeros(n)
    ub = None
    model = Model(c, A, l, u, lb, ub)
    # turn off output
    model.setParams(OutputFlag=False)
    # set iteration limit
    model.setParams(IterationLimit=25)
    # optimize
    model.optimize()
    # check status
    assert hasattr(model, "Status"), "Model.Status not exposed."
    assert model.Status == "ITERATION_LIMIT", f"Unexpected termination status: {model.Status}"
    # check solving time
    assert hasattr(model, "IterCount"), "Model.IterCount not exposed."
    assert model.IterCount < 50, f"Internal iteration count exceeded limit a lot: {model.IterCount} seconds"