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

import numpy as np
import scipy.sparse as sp
from cupdlpx import Model

SEED = 42

def test_random_sparse_lp(atol):
    """
    Test random large sparse LP.
    """
    # setup model
    rng = np.random.default_rng(seed=42)
    m, n = 1000, 800
    A = sp.rand(m, n, density=0.02, format="csr", random_state=rng)
    c = rng.standard_normal(n)
    l = None
    u = rng.random(m)
    lb = np.zeros(n)
    ub = None
    model = Model(c, A, l, u, lb, ub)
    # turn off output
    model.setParams(OutputFlag=False, Presolve=False)
    # optimize
    model.optimize()
    # check status
    assert hasattr(model, "Status"), "Model.Status not exposed."
    assert model.Status == "OPTIMAL", f"Unexpected termination status: {model.Status}"
    # primal objective
    obj_primal = c @ model.X
    # dual objective
    obj_dual = model.Pi @ u
    # check objective
    assert np.isclose(obj_primal, obj_dual, atol=atol), f"Primal and dual objectives do not match: {obj_primal} != {obj_dual}"
    assert np.isclose(obj_primal, model.ObjVal, atol=atol), f"Primal and ObjVal do not match: {obj_primal} != {model.ObjVal}"
    assert np.isclose(obj_dual, model.ObjVal, atol=atol), f"Dual and ObjVal do not match: {obj_dual} != {model.ObjVal}"
    # primal feasibility: A x <= u, x >= 0
    lhs = A @ model.X
    assert np.all(lhs <= u + atol), "Primal solution is not feasible."
    assert np.all(model.X >= -atol), "Primal solution is not feasible."
    # dual feasibility: A' y >= c, y >= 0
    lhs = A.T @ model.Pi
    assert np.all(lhs <= c + atol), "Dual solution is not feasible."
    assert np.all(model.Pi <= atol), "Dual solution is not feasible."