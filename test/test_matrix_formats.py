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
from cupdlpx import Model
import scipy.sparse as sp

def test_csr(base_lp_data, atol):
    """
    Test CSR constraint matrix for the baseline minimization problem.
    """
    # setup model
    c, A, l, u, lb, ub = base_lp_data
    A = sp.csr_matrix(A) # convert to CSR
    model = Model(c, A, l, u, lb, ub)
    # turn off output
    model.setParams(OutputFlag=False, Presolve=False)
    # optimize
    model.optimize()
    # check status
    assert hasattr(model, "Status"), "Model.Status not exposed."
    assert model.Status == "OPTIMAL", f"Unexpected termination status: {model.Status}"
    # check primal solution
    assert hasattr(model, "X"), "Model.X (primal solution) not exposed."
    assert np.allclose(model.X, [1, 2], atol=atol), f"Unexpected primal solution: {model.X}"
    # check dual solution
    assert hasattr(model, "Pi"), "Model.Pi (dual solution) not exposed."
    assert np.allclose(model.Pi, [1, -1, 0], atol=atol), f"Unexpected dual solution: {model.Pi}"
    # check objective
    assert hasattr(model, "ObjVal"), "Model.ObjVal (objective value) not exposed."
    assert np.isclose(model.ObjVal, 3, atol=atol), f"Unexpected objective value: {model.ObjVal}"


def test_csc(base_lp_data, atol):
    """
    Test CSC constraint matrix for the baseline minimization problem.
    """
    # setup model
    c, A, l, u, lb, ub = base_lp_data
    A = sp.csc_matrix(A) # convert to CSC
    model = Model(c, A, l, u, lb, ub)
    # turn off output
    model.setParams(OutputFlag=False, Presolve=False)
    # optimize
    model.optimize()
    # check status
    assert hasattr(model, "Status"), "Model.Status not exposed."
    assert model.Status == "OPTIMAL", f"Unexpected termination status: {model.Status}"
    # check primal solution
    assert hasattr(model, "X"), "Model.X (primal solution) not exposed."
    assert np.allclose(model.X, [1, 2], atol=atol), f"Unexpected primal solution: {model.X}"
    # check dual solution
    assert hasattr(model, "Pi"), "Model.Pi (dual solution) not exposed."
    assert np.allclose(model.Pi, [1, -1, 0], atol=atol), f"Unexpected dual solution: {model.Pi}"
    # check objective
    assert hasattr(model, "ObjVal"), "Model.ObjVal (objective value) not exposed."
    assert np.isclose(model.ObjVal, 3, atol=atol), f"Unexpected objective value: {model.ObjVal}"


def test_coo(base_lp_data, atol):
    """
    Test COO constraint matrix for the baseline minimization problem.
    """
    # setup model
    c, A, l, u, lb, ub = base_lp_data
    A = sp.coo_matrix(A) # convert to COO
    model = Model(c, A, l, u, lb, ub)
    # turn off output
    model.setParams(OutputFlag=False, Presolve=False)
    # optimize
    model.optimize()
    # check status
    assert hasattr(model, "Status"), "Model.Status not exposed."
    assert model.Status == "OPTIMAL", f"Unexpected termination status: {model.Status}"
    # check primal solution
    assert hasattr(model, "X"), "Model.X (primal solution) not exposed."
    assert np.allclose(model.X, [1, 2], atol=atol), f"Unexpected primal solution: {model.X}"
    # check dual solution
    assert hasattr(model, "Pi"), "Model.Pi (dual solution) not exposed."
    assert np.allclose(model.Pi, [1, -1, 0], atol=atol), f"Unexpected dual solution: {model.Pi}"
    # check objective
    assert hasattr(model, "ObjVal"), "Model.ObjVal (objective value) not exposed."
    assert np.isclose(model.ObjVal, 3, atol=atol), f"Unexpected objective value: {model.ObjVal}"