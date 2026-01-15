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

import pytest
import warnings

import numpy as np
from cupdlpx import Model, PDLP

def test_warm_start(base_lp_data, atol):
    """
    Verify that warm start works correctly.
    We use the optimal primal and dual solutions as the warm start.
    Minimize  x1 + x2
    Subject to
        x1 + 2*x2 == 5
               x2 <= 2
      3*x1 + 2*x2 <= 8
           x1, x2 >= 0
    Optimal solution: x* = (1, 2), y* = (1, -1, 0), objective = 3
    The solution should be the same.
    """
    # setup model
    c, A, l, u, lb, ub = base_lp_data
    model = Model(c, A, l, u, lb, ub)
    # turn off output
    model.setParams(OutputFlag=False, Presolve=False)
    # cold start baseline
    model.optimize()
    assert hasattr(model, "Status"), "Model.Status not exposed."
    assert model.Status == "OPTIMAL", f"Unexpected termination status (cold): {model.Status}"
    assert hasattr(model, "IterCount"), "Model.IterCount not exposed."
    baseline_iters = model.IterCount
    # set warm start values
    model.setWarmStart(primal=[1, 2], dual=[1, -1, 0])
    # optimize with warm start
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
    # check iteration count
    assert hasattr(model, "IterCount"), "Model.IterCount not exposed (warm)."
    assert model.IterCount <= baseline_iters, (
        f"Warm start took more iterations than cold start: {model.IterCount} > {baseline_iters}"
    )


def test_warm_start_primal(base_lp_data, atol):
    """
    Verify that warm start works correctly.
    We use the optimal primal solution as the warm start.
    Minimize  x1 + x2
    Subject to
        x1 + 2*x2 == 5
               x2 <= 2
      3*x1 + 2*x2 <= 8
           x1, x2 >= 0
    Optimal solution: x* = (1, 2), y* = (1, -1, 0), objective = 3
    The solution should be the same.
    """
    # setup model
    c, A, l, u, lb, ub = base_lp_data
    model = Model(c, A, l, u, lb, ub)
    # turn off output
    model.setParams(OutputFlag=False, Presolve=False)
    # set warm start values
    model.setWarmStart(primal=[1, 2])
    # optimize with warm start
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


def test_warm_start_dual(base_lp_data, atol):
    """
    Verify that warm start works correctly.
    We use the optimal dual solutions as the warm start.
    Minimize  x1 + x2
    Subject to
        x1 + 2*x2 == 5
               x2 <= 2
      3*x1 + 2*x2 <= 8
           x1, x2 >= 0
    Optimal solution: x* = (1, 2), y* = (1, -1, 0), objective = 3
    The solution should be the same.
    """
    # setup model
    c, A, l, u, lb, ub = base_lp_data
    model = Model(c, A, l, u, lb, ub)
    # turn off output
    model.setParams(OutputFlag=False, Presolve=False)
    # set warm start values
    model.setWarmStart(dual=[1, -1, 0])
    # optimize with warm start
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


def test_clear_warm_start(base_lp_data, atol):
    """
    Verify that clearWarmStart correctly resets the warm start values.
    """
    # setup model
    c, A, l, u, lb, ub = base_lp_data
    model = Model(c, A, l, u, lb, ub)
    # turn off output
    model.setParams(OutputFlag=False, Presolve=False)
    # set warm start values
    model.setWarmStart(primal=[1, 2], dual=[1, -1, 0])
    # clear warm start values
    model.clearWarmStart()
    assert model._primal_start is None, "Primal warm start not cleared."
    assert model._dual_start is None, "Dual warm start not cleared."
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

def test_warm_start_wrong_size_fallback(base_lp_data, atol):
    """
    Verify that warm start with wrong size falls back to cold start with a warning.
    """
    # setup model
    c, A, l, u, lb, ub = base_lp_data
    model = Model(c, A, l, u, lb, ub)
    # turn off output
    model.setParams(OutputFlag=False, Presolve=False)
    # set warm start values with wrong size
    with pytest.warns(RuntimeWarning):
        model.setWarmStart(primal=[1], dual=[1, 1]) # wrong sizes
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