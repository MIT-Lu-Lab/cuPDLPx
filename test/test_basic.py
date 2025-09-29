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
from cupdlpx import Model, PDLP

def test_smoke_optimize_runs(base_lp_data):
    """
    Smoke test to check if optimize runs without error
    """
    # setup model
    c, A, l, u, lb, ub = base_lp_data
    model = Model(c, A, l, u, lb, ub)
    # turn off output
    model.setParams(OutputFlag=False)
    # optimize
    model.optimize()

def test_minimize_solution_correct(base_lp_data, atol):
    """
    Verify the status optimal solution and objective for a minimization problem.
    Minimize  x1 + x2
    Subject to
        x1 + 2*x2 == 5
               x2 <= 2
      3*x1 + 2*x2 <= 8
           x1, x2 >= 0
    Optimal solution: x* = (1, 2), y* = (1, -1, 0), objective = 3
    """
    # setup model
    c, A, l, u, lb, ub = base_lp_data
    model = Model(c, A, l, u, lb, ub)
    # turn off output
    model.setParams(OutputFlag=False)
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


def test_maximize_solution_correct(base_lp_data, atol):
    """
    Verify the status optimal solution and objective for a maximization problem.
    Maximize  x1 + x2
    Subject to
        x1 + 2*x2 == 5
               x2 <= 2
      3*x1 + 2*x2 <= 8
           x1, x2 >= 0
    Optimal solution: x* = (1.5, 1.75), y* = (-0.25, 0, -0.25), objective = 3.25
    """
    # setup model
    c, A, l, u, lb, ub = base_lp_data
    model = Model(c, A, l, u, lb, ub)
    # model sense
    try:
        model.ModelSense = PDLP.MAXIMIZE
    except Exception as e:
        print(f"cuPDLPx: failed to set model sense to MAXIMIZE.")
    # turn off output
    model.setParams(OutputFlag=False)
    # optimize
    model.optimize()
    # check status
    assert hasattr(model, "Status"), "Model.Status not exposed."
    assert model.Status == "OPTIMAL", f"Unexpected termination status: {model.Status}"
    # check primal solution
    assert hasattr(model, "X"), "Model.X (primal solution) not exposed."
    assert np.allclose(model.X, [1.5, 1.75], atol=atol), f"Unexpected primal solution: {model.X}"
    # check dual solution
    assert hasattr(model, "Pi"), "Model.Pi (dual solution) not exposed."
    assert np.allclose(model.Pi, [-0.25, 0, -0.25], atol=atol), f"Unexpected dual solution: {model.Pi}"
    # check objective
    assert hasattr(model, "ObjVal"), "Model.ObjVal (objective value) not exposed."
    assert np.isclose(model.ObjVal, 3.25, atol=atol), f"Unexpected objective value: {model.ObjVal}"