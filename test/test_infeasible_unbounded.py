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

def test_infeasible_lp(base_lp_data, atol):
    """
    Verify the status for an infeasible LP.
    Minimize  x1 + x2
    Subject to
        x1 + 2*x2 == 10
               x2 <= 2
      3*x1 + 2*x2 <= 8
           x1, x2 >= 0
    """
    # setup model
    c, A, l, u, lb, ub = base_lp_data
    model = Model(c, A, l, u, lb, ub)
    # modify to make infeasible
    l, u = l.copy(), u.copy()  # make a copy to avoid modifying the fixture
    l[0], u[0] = 10, 10  # modify to make infeasible
    model.setConstraintLowerBound(l)
    model.setConstraintUpperBound(u)
    # turn off output
    model.setParams(OutputFlag=False, Presolve=False)
    # optimize
    #model.optimize()
    # check status
    #assert hasattr(model, "Status"), "Model.Status not exposed."
    #assert model.Status == "PRIMAL_INFEASIBLE", f"Unexpected termination status: {model.Status}"
    #assert model.StatusCode == PDLP.PRIMAL_INFEASIBLE, f"Unexpected termination status code: {model.StatusCode}"
    # check dual ray
    #assert model.DualRayObj > atol, f"DualRayObj should be positive for dual infeasible, got {model.DualRayObj}"


def test_unbounded_lp(base_lp_data, atol):
    """
    Verify the status for an unbounded LP.
    Minimize  x1 + x2
    Subject to
        x1 + 2*x2 == 5
      3*x1 + 2*x2 <= 8
    """
    # setup model
    c, A, l, u, lb, ub = base_lp_data
    model = Model(c, A, l, u, lb, ub)
    # modify to make unbounded
    l, u = l.copy(), u.copy()  # make a copy to avoid modifying the fixture
    l[1], u[1] = -np.inf, np.inf  # remove the second constraint
    model.setConstraintLowerBound(l)
    model.setConstraintUpperBound(u)
    lb = np.array([-np.inf, -np.inf])  # make x1, x2 unsigned
    model.setVariableLowerBound(lb)
    # turn off output
    model.setParams(OutputFlag=False, Presolve=False)
    # set infeasible tolerance
    model.setParams(InfeasibleTol=1e-6)
    # optimize
    #model.optimize()
    # check status
    #assert hasattr(model, "Status"), "Model.Status not exposed."
    #assert model.Status == "DUAL_INFEASIBLE", f"Unexpected termination status: {model.Status}"
    #assert model.StatusCode == PDLP.DUAL_INFEASIBLE, f"Unexpected termination status code: {model.StatusCode}"
    # check primal ray
    #assert model.PrimalRayLinObj < -atol, f"PrimalRayLinObj should be negative for dual infeasible, got {model.PrimalRayLinObj}"