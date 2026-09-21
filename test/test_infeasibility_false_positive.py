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

"""
Regression tests for false infeasibility claims.

The PDLP ray criterion (constraint violation relative to the ray objective)
is not scale invariant: on a bounded, feasible LP whose objective vector or
matrix contains huge coefficients, a direction that violates the constraints
badly can still look like a certificate. These LPs must solve to OPTIMAL with
the default infeasibility tolerance, with and without presolve.
"""

import numpy as np
import pytest
from cupdlpx import Model, PDLP


def _solve(c, A, l, u, lb, ub, presolve):
    # Model defaults variable bounds to (-inf, inf); these LPs are stated with x >= 0.
    n = len(c)
    if lb is None:
        lb = np.zeros(n)
    if ub is None:
        ub = np.full(n, np.inf)
    model = Model(c, A, l, u, lb, ub)
    model.setParams(OutputFlag=False, Presolve=presolve, TimeLimit=30.0)
    model.optimize()
    return model


@pytest.mark.parametrize("presolve", [False, True])
def test_huge_objective_with_dummy_equality(presolve, atol):
    """
    Minimize  -1e10*x - 1.001e10*y
    Subject to  x + y <= 2
                1e10*z == 0
                x, y, z >= 0
    Optimum (0, 2, 0) with objective -2.002e10. The direction (0, 1, 0)
    improves the objective by 1e10 per unit while violating x + y <= 0 by 1;
    the large unrelated coefficient on z must not hide that violation.
    """
    c = np.array([-1e10, -1.001e10, 0.0])
    A = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1e10]])
    l = np.array([-np.inf, 0.0])
    u = np.array([2.0, 0.0])
    model = _solve(c, A, l, u, None, None, presolve)
    assert model.Status == PDLP.OPTIMAL, f"Unexpected termination status: {model.Status}"
    assert abs(model.ObjVal / -2.002e10 - 1.0) < atol


@pytest.mark.parametrize("presolve", [False, True])
def test_huge_objective_and_huge_row(presolve, atol):
    """
    Same LP, but the large coefficients sit in a real constraint:
    Minimize  -1e10*x - 1.001e10*y
    Subject to  x + y <= 2
                -1e10*x + 1e10*y >= 0
                x, y >= 0
    Optimum (0, 2) with objective -2.002e10.
    """
    c = np.array([-1e10, -1.001e10])
    A = np.array([[1.0, 1.0], [-1e10, 1e10]])
    l = np.array([-np.inf, 0.0])
    u = np.array([2.0, np.inf])
    model = _solve(c, A, l, u, None, None, presolve)
    assert model.Status == PDLP.OPTIMAL, f"Unexpected termination status: {model.Status}"
    assert abs(model.ObjVal / -2.002e10 - 1.0) < atol


@pytest.mark.parametrize("presolve", [False, True])
@pytest.mark.parametrize("seed", range(6))
def test_random_bounded_lp_with_dummy_equality(presolve, seed, atol):
    """
    Random bounded LPs: positive constraint coefficients, x = 1 feasible,
    objective scaled by 1e10, plus an unrelated equality with a 1e12
    coefficient. All are feasible and bounded.
    """
    rng = np.random.default_rng(seed)
    m, n = 6, 4
    A_rows = rng.uniform(0.1, 2.0, (m, n))
    b = A_rows @ np.ones(n)
    c = np.concatenate([-rng.uniform(0.1, 2.0, n) * 1e10, [0.0]])
    A = np.zeros((m + 1, n + 1))
    A[:m, :n] = A_rows
    A[m, n] = 1e12
    l = np.concatenate([np.full(m, -np.inf), [0.0]])
    u = np.concatenate([b, [0.0]])
    model = _solve(c, A, l, u, None, None, presolve)
    assert model.Status == PDLP.OPTIMAL, f"Unexpected termination status: {model.Status}"
    # a feasible point (x = 1, z = 0) bounds the optimum from above
    assert model.ObjVal <= c[:n].sum() * (1.0 - atol) + atol * abs(c[:n].sum())


@pytest.mark.parametrize("presolve", [False, True])
@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e8])
def test_infeasible_lp_detection_is_scale_invariant(presolve, scale, base_lp_data):
    """
    The infeasible LP from test_infeasible_unbounded.py with the objective
    scaled must still be declared primal infeasible with the default tolerance.
    """
    c, A, l, u, lb, ub = base_lp_data
    l, u = l.copy(), u.copy()
    l[0], u[0] = 10, 10
    model = _solve(c * scale, A, l, u, lb, ub, presolve)
    assert model.Status == PDLP.PRIMAL_INFEASIBLE, f"Unexpected termination status: {model.Status}"
