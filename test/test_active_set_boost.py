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
import pytest
import scipy.sparse as sp

from cupdlpx import Model, PDLP
from cupdlpx._core import get_default_params, solve_once, validate_params

# alias -> (backend key, default)
ASB_PARAMS = {
    "ActiveSetBoost": ("active_set_boost", True),
    "ASBActivationTol": ("asb_activation_tol", 1e-4),
    "ASBWindowIter": ("asb_window_iter", 10000),
    "ASBSafetyFactor": ("asb_safety_factor", 0.9),
    "ASBMaxReverts": ("asb_max_reverts", 2),
    "ASBMinRaiseRatio": ("asb_min_raise_ratio", 1.1),
    "ASBReestimateChangeRatio": ("asb_reestimate_change_ratio", 0.01),
    "ASBConstraintTol": ("asb_constraint_tol", 1e-8),
    "ASBVariableTol": ("asb_variable_tol", 1e-8),
    "ASBDivergenceCeilingRatio": ("asb_divergence_ceiling_ratio", 0.7),
    "ASBDivergenceMargin": ("asb_divergence_margin", 0.05),
}
ASB_INFO_KEYS = ("ASBRaiseCount", "ASBRevertCount", "ASBPowerIterations")


def _model(base_lp_data, **params):
    c, A, l, u, lb, ub = base_lp_data
    model = Model(c, A, l, u, lb, ub)
    model.setParams(OutputFlag=False, Presolve=False, **params)
    return model


def test_defaults_exposed_through_every_layer(base_lp_data):
    defaults = get_default_params()
    assert defaults["debug"] is False
    model = _model(base_lp_data)
    for alias, (key, default) in ASB_PARAMS.items():
        assert PDLP._PARAM_ALIAS[alias] == key
        assert defaults[key] == default
        assert model.getParam(alias) == default
        assert model.getParam(key) == default
    assert PDLP._PARAM_ALIAS["Debug"] == "debug"
    assert model.getParam("Debug") is False


@pytest.mark.parametrize(
    "alias, value",
    [
        ("ASBWindowIter", 0),
        ("ASBMaxReverts", -1),
        ("ASBActivationTol", -1e-9),
        ("ASBSafetyFactor", 0.0),
        ("ASBMinRaiseRatio", 0.5),
        ("ASBReestimateChangeRatio", -0.1),
        ("ASBConstraintTol", -1e-9),
        ("ASBVariableTol", -1e-9),
        ("ASBDivergenceCeilingRatio", 0.0),
        ("ASBDivergenceCeilingRatio", 1.5),
        ("ASBDivergenceMargin", -0.1),
        ("TermCheckFreq", 2),
    ],
)
def test_out_of_range_values_fail_at_set_param(base_lp_data, alias, value):
    """setParam runs the C validator, so a bad value is rejected before optimize()
    with the backend's message and the stored parameters stay untouched."""
    model = _model(base_lp_data)
    before = dict(model.Params.items())
    key = PDLP._PARAM_ALIAS[alias]
    with pytest.raises(ValueError, match=key):
        model.setParam(alias, value)
    assert dict(model.Params.items()) == before
    with pytest.raises(ValueError, match=key):
        validate_params({key: value})


def test_boundary_values_accepted(base_lp_data):
    model = _model(base_lp_data)
    model.setParam("ASBMinRaiseRatio", 1.0)
    model.setParam("ASBDivergenceCeilingRatio", 1.0)
    model.setParam("ASBMaxReverts", 0)
    model.setParam("ASBActivationTol", 0.0)
    model.setParam("TermCheckFreq", 3)
    model.setParam("ActiveSetBoost", 0)
    assert model.getParam("ActiveSetBoost") is False
    model.optimize()
    assert model.Status == PDLP.OPTIMAL


@pytest.mark.parametrize(
    "params",
    [
        {"asb_min_raise_ratio": 0.5},
        {"asb_divergence_ceiling_ratio": 1.5},
        {"asb_window_iter": 0},
        {"asb_safety_factor": float("nan")},
    ],
)
def test_core_rejects_out_of_range_values(base_lp_data, params):
    c, A, l, u, lb, ub = base_lp_data
    with pytest.raises(ValueError):
        solve_once(A, c, None, lb, ub, l, u, params=params)


def _solve_info(base_lp_data, **params):
    c, A, l, u, lb, ub = base_lp_data
    base = {"verbose": False, "presolve": False}
    base.update(params)
    return solve_once(A, c, None, lb, ub, l, u, params=base)


def test_info_reports_controller_statistics(base_lp_data):
    info = _solve_info(base_lp_data)
    assert info["Status"] == "OPTIMAL"
    for key in ASB_INFO_KEYS:
        assert isinstance(info[key], int)
        assert info[key] >= 0


def test_boost_off_reports_zero_statistics(base_lp_data):
    info = _solve_info(base_lp_data, active_set_boost=False)
    assert info["Status"] == "OPTIMAL"
    assert all(info[key] == 0 for key in ASB_INFO_KEYS)


def test_boost_on_and_off_reach_the_same_solution(base_lp_data, atol):
    on = _solve_info(base_lp_data, active_set_boost=True)
    off = _solve_info(base_lp_data, active_set_boost=False)
    assert on["Status"] == off["Status"] == "OPTIMAL"
    assert np.allclose(on["X"], off["X"], atol=atol)
    assert abs(on["PrimalObj"] - off["PrimalObj"]) <= atol


def test_activated_controller_solves_to_tight_tolerance(base_lp_data, atol):
    """Activate immediately and solve tightly so the controller runs its full path."""
    info = _solve_info(
        base_lp_data,
        eps_optimal_relative=1e-8,
        eps_feasible_relative=1e-8,
        asb_activation_tol=1.0,
        asb_window_iter=1,
    )
    assert info["Status"] == "OPTIMAL"
    assert info["ASBRevertCount"] <= get_default_params()["asb_max_reverts"]
    assert np.allclose(info["X"], [1.0, 2.0], atol=atol)


def test_debug_implies_verbose_and_solves(base_lp_data):
    model = _model(base_lp_data, Debug=True)
    assert model.getParam("Debug") is True
    assert model.getParam("OutputFlag") is False
    model.optimize()
    assert model.Status == PDLP.OPTIMAL


def _hadamard(k):
    H = np.array([[1.0]])
    while H.shape[0] < k:
        H = np.block([[H, H], [H, -H]])
    return H


def _column_mask_lp(k=16):
    """
    Rows  H x + J y = b  with H Hadamard (k x k) and J all-ones (k x k).
    Minimize -sum(y),  0 <= x, y <= 1.
    At the optimum x = 0.5 is interior (free) and y = 1 sits on its upper
    bound (clamped). All rows are equalities, so every row is binding; the
    dense block J lives only in the clamped columns. After rescaling the full
    matrix has sigma ~ 0.71 while the masked matrix H alone has 1/sqrt(2k),
    so a raise happens only if the column mask correctly drops y.
    """
    H = _hadamard(k)
    J = np.ones((k, k))
    x_star = np.full(k, 0.5)
    y_star = np.ones(k)
    b = H @ x_star + J @ y_star
    A = sp.csr_matrix(np.hstack([H, J]))
    c = np.concatenate([np.zeros(k), -np.ones(k)])
    lb = np.zeros(2 * k)
    ub = np.ones(2 * k)
    return c, A, b.copy(), b.copy(), lb, ub, np.concatenate([x_star, y_star])


def test_raise_follows_the_column_mask(atol):
    """The boost must raise the step on an instance whose gain comes from clamped columns.

    Regression guard: the infeasibility-certificate routine once reused
    ``dual_slack`` as scratch, which wiped the clamping signal the column
    mask reads; the masked matrix then equalled the full one and no raise
    was possible.
    """
    c, A, l, u, lb, ub, x_star = _column_mask_lp()
    params = {
        "verbose": False,
        "presolve": False,
        "eps_optimal_relative": 1e-8,
        "eps_feasible_relative": 1e-8,
        "termination_evaluation_frequency": 10,
        "asb_activation_tol": 1.0,
        "asb_window_iter": 1,
    }
    info = solve_once(A, c, None, lb, ub, l, u, params=params)
    assert info["Status"] == "OPTIMAL"
    assert np.allclose(info["X"], x_star, atol=atol)
    assert info["ASBPowerIterations"] >= 1
    assert info["ASBRaiseCount"] >= 1

    off = solve_once(A, c, None, lb, ub, l, u, params={**params, "active_set_boost": False})
    assert off["Status"] == "OPTIMAL"
    assert np.allclose(off["X"], x_star, atol=atol)
    assert off["ASBRaiseCount"] == 0
