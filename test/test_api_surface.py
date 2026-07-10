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
import scipy.sparse as sp
import pytest

import cupdlpx
from cupdlpx import Model, PDLP, read
from cupdlpx._core import solve_once


def test_public_exports_include_documented_api():
    assert cupdlpx.PDLP is PDLP
    assert cupdlpx.Model is Model
    assert cupdlpx.read is read
    assert isinstance(cupdlpx.__version__, str)
    assert set(cupdlpx.__all__) == {"Model", "PDLP", "read", "__version__"}


def _model(base_lp_data):
    c, A, l, u, lb, ub = base_lp_data
    model = Model(c, A, l, u, lb, ub)
    model.setParams(OutputFlag=False, Presolve=False)
    return model


def test_data_property_reads(base_lp_data):
    """Read-side of the model-data properties."""
    model = _model(base_lp_data)
    assert model.c.shape == (2,)
    assert model.c0 == 0.0
    assert model.A.shape == (3, 2)
    assert model.constr_lb.shape == (3,)
    assert model.constr_ub.shape == (3,)
    assert model.lb is None
    assert model.ub is None
    assert model.ModelSense == PDLP.MINIMIZE


def test_data_property_writes(base_lp_data):
    """Direct assignment routes through the validating setters."""
    model = _model(base_lp_data)
    model.c = [2.0, 3.0]
    assert np.allclose(model.c, [2.0, 3.0])
    model.c0 = 4.0
    assert model.c0 == 4.0
    model.lb = [0.0, 0.0]
    assert np.allclose(model.lb, [0.0, 0.0])
    model.ub = [10.0, 10.0]
    assert np.allclose(model.ub, [10.0, 10.0])
    model.constr_lb = [1.0, 2.0, 3.0]
    assert np.allclose(model.constr_lb, [1.0, 2.0, 3.0])
    model.constr_ub = [4.0, 5.0, 6.0]
    assert np.allclose(model.constr_ub, [4.0, 5.0, 6.0])
    model.A = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    assert model.A.shape == (3, 2)
    model.ModelSense = PDLP.MAXIMIZE
    assert model.ModelSense == PDLP.MAXIMIZE


def test_bounds_none_clears(base_lp_data):
    """Passing None clears optional bounds."""
    model = _model(base_lp_data)
    model.lb = [0.0, 0.0]
    model.lb = None
    assert model.lb is None
    model.ub = None
    assert model.ub is None
    model.constr_lb = None
    assert model.constr_lb is None
    model.constr_ub = None
    assert model.constr_ub is None


def test_property_validation_errors(base_lp_data):
    """Invalid assignments raise immediately instead of failing in the solver."""
    model = _model(base_lp_data)
    with pytest.raises(ValueError):
        model.c = [1.0, 2.0, 3.0]          # wrong length
    with pytest.raises(ValueError):
        model.c = [[1.0, 2.0]]             # not 1D
    with pytest.raises(TypeError):
        model.A = "not a matrix"
    with pytest.raises(ValueError):
        model.A = np.ones(3)               # not 2D
    with pytest.raises(ValueError):
        model.A = np.ones((3, 5))          # wrong number of variables
    with pytest.raises(ValueError):
        model.lb = [0.0]                   # wrong length
    with pytest.raises(ValueError):
        model.ub = [0.0]
    with pytest.raises(ValueError):
        model.constr_lb = [0.0, 0.0]       # need 3
    with pytest.raises(ValueError):
        model.constr_ub = [0.0, 0.0]
    with pytest.raises(ValueError):
        model.ModelSense = 99


def test_bound_nan_validation(base_lp_data):
    model = _model(base_lp_data)
    with pytest.raises(ValueError):
        model.lb = [0.0, np.nan]
    with pytest.raises(ValueError):
        model.ub = [np.nan, 1.0]
    with pytest.raises(ValueError):
        model.constr_lb = [0.0, np.nan, -np.inf]
    with pytest.raises(ValueError):
        model.constr_ub = [0.0, 1.0, np.nan]

    model.lb = [0.0, -np.inf]
    model.ub = [np.inf, 1.0]
    model.constr_lb = [5.0, -np.inf, -np.inf]
    model.constr_ub = [5.0, np.inf, 8.0]


def test_invalid_objective_assignment_preserves_previous_value(base_lp_data):
    model = _model(base_lp_data)
    original = model.c.copy()
    with pytest.raises(ValueError):
        model.c = [1.0, 2.0, 3.0]
    assert np.allclose(model.c, original)


def test_dense_inputs_are_copied(base_lp_data):
    c, A, l, u, lb, ub = base_lp_data
    c = c.copy()
    A = A.copy()
    l = l.copy()
    u = u.copy()
    model = Model(c, A, l, u, lb, ub)
    c[0] = 99.0
    A[0, 0] = 99.0
    l[0] = 99.0
    u[0] = 99.0
    assert np.allclose(model.c, [1.0, 1.0])
    assert model.A[0, 0] == 1.0
    assert model.constr_lb[0] == 5.0
    assert model.constr_ub[0] == 5.0


def test_model_arrays_are_read_only(base_lp_data):
    c, A, l, u, lb, ub = base_lp_data
    model = Model(c, A, l, u, lb, ub)
    with pytest.raises(ValueError):
        model.c[0] = 99.0
    with pytest.raises(ValueError):
        model.A[0, 0] = 99.0
    with pytest.raises(ValueError):
        model.constr_lb[0] = 99.0
    with pytest.raises(ValueError):
        model.constr_ub[0] = 99.0

    model.c = [2.0, 3.0]
    assert np.allclose(model.c, [2.0, 3.0])


def test_sparse_model_arrays_are_read_only(base_lp_data):
    c, A, l, u, lb, ub = base_lp_data
    model = Model(c, sp.csr_matrix(A), l, u, lb, ub)
    with pytest.raises(ValueError):
        model.A.data[0] = 99.0
    with pytest.raises(ValueError):
        model.A.indices[0] = 99
    with pytest.raises(ValueError):
        model.A.indptr[0] = 99


def test_bound_order_validation(base_lp_data):
    model = _model(base_lp_data)
    model.lb = [2.0, 0.0]
    model.ub = [1.0, 1.0]
    with pytest.raises(ValueError):
        model.optimize()

    with pytest.raises(ValueError):
        Model(
            [1.0],
            np.array([[1.0]]),
            constraint_lower_bound=[2.0],
            constraint_upper_bound=[1.0],
        )


def test_set_params_is_transactional(base_lp_data):
    model = _model(base_lp_data)
    old_time_limit = model.getParam("TimeLimit")
    with pytest.raises(KeyError):
        model.setParams(TimeLimit=123.0, DefinitelyNotAParam=1)
    assert model.getParam("TimeLimit") == old_time_limit


def test_param_value_validation(base_lp_data):
    model = _model(base_lp_data)
    # float params accept ints and numpy numbers
    model.setParam("TimeLimit", 12)
    assert model.getParam("TimeLimit") == 12.0
    model.setParam("TimeLimit", np.float64(30.0))
    assert model.getParam("TimeLimit") == 30.0
    model.setParam("OptimalityNorm", "LINF")
    assert model.getParam("OptimalityNorm") == "linf"

    # bool params accept real/numpy bools and 0/1 (coerced to a Python bool)
    model.setParam("OutputFlag", 1)
    assert model.getParam("OutputFlag") is True
    model.setParam("OutputFlag", 0)
    assert model.getParam("OutputFlag") is False
    model.setParam("OutputFlag", np.bool_(True))
    assert model.getParam("OutputFlag") is True

    # int params accept numpy ints and integer-valued floats (coerced to a Python int)
    model.setParam("IterationLimit", np.int64(1000))
    assert model.getParam("IterationLimit") == 1000 and isinstance(model.getParam("IterationLimit"), int)
    model.setParam("IterationLimit", 2000.0)
    assert model.getParam("IterationLimit") == 2000

    # still-invalid inputs
    with pytest.raises(TypeError):
        model.setParam("IterationLimit", False)   # bool is not an int here
    with pytest.raises(TypeError):
        model.setParam("IterationLimit", 1.5)     # non-integer float
    with pytest.raises(TypeError):
        model.setParam("OutputFlag", "yes")       # string is not a bool
    with pytest.raises(ValueError):
        model.setParam("OutputFlag", 2)           # only 0/1 allowed as int
    with pytest.raises(ValueError):
        model.setParam("IterationLimit", -1)
    with pytest.raises(ValueError):
        model.setParam("TermCheckFreq", 0)
    with pytest.raises(ValueError):
        model.setParam("OptimalityTol", 0.0)
    with pytest.raises(ValueError):
        model.setParam("TimeLimit", -1.0)
    with pytest.raises(ValueError):
        model.setParam("OptimalityNorm", "l1")


def test_direct_core_param_value_validation(base_lp_data):
    c, A, l, u, lb, ub = base_lp_data
    with pytest.raises(ValueError):
        solve_once(A, c, None, lb, ub, l, u, params={"verbose": 1})
    with pytest.raises(ValueError):
        solve_once(A, c, None, lb, ub, l, u, params={"iteration_limit": False})
    with pytest.raises(ValueError):
        solve_once(A, c, None, lb, ub, l, u, params={"eps_optimal_relative": 0.0})
    with pytest.raises(ValueError):
        solve_once(A, c, None, lb, ub, l, u, params={"time_sec_limit": float("nan")})
    with pytest.raises(ValueError):
        solve_once(A, c, None, lb, ub, l, u, params={"optimality_norm": "l1"})


def test_direct_core_model_data_validation(base_lp_data):
    c, A, l, u, lb, ub = base_lp_data
    with pytest.raises(ValueError):
        solve_once(A, [np.nan, 1.0], None, lb, ub, l, u)
    with pytest.raises(ValueError):
        solve_once([[np.nan, 1.0], [0.0, 1.0], [3.0, 2.0]], c, None, lb, ub, l, u)
    with pytest.raises(ValueError):
        solve_once(A, c, np.nan, lb, ub, l, u)
    with pytest.raises(ValueError):
        solve_once(A, c, None, [0.0, np.nan], ub, l, u)
    with pytest.raises(ValueError):
        solve_once(A, c, None, [2.0, 0.0], [1.0, 1.0], l, u)
    with pytest.raises(ValueError):
        solve_once(A, c, None, lb, ub, l, u, primal_start=[np.nan, 0.0])


def test_set_params_value_validation_is_transactional(base_lp_data):
    model = _model(base_lp_data)
    old_time_limit = model.getParam("TimeLimit")
    with pytest.raises(TypeError):
        model.setParams(TimeLimit=123.0, OutputFlag="yes")  # OutputFlag invalid
    assert model.getParam("TimeLimit") == old_time_limit


def test_reset_params(base_lp_data):
    c, A, l, u, lb, ub = base_lp_data
    model = Model(c, A, l, u, lb, ub)
    default_time_limit = model.getParam("TimeLimit")
    default_output_flag = model.getParam("OutputFlag")

    model.setParams(TimeLimit=12.0, OutputFlag=not default_output_flag)
    model.resetParams()
    assert model.getParam("TimeLimit") == default_time_limit
    assert model.getParam("OutputFlag") is default_output_flag


def test_matrix_row_change_conflicts_with_bounds(base_lp_data):
    """Reassigning A with a different row count than existing bounds raises."""
    model = _model(base_lp_data)
    # existing constraint bounds have 3 rows; a 2-row matrix must be rejected
    # (constraint_lower_bound check)
    two_row = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        model.A = two_row
    # now clear the lower bound so the upper-bound check is the one that fires
    model.constr_lb = None
    with pytest.raises(ValueError):
        model.A = two_row


def test_params_view(base_lp_data):
    """_ParamsView get/set via attribute and mapping-style access."""
    model = _model(base_lp_data)
    model.Params.TimeLimit = 123.0
    assert model.Params.TimeLimit == 123.0
    model.Params["OptimalityTol"] = 1e-5
    assert model.Params["OptimalityTol"] == 1e-5
    assert model.getParam("TimeLimit") == 123.0
    assert "TimeLimit" in model.Params
    assert "time_sec_limit" in model.Params
    assert "DefinitelyNotAParam" not in model.Params
    assert len(list(model.Params.keys())) > 0
    assert len(list(model.Params.values())) == len(list(model.Params.keys()))
    assert dict(model.Params.items())["time_sec_limit"] == 123.0
    with pytest.raises(AttributeError):
        _ = model.Params.NoSuchParameter


def test_param_aliases_resolve_consistently(base_lp_data):
    """Every public parameter alias resolves to the same backend parameter."""
    model = _model(base_lp_data)
    for alias, key in PDLP._PARAM_ALIAS.items():
        assert alias in model.Params
        assert key in model.Params
        value = model.getParam(key)
        assert model.getParam(alias) == value
        model.setParam(alias, value)
        assert getattr(model.Params, alias) == value
        assert model.Params[alias] == value


def test_param_name_validation(base_lp_data):
    """Unknown parameter names are rejected in both set and get."""
    model = _model(base_lp_data)
    with pytest.raises(KeyError):
        model.setParam("DefinitelyNotAParam", 1)
    with pytest.raises(KeyError):
        model.getParam("DefinitelyNotAParam")


def test_non_contiguous_and_typed_inputs(base_lp_data):
    """Helper conversions handle non-contiguous dense and non-float64 sparse."""
    _, _, l, u, _, _ = base_lp_data
    # non-C-contiguous objective vector (a strided column view)
    c_nc = np.ones((2, 4))[:, 1]
    assert not c_nc.flags["C_CONTIGUOUS"]
    # integer-typed sparse matrix (exercises the float64 upcast path)
    A = sp.csr_matrix(np.array([[1, 2], [0, 1], [3, 2]], dtype=np.int64))
    model = Model(c_nc, A, l, u)
    assert model.c.dtype == np.float64
    assert model.A.dtype == np.float64
    assert model.A.indices.dtype == np.int32


def test_int64_index_downcast(base_lp_data):
    """A float64 CSR with int64 indices is downcast to int32 without mutating it."""
    _, _, l, u, _, _ = base_lp_data
    c = np.array([1.0, 1.0])
    A = sp.csr_matrix(np.array([[1.0, 2.0], [0.0, 1.0], [3.0, 2.0]]))
    A.indices = A.indices.astype(np.int64)
    A.indptr = A.indptr.astype(np.int64)
    model = Model(c, A, l, u)
    assert model.A.indices.dtype == np.int32
    assert model.A.indptr.dtype == np.int32
    # caller's matrix must be untouched
    assert A.indices.dtype == np.int64
    assert A.indptr.dtype == np.int64


def test_init_requires_2d_matrix():
    with pytest.raises(ValueError):
        Model(np.ones(2), np.ones(3), None, None)  # 1D "matrix"


def test_solution_attributes_accessible(base_lp_data):
    """Every result attribute is readable after optimize()."""
    model = _model(base_lp_data)
    model.optimize()
    # touching each property exercises its getter
    attrs = [
        model.X, model.Pi, model.RC, model.ObjVal, model.DualObj,
        model.Gap, model.RelGap, model.Status, model.StatusName,
        model.IterCount, model.Runtime, model.RescalingTime,
        model.RelPrimalResidual, model.RelDualResidual,
        model.MaxPrimalRayInfeas, model.MaxDualRayInfeas,
        model.PrimalRayLinObj, model.DualRayObj,
        model.PrimalInfeas, model.DualInfeas,
    ]
    assert model.Status == PDLP.OPTIMAL
    assert model.StatusName == "OPTIMAL"
    assert len(attrs) == 20


def test_optimize_rejects_bad_sense(base_lp_data):
    """The defensive sense check inside optimize() rejects a corrupted sense."""
    model = _model(base_lp_data)
    model._model_sense = 999  # bypass the property to hit optimize()'s guard
    with pytest.raises(ValueError):
        model.optimize()


def test_read_mps_roundtrip():
    """read() loads an MPS file into a usable Model."""
    path = os.path.join(os.path.dirname(__file__), "cplex2.mps")
    model = read(path)
    assert model.num_vars > 0
    assert model.num_constrs > 0
    assert model.A.shape == (model.num_constrs, model.num_vars)
    assert model.ModelSense in (PDLP.MINIMIZE, PDLP.MAXIMIZE)


def test_read_mps_missing_file():
    with pytest.raises(FileNotFoundError):
        read("this_file_does_not_exist.mps")
