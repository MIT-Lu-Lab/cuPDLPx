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

"""Model interface for the cuPDLPx LP solver."""

from __future__ import annotations
import os
from typing import Any, Optional, Union

import numpy as np
import scipy.sparse as sp

from ._core import solve_once, get_default_params, read_mps
from . import PDLP

# array-like type
ArrayLike = Union[np.ndarray, list, tuple]

# sentinel for "argument not provided" (distinct from None, which means "clear")
_UNSET = object()

_BOOL_PARAMS = frozenset(
    {
        "verbose",
        "has_pock_chambolle_alpha",
        "bound_objective_rescaling",
        "feasibility_polishing",
        "presolve",
    }
)
_INT_PARAMS = frozenset(
    {
        "termination_evaluation_frequency",
        "iteration_limit",
        "l_inf_ruiz_iterations",
        "sv_max_iter",
    }
)
_POSITIVE_INT_PARAMS = frozenset({"termination_evaluation_frequency", "sv_max_iter"})
_FLOAT_PARAMS = frozenset(
    {
        "eps_optimal_relative",
        "eps_feasible_relative",
        "time_sec_limit",
        "pock_chambolle_alpha",
        "artificial_restart_threshold",
        "sufficient_reduction_for_restart",
        "necessary_reduction_for_restart",
        "k_p",
        "reflection_coefficient",
        "eps_feas_polish_relative",
        "sv_tol",
        "matrix_zero_tol",
    }
)
_POSITIVE_FLOAT_PARAMS = frozenset(
    {
        "eps_optimal_relative",
        "eps_feasible_relative",
        "eps_feas_polish_relative",
        "sv_tol",
    }
)
_NONNEGATIVE_FLOAT_PARAMS = frozenset({"time_sec_limit", "matrix_zero_tol"})
_STRING_PARAMS = frozenset({"optimality_norm"})

# int params are stored as C int32 on the backend
_INT32_MAX = np.iinfo(np.int32).max

# every backend param must fall in one typed set, else it goes unvalidated
_CLASSIFIED_PARAMS = _BOOL_PARAMS | _INT_PARAMS | _FLOAT_PARAMS | _STRING_PARAMS

# refinement sets must be subsets of their base type sets
assert _POSITIVE_INT_PARAMS <= _INT_PARAMS
assert _POSITIVE_FLOAT_PARAMS <= _FLOAT_PARAMS
assert _NONNEGATIVE_FLOAT_PARAMS <= _FLOAT_PARAMS

def _as_dense_f64_c(a: ArrayLike) -> np.ndarray:
    """
    Convert input to an owned C-contiguous numpy array of float64.
    """
    return np.array(a, dtype=np.float64, order="C", copy=True)

def _readonly_array(arr: np.ndarray) -> np.ndarray:
    arr.setflags(write=False)
    return arr

def _readonly_matrix(A):
    if sp.issparse(A):
        A.data.setflags(write=False)
        A.indices.setflags(write=False)
        A.indptr.setflags(write=False)
    else:
        A.setflags(write=False)
    return A

def _require_finite(name: str, arr: np.ndarray) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")

def _require_no_nan(name: str, arr: np.ndarray) -> None:
    if np.any(np.isnan(arr)):
        raise ValueError(f"{name} must not contain NaN")

def _check_bounds(lower: Optional[np.ndarray], upper: Optional[np.ndarray], name: str) -> None:
    if lower is not None and upper is not None and np.any(lower > upper):
        raise ValueError(f"{name}: lower bounds must be <= upper bounds")

def _as_csr_f64_i32(A) -> sp.csr_matrix:
    """
    Convert input sparse matrix/array to CSR format with float64 values and
    int32 indices. Never mutates the caller's matrix.
    """
    csr = A.tocsr()
    if csr is A:
        csr = csr.copy()
    if csr.dtype != np.float64:
        csr = csr.astype(np.float64)
    _require_finite("constraint_matrix", csr.data)
    # merge duplicate (row, col) entries and sort indices within each row
    csr.sum_duplicates()
    csr.sort_indices()
    # force int32 indices (common C/CUDA req); int32 arrays can't overflow
    if csr.indptr.dtype != np.int32:
        # indptr is non-decreasing, so its last entry (== nnz) is the maximum
        if csr.indptr[-1] > _INT32_MAX:
            raise OverflowError("constraint_matrix CSR indptr exceeds int32 range")
        csr.indptr = csr.indptr.astype(np.int32, copy=False)
    if csr.indices.dtype != np.int32:
        if csr.indices.size and csr.indices.max() > _INT32_MAX:
            raise OverflowError("constraint_matrix CSR indices exceed int32 range")
        csr.indices = csr.indices.astype(np.int32, copy=False)
    return _readonly_matrix(csr)


def read(filename: Union[str, os.PathLike]) -> "Model":
    """
    Read a linear program from an MPS file (plain or gzip-compressed) and
    return a Model.

    Parameters:
    - filename: Path to a .mps or .mps.gz file.
    """
    # normalize path and check existence
    filename = os.fspath(filename)
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"No such MPS file: {filename}")
    # parse the MPS file into raw problem data
    data = read_mps(str(filename))
    # rebuild the constraint matrix in CSR form
    m = int(data["num_constraints"])
    n = int(data["num_variables"])
    A = sp.csr_matrix(
        (data["values"], data["col_ind"], data["row_ptr"]), shape=(m, n)
    )
    # assemble the model
    model = Model(
        objective_vector=data["objective_vector"],
        constraint_matrix=A,
        constraint_lower_bound=data["constraint_lower_bound"],
        constraint_upper_bound=data["constraint_upper_bound"],
        variable_lower_bound=data["variable_lower_bound"],
        variable_upper_bound=data["variable_upper_bound"],
        objective_constant=data["objective_constant"],
    )
    # set objective sense
    model.ModelSense = PDLP.MAXIMIZE if data["maximize"] else PDLP.MINIMIZE
    return model


class _ParamsView:
    """
    A view of the model parameters that allows getting/setting via attributes or keys.
    """
    def __init__(self, model: "Model"):
        object.__setattr__(self, "_m", model)

    def __getattr__(self, name: str):
        key = PDLP._PARAM_ALIAS.get(name, name)
        if key in self._m._params:
            return self._m._params[key]
        raise AttributeError(f"Unknown parameter '{name}'")

    def __setattr__(self, name: str, value):
        self._m.setParam(name, value)

    def __getitem__(self, name: str):
        return getattr(self, name)

    def __setitem__(self, name: str, value):
        self._m.setParam(name, value)

    def keys(self):
        return self._m._params.keys()

    def values(self):
        return self._m._params.values()

    def items(self):
        return self._m._params.items()

    def __contains__(self, name: str):
        key = PDLP._PARAM_ALIAS.get(name, name)
        return key in self._m._params

    def __iter__(self):
        return iter(self._m._params)

    def __len__(self):
        return len(self._m._params)

    def __repr__(self):
        return f"ParamsView({dict(self._m._params)!r})"


class Model:
    """
    A class representing a linear programming model.
    """
    def __init__(
        self,
        objective_vector: ArrayLike,
        constraint_matrix: Union[np.ndarray, sp.spmatrix],
        constraint_lower_bound: Optional[ArrayLike],
        constraint_upper_bound: Optional[ArrayLike],
        variable_lower_bound: Optional[ArrayLike] = None,
        variable_upper_bound: Optional[ArrayLike] = None,
        objective_constant: float = 0.0,
    ):
        """
        Initialize the Model with the given parameters.

        Parameters:
        - objective_vector: Coefficients of the objective function.
        - constraint_matrix: Constraint coefficient matrix (2D dense or scipy.sparse).
        - constraint_lower_bound: Lower bounds for the constraints.
        - constraint_upper_bound: Upper bounds for the constraints.
        - variable_lower_bound: Lower bounds for the decision variables (default -inf).
        - variable_upper_bound: Upper bounds for the decision variables (default +inf).
        - objective_constant: Constant term in the objective function.

        The objective sense defaults to PDLP.MINIMIZE; set model.ModelSense to
        PDLP.MAXIMIZE to maximize. Constraint bounds may be None, meaning -inf
        (lower) or +inf (upper).
        """
        # problem dimensions
        if not hasattr(constraint_matrix, "shape") or len(constraint_matrix.shape) != 2:
            raise ValueError("constraint_matrix must be a 2D numpy.ndarray or scipy.sparse matrix.")
        m, n = constraint_matrix.shape
        self.num_vars = int(n)
        self.num_constrs = int(m)
        # model data storage (populated by the setters below; exposed via properties)
        self._A = None
        self._c: Optional[np.ndarray] = None
        self._c0: float = 0.0
        self._lb: Optional[np.ndarray] = None
        self._ub: Optional[np.ndarray] = None
        self._constr_lb: Optional[np.ndarray] = None
        self._constr_ub: Optional[np.ndarray] = None
        # objective sense (default minimize)
        self._model_sense = PDLP.MINIMIZE
        # always start from backend defaults PDLP params
        self._default_params: dict[str, Any] = dict(get_default_params())
        self._params: dict[str, Any] = dict(self._default_params)
        # canonical set of backend parameter keys, used to reject typos in setParam
        self._valid_param_keys = frozenset(self._params)
        # fail loudly if the backend exposes a param the typed sets don't cover
        unclassified = self._valid_param_keys - _CLASSIFIED_PARAMS
        if unclassified:
            raise RuntimeError(f"Unclassified solver parameters (update model.py): {sorted(unclassified)}")
        self.Params = _ParamsView(self)
        # initialize warm start values
        self._primal_start: Optional[np.ndarray] = None # warm start primal solution
        self._dual_start: Optional[np.ndarray] = None # warm start dual solution
        # initialize solution attributes before the setters below
        self._x: Optional[np.ndarray] = None # primal solution
        self._y: Optional[np.ndarray] = None # dual solution
        self._rc: Optional[np.ndarray] = None # reduced costs
        self._objval: Optional[float] = None # objective value
        self._dualobj: Optional[float] = None # dual objective value
        self._gap: Optional[float] = None # primal-dual gap
        self._rel_gap: Optional[float] = None # relative gap
        self._status_name: Optional[str] = None # solution status name (str)
        self._status_code: Optional[int] = None # solution status code (int)
        self._iter: Optional[int] = None # number of iterations
        self._runtime: Optional[float] = None # runtime
        self._rescale_time: Optional[float] = None # rescale time
        self._rel_p_res: Optional[float] = None # relative primal residual
        self._rel_d_res: Optional[float] = None # relative dual residual
        self._max_p_ray: Optional[float] = None # maximum primal ray
        self._max_d_ray: Optional[float] = None # maximum dual ray
        self._p_ray_lin_obj: Optional[float] = None # primal ray linear objective
        self._d_ray_obj: Optional[float] = None # dual ray objective
        # set coefficients and bounds
        self.setObjectiveVector(objective_vector)
        self.setObjectiveConstant(objective_constant)
        self.setConstraintMatrix(constraint_matrix)
        self.setConstraintLowerBound(constraint_lower_bound)
        self.setConstraintUpperBound(constraint_upper_bound)
        self.setVariableLowerBound(variable_lower_bound)
        self.setVariableUpperBound(variable_upper_bound)
        self._validate_bounds()

    def setObjectiveVector(self, c: ArrayLike) -> None:
        """
        Overwrite objective vector c.
        """
        c_arr = _as_dense_f64_c(c)
        if c_arr.ndim != 1:
            raise ValueError(f"setObjectiveVector: c must be 1D, got shape {c_arr.shape}")
        if c_arr.size != self.num_vars:
            raise ValueError(f"setObjectiveVector: length {c_arr.size} != self.num_vars ({self.num_vars})")
        _require_finite("objective_vector", c_arr)
        self._c = _readonly_array(c_arr)
        # clear cached solution
        self._clear_solution_cache()

    def setObjectiveConstant(self, c0: float) -> None:
        """
        Overwrite objective constant term.
        Minimal check: convert to float.
        """
        c0 = float(c0)
        if not np.isfinite(c0):
            raise ValueError("objective_constant must be finite")
        self._c0 = c0
        # clear cached solution
        self._clear_solution_cache()

    def setConstraintMatrix(self, A_like: Union[np.ndarray, sp.spmatrix]) -> None:
        """
        Overwrite constraint matrix A.
        """
        if not (sp.issparse(A_like) or isinstance(A_like, np.ndarray)):
            raise TypeError("setConstraintMatrix: A must be a numpy.ndarray or scipy.sparse matrix/array")
        if len(A_like.shape) != 2:
            raise ValueError(f"setConstraintMatrix: A must be 2D, got shape {A_like.shape}")
        if A_like.shape[1] != self.num_vars:
            raise ValueError(f"setConstraintMatrix: A shape {A_like.shape} does not match number of variables ({self.num_vars})")
        # convert to backend layout (do not mutate self until all checks pass)
        if sp.issparse(A_like):
            A = _as_csr_f64_i32(A_like)
        else:
            A = _as_dense_f64_c(A_like)
            _require_finite("constraint_matrix", A)
        m = int(A.shape[0])
        # validate existing constraint bounds against the new row count before committing
        l = self._constr_lb
        if l is not None:
            n_l = np.asarray(l).ravel().size
            if n_l != m:
                raise ValueError(
                    f"setConstraintMatrix: constraint_lower_bound length {n_l} != rows {m}. "
                    f"Call setConstraintLowerBound(...) to update it."
                )
        u = self._constr_ub
        if u is not None:
            n_u = np.asarray(u).ravel().size
            if n_u != m:
                raise ValueError(
                    f"setConstraintMatrix: constraint_upper_bound length {n_u} != rows {m}. "
                    f"Call setConstraintUpperBound(...) to update it."
                )
        # commit
        self._A = _readonly_matrix(A)
        self.num_constrs = m
        # drop a dual warm start that no longer matches the row count
        if self._dual_start is not None and self._dual_start.size != m:
            self._dual_start = None
        # clear cached solution
        self._clear_solution_cache()

    def setConstraintLowerBound(self, constr_lb: Optional[ArrayLike]) -> None:
        """
        Overwrite constraint lower bounds.
        """
        # check if the input is None
        if constr_lb is None:
            self._constr_lb = None
            # clear cached solution
            self._clear_solution_cache()
            return
        constr_lb_arr = _as_dense_f64_c(constr_lb).ravel()
        if constr_lb_arr.size != self.num_constrs:
            raise ValueError(
                f"setConstraintLowerBound: length {constr_lb_arr.size} != self.num_constrs ({self.num_constrs})"
            )
        _require_no_nan("constraint_lower_bound", constr_lb_arr)
        self._constr_lb = _readonly_array(constr_lb_arr)
        # clear cached solution
        self._clear_solution_cache()

    def setConstraintUpperBound(self, constr_ub: Optional[ArrayLike]) -> None:
        """
        Overwrite constraint upper bounds.
        """
        # check if the input is None
        if constr_ub is None:
            self._constr_ub = None
            # clear cached solution
            self._clear_solution_cache()
            return
        constr_ub_arr = _as_dense_f64_c(constr_ub).ravel()
        if constr_ub_arr.size != self.num_constrs:
            raise ValueError(
                f"setConstraintUpperBound: length {constr_ub_arr.size} != self.num_constrs ({self.num_constrs})"
            )
        _require_no_nan("constraint_upper_bound", constr_ub_arr)
        self._constr_ub = _readonly_array(constr_ub_arr)
        # clear cached solution
        self._clear_solution_cache()

    def setVariableLowerBound(self, lb: Optional[ArrayLike]) -> None:
        """
        Overwrite variable lower bounds.
        """
        # check if the input is None
        if lb is None:
            self._lb = None
            # clear cached solution
            self._clear_solution_cache()
            return
        lb_arr = _as_dense_f64_c(lb).ravel()
        if lb_arr.size != self.num_vars:
            raise ValueError(
                f"setVariableLowerBound: length {lb_arr.size} != self.num_vars ({self.num_vars})"
            )
        _require_no_nan("variable_lower_bound", lb_arr)
        self._lb = _readonly_array(lb_arr)
        # clear cached solution
        self._clear_solution_cache()

    def setVariableUpperBound(self, ub: Optional[ArrayLike]) -> None:
        """
        Overwrite variable upper bounds.
        """
        # check if the input is None
        if ub is None:
            self._ub = None
            # clear cached solution
            self._clear_solution_cache()
            return
        ub_arr = _as_dense_f64_c(ub).ravel()
        if ub_arr.size != self.num_vars:
            raise ValueError(
                f"setVariableUpperBound: length {ub_arr.size} != self.num_vars ({self.num_vars})"
            )
        _require_no_nan("variable_upper_bound", ub_arr)
        self._ub = _readonly_array(ub_arr)
        # clear cached solution
        self._clear_solution_cache()

    def setWarmStart(self, primal: Optional[ArrayLike] = _UNSET, dual: Optional[ArrayLike] = _UNSET) -> None:
        """
        Set warm start values for the primal and/or dual solutions.

        For each of primal/dual: pass an array to set it, None to clear it, or
        omit the argument to leave the current value unchanged. Raises
        ValueError on a size mismatch.
        """
        next_primal = self._primal_start
        next_dual = self._dual_start
        # primal warm start
        if primal is not _UNSET:
            if primal is None:
                next_primal = None
            else:
                primal_arr = _as_dense_f64_c(primal).ravel()
                if primal_arr.size != self.num_vars:
                    raise ValueError(
                        f"setWarmStart: primal size mismatch (expected {self.num_vars}, got {primal_arr.size})."
                    )
                _require_finite("primal warm start", primal_arr)
                next_primal = _readonly_array(primal_arr)
        # dual warm start
        if dual is not _UNSET:
            if dual is None:
                next_dual = None
            else:
                dual_arr = _as_dense_f64_c(dual).ravel()
                if dual_arr.size != self.num_constrs:
                    raise ValueError(
                        f"setWarmStart: dual size mismatch (expected {self.num_constrs}, got {dual_arr.size})."
                    )
                _require_finite("dual warm start", dual_arr)
                next_dual = _readonly_array(dual_arr)
        self._primal_start = next_primal
        self._dual_start = next_dual

    def clearWarmStart(self) -> None:
        """
        Clear any existing warm start values.
        """
        self.setWarmStart(primal=None, dual=None)

    def _resolve_param_key(self, name: str) -> str:
        """
        Map a user-facing parameter name (alias or backend key) to its backend
        key, raising KeyError for unknown names instead of silently accepting them.
        """
        # map alias to backend key
        key = PDLP._PARAM_ALIAS.get(name, name)
        # reject unknown names
        if key not in self._valid_param_keys:
            valid = sorted(PDLP._PARAM_ALIAS.keys()) + sorted(self._valid_param_keys)
            raise KeyError(f"Unknown parameter '{name}'. Valid names: {valid}")
        return key

    def _validate_param_value(self, key: str, value: Any) -> Any:
        if key in _BOOL_PARAMS:
            # accept a real bool, numpy bool, or an integer 0/1; store a Python bool
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            if isinstance(value, (int, np.integer)):
                if int(value) not in (0, 1):
                    raise ValueError(f"Parameter '{key}' must be 0 or 1 when given as an int.")
                return bool(value)
            raise TypeError(f"Parameter '{key}' must be a bool (or 0/1).")

        if key in _INT_PARAMS:
            # accept any Python/numpy integer or an integer-valued float; store a Python int
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(f"Parameter '{key}' must be an int.")
            if isinstance(value, (int, np.integer)):
                value = int(value)
            elif isinstance(value, (float, np.floating)) and float(value).is_integer():
                value = int(value)
            else:
                raise TypeError(f"Parameter '{key}' must be an int.")
            if key in _POSITIVE_INT_PARAMS and value <= 0:
                raise ValueError(f"Parameter '{key}' must be positive.")
            if key not in _POSITIVE_INT_PARAMS and value < 0:
                raise ValueError(f"Parameter '{key}' must be nonnegative.")
            if value > _INT32_MAX:
                raise ValueError(f"Parameter '{key}' must not exceed {_INT32_MAX} (int32 range).")
            return value

        if key in _FLOAT_PARAMS:
            # accept any real number (Python/numpy int or float); store a Python float
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(f"Parameter '{key}' must be a number.")
            if not isinstance(value, (int, float, np.integer, np.floating)):
                raise TypeError(f"Parameter '{key}' must be a number.")
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"Parameter '{key}' must be finite.")
            if key in _POSITIVE_FLOAT_PARAMS and value <= 0.0:
                raise ValueError(f"Parameter '{key}' must be positive.")
            if key in _NONNEGATIVE_FLOAT_PARAMS and value < 0.0:
                raise ValueError(f"Parameter '{key}' must be nonnegative.")
            return value

        if key in _STRING_PARAMS:
            if not isinstance(value, str):
                raise TypeError(f"Parameter '{key}' must be str.")
            value = value.lower()
            if value not in ("l2", "linf"):
                raise ValueError("Parameter 'optimality_norm' must be 'l2' or 'linf'.")
            return value

        return value

    def setParam(self, name: str, value: Any) -> None:
        """
        Set the value of a solver parameter by name.
        """
        # resolve name and store
        key = self._resolve_param_key(name)
        self._params[key] = self._validate_param_value(key, value)

    def getParam(self, name: str) -> Any:
        """
        Get the value of a solver parameter by name.
        """
        # resolve name and return
        key = self._resolve_param_key(name)
        return self._params[key]

    def setParams(self, /, **kwargs) -> None:
        """
        Set multiple solver parameters by name. 
        """
        updates = {}
        for k, v in kwargs.items():
            key = self._resolve_param_key(k)
            updates[key] = self._validate_param_value(key, v)
        self._params.update(updates)

    def resetParams(self) -> None:
        """
        Reset all solver parameters to their backend default values.
        """
        self._params = dict(self._default_params)

    def optimize(self):
        """
        Solve the linear programming problem using the cuPDLPx solver.
        """
        # clear cached solution
        self._clear_solution_cache()
        # check model sense
        if self.ModelSense not in (PDLP.MINIMIZE, PDLP.MAXIMIZE):
            raise ValueError("model_sense must be PDLP.MINIMIZE or PDLP.MAXIMIZE")
        self._validate_bounds()
        minimize = self.ModelSense == PDLP.MINIMIZE
        # call the core solver
        info = solve_once(
            self.A,
            self.c,
            self.c0,
            self.lb,
            self.ub,
            self.constr_lb,
            self.constr_ub,
            params=self._params,
            primal_start=self._primal_start,
            dual_start=self._dual_start,
            minimize=minimize,
        )
        # solutions
        x = info.get("X")
        y = info.get("Pi")
        rc = info.get("RC")
        self._x = _readonly_array(np.asarray(x)) if x is not None else None
        self._y = _readonly_array(np.asarray(y)) if y is not None else None
        self._rc = _readonly_array(np.asarray(rc)) if rc is not None else None
        # objectives & gaps
        self._objval = info.get("PrimalObj")
        self._dualobj = info.get("DualObj")
        self._gap = info.get("ObjectiveGap")
        self._rel_gap = info.get("RelativeObjectiveGap")
        # status & counters
        status = info.get("Status")
        status_code = info.get("StatusCode")
        iters = info.get("Iterations")
        self._status_name = str(status) if status is not None else None
        self._status_code = int(status_code) if status_code is not None else None
        self._iter = int(iters) if iters is not None else None
        self._runtime = info.get("RuntimeSec")
        self._rescale_time = info.get("RescalingTimeSec")
        # residuals
        self._rel_p_res = info.get("RelativePrimalResidual")
        self._rel_d_res = info.get("RelativeDualResidual")
        # rays
        self._max_p_ray = info.get("MaxPrimalRayInfeas")
        self._max_d_ray = info.get("MaxDualRayInfeas")
        self._p_ray_lin_obj = info.get("PrimalRayLinObj")
        self._d_ray_obj = info.get("DualRayObj")
        return self

    def _clear_solution_cache(self) -> None:
        """
        Clear cached solution attributes.
        """
        self._x = self._y = self._rc = None
        self._objval = self._dualobj = None
        self._gap = self._rel_gap = None
        self._status_name = None
        self._status_code = None
        self._iter = None
        self._runtime = self._rescale_time = None
        self._rel_p_res = None
        self._rel_d_res = None
        self._max_p_ray = self._max_d_ray = None
        self._p_ray_lin_obj = self._d_ray_obj = None

    def _validate_bounds(self) -> None:
        _check_bounds(self._lb, self._ub, "variable bounds")
        _check_bounds(self._constr_lb, self._constr_ub, "constraint bounds")

    # model data (read/write; assignment reroutes through the validating setters)
    @property
    def c(self) -> Optional[np.ndarray]:
        """Objective coefficient vector."""
        return self._c

    @c.setter
    def c(self, value: ArrayLike) -> None:
        self.setObjectiveVector(value)

    @property
    def c0(self) -> float:
        """Objective constant term."""
        return self._c0

    @c0.setter
    def c0(self, value: float) -> None:
        self.setObjectiveConstant(value)

    @property
    def A(self):
        """Constraint matrix (CSR for sparse input, dense ndarray otherwise)."""
        return self._A

    @A.setter
    def A(self, value) -> None:
        self.setConstraintMatrix(value)

    @property
    def lb(self) -> Optional[np.ndarray]:
        """Variable lower bounds (None means -inf)."""
        return self._lb

    @lb.setter
    def lb(self, value: Optional[ArrayLike]) -> None:
        self.setVariableLowerBound(value)

    @property
    def ub(self) -> Optional[np.ndarray]:
        """Variable upper bounds (None means +inf)."""
        return self._ub

    @ub.setter
    def ub(self, value: Optional[ArrayLike]) -> None:
        self.setVariableUpperBound(value)

    @property
    def constr_lb(self) -> Optional[np.ndarray]:
        """Constraint lower bounds (None means -inf)."""
        return self._constr_lb

    @constr_lb.setter
    def constr_lb(self, value: Optional[ArrayLike]) -> None:
        self.setConstraintLowerBound(value)

    @property
    def constr_ub(self) -> Optional[np.ndarray]:
        """Constraint upper bounds (None means +inf)."""
        return self._constr_ub

    @constr_ub.setter
    def constr_ub(self, value: Optional[ArrayLike]) -> None:
        self.setConstraintUpperBound(value)

    @property
    def ModelSense(self) -> int:
        """Objective sense: PDLP.MINIMIZE or PDLP.MAXIMIZE."""
        return self._model_sense

    @ModelSense.setter
    def ModelSense(self, value: int) -> None:
        # validate sense
        if value not in (PDLP.MINIMIZE, PDLP.MAXIMIZE):
            raise ValueError("ModelSense must be PDLP.MINIMIZE or PDLP.MAXIMIZE")
        self._model_sense = value
        # clear cached solution
        self._clear_solution_cache()

    @property
    def X(self) -> Optional[np.ndarray]:
        return self._x

    @property
    def Pi(self) -> Optional[np.ndarray]:
        return self._y
    
    @property
    def RC(self) -> Optional[np.ndarray]:
        return self._rc

    @property
    def ObjVal(self) -> Optional[float]:
        return self._objval

    @property
    def DualObj(self) -> Optional[float]:
        return self._dualobj

    @property
    def Gap(self) -> Optional[float]:
        return self._gap

    @property
    def RelGap(self) -> Optional[float]:
        return self._rel_gap
    
    @property
    def Status(self) -> Optional[int]:
        """
        Integer termination status code. Compare against the constants in
        cupdlpx.PDLP, e.g. ``model.Status == PDLP.OPTIMAL``.
        """
        return self._status_code

    @property
    def StatusName(self) -> Optional[str]:
        """Human-readable termination status name, e.g. ``'OPTIMAL'``."""
        return self._status_name

    @property
    def IterCount(self) -> Optional[int]:
        return self._iter

    @property
    def Runtime(self) -> Optional[float]:
        return self._runtime

    @property
    def RescalingTime(self) -> Optional[float]:
        return self._rescale_time

    @property
    def RelPrimalResidual(self) -> Optional[float]:
        return self._rel_p_res

    @property
    def RelDualResidual(self) -> Optional[float]:
        return self._rel_d_res

    @property
    def MaxPrimalRayInfeas(self) -> Optional[float]:
        return self._max_p_ray

    @property
    def MaxDualRayInfeas(self) -> Optional[float]:
        return self._max_d_ray

    @property
    def PrimalRayLinObj(self) -> Optional[float]:
        return self._p_ray_lin_obj

    @property
    def DualRayObj(self) -> Optional[float]:
        return self._d_ray_obj

    @property
    def PrimalInfeas(self) -> Optional[float]:
        """Alias of RelPrimalResidual (relative, not absolute infeasibility)."""
        return self._rel_p_res

    @property
    def DualInfeas(self) -> Optional[float]:
        """Alias of RelDualResidual (relative, not absolute infeasibility)."""
        return self._rel_d_res
