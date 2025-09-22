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

from __future__ import annotations
from typing import Any, Optional, Union

import numpy as np
import scipy.sparse as sp

from ._core import solve_once, get_default_params
from . import PDLP

# array-like type
ArrayLike = Union[np.ndarray, list, tuple]

def _as_dense_f64_c(a: ArrayLike) -> np.ndarray:
    """
    Convert input to a C-contiguous numpy array of float64.
    """
    arr = np.asarray(a, dtype=np.float64)
    # ensure C-contiguous
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr, dtype=np.float64)
    return arr

def _as_csr_f64_i32(A: sp.spmatrix) -> sp.csr_matrix:
    """
    Convert input sparse matrix to CSR format with float64 values and int32 indices.
    """
    csr = A.tocsr().astype(np.float64, copy=False)
    # force int32 indices (common C/CUDA req)
    if csr.indptr.dtype != np.int32:
        csr.indptr = csr.indptr.astype(np.int32, copy=True)
    if csr.indices.dtype != np.int32:
        csr.indices = csr.indices.astype(np.int32, copy=True)
    csr.sort_indices()
    return csr


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
        - constraint_matrix: Coefficients of the constraints.
        - lower_bounds: Lower bounds for the decision variables.
        - upper_bounds: Upper bounds for the decision variables.
        - constraint_lower_bounds: Lower bounds for the constraints.
        - constraint_upper_bounds: Upper bounds for the constraints.
        - objective_constant: Constant term in the objective function.
        - model_sense: PDLP.MINIMIZE or PDLP.MAXIMIZE.
        If variable bounds are not provided, they default to -inf and +inf respectively.    
        """
        # problem dimensions
        if not hasattr(constraint_matrix, "shape") or len(constraint_matrix.shape) != 2:
            raise ValueError("constraint_matrix must be a 2D numpy.ndarray or scipy.sparse matrix.")
        m, n = constraint_matrix.shape
        self.num_vars = int(n)
        self.num_constrs = int(m)
        # sense
        self.ModelSense = PDLP.MINIMIZE
        # always start from backend defaults PDLP params
        self._params: dict[str, Any] = dict(get_default_params())
        self.Params = _ParamsView(self)
        # set coefficients and bounds
        self.setObjectiveVector(objective_vector)
        self.setObjectiveConstant(objective_constant)
        self.setConstraintMatrix(constraint_matrix)
        self.setConstraintLowerBound(constraint_lower_bound)
        self.setConstraintUpperBound(constraint_upper_bound)
        self.setVariableLowerBound(variable_lower_bound)
        self.setVariableUpperBound(variable_upper_bound)
        # initialize solution attributes
        self._x: Optional[np.ndarray] = None # primal solution
        self._y: Optional[np.ndarray] = None # dual solution
        self._objval: Optional[float] = None # objective value
        self._dualobj: Optional[float] = None # dual objective value
        self._gap: Optional[float] = None # primal-dual gap
        self._rel_gap: Optional[float] = None # relative gap
        self._status: Optional[str] = None # solution status
        self._status_code: Optional[int] = None # solution status code
        self._iter: Optional[int] = None # number of iterations
        self._runtime: Optional[float] = None # runtime
        self._rescale_time: Optional[float] = None # rescale time
        self._abs_p_res: Optional[float] = None # absolute primal residual
        self._rel_p_res: Optional[float] = None # relative primal residual
        self._abs_d_res: Optional[float] = None # absolute dual residual
        self._rel_d_res: Optional[float] = None # relative dual residual
        self._max_p_ray: Optional[float] = None # maximum primal ray
        self._max_d_ray: Optional[float] = None # maximum dual ray
        self._p_ray_lin_obj: Optional[float] = None # primal ray linear objective
        self._d_ray_obj: Optional[float] = None # dual ray objective

    def setObjectiveVector(self, c: ArrayLike) -> None:
        """
        Overwrite objective vector c.
        """
        # store as float64
        self.c = _as_dense_f64_c(c)
        # check dimensions
        if self.c.ndim != 1:
            raise ValueError(f"setObjectiveVector: c must be 1D, got shape {self.c.shape}")
        if self.c.size != self.num_vars:
            raise ValueError(f"setObjectiveVector: length {self.c.size} != self.num_vars ({self.num_vars})")
        # clear cached solution
        self._clear_solution_cache()

    def setObjectiveConstant(self, c0: float) -> None:
        """
        Overwrite objective constant term.
        Minimal check: convert to float.
        """
        self.c0 = float(c0)
        # clear cached solution
        self._clear_solution_cache()

    def setConstraintMatrix(self, A_like: ArrayLike) -> None:
        """
        Overwrite constraint matrix A.
        """
        if not isinstance(A_like, (np.ndarray, sp.spmatrix)):
            raise TypeError("setConstraintMatrix: A must be a numpy.ndarray or scipy.sparse matrix")
        if len(A_like.shape) != 2:
            raise ValueError(f"setConstraintMatrix: A must be 2D, got shape {A_like.shape}")
        if A_like.shape[1] != self.num_vars:
            raise ValueError(f"setConstraintMatrix: A shape {A_like.shape} does not match number of variables ({self.num_vars})")
        # store as float64
        if sp.issparse(A_like):
            self.A = _as_csr_f64_i32(A_like)
        else:
            self.A = _as_dense_f64_c(A_like)
        # problem dimensions
        if not hasattr(self.A, "shape") or len(self.A.shape) != 2:
            raise ValueError("constraint_matrix must be a 2D numpy.ndarray or scipy.sparse matrix.")
        m, _ = self.A.shape
        self.num_constrs = int(m)
        # check constraint bounds
        l = getattr(self, "constr_lb", None)
        if l is not None:
            l = np.asarray(l, dtype=np.float64).ravel()
            if l.size != self.num_constrs:
                raise ValueError(
                    f"setConstraintMatrix: constraint_lower_bound length {l.size} != rows {self.num_constrs}. "
                    f"Call setConstraintLowerBound(...) to update it."
                )
        u = getattr(self, "constr_ub", None)
        if u is not None:
            u = np.asarray(u, dtype=np.float64).ravel()
            if u.size != self.num_constrs:
                raise ValueError(
                    f"setConstraintMatrix: constraint_upper_bound length {u.size} != rows {self.num_constrs}. "
                    f"Call setConstraintUpperBound(...) to update it."
               )
        # clear cached solution
        self._clear_solution_cache()

    def setConstraintLowerBound(self, constr_lb: Optional[ArrayLike]) -> None:
        """
        Overwrite constraint lower bounds.
        """
        # check if the input is None
        if constr_lb is None:
            self.constr_lb = None
            # clear cached solution
            self._clear_solution_cache()
            return
        # convert to numpy array
        constr_lb = _as_dense_f64_c(constr_lb).ravel()
        if constr_lb.size != self.num_constrs:
            raise ValueError(
                f"setConstraintLowerBound: length {constr_lb.size} != self.num_constrs ({self.num_constrs})"
            )
        self.constr_lb = constr_lb
        # clear cached solution
        self._clear_solution_cache()

    def setConstraintUpperBound(self, constr_ub: Optional[ArrayLike]) -> None:
        """
        Overwrite constraint upper bounds.
        """
        # check if the input is None
        if constr_ub is None:
            self.constr_ub = None
            # clear cached solution
            self._clear_solution_cache()
            return
        # convert to numpy array
        constr_ub = _as_dense_f64_c(constr_ub).ravel()
        if constr_ub.size != self.num_constrs:
            raise ValueError(
                f"setConstraintUpperBound: length {constr_ub.size} != self.num_constrs ({self.num_constrs})"
            )
        self.constr_ub = constr_ub
        # clear cached solution
        self._clear_solution_cache()

    def setVariableLowerBound(self, lb: Optional[ArrayLike]) -> None:
        """
        Overwrite variable lower bounds.
        """
        # check if the input is None
        if lb is None:
            self.lb = None
            # clear cached solution
            self._clear_solution_cache()
            return
        # convert to numpy array
        lb = _as_dense_f64_c(lb).ravel()
        if lb.size != self.num_vars:
            raise ValueError(
                f"setVariableLowerBound: length {lb.size} != self.num_vars ({self.num_vars})"
            )
        self.lb = lb
        # clear cached solution
        self._clear_solution_cache()

    def setVariableUpperBound(self, ub: Optional[ArrayLike]) -> None:
        """
        Overwrite variable upper bounds.
        """
        # check if the input is None
        if ub is None:
            self.ub = None
            # clear cached solution
            self._clear_solution_cache()
            return
        # convert to numpy array
        ub = _as_dense_f64_c(ub).ravel()
        if ub.size != self.num_vars:
            raise ValueError(
                f"setVariableUpperBound: length {ub.size} != self.num_vars ({self.num_vars})"
            )
        self.ub = ub
        # clear cached solution
        self._clear_solution_cache()

    def setParam(self, name: str, value: Any) -> None:
        """
        Set the value of a solver parameter by name.
        """
        key = PDLP._PARAM_ALIAS.get(name, name)
        self._params[key] = value

    def getParam(self, name: str) -> Any:
        """
        Get the value of a solver parameter by name.
        """
        key = PDLP._PARAM_ALIAS.get(name, name)
        return self._params.get(key)

    def setParams(self, /, **kwargs) -> None:
        """
        Set multiple solver parameters by name. 
        """
        for k, v in kwargs.items():
            self.setParam(k, v)

    def optimize(self):
        """
        Solve the linear programming problem using the cuPDLPx solver.
        """
        # clear cached solution
        self._clear_solution_cache()
        # check model sense
        if self.ModelSense not in (PDLP.MINIMIZE, PDLP.MAXIMIZE):
            raise ValueError("model_sense must be PDLP.MINIMIZE or PDLP.MAXIMIZE")
        # determine sign
        sign = 1 if self.ModelSense == PDLP.MINIMIZE else -1
        # effective objective based on sense
        c_eff  = sign * self.c if self.c is not None else None
        c0_eff = sign * self.c0 if self.c0 is not None else None
        # call the core solver
        info = solve_once(
            self.A,
            c_eff,
            c0_eff,
            self.lb,
            self.ub,
            self.constr_lb,
            self.constr_ub,
            zero_tolerance=0.0,
            params=self._params
        )
        # solutions
        self._x = np.asarray(info.get("X")) if info.get("X") is not None else None
        self._y = np.asarray(info.get("Pi")) if info.get("Pi") is not None else None
        # objectives & gaps
        primal_obj_eff = info.get("PrimalObj")
        dual_obj_eff   = info.get("DualObj")
        self._objval = sign * primal_obj_eff if primal_obj_eff is not None else None
        self._dualobj = sign * dual_obj_eff if dual_obj_eff is not None else None
        self._gap = info.get("ObjectiveGap")
        self._rel_gap = info.get("RelativeObjectiveGap")
        # status & counters
        self._status = str(info.get("Status")) if info.get("Status") is not None else None
        self._status_code = int(info.get("StatusCode")) if info.get("StatusCode") is not None else None
        self._iter = int(info.get("Iterations")) if info.get("Iterations") is not None else None
        self._runtime = info.get("RuntimeSec")
        self._rescale_time = info.get("RescalingTimeSec")
        # residuals
        self._abs_p_res = info.get("AbsolutePrimalResidual")
        self._rel_p_res = info.get("RelativePrimalResidual")
        self._abs_d_res = info.get("AbsoluteDualResidual")
        self._rel_d_res = info.get("RelativeDualResidual")
        # rays
        self._max_p_ray = info.get("MaxPrimalRayInfeas")
        self._max_d_ray = info.get("MaxDualRayInfeas")
        p_ray_lin_eff  = info.get("PrimalRayLinObj")
        d_ray_obj_eff  = info.get("DualRayObj")
        self._p_ray_lin_obj = sign * p_ray_lin_eff if p_ray_lin_eff is not None else None
        self._d_ray_obj = sign * d_ray_obj_eff if d_ray_obj_eff is not None else None

    def _clear_solution_cache(self) -> None:
        """
        Clear cached solution attributes.
        """
        self._x = self._y = None
        self._objval = self._dualobj = None
        self._gap = self._rel_gap = None
        self._status = None
        self._status_code = None
        self._iter = None
        self._runtime = self._rescale_time = None
        self._abs_p_res = self._rel_p_res = None
        self._abs_d_res = self._rel_d_res = None
        self._max_p_ray = self._max_d_ray = None
        self._p_ray_lin_obj = self._d_ray_obj = None

    @property
    def X(self) -> Optional[np.ndarray]:
        return self._x

    @property
    def Pi(self) -> Optional[np.ndarray]:
        return self._y

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
    def Status(self) -> Optional[str]:
        return self._status

    @property
    def StatusCode(self) -> Optional[int]:
        return self._status_code

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
    def AbsPrimalResidual(self) -> Optional[float]:
        return self._abs_p_res

    @property
    def RelPrimalResidual(self) -> Optional[float]:
        return self._rel_p_res

    @property
    def AbsDualResidual(self) -> Optional[float]:
        return self._abs_d_res

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
        return self._rel_p_res

    @property
    def DualInfeas(self) -> Optional[float]:
        return self._rel_d_res