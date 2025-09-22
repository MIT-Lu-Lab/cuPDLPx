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

from ._core import solve_once

# array-like type
ArrayLike = Union[np.ndarray, list, tuple]

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
        objective_constant: float = 0.0
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
        If variable bounds are not provided, they default to -inf and +inf respectively.    
        """
        # problem dimensions
        if not hasattr(constraint_matrix, "shape") or len(constraint_matrix.shape) != 2:
            raise ValueError("constraint_matrix must be a 2D numpy.ndarray or scipy.sparse matrix.")
        m, n = constraint_matrix.shape
        self.num_vars = int(n)
        self.num_constrs = int(m)
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

    def setObjectiveVector(self, c: np.ndarray) -> None:
        """
        Overwrite objective vector c.
        """
        if not isinstance(c, np.ndarray):
            raise TypeError("setObjectiveVector: c must be a numpy.ndarray")
        if c.ndim != 1:
            raise ValueError(f"setObjectiveVector: c must be 1D, got shape {c.shape}")
        if c.size != self.num_vars:
            raise ValueError(f"setObjectiveVector: length {c.size} != self.num_vars ({self.num_vars})")
        # store as float64
        self.c = np.asarray(c, dtype=np.float64)

    def setObjectiveConstant(self, c0: float) -> None:
        """
        Overwrite objective constant term.
        Minimal check: convert to float.
        """
        self.c0 = float(c0)

    def setConstraintMatrix(self, A_like: ArrayLike) -> None:
        """
        Overwrite constraint matrix A.
        """
        if not isinstance(A_like, (np.ndarray, sp.spmatrix)):
            raise TypeError("setConstraintMatrix: A must be a numpy.ndarray or scipy.sparse matrix")
        if len(A_like.shape) != 2:
            raise ValueError(f"setConstraintMatrix: A must be 2D, got shape {A_like.shape}")
        if A_like.shape[1] != self.num_vars:
            raise ValueError(f"setConstraintMatrix: A shape {A_like.shape} does not match number of variables ( {self.num_vars})")
        # store as float64
        if sp.issparse(A_like):
            self.A = A_like.astype(np.float64)
        else:
            self.A = np.asarray(A_like, dtype=np.float64)
        # problem dimensions
        if not hasattr(self.A, "shape") or len(self.A.shape) != 2:
            raise ValueError("constraint_matrix must be a 2D numpy.ndarray or scipy.sparse matrix.")
        m, _ = self.A.shape
        self.num_constrs = int(m)
        # check constraint bounds
        if self.constr_lb is not None:
            l = np.asarray(self.constr_lb, dtype=np.float64).ravel()
            if l.size != self.num_constrs:
                raise ValueError(
                    f"setConstraintMatrix: constraint_lower_bound length {l.size} != rows {self.num_constrs}. "
                    f"Call setConstraintLowerBound(...) to update it."
                )
        if self.constr_ub is not None:
            u = np.asarray(self.constr_ub, dtype=np.float64).ravel()
            if u.size != self.num_constrs:
                raise ValueError(
                    f"setConstraintMatrix: constraint_upper_bound length {u.size} != rows {self.num_constrs}. "
                    f"Call setConstraintUpperBound(...) to update it."
               )

    def setConstraintLowerBound(self, constr_lb: Optional[ArrayLike]) -> None:
        """
        Overwrite constraint lower bounds.
        """
        # check if the input is None
        if constr_lb is None:
            self.constr_lb = None
            return
        # convert to numpy array
        constr_lb = np.asarray(constr_lb, dtype=np.float64).ravel()
        if constr_lb.size != self.num_constrs:
            raise ValueError(
                f"setConstraintLowerBound: length {constr_lb.size} != self.num_constrs ({self.num_constrs})"
            )
        self.constr_lb = constr_lb

    def setConstraintUpperBound(self, constr_ub: Optional[ArrayLike]) -> None:
        """
        Overwrite constraint upper bounds.
        """
        # check if the input is None
        if constr_ub is None:
            self.constr_ub = None
            return
        # convert to numpy array
        constr_ub = np.asarray(constr_ub, dtype=np.float64).ravel()
        if constr_ub.size != self.num_constrs:
            raise ValueError(
                f"setConstraintUpperBound: length {constr_ub.size} != self.num_constrs ({self.num_constrs})"
            )
        self.constr_ub = constr_ub

    def setVariableLowerBound(self, lb: Optional[ArrayLike]) -> None:
        """
        Overwrite variable lower bounds.
        """
        # check if the input is None
        if lb is None:
            self.lb = None
            return
        # convert to numpy array
        lb = np.asarray(lb, dtype=np.float64).ravel()
        if lb.size != self.num_vars:
            raise ValueError(
                f"setVariableLowerBound: length {lb.size} != self.num_vars ({self.num_vars})"
            )
        self.lb = lb

    def setVariableUpperBound(self, ub: Optional[ArrayLike]) -> None:
        """
        Overwrite variable upper bounds.
        """
        # check if the input is None
        if ub is None:
            self.ub = None
            return
        # convert to numpy array
        ub = np.asarray(ub, dtype=np.float64).ravel()
        if ub.size != self.num_vars:
            raise ValueError(
                f"setVariableUpperBound: length {ub.size} != self.num_vars ({self.num_vars})"
            )
        self.ub = ub

    def optimize(self):
        """
        Solve the linear programming problem using the cuPDLPx solver.
        """
        # call the core solver
        info = solve_once(
            self.A,
            self.c,
            self.c0,
            self.lb,
            self.ub,
            self.constr_lb,
            self.constr_ub,
            zero_tolerance=0.0,
        )
        # solutions
        self._x = np.asarray(info.get("X")) if info.get("X") is not None else None
        self._y = np.asarray(info.get("Pi")) if info.get("Pi") is not None else None
        # objectives & gaps
        self._objval = info.get("PrimalObj")
        self._dualobj = info.get("DualObj")
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
        self._p_ray_lin_obj = info.get("PrimalRayLinObj")
        self._d_ray_obj = info.get("DualRayObj")

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