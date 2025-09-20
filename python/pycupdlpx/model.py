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

from ._core import solve_once

class Model:
    """
    A class representing a linear programming model.
    """
    def __init__(self, objective_vector, constraint_matrix, 
                 constraint_lower_bound, constraint_upper_bound,
                 variable_lower_bound=None, variable_upper_bound=None,
                 objective_constant=0.0):
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
        self.num_vars = len(objective_vector)
        self.num_constrs = len(constraint_lower_bound)
        # set coefficients
        self.c = objective_vector
        self.A = constraint_matrix
        self.lb = variable_lower_bound
        self.ub = variable_upper_bound
        self.constr_lb = constraint_lower_bound
        self.constr_ub = constraint_upper_bound
        self.c0 = objective_constant
        # initialize solution attributes
        self._status = None
        self._objval = None
        self._dualobj = None
        self._iter = None
        self._runtime = None
        self._x = None
        self._y = None
        self._pinf = None
        self._dinf = None

    def optimize(self):
        """
        Solve the linear programming problem using the cuPDLPx solver.
        """
        pass