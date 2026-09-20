/*
Copyright 2025 Haihao Lu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#pragma once

#include "utils.h"

#ifdef __cplusplus
extern "C"
{
#endif

    // The termination check: fixed point error, residuals and objective values
    // of the current iterate, and the criteria that decide whether the solver
    // stops. The infeasibility check (infeasibility.h) follows the same
    // protocol.
    //
    // Every scalar of a check is produced on the device by the reductions of
    // reduce.h and lands in state->scalars_d. A check is computed in halves so
    // that it synchronizes once: a _device half only queues kernels on the
    // stream and never waits, so the host can queue a whole check while the
    // GPU is still running the iterations before it; a _host half only does
    // host arithmetic on state->scalars_h. In between, the caller calls
    // fetch_device_scalars. The functions without a suffix run a whole check,
    // synchronization included.

    // Copies the whole scalar struct to the host and waits for the stream:
    // the one synchronization of a check.
    void fetch_device_scalars(pdhg_solver_state_t *state);

    // state->delta_primal_solution = to_primal - from_primal, and the same for
    // the dual vectors.
    void compute_solution_difference(pdhg_solver_state_t *state,
                                     const double *from_primal,
                                     const double *to_primal,
                                     const double *from_dual,
                                     const double *to_dual);

    // Residual vectors and their reductions from the products already in the
    // state (A x in primal_product, A^T y in dual_product): the primal
    // residual, the dual residual, the primal slack (the contribution of y to
    // the dual objective against the finite constraint bounds given), c^T x
    // and (dual slack)^T x. The optimality check passes the state's own
    // vectors; feasibility polishing, whose states carry a modified problem,
    // passes the original problem's objective vector, primal solution and
    // finite bounds where its formulas need them.
    void launch_residual_check(pdhg_solver_state_t *state,
                               const double *objective_vector,
                               const double *primal_solution,
                               const double *constraint_lower_bound_finite,
                               const double *constraint_upper_bound_finite);

    void compute_residual_device(pdhg_solver_state_t *state);

    void compute_residual_host(pdhg_solver_state_t *state, norm_type_t optimality_norm);

    void compute_residual(pdhg_solver_state_t *state, norm_type_t optimality_norm);

    void compute_fixed_point_error_device(pdhg_solver_state_t *state);

    void compute_fixed_point_error_host(pdhg_solver_state_t *state);

    void compute_fixed_point_error(pdhg_solver_state_t *state);

    void check_termination_criteria(pdhg_solver_state_t *solver_state, const termination_criteria_t *criteria);

#ifdef __cplusplus
}
#endif
