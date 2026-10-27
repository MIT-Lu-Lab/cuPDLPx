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

    // A check runs in two halves so that it synchronizes once: a _device half
    // only queues kernels, a _host half only does arithmetic on
    // state->scalars_h, and the caller calls fetch_device_scalars in between.
    // The functions without a suffix run all three.

    // Copies the scalar struct to the host and waits.
    void fetch_device_scalars(pdhg_solver_state_t *state);

    // state->delta_primal_solution = to_primal - from_primal, and the same for
    // the dual vectors.
    void compute_solution_difference(pdhg_solver_state_t *state,
                                     const double *from_primal,
                                     const double *to_primal,
                                     const double *from_dual,
                                     const double *to_dual);

    // Residuals and objective values from A x and A^T y already in the state.
    // Feasibility polishing passes vectors of the original problem where its own
    // state holds a modified one.
    void launch_residual_check(pdhg_solver_state_t *state,
                               const double *objective_vector,
                               const double *primal_solution,
                               const double *constraint_lower_bound_finite,
                               const double *constraint_upper_bound_finite);

    // The reductions of the residual and fixed point checks; see reduce.h for
    // required_bytes.
    void launch_residual_reductions(const pdhg_solver_state_t *state,
                                    size_t *required_bytes,
                                    const double *objective_vector,
                                    const double *primal_solution);
    void launch_fixed_point_reductions(const pdhg_solver_state_t *state, size_t *required_bytes);

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
