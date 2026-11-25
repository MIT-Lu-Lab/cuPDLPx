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

// Termination check: fixed point error, residuals, objective values, and the
// stopping criteria.

#include "reduce.h"
#include "termination.h"
#include "utils.h"

#include <math.h>
#include <time.h>

__global__ void compute_delta_dual_solution_kernel(const double *__restrict__ from_dual,
                                                   const double *__restrict__ to_dual,
                                                   double *__restrict__ delta_dual,
                                                   int n_cons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_cons)
    {
        delta_dual[i] = (to_dual[i] - from_dual[i]);
    }
}

void fetch_device_scalars(pdhg_solver_state_t *state)
{
    CUDA_CHECK(cudaMemcpyAsync(
        state->scalars_h, state->scalars_d, sizeof(device_scalars_t), cudaMemcpyDeviceToHost, state->stream));
    CUDA_CHECK(cudaStreamSynchronize(state->stream));
}

void launch_primal_residual_reductions(const pdhg_solver_state_t *state,
                                       size_t *required_bytes,
                                       const double *objective_vector,
                                       const double *primal_solution,
                                       const double *constraint_lower_bound,
                                       const double *constraint_upper_bound)
{
    device_scalars_t *out = state->scalars_d;
    const int m = state->num_constraints, n = state->num_variables;
    const double *primal_product = state->primal_product, *constraint_rescaling = state->constraint_rescaling;
    const double *constraint_lower = constraint_lower_bound, *constraint_upper = constraint_upper_bound;
    const double *objective = objective_vector, *x = primal_solution;
    auto primal_residual = DEVICE_FN(i)
    {
        double clamped_val = fmax(constraint_lower[i], fmin(primal_product[i], constraint_upper[i]));
        return (primal_product[i] - clamped_val) * constraint_rescaling[i];
    };
    reduce_sum(
        state,
        required_bytes,
        m,
        DEVICE_FN(i)
        {
            double r = primal_residual(i);
            return r * r;
        },
        &out->primal_res_sq);
    reduce_max(state, required_bytes, m, DEVICE_FN(i) { return fabs(primal_residual(i)); }, &out->primal_res_max);
    reduce_sum(state, required_bytes, n, DEVICE_FN(j) { return objective[j] * x[j]; }, &out->objective_dot);
}

void launch_dual_residual_reductions(const pdhg_solver_state_t *state,
                                     size_t *required_bytes,
                                     const double *objective_vector,
                                     const double *primal_solution,
                                     const double *constraint_lower_bound,
                                     const double *constraint_upper_bound)
{
    device_scalars_t *out = state->scalars_d;
    const int m = state->num_constraints, n = state->num_variables;
    const double *dual_product = state->dual_product, *variable_rescaling = state->variable_rescaling;
    const double *constraint_lower = constraint_lower_bound, *constraint_upper = constraint_upper_bound;
    const double *y = state->pdhg_dual_solution, *dual_slack = state->dual_slack;
    const double *objective = objective_vector, *x = primal_solution;
    auto primal_slack = DEVICE_FN(i)
    {
        return fmax(y[i], 0.0) * finite_or_zero(constraint_lower[i]) +
            fmin(y[i], 0.0) * finite_or_zero(constraint_upper[i]);
    };
    auto dual_residual =
        DEVICE_FN(j) { return (objective[j] - dual_product[j] - dual_slack[j]) * variable_rescaling[j]; };
    reduce_sum(state, required_bytes, m, primal_slack, &out->primal_slack_sum);
    reduce_sum(
        state,
        required_bytes,
        n,
        DEVICE_FN(j)
        {
            double r = dual_residual(j);
            return r * r;
        },
        &out->dual_res_sq);
    reduce_max(state, required_bytes, n, DEVICE_FN(j) { return fabs(dual_residual(j)); }, &out->dual_res_max);
    reduce_sum(state, required_bytes, n, DEVICE_FN(j) { return dual_slack[j] * x[j]; }, &out->dual_slack_dot);
}

void compute_residual_device(pdhg_solver_state_t *state)
{
    cupdlpx_spmv_Ax(state->sparse_handle, state->spmv_ctx, state->pdhg_primal_solution, state->primal_product);
    cupdlpx_spmv_ATx(state->sparse_handle, state->spmv_ctx, state->pdhg_dual_solution, state->dual_product);
    launch_primal_residual_reductions(state,
                                      NULL,
                                      state->objective_vector,
                                      state->pdhg_primal_solution,
                                      state->constraint_lower_bound,
                                      state->constraint_upper_bound);
    launch_dual_residual_reductions(state,
                                    NULL,
                                    state->objective_vector,
                                    state->pdhg_primal_solution,
                                    state->constraint_lower_bound,
                                    state->constraint_upper_bound);
}

// Turns the fetched scalars into the reported residuals and objective values.
void compute_residual_host(pdhg_solver_state_t *state, norm_type_t optimality_norm)
{
    const device_scalars_t *v = state->scalars_h;
    bool linf = optimality_norm == NORM_TYPE_L_INF;

    state->absolute_primal_residual =
        (linf ? v->primal_res_max : sqrt(v->primal_res_sq)) / state->constraint_bound_rescaling;
    state->absolute_dual_residual = (linf ? v->dual_res_max : sqrt(v->dual_res_sq)) / state->objective_vector_rescaling;

    double scale = state->constraint_bound_rescaling * state->objective_vector_rescaling;
    state->primal_objective_value = v->objective_dot / scale + state->objective_constant;
    state->dual_objective_value = (v->dual_slack_dot + v->primal_slack_sum) / scale + state->objective_constant;

    state->relative_primal_residual = state->absolute_primal_residual / (1.0 + state->constraint_bound_norm);
    state->relative_dual_residual = state->absolute_dual_residual / (1.0 + state->objective_vector_norm);
    state->objective_gap = fabs(state->primal_objective_value - state->dual_objective_value);
    state->relative_objective_gap =
        state->objective_gap / (1.0 + fabs(state->primal_objective_value) + fabs(state->dual_objective_value));
}

void compute_residual(pdhg_solver_state_t *state, norm_type_t optimality_norm)
{
    compute_residual_device(state);
    fetch_device_scalars(state);
    compute_residual_host(state, optimality_norm);
}

void launch_fixed_point_reductions(const pdhg_solver_state_t *state, size_t *required_bytes)
{
    // the squared norms of the two steps and the interaction term of the
    // primal step with A^T times the dual step
    device_scalars_t *out = state->scalars_d;
    const int m = state->num_constraints, n = state->num_variables;
    const double *from_primal = state->pdhg_primal_solution, *to_primal = state->reflected_primal_solution;
    const double *delta_dual = state->delta_dual_solution;
    const double *delta_dual_product = state->delta_dual_product;
    auto delta_primal = DEVICE_FN(j) { return to_primal[j] - from_primal[j]; };
    reduce_sum(state, required_bytes, m, DEVICE_FN(i) { return delta_dual[i] * delta_dual[i]; }, &out->fixed_point_dual_sq);
    reduce_sum(
        state,
        required_bytes,
        n,
        DEVICE_FN(j)
        {
            double d = delta_primal(j);
            return d * d;
        },
        &out->fixed_point_primal_sq);
    reduce_sum(
        state,
        required_bytes,
        n,
        DEVICE_FN(j) { return delta_dual_product[j] * delta_primal(j); },
        &out->fixed_point_cross);
}

// Queues the fixed point error of the current iterate.
void compute_fixed_point_error_device(pdhg_solver_state_t *state)
{
    compute_delta_dual_solution_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
        state->pdhg_dual_solution, state->reflected_dual_solution, state->delta_dual_solution, state->num_constraints);

    cupdlpx_spmv_ATx(state->sparse_handle, state->spmv_ctx, state->delta_dual_solution, state->delta_dual_product);

    launch_fixed_point_reductions(state, NULL);
}

void compute_fixed_point_error_host(pdhg_solver_state_t *state)
{
    const device_scalars_t *v = state->scalars_h;
    double movement = v->fixed_point_primal_sq * state->primal_weight + v->fixed_point_dual_sq / state->primal_weight;
    // measure in the M-norm of the smaller (global) step: M is indefinite for a
    // controller-boosted step
    double norm_step = fmin(state->step_size, state->base_step_size);
    double interaction = 2.0 * norm_step * v->fixed_point_cross;
    state->fixed_point_error = sqrt(movement + interaction);
}

void compute_fixed_point_error(pdhg_solver_state_t *state)
{
    compute_fixed_point_error_device(state);
    fetch_device_scalars(state);
    compute_fixed_point_error_host(state);
}

static bool optimality_criteria_met(const pdhg_solver_state_t *state, double rel_opt_tol, double rel_feas_tol)
{
    return state->relative_dual_residual < rel_feas_tol && state->relative_primal_residual < rel_feas_tol &&
        state->relative_objective_gap < rel_opt_tol;
}

static bool primal_infeasibility_criteria_met(const pdhg_solver_state_t *state, double eps)
{
    return state->dual_ray_objective > 0.0 && state->dual_ray_certificate_error <= eps;
}

static bool dual_infeasibility_criteria_met(const pdhg_solver_state_t *state, double eps)
{
    return state->primal_ray_linear_objective < 0.0 && state->primal_ray_certificate_error <= eps;
}

void check_termination_criteria(pdhg_solver_state_t *solver_state, const termination_criteria_t *criteria)
{
    solver_state->cumulative_time_sec = (double)(clock() - solver_state->start_time) / CLOCKS_PER_SEC;
    if (optimality_criteria_met(solver_state, criteria->eps_optimal_relative, criteria->eps_feasible_relative))
    {
        solver_state->termination_reason = TERMINATION_REASON_OPTIMAL;
        return;
    }
    if (primal_infeasibility_criteria_met(solver_state, criteria->eps_infeasible_relative))
    {
        solver_state->termination_reason = TERMINATION_REASON_PRIMAL_INFEASIBLE;
        return;
    }
    if (dual_infeasibility_criteria_met(solver_state, criteria->eps_infeasible_relative))
    {
        solver_state->termination_reason = TERMINATION_REASON_DUAL_INFEASIBLE;
        return;
    }
    if (solver_state->total_count >= criteria->iteration_limit)
    {
        solver_state->termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
        return;
    }
    if (solver_state->cumulative_time_sec >= criteria->time_sec_limit)
    {
        solver_state->termination_reason = TERMINATION_REASON_TIME_LIMIT;
        return;
    }
}
