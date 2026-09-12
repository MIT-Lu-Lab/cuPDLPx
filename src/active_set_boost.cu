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

#include "active_set_boost.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>

static inline bool asb_raise_possible(double target, double step, double min_raise_ratio)
{
    return target > step && target >= min_raise_ratio * step;
}

__global__ void asb_var_window_kernel(const double *__restrict__ lb,
                                      const double *__restrict__ ub,
                                      const double *__restrict__ dual_slack,
                                      const double *__restrict__ objective,
                                      double variable_tol,
                                      int *__restrict__ last_free,
                                      bool *__restrict__ mask,
                                      int now,
                                      int window_start,
                                      int *__restrict__ delta_count,
                                      int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    /* The sign of dual_slack = (xbar - projection_input) / step certifies clamping. */
    double tol = variable_tol * fmax(1.0, fabs(objective[i]));
    bool fixed = lb[i] == ub[i];
    bool lower_projected = isfinite(lb[i]) && dual_slack[i] > tol;
    bool upper_projected = isfinite(ub[i]) && dual_slack[i] < -tol;
    bool confidently_clamped = fixed || lower_projected || upper_projected;
    if (!confidently_clamped)
        last_free[i] = now;
    bool m = last_free[i] >= window_start;
    if (m != mask[i])
    {
        mask[i] = m;
        atomicAdd(&delta_count[m ? 0 : 1], 1);
    }
}

__global__ void asb_row_window_kernel(const double *__restrict__ dual_projection_input,
                                      const double *__restrict__ lb,
                                      const double *__restrict__ ub,
                                      double constraint_tol,
                                      int *__restrict__ last_binding,
                                      bool *__restrict__ mask,
                                      int now,
                                      int window_start,
                                      int *__restrict__ delta_count,
                                      int m_rows)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m_rows)
        return;
    /* The row is inactive exactly when the dual projection input q lies safely
       inside [-ub,-lb]; a margin retains boundary and uncertain rows. */
    double q = dual_projection_input[j];
    bool binding = lb[j] == ub[j];
    if (!binding)
    {
        bool safely_inside = true;
        if (isfinite(ub[j]))
        {
            double margin = constraint_tol * fmax(1.0, fmax(fabs(q), fabs(ub[j])));
            safely_inside = safely_inside && (q + ub[j] > margin);
        }
        if (isfinite(lb[j]))
        {
            double margin = constraint_tol * fmax(1.0, fmax(fabs(q), fabs(lb[j])));
            safely_inside = safely_inside && (-lb[j] - q > margin);
        }
        binding = !safely_inside;
    }
    if (binding)
        last_binding[j] = now;
    bool m = last_binding[j] >= window_start;
    if (m != mask[j])
    {
        mask[j] = m;
        atomicAdd(&delta_count[m ? 2 : 3], 1);
    }
}

void active_set_boost_init(pdhg_solver_state_t *state)
{
    state->asb_step_ceiling = INFINITY;
    state->asb_anchor_best_pd_residual_gap = INFINITY;
    size_t n = (size_t)state->num_variables;
    size_t m = (size_t)state->num_constraints;
    CUDA_CHECK(cudaMalloc(&state->d_asb_var_last_free, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state->d_asb_row_last_binding, m * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state->d_asb_col_mask, n * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&state->d_asb_row_mask, m * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&state->d_asb_primal_anchor, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_asb_dual_anchor, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_asb_dual_slack_anchor, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_asb_dual_projection_input, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_asb_count, 4 * sizeof(int)));
    CUDA_CHECK(cudaMemset(state->d_asb_var_last_free, 0xff, n * sizeof(int)));    /* -1 */
    CUDA_CHECK(cudaMemset(state->d_asb_row_last_binding, 0xff, m * sizeof(int))); /* -1 */
    CUDA_CHECK(cudaMemset(state->d_asb_col_mask, 0, n * sizeof(bool)));
    CUDA_CHECK(cudaMemset(state->d_asb_row_mask, 0, m * sizeof(bool)));
}

void active_set_boost_free(pdhg_solver_state_t *state)
{
    if (state->asb_sv_ctx)
    {
        sv_estimator_free(state->asb_sv_ctx);
        state->asb_sv_ctx = NULL;
    }
    if (state->d_asb_var_last_free)
        cudaFree(state->d_asb_var_last_free);
    if (state->d_asb_row_last_binding)
        cudaFree(state->d_asb_row_last_binding);
    if (state->d_asb_col_mask)
        cudaFree(state->d_asb_col_mask);
    if (state->d_asb_row_mask)
        cudaFree(state->d_asb_row_mask);
    if (state->d_asb_primal_anchor)
        cudaFree(state->d_asb_primal_anchor);
    if (state->d_asb_dual_anchor)
        cudaFree(state->d_asb_dual_anchor);
    if (state->d_asb_dual_slack_anchor)
        cudaFree(state->d_asb_dual_slack_anchor);
    if (state->d_asb_dual_projection_input)
        cudaFree(state->d_asb_dual_projection_input);
    if (state->d_asb_count)
        cudaFree(state->d_asb_count);
}

static void asb_restore_anchor(pdhg_solver_state_t *state)
{
    double *p_dst[3] = {state->initial_primal_solution, state->current_primal_solution, state->pdhg_primal_solution};
    double *d_dst[3] = {state->initial_dual_solution, state->current_dual_solution, state->pdhg_dual_solution};
    for (int k = 0; k < 3; ++k)
    {
        CUDA_CHECK(cudaMemcpyAsync(p_dst[k],
                                   state->d_asb_primal_anchor,
                                   state->num_variables * sizeof(double),
                                   cudaMemcpyDeviceToDevice,
                                   state->stream));
        CUDA_CHECK(cudaMemcpyAsync(d_dst[k],
                                   state->d_asb_dual_anchor,
                                   state->num_constraints * sizeof(double),
                                   cudaMemcpyDeviceToDevice,
                                   state->stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(state->dual_slack,
                               state->d_asb_dual_slack_anchor,
                               state->num_variables * sizeof(double),
                               cudaMemcpyDeviceToDevice,
                               state->stream));
    CUDA_CHECK(cudaStreamSynchronize(state->stream));
}

static void
asb_update_window(pdhg_solver_state_t *state, const pdhg_parameters_t *params, int *delta_add, int *delta_remove)
{
    int now = state->total_count;
    int window_start = now - params->asb_window_iter;
    CUDA_CHECK(cudaMemsetAsync(state->d_asb_count, 0, 4 * sizeof(int), state->stream));
    asb_var_window_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream>>>(
        state->variable_lower_bound,
        state->variable_upper_bound,
        state->dual_slack,
        state->objective_vector,
        params->asb_variable_tol,
        state->d_asb_var_last_free,
        state->d_asb_col_mask,
        now,
        window_start,
        state->d_asb_count,
        state->num_variables);
    asb_row_window_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
        state->d_asb_dual_projection_input,
        state->constraint_lower_bound,
        state->constraint_upper_bound,
        params->asb_constraint_tol,
        state->d_asb_row_last_binding,
        state->d_asb_row_mask,
        now,
        window_start,
        state->d_asb_count,
        state->num_constraints);
    int counts[4] = {0, 0, 0, 0};
    CUDA_CHECK(cudaMemcpyAsync(counts, state->d_asb_count, 4 * sizeof(int), cudaMemcpyDeviceToHost, state->stream));
    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    state->asb_free_variables += counts[0] - counts[1];
    state->asb_binding_constraints += counts[2] - counts[3];
    *delta_add = counts[0] + counts[2];
    *delta_remove = counts[1] + counts[3];
}

typedef enum
{
    ASB_SV_OK = 0,       /* converged; current estimate stored in state->asb_sv */
    ASB_SV_NO_RAISE = 1, /* early PI termination proved that no step increase is possible */
    ASB_SV_FAILED = 2,   /* did not converge or produced an invalid result */
} asb_sv_status_t;

static double asb_pi_abort_threshold(const pdhg_solver_state_t *state, const pdhg_parameters_t *params)
{
    return params->asb_safety_factor / (params->asb_min_raise_ratio * state->step_size);
}

static asb_sv_status_t
asb_estimate_max_singular_value(pdhg_solver_state_t *state, const pdhg_parameters_t *params, int *out_pi_iterations)
{
    if (!state->asb_sv_ctx)
    {
        state->asb_sv_ctx = sv_estimator_create(
            state->sparse_handle, state->blas_handle, state->constraint_matrix, state->constraint_matrix_t);
    }
    sv_estimator_opts_t opts = {};
    opts.max_iterations = params->sv_max_iter;
    opts.tolerance = params->sv_tol;
    opts.d_row_mask = state->d_asb_row_mask;
    opts.d_col_mask = state->d_asb_col_mask;
    opts.abort_singular_value_threshold = asb_pi_abort_threshold(state, params);
    sv_estimator_result_t r = sv_estimator_run(state->asb_sv_ctx, &opts);
    state->asb_pi_iterations += r.iterations;
    *out_pi_iterations = r.iterations;
    if (r.status == SV_ESTIMATOR_ABORTED)
    {
        state->asb_pi_early_exit_count++;
        if (params->debug)
        {
            printf("[active-set boost] iter %d: sv estimate aborted early (PI %d iters, lower bound %.3e)\n",
                   state->total_count,
                   r.iterations,
                   r.max_singular_value);
        }
        return ASB_SV_NO_RAISE;
    }
    if (r.status != SV_ESTIMATOR_CONVERGED)
    {
        if (state->step_size <= state->base_step_size)
        {
            state->asb_sv = 0.0;
        }
        return ASB_SV_FAILED;
    }
    state->asb_sv = r.max_singular_value;
    return ASB_SV_OK;
}

static asb_action_t asb_revert(pdhg_solver_state_t *state, const pdhg_parameters_t *params)
{
    state->restart_count++;
    double diverged_step = state->step_size;
    state->asb_step_ceiling = params->asb_divergence_ceiling_ratio * diverged_step;
    asb_restore_anchor(state);
    compute_residual(state, params->optimality_norm);
    /* The infeasibility rays describe the discarded iterate; the anchor has none. */
    state->max_primal_ray_infeasibility = 0.0;
    state->max_dual_ray_infeasibility = 0.0;
    state->primal_ray_linear_objective = 0.0;
    state->dual_ray_objective = 0.0;
    state->step_size = state->base_step_size;
    state->primal_weight = state->asb_anchor_primal_weight;
    state->primal_weight_error_sum = state->asb_anchor_pw_error_sum;
    state->primal_weight_last_error = state->asb_anchor_pw_last_error;
    state->best_primal_weight = state->asb_anchor_best_pw;
    state->best_primal_dual_residual_gap = state->asb_anchor_best_pd_residual_gap;
    state->inner_count = 0;
    state->last_trial_fixed_point_error = INFINITY;
    state->asb_sv = 0.0;
    state->asb_no_raise_certified = false;
    sync_step_sizes_to_gpu(state);
    state->asb_revert_count++;
    if (state->asb_revert_count >= params->asb_max_reverts)
    {
        state->asb_phase = ASB_PHASE_OFF;
    }
    if (params->debug)
    {
        printf("[active-set boost] iter %d: divergence #%d, stepsize reverted "
               "(fixed-point error %.2e vs span initial %.2e, new stepsize ceiling %.3e)%s\n",
               state->total_count,
               state->asb_revert_count,
               state->fixed_point_error,
               state->initial_fixed_point_error,
               state->asb_step_ceiling,
               state->asb_revert_count >= params->asb_max_reverts ? ", controller off" : "");
    }
    else if (params->verbose)
    {
        printf("[active-set boost] iter %d: divergence #%d, stepsize reverted%s\n",
               state->total_count,
               state->asb_revert_count,
               state->asb_revert_count >= params->asb_max_reverts ? ", controller off" : "");
    }
    return ASB_ACTION_REVERT;
}

static double asb_worst_residual(const pdhg_solver_state_t *state)
{
    return fmax(state->relative_primal_residual, fmax(state->relative_dual_residual, state->relative_objective_gap));
}

static double asb_compute_target(const pdhg_solver_state_t *state, const pdhg_parameters_t *params)
{
    double global_step = state->base_step_size;
    double target = (state->asb_sv > 0.0) ? fmax(global_step, params->asb_safety_factor / state->asb_sv) : global_step;
    return fmax(global_step, fmin(target, state->asb_step_ceiling));
}

static bool asb_change_ready(const pdhg_solver_state_t *state, const pdhg_parameters_t *params)
{
    return state->asb_changes_since_estimate > 0 &&
        (double)state->asb_changes_since_estimate >=
        params->asb_reestimate_change_ratio * (double)(state->asb_free_variables + state->asb_binding_constraints);
}

static double asb_refresh_target(pdhg_solver_state_t *state, const pdhg_parameters_t *params, int *out_pi_iterations)
{
    if (!((state->asb_sv <= 0.0 && !state->asb_no_raise_certified) || asb_change_ready(state, params)))
        return state->step_size;
    /* The ceiling already precludes an increase: certify without running the power iteration. */
    if (!asb_raise_possible(
            fmax(state->base_step_size, state->asb_step_ceiling), state->step_size, params->asb_min_raise_ratio))
    {
        state->asb_no_raise_certified = true;
        state->asb_changes_since_estimate = 0;
        return state->step_size;
    }
    asb_sv_status_t status = asb_estimate_max_singular_value(state, params, out_pi_iterations);
    if (status == ASB_SV_OK)
    {
        state->asb_changes_since_estimate = 0;
        state->asb_no_raise_certified = false;
        return asb_compute_target(state, params);
    }
    if (status == ASB_SV_FAILED)
    {
        state->asb_sv_failed_count++;
    }
    else
    {
        state->asb_no_raise_certified = true;
        state->asb_changes_since_estimate = 0;
    }
    return state->step_size;
}

void active_set_boost_update_window(pdhg_solver_state_t *state, const pdhg_parameters_t *params)
{
    if (state->asb_phase == ASB_PHASE_OFF)
        return;
    int delta_add = 0, delta_remove = 0;
    asb_update_window(state, params, &delta_add, &delta_remove);
    state->asb_changes_since_estimate += delta_add + delta_remove;
}

asb_action_t active_set_boost_check(pdhg_solver_state_t *state, const pdhg_parameters_t *params)
{
    if (state->asb_phase == ASB_PHASE_OFF)
        return ASB_ACTION_NONE;
    /* A stop certified by the residuals keeps its iterate; only a limit stop may still revert. */
    if (state->termination_reason != TERMINATION_REASON_UNSPECIFIED &&
        state->termination_reason != TERMINATION_REASON_ITERATION_LIMIT &&
        state->termination_reason != TERMINATION_REASON_TIME_LIMIT)
        return ASB_ACTION_NONE;

    bool residuals_finite = isfinite(state->relative_primal_residual) && isfinite(state->relative_dual_residual) &&
        isfinite(state->relative_objective_gap);
    double res = asb_worst_residual(state);

    if (state->asb_phase == ASB_PHASE_WAITING)
    {
        if (!residuals_finite || res >= params->asb_activation_tol)
            return ASB_ACTION_NONE;
        if (state->asb_free_variables == state->num_variables &&
            state->asb_binding_constraints == state->num_constraints)
            return ASB_ACTION_NONE;
        state->asb_phase = ASB_PHASE_ACTIVE;
    }

    bool boosted = state->step_size > state->base_step_size;
    double fpe = state->fixed_point_error;
    bool tripped = boosted &&
        (!residuals_finite || !isfinite(fpe) ||
         (state->initial_fixed_point_error > 0.0 &&
          fpe > (1.0 + params->asb_divergence_margin) * state->initial_fixed_point_error));
    if (tripped)
    {
        return asb_revert(state, params);
    }

    return ASB_ACTION_NONE;
}

static void asb_snapshot_anchor(pdhg_solver_state_t *state)
{
    CUDA_CHECK(cudaMemcpyAsync(state->d_asb_primal_anchor,
                               state->initial_primal_solution,
                               state->num_variables * sizeof(double),
                               cudaMemcpyDeviceToDevice,
                               state->stream));
    CUDA_CHECK(cudaMemcpyAsync(state->d_asb_dual_anchor,
                               state->initial_dual_solution,
                               state->num_constraints * sizeof(double),
                               cudaMemcpyDeviceToDevice,
                               state->stream));
    /* dual_slack describes the iterate that became the anchor; compute_residual
       needs it to reproduce the anchor's residuals after a revert */
    CUDA_CHECK(cudaMemcpyAsync(state->d_asb_dual_slack_anchor,
                               state->dual_slack,
                               state->num_variables * sizeof(double),
                               cudaMemcpyDeviceToDevice,
                               state->stream));
    state->asb_anchor_primal_weight = state->primal_weight;
    state->asb_anchor_pw_error_sum = state->primal_weight_error_sum;
    state->asb_anchor_pw_last_error = state->primal_weight_last_error;
    state->asb_anchor_best_pw = state->best_primal_weight;
    state->asb_anchor_best_pd_residual_gap = state->best_primal_dual_residual_gap;
}

void active_set_boost_on_restart(pdhg_solver_state_t *state, const pdhg_parameters_t *params)
{
    if (state->asb_phase != ASB_PHASE_ACTIVE)
        return;
    int pi_iterations = 0;
    double target = asb_refresh_target(state, params, &pi_iterations);
    if (asb_raise_possible(target, state->step_size, params->asb_min_raise_ratio))
    {
        if (params->debug)
        {
            printf("[active-set boost] iter %d: stepsize %.3e -> %.3e (sv %.3e, PI iter %d)\n",
                   state->total_count,
                   state->step_size,
                   target,
                   state->asb_sv,
                   pi_iterations);
        }
        state->step_size = target;
        state->asb_raise_count++;
        asb_snapshot_anchor(state);
    }
}
