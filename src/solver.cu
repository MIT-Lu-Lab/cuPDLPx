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

#include "cupdlpx.h"
#include "feasibility_polish.h"
#include "internal_types.h"
#include "preconditioner.h"
#include "presolve.h"
#include "solver.h"
#include "utils.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void build_row_ind(const int *__restrict__ row_ptr, int num_rows, int *__restrict__ row_ind);
__global__ void build_transpose_map(const int *__restrict__ A_row_ind,
                                    const int *__restrict__ A_col_ind,
                                    const int *__restrict__ At_row_ptr,
                                    const int *__restrict__ At_col_ind,
                                    int nnz,
                                    int *__restrict__ A_to_At);
__global__ void fill_finite_bounds_kernel(const double *__restrict__ lower_bound,
                                          const double *__restrict__ upper_bound,
                                          double *__restrict__ lower_bound_finite_val,
                                          double *__restrict__ upper_bound_finite_val,
                                          int num_elements);
__global__ void rescale_solution_kernel(double *__restrict__ primal_solution,
                                        double *__restrict__ dual_solution,
                                        const double *__restrict__ variable_rescaling,
                                        const double *__restrict__ constraint_rescaling,
                                        const double objective_vector_rescaling,
                                        const double constraint_bound_rescaling,
                                        int n_vars,
                                        int n_cons);
__global__ void compute_delta_solution_kernel(const double *__restrict__ initial_primal,
                                              const double *__restrict__ pdhg_primal,
                                              double *__restrict__ delta_primal,
                                              const double *__restrict__ initial_dual,
                                              const double *__restrict__ pdhg_dual,
                                              double *__restrict__ delta_dual,
                                              int n_vars,
                                              int n_cons);
static cupdlpx_result_t *create_result_from_state(pdhg_solver_state_t *state, const lp_problem_t *original_problem);
static void perform_restart(pdhg_solver_state_t *state, const pdhg_parameters_t *params);
static void initialize_step_size_and_primal_weight(pdhg_solver_state_t *state, const pdhg_parameters_t *params);
static pdhg_solver_state_t *initialize_solver_state(const lp_problem_t *working_problem,
                                                    const pdhg_parameters_t *params);
static void compute_fixed_point_error(pdhg_solver_state_t *state);
void pdhg_solver_state_free(pdhg_solver_state_t *state);
void rescale_info_free(rescale_info_t *info);

__global__ void compute_next_primal_solution_kernel(double *__restrict__ current_primal,
                                                    double *__restrict__ reflected_primal,
                                                    const double *__restrict__ initial_primal,
                                                    const double *__restrict__ dual_product,
                                                    const double *__restrict__ objective,
                                                    const double *__restrict__ var_lb,
                                                    const double *__restrict__ var_ub,
                                                    int n,
                                                    const double *__restrict__ d_step_size,
                                                    const int *__restrict__ d_base_count,
                                                    int k_offset,
                                                    double reflection_coeff);
__global__ void compute_next_primal_solution_major_kernel(double *__restrict__ current_primal,
                                                          double *__restrict__ pdhg_primal,
                                                          double *__restrict__ reflected_primal,
                                                          const double *__restrict__ initial_primal,
                                                          const double *__restrict__ dual_product,
                                                          const double *__restrict__ objective,
                                                          const double *__restrict__ var_lb,
                                                          const double *__restrict__ var_ub,
                                                          int n,
                                                          const double *__restrict__ d_step_size,
                                                          double *__restrict__ dual_slack,
                                                          const int *__restrict__ d_base_count,
                                                          int k_offset,
                                                          double reflection_coeff);
__global__ void compute_next_dual_solution_kernel(double *__restrict__ current_dual,
                                                  const double *__restrict__ initial_dual,
                                                  const double *__restrict__ primal_product,
                                                  const double *__restrict__ const_lb,
                                                  const double *__restrict__ const_ub,
                                                  int n,
                                                  const double *__restrict__ d_step_size,
                                                  const int *__restrict__ d_base_count,
                                                  int k_offset,
                                                  double reflection_coeff);
__global__ void compute_next_dual_solution_major_kernel(double *__restrict__ current_dual,
                                                        double *__restrict__ pdhg_dual,
                                                        double *__restrict__ reflected_dual,
                                                        const double *__restrict__ initial_dual,
                                                        const double *__restrict__ primal_product,
                                                        const double *__restrict__ const_lb,
                                                        const double *__restrict__ const_ub,
                                                        int n,
                                                        const double *__restrict__ d_step_size,
                                                        const int *__restrict__ d_base_count,
                                                        int k_offset,
                                                        double reflection_coeff);
void compute_next_primal_solution(pdhg_solver_state_t *state,
                                  const int k_offset,
                                  const double reflection_coefficient,
                                  bool is_major);
void compute_next_dual_solution(pdhg_solver_state_t *state,
                                const int k_offset,
                                const double reflection_coefficient,
                                bool is_major);
static void sync_step_sizes_to_gpu(pdhg_solver_state_t *state);
void sync_inner_count_to_gpu(pdhg_solver_state_t *state);
static void check_params_validity(const pdhg_parameters_t *params);

cupdlpx_result_t *optimize(const pdhg_parameters_t *params, lp_problem_t *original_problem)
{
    check_params_validity(params);
    print_initial_info(params, original_problem);

    cupdlpx_presolve_info_t *presolve_info = NULL;
    const lp_problem_t *working_problem = original_problem;

    if (params->presolve)
    {
        presolve_info = pslp_presolve(original_problem, params);
        if (presolve_info->problem_solved_during_presolve)
        {
            cupdlpx_result_t *result = create_result_from_presolve(presolve_info, original_problem);
            cupdlpx_presolve_info_free(presolve_info);
            pdhg_final_log(result, params);
            return result;
        }
        working_problem = presolve_info->reduced_problem;
    }

    pdhg_solver_state_t *state = initialize_solver_state(working_problem, params);
    display_iteration_stats(state, params->verbose);

    initialize_step_size_and_primal_weight(state, params);
    sync_step_sizes_to_gpu(state);

    state->start_time = clock();
    bool do_restart = false;

    cudaGraphExec_t graphExec = NULL;
    bool graph_created = false;

    while (state->total_count < params->termination_criteria.iteration_limit)
    {
        sync_inner_count_to_gpu(state);
        compute_next_primal_solution(state, 1, params->reflection_coefficient, true);
        compute_next_dual_solution(state, 1, params->reflection_coefficient, true);

        if (do_restart)
        {
            compute_fixed_point_error(state);
            state->initial_fixed_point_error = state->fixed_point_error;
            do_restart = false;
        }

        if (!graph_created)
        {
            // Start CUDA graph capture
            cudaStreamBeginCapture(state->stream, cudaStreamCaptureModeGlobal);

            for (int i = 2; i <= params->termination_evaluation_frequency - 1; i++)
            {
                compute_next_primal_solution(state, i, params->reflection_coefficient, false);
                compute_next_dual_solution(state, i, params->reflection_coefficient, false);
            }

            compute_next_primal_solution(
                state, params->termination_evaluation_frequency, params->reflection_coefficient, true);
            compute_next_dual_solution(
                state, params->termination_evaluation_frequency, params->reflection_coefficient, true);
            // end CUDA graph capture

            cudaGraph_t graph;
            CUDA_CHECK(cudaStreamEndCapture(state->stream, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
            CUDA_CHECK(cudaGraphDestroy(graph));
            graph_created = true;
        }
        CUDA_CHECK(cudaGraphLaunch(graphExec, state->stream));
        compute_fixed_point_error(state);

        compute_residual(state, params->optimality_norm);
        state->inner_count += params->termination_evaluation_frequency;
        state->total_count += params->termination_evaluation_frequency;

        // Logging
        if (state->total_count % get_print_frequency(state->total_count) == 0)
        {
            display_iteration_stats(state, params->verbose);
        }

        // Check Termination
        check_termination_criteria(state, &params->termination_criteria);
        if (state->termination_reason != TERMINATION_REASON_UNSPECIFIED)
        {
            break;
        }

        // Check Adaptive Restart
        do_restart =
            should_do_adaptive_restart(state, &params->restart_params, params->termination_evaluation_frequency);
        if (do_restart)
        {
            perform_restart(state, params);
            sync_step_sizes_to_gpu(state);
        }
    }

    if (graphExec)
    {
        CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    }

    if (state->termination_reason == TERMINATION_REASON_UNSPECIFIED)
    {
        state->termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
        compute_residual(state, params->optimality_norm);
        display_iteration_stats(state, params->verbose);
    }

    if (params->feasibility_polishing && state->termination_reason != TERMINATION_REASON_DUAL_INFEASIBLE &&
        state->termination_reason != TERMINATION_REASON_PRIMAL_INFEASIBLE)
    {
        feasibility_polish(params, state);
    }

    cupdlpx_result_t *result = create_result_from_state(state, original_problem);

    if (params->presolve && presolve_info)
    {
        pslp_postsolve(presolve_info, result, original_problem);
        cupdlpx_presolve_info_free(presolve_info);
    }

    pdhg_final_log(result, params);
    pdhg_solver_state_free(state);
    CUDA_CHECK(cudaGetLastError());
    return result;
}

static void sync_step_sizes_to_gpu(pdhg_solver_state_t *state)
{
    double current_primal_step = state->step_size / state->primal_weight;
    double current_dual_step = state->step_size * state->primal_weight;

    CUDA_CHECK(cudaMemcpyAsync(
        state->d_primal_step_size, &current_primal_step, sizeof(double), cudaMemcpyHostToDevice, state->stream));
    CUDA_CHECK(cudaMemcpyAsync(
        state->d_dual_step_size, &current_dual_step, sizeof(double), cudaMemcpyHostToDevice, state->stream));
}

void sync_inner_count_to_gpu(pdhg_solver_state_t *state)
{
    CUDA_CHECK(
        cudaMemcpyAsync(state->d_inner_count, &state->inner_count, sizeof(int), cudaMemcpyHostToDevice, state->stream));
}

static void check_params_validity(const pdhg_parameters_t *params)
{
    if (params->termination_evaluation_frequency < 3)
    {
        fprintf(stderr,
                "Error: termination_evaluation_frequency must be >= 3 (got %d).\n",
                params->termination_evaluation_frequency);
        exit(EXIT_FAILURE);
    }
}

__global__ void compute_and_rescale_reduced_cost_kernel(double *__restrict__ reduced_cost,
                                                        const double *__restrict__ objective,
                                                        const double *__restrict__ dual_product,
                                                        const double *__restrict__ variable_rescaling,
                                                        const double objective_vector_rescaling,
                                                        const double constraint_bound_rescaling,
                                                        const double *__restrict__ variable_lower_bound,
                                                        const double *__restrict__ variable_upper_bound,
                                                        int n_vars)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        double rc = (objective[i] - dual_product[i]) * variable_rescaling[i] / objective_vector_rescaling;

        if (!isfinite(variable_lower_bound[i]))
        {
            rc = fmin(rc, 0.0);
        }
        if (!isfinite(variable_upper_bound[i]))
        {
            rc = fmax(rc, 0.0);
        }
        reduced_cost[i] = rc;
    }
}

static pdhg_solver_state_t *initialize_solver_state(const lp_problem_t *working_problem,
                                                    const pdhg_parameters_t *params)
{
    pdhg_solver_state_t *state = (pdhg_solver_state_t *)safe_calloc(1, sizeof(pdhg_solver_state_t));

    int n_vars = working_problem->num_variables;
    int n_cons = working_problem->num_constraints;
    int nnz = working_problem->constraint_matrix_num_nonzeros;
    size_t var_bytes = n_vars * sizeof(double);
    size_t con_bytes = n_cons * sizeof(double);

    state->num_variables = n_vars;
    state->num_constraints = n_cons;
    state->objective_constant = working_problem->objective_constant;

    state->constraint_matrix = (cu_sparse_matrix_csr_t *)safe_malloc(sizeof(cu_sparse_matrix_csr_t));
    state->constraint_matrix_t = (cu_sparse_matrix_csr_t *)safe_malloc(sizeof(cu_sparse_matrix_csr_t));

    state->constraint_matrix->num_rows = n_cons;
    state->constraint_matrix->num_cols = n_vars;
    state->constraint_matrix->num_nonzeros = nnz;

    state->constraint_matrix_t->num_rows = n_vars;
    state->constraint_matrix_t->num_cols = n_cons;
    state->constraint_matrix_t->num_nonzeros = nnz;

    state->termination_reason = TERMINATION_REASON_UNSPECIFIED;

    state->num_blocks_primal = (state->num_variables + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    state->num_blocks_dual = (state->num_constraints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    state->num_blocks_primal_dual =
        (state->num_variables + state->num_constraints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    state->num_blocks_nnz = (state->constraint_matrix->num_nonzeros + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    CUSPARSE_CHECK(cusparseCreate(&state->sparse_handle));
    CUBLAS_CHECK(cublasCreate(&state->blas_handle));
    CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_HOST));

#define ALLOC_AND_COPY(dest, src, bytes)                                                                               \
    CUDA_CHECK(cudaMalloc(&dest, bytes));                                                                              \
    CUDA_CHECK(cudaMemcpy(dest, src, bytes, cudaMemcpyHostToDevice));

    ALLOC_AND_COPY(
        state->constraint_matrix->row_ptr, working_problem->constraint_matrix_row_pointers, (n_cons + 1) * sizeof(int));
    ALLOC_AND_COPY(state->constraint_matrix->col_ind,
                   working_problem->constraint_matrix_col_indices,
                   working_problem->constraint_matrix_num_nonzeros * sizeof(int));
    ALLOC_AND_COPY(state->constraint_matrix->val,
                   working_problem->constraint_matrix_values,
                   working_problem->constraint_matrix_num_nonzeros * sizeof(double));

    CUDA_CHECK(cudaMalloc(&state->constraint_matrix->row_ind, nnz * sizeof(int)));
    build_row_ind<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
        state->constraint_matrix->row_ptr, n_cons, state->constraint_matrix->row_ind);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMalloc(&state->constraint_matrix_t->row_ptr, (n_vars + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state->constraint_matrix_t->col_ind,
                          working_problem->constraint_matrix_num_nonzeros * sizeof(int)));
    CUDA_CHECK(
        cudaMalloc(&state->constraint_matrix_t->val, working_problem->constraint_matrix_num_nonzeros * sizeof(double)));

    size_t buffer_size = 0;
    void *buffer = nullptr;
    CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(state->sparse_handle,
                                                 state->constraint_matrix->num_rows,
                                                 state->constraint_matrix->num_cols,
                                                 state->constraint_matrix->num_nonzeros,
                                                 state->constraint_matrix->val,
                                                 state->constraint_matrix->row_ptr,
                                                 state->constraint_matrix->col_ind,
                                                 state->constraint_matrix_t->val,
                                                 state->constraint_matrix_t->row_ptr,
                                                 state->constraint_matrix_t->col_ind,
                                                 CUDA_R_64F,
                                                 CUSPARSE_ACTION_NUMERIC,
                                                 CUSPARSE_INDEX_BASE_ZERO,
                                                 CUSPARSE_CSR2CSC_ALG_DEFAULT,
                                                 &buffer_size));
    CUDA_CHECK(cudaMalloc(&buffer, buffer_size));

    CUSPARSE_CHECK(cusparseCsr2cscEx2(state->sparse_handle,
                                      state->constraint_matrix->num_rows,
                                      state->constraint_matrix->num_cols,
                                      state->constraint_matrix->num_nonzeros,
                                      state->constraint_matrix->val,
                                      state->constraint_matrix->row_ptr,
                                      state->constraint_matrix->col_ind,
                                      state->constraint_matrix_t->val,
                                      state->constraint_matrix_t->row_ptr,
                                      state->constraint_matrix_t->col_ind,
                                      CUDA_R_64F,
                                      CUSPARSE_ACTION_NUMERIC,
                                      CUSPARSE_INDEX_BASE_ZERO,
                                      CUSPARSE_CSR2CSC_ALG_DEFAULT,
                                      buffer));

    CUDA_CHECK(cudaFree(buffer));

    CUDA_CHECK(cudaMalloc(&state->constraint_matrix_t->row_ind, nnz * sizeof(int)));
    build_row_ind<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->constraint_matrix_t->row_ptr, n_vars, state->constraint_matrix_t->row_ind);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMalloc(&state->constraint_matrix->transpose_map, nnz * sizeof(int)));
    state->constraint_matrix_t->transpose_map = NULL;
    build_transpose_map<<<state->num_blocks_nnz, THREADS_PER_BLOCK>>>(state->constraint_matrix->row_ind,
                                                                      state->constraint_matrix->col_ind,
                                                                      state->constraint_matrix_t->row_ptr,
                                                                      state->constraint_matrix_t->col_ind,
                                                                      nnz,
                                                                      state->constraint_matrix->transpose_map);
    CUDA_CHECK(cudaGetLastError());

    ALLOC_AND_COPY(state->variable_lower_bound, working_problem->variable_lower_bound, var_bytes);
    ALLOC_AND_COPY(state->variable_upper_bound, working_problem->variable_upper_bound, var_bytes);
    ALLOC_AND_COPY(state->objective_vector, working_problem->objective_vector, var_bytes);
    ALLOC_AND_COPY(state->constraint_lower_bound, working_problem->constraint_lower_bound, con_bytes);
    ALLOC_AND_COPY(state->constraint_upper_bound, working_problem->constraint_upper_bound, con_bytes);

#define ALLOC_ZERO(dest, bytes)                                                                                        \
    CUDA_CHECK(cudaMalloc(&dest, bytes));                                                                              \
    CUDA_CHECK(cudaMemset(dest, 0, bytes));

    ALLOC_ZERO(state->initial_primal_solution, var_bytes);
    ALLOC_ZERO(state->current_primal_solution, var_bytes);
    ALLOC_ZERO(state->pdhg_primal_solution, var_bytes);
    ALLOC_ZERO(state->reflected_primal_solution, var_bytes);
    ALLOC_ZERO(state->dual_product, var_bytes);
    ALLOC_ZERO(state->dual_slack, var_bytes);
    ALLOC_ZERO(state->dual_residual, var_bytes);
    ALLOC_ZERO(state->delta_primal_solution, var_bytes);

    ALLOC_ZERO(state->initial_dual_solution, con_bytes);
    ALLOC_ZERO(state->current_dual_solution, con_bytes);
    ALLOC_ZERO(state->pdhg_dual_solution, con_bytes);
    ALLOC_ZERO(state->reflected_dual_solution, con_bytes);
    ALLOC_ZERO(state->primal_product, con_bytes);
    ALLOC_ZERO(state->primal_slack, con_bytes);
    ALLOC_ZERO(state->primal_residual, con_bytes);
    ALLOC_ZERO(state->delta_dual_solution, con_bytes);

    if (working_problem->primal_start)
    {
        CUDA_CHECK(cudaMemcpy(
            state->initial_primal_solution, working_problem->primal_start, var_bytes, cudaMemcpyHostToDevice));
    }
    if (working_problem->dual_start)
    {
        CUDA_CHECK(
            cudaMemcpy(state->initial_dual_solution, working_problem->dual_start, con_bytes, cudaMemcpyHostToDevice));
    }

    rescale_info_t *rescale_info = rescale_problem(params, state);

    state->constraint_rescaling = rescale_info->con_rescale;
    state->variable_rescaling = rescale_info->var_rescale;
    state->constraint_bound_rescaling = rescale_info->con_bound_rescale;
    state->objective_vector_rescaling = rescale_info->obj_vec_rescale;
    state->rescaling_time_sec = rescale_info->rescaling_time_sec;

    rescale_info->con_rescale = NULL;
    rescale_info->var_rescale = NULL;
    rescale_info_free(rescale_info);

    CUDA_CHECK(cudaMemcpy(
        state->current_primal_solution, state->initial_primal_solution, var_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(
        cudaMemcpy(state->current_dual_solution, state->initial_dual_solution, con_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(
        cudaMemcpy(state->pdhg_primal_solution, state->initial_primal_solution, var_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(
        cudaMemcpy(state->pdhg_dual_solution, state->initial_dual_solution, con_bytes, cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaMalloc(&state->constraint_lower_bound_finite_val, con_bytes));
    CUDA_CHECK(cudaMalloc(&state->constraint_upper_bound_finite_val, con_bytes));
    CUDA_CHECK(cudaMalloc(&state->variable_lower_bound_finite_val, var_bytes));
    CUDA_CHECK(cudaMalloc(&state->variable_upper_bound_finite_val, var_bytes));

    fill_finite_bounds_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(state->constraint_lower_bound,
                                                                             state->constraint_upper_bound,
                                                                             state->constraint_lower_bound_finite_val,
                                                                             state->constraint_upper_bound_finite_val,
                                                                             n_cons);

    fill_finite_bounds_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(state->variable_lower_bound,
                                                                               state->variable_upper_bound,
                                                                               state->variable_lower_bound_finite_val,
                                                                               state->variable_upper_bound_finite_val,
                                                                               n_vars);

    CUDA_CHECK(cudaFree(state->constraint_matrix->row_ind));
    state->constraint_matrix->row_ind = NULL;
    CUDA_CHECK(cudaFree(state->constraint_matrix_t->row_ind));
    state->constraint_matrix_t->row_ind = NULL;
    CUDA_CHECK(cudaFree(state->constraint_matrix->transpose_map));
    state->constraint_matrix->transpose_map = NULL;

    double sum_of_squares = 0.0;
    double max_val = 0.0;
    double val = 0.0;
    for (int i = 0; i < n_vars; ++i)
    {
        if (params->optimality_norm == NORM_TYPE_L_INF)
        {
            val = fabs(working_problem->objective_vector[i]);
            if (val > max_val)
                max_val = val;
        }
        else
        {
            sum_of_squares += working_problem->objective_vector[i] * working_problem->objective_vector[i];
        }
    }

    if (params->optimality_norm == NORM_TYPE_L_INF)
    {
        state->objective_vector_norm = max_val;
    }
    else
    {
        state->objective_vector_norm = sqrt(sum_of_squares);
    }

    sum_of_squares = 0.0;
    max_val = 0.0;
    val = 0.0;
    for (int i = 0; i < n_cons; ++i)
    {
        double lower = working_problem->constraint_lower_bound[i];
        double upper = working_problem->constraint_upper_bound[i];

        if (params->optimality_norm == NORM_TYPE_L_INF)
        {
            if (isfinite(lower) && (lower != upper))
            {
                val = fabs(lower);
                if (val > max_val)
                    max_val = val;
            }
            if (isfinite(upper))
            {
                val = fabs(upper);
                if (val > max_val)
                    max_val = val;
            }
        }
        else
        {
            if (isfinite(lower) && (lower != upper))
            {
                sum_of_squares += lower * lower;
            }
            if (isfinite(upper))
            {
                sum_of_squares += upper * upper;
            }
        }
    }
    if (params->optimality_norm == NORM_TYPE_L_INF)
    {
        state->constraint_bound_norm = max_val;
    }
    else
    {
        state->constraint_bound_norm = sqrt(sum_of_squares);
    }

    state->best_primal_dual_residual_gap = INFINITY;
    state->last_trial_fixed_point_error = INFINITY;
    state->step_size = 0.0;
    state->is_this_major_iteration = false;

    CUDA_CHECK(cudaMalloc(&state->d_primal_step_size, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_dual_step_size, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_inner_count, sizeof(int)));

    CUDA_CHECK(cudaMemset(state->d_primal_step_size, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(state->d_dual_step_size, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(state->d_inner_count, 0, sizeof(int)));

    CUDA_CHECK(cudaMalloc(&state->ones_primal_d, state->num_variables * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->ones_dual_d, state->num_constraints * sizeof(double)));

    double *ones_primal_h = (double *)safe_malloc(state->num_variables * sizeof(double));
    for (int i = 0; i < state->num_variables; ++i)
        ones_primal_h[i] = 1.0;
    CUDA_CHECK(
        cudaMemcpy(state->ones_primal_d, ones_primal_h, state->num_variables * sizeof(double), cudaMemcpyHostToDevice));
    free(ones_primal_h);

    double *ones_dual_h = (double *)safe_malloc(state->num_constraints * sizeof(double));
    for (int i = 0; i < state->num_constraints; ++i)
        ones_dual_h[i] = 1.0;
    CUDA_CHECK(
        cudaMemcpy(state->ones_dual_d, ones_dual_h, state->num_constraints * sizeof(double), cudaMemcpyHostToDevice));

    // --- CUDA Graph Initialization ---
    CUDA_CHECK(cudaStreamCreate(&state->stream));
    CUSPARSE_CHECK(cusparseSetStream(state->sparse_handle, state->stream));
    CUBLAS_CHECK(cublasSetStream(state->blas_handle, state->stream));

    state->spmv_ctx = cupdlpx_spmv_ctx_create(state->sparse_handle,
                                              state->constraint_matrix,
                                              state->constraint_matrix_t,
                                              state->pdhg_primal_solution,
                                              state->primal_product,
                                              state->pdhg_dual_solution,
                                              state->dual_product);

    free(ones_dual_h);
    if (params->verbose)
    {
        printf("---------------------------------------------------------------------"
               "------------------\n");
        printf("%s | %s | %s | %s \n",
               "   runtime    ",
               "    objective     ",
               "  absolute residuals   ",
               "  relative residuals   ");
        printf("%s %s | %s %s | %s %s %s | %s %s %s \n",
               "  iter",
               "  time ",
               " pr obj ",
               "  du obj ",
               " pr res",
               " du res",
               "  gap  ",
               " pr res",
               " du res",
               "  gap  ");
        printf("---------------------------------------------------------------------"
               "------------------\n");
    }

    return state;
}

__global__ void build_row_ind(const int *__restrict__ row_ptr, int num_rows, int *__restrict__ row_ind)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rows)
        return;

    int s = row_ptr[i];
    int e = row_ptr[i + 1];

    for (int k = s; k < e; ++k)
    {
        row_ind[k] = i;
    }
}

__global__ void build_transpose_map(const int *__restrict__ A_row_ind,
                                    const int *__restrict__ A_col_ind,
                                    const int *__restrict__ At_row_ptr,
                                    const int *__restrict__ At_col_ind,
                                    int nnz,
                                    int *__restrict__ A_to_At)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nnz)
        return;

    int i = A_row_ind[k];
    int j = A_col_ind[k];

    int start = At_row_ptr[j];
    int end = At_row_ptr[j + 1];

    int pos = -1;
    for (int idx = start; idx < end; ++idx)
    {
        if (At_col_ind[idx] == i)
        {
            pos = idx;
            break;
        }
    }

    if (pos < 0)
        return;

    A_to_At[k] = pos;
}

__global__ void fill_finite_bounds_kernel(const double *__restrict__ lb,
                                          const double *__restrict__ ub,
                                          double *__restrict__ lb_finite,
                                          double *__restrict__ ub_finite,
                                          int num_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements)
        return;

    double Li = lb[i];
    double Ui = ub[i];

    lb_finite[i] = isfinite(Li) ? Li : 0.0;
    ub_finite[i] = isfinite(Ui) ? Ui : 0.0;
}

__global__ void compute_next_primal_solution_kernel(double *__restrict__ current_primal,
                                                    double *__restrict__ reflected_primal,
                                                    const double *__restrict__ initial_primal,
                                                    const double *__restrict__ dual_product,
                                                    const double *__restrict__ objective,
                                                    const double *__restrict__ var_lb,
                                                    const double *__restrict__ var_ub,
                                                    int n,
                                                    const double *__restrict__ d_step_size,
                                                    const int *__restrict__ d_base_count,
                                                    int k_offset,
                                                    double reflection_coeff)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double step_size = *d_step_size;
    int current_k = *d_base_count + k_offset;
    double weight = (double)(current_k) / (double)(current_k + 1);
    if (i < n)
    {
        double temp = current_primal[i] - step_size * (objective[i] - dual_product[i]);
        double temp_proj = fmax(var_lb[i], fmin(temp, var_ub[i]));
        reflected_primal[i] = 2.0 * temp_proj - current_primal[i];
        double reflected = reflection_coeff * reflected_primal[i] + (1.0 - reflection_coeff) * current_primal[i];
        current_primal[i] = weight * reflected + (1.0 - weight) * initial_primal[i];
    }
}

__global__ void compute_next_primal_solution_major_kernel(double *__restrict__ current_primal,
                                                          double *__restrict__ pdhg_primal,
                                                          double *__restrict__ reflected_primal,
                                                          const double *__restrict__ initial_primal,
                                                          const double *__restrict__ dual_product,
                                                          const double *__restrict__ objective,
                                                          const double *__restrict__ var_lb,
                                                          const double *__restrict__ var_ub,
                                                          int n,
                                                          const double *__restrict__ d_step_size,
                                                          double *__restrict__ dual_slack,
                                                          const int *__restrict__ d_base_count,
                                                          int k_offset,
                                                          double reflection_coeff)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double step_size = *d_step_size;
    int current_k = *d_base_count + k_offset;
    double weight = (double)(current_k) / (double)(current_k + 1);
    if (i < n)
    {
        double temp = current_primal[i] - step_size * (objective[i] - dual_product[i]);
        pdhg_primal[i] = fmax(var_lb[i], fmin(temp, var_ub[i]));
        dual_slack[i] = (pdhg_primal[i] - temp) / step_size;
        reflected_primal[i] = 2.0 * pdhg_primal[i] - current_primal[i];
        double reflected = reflection_coeff * reflected_primal[i] + (1.0 - reflection_coeff) * current_primal[i];
        current_primal[i] = weight * reflected + (1.0 - weight) * initial_primal[i];
    }
}

__global__ void compute_next_dual_solution_kernel(double *__restrict__ current_dual,
                                                  const double *__restrict__ initial_dual,
                                                  const double *__restrict__ primal_product,
                                                  const double *__restrict__ const_lb,
                                                  const double *__restrict__ const_ub,
                                                  int n,
                                                  const double *__restrict__ d_step_size,
                                                  const int *__restrict__ d_base_count,
                                                  int k_offset,
                                                  double reflection_coeff)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double step_size = *d_step_size;
    int current_k = *d_base_count + k_offset;
    double weight = (double)(current_k) / (double)(current_k + 1);
    if (i < n)
    {
        double temp = current_dual[i] / step_size - primal_product[i];
        double temp_proj = fmax(-const_ub[i], fmin(temp, -const_lb[i]));
        double reflected = reflection_coeff * (2.0 * (temp - temp_proj) * step_size - current_dual[i]) +
            (1.0 - reflection_coeff) * current_dual[i];
        current_dual[i] = weight * reflected + (1.0 - weight) * initial_dual[i];
    }
}

__global__ void compute_next_dual_solution_major_kernel(double *__restrict__ current_dual,
                                                        double *__restrict__ pdhg_dual,
                                                        double *__restrict__ reflected_dual,
                                                        const double *__restrict__ initial_dual,
                                                        const double *__restrict__ primal_product,
                                                        const double *__restrict__ const_lb,
                                                        const double *__restrict__ const_ub,
                                                        int n,
                                                        const double *__restrict__ d_step_size,
                                                        const int *__restrict__ d_base_count,
                                                        int k_offset,
                                                        double reflection_coeff)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double step_size = *d_step_size;
    int current_k = *d_base_count + k_offset;
    double weight = (double)(current_k) / (double)(current_k + 1);
    if (i < n)
    {
        double temp = current_dual[i] / step_size - primal_product[i];
        double temp_proj = fmax(-const_ub[i], fmin(temp, -const_lb[i]));
        pdhg_dual[i] = (temp - temp_proj) * step_size;
        reflected_dual[i] = 2.0 * pdhg_dual[i] - current_dual[i];
        double reflected = reflection_coeff * reflected_dual[i] + (1.0 - reflection_coeff) * current_dual[i];
        current_dual[i] = weight * reflected + (1.0 - weight) * initial_dual[i];
    }
}

__global__ void rescale_solution_kernel(double *__restrict__ primal_solution,
                                        double *__restrict__ dual_solution,
                                        const double *__restrict__ variable_rescaling,
                                        const double *__restrict__ constraint_rescaling,
                                        const double objective_vector_rescaling,
                                        const double constraint_bound_rescaling,
                                        int n_vars,
                                        int n_cons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        primal_solution[i] = primal_solution[i] / variable_rescaling[i] / constraint_bound_rescaling;
    }
    else if (i < n_vars + n_cons)
    {
        int idx = i - n_vars;
        dual_solution[idx] = dual_solution[idx] / constraint_rescaling[idx] / objective_vector_rescaling;
    }
}

__global__ void compute_delta_solution_kernel(const double *__restrict__ initial_primal,
                                              const double *__restrict__ pdhg_primal,
                                              double *__restrict__ delta_primal,
                                              const double *__restrict__ initial_dual,
                                              const double *__restrict__ pdhg_dual,
                                              double *__restrict__ delta_dual,
                                              int n_vars,
                                              int n_cons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        delta_primal[i] = (pdhg_primal[i] - initial_primal[i]);
    }
    else if (i < n_vars + n_cons)
    {
        int idx = i - n_vars;
        delta_dual[idx] = (pdhg_dual[idx] - initial_dual[idx]);
    }
}

void compute_next_primal_solution(pdhg_solver_state_t *state,
                                  const int k_offset,
                                  const double reflection_coefficient,
                                  bool is_major)
{
    cupdlpx_spmv_ATx(state->sparse_handle, state->spmv_ctx, state->current_dual_solution, state->dual_product);

    double step = state->step_size / state->primal_weight;

    if (is_major)
    {
        compute_next_primal_solution_major_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream>>>(
            state->current_primal_solution,
            state->pdhg_primal_solution,
            state->reflected_primal_solution,
            state->initial_primal_solution,
            state->dual_product,
            state->objective_vector,
            state->variable_lower_bound,
            state->variable_upper_bound,
            state->num_variables,
            state->d_primal_step_size,
            state->dual_slack,
            state->d_inner_count,
            k_offset,
            reflection_coefficient);
    }
    else
    {
        compute_next_primal_solution_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream>>>(
            state->current_primal_solution,
            state->reflected_primal_solution,
            state->initial_primal_solution,
            state->dual_product,
            state->objective_vector,
            state->variable_lower_bound,
            state->variable_upper_bound,
            state->num_variables,
            state->d_primal_step_size,
            state->d_inner_count,
            k_offset,
            reflection_coefficient);
    }
}

void compute_next_dual_solution(pdhg_solver_state_t *state,
                                const int k_offset,
                                const double reflection_coefficient,
                                bool is_major)
{
    cupdlpx_spmv_Ax(state->sparse_handle, state->spmv_ctx, state->reflected_primal_solution, state->primal_product);

    double step = state->step_size * state->primal_weight;

    if (is_major)
    {
        compute_next_dual_solution_major_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
            state->current_dual_solution,
            state->pdhg_dual_solution,
            state->reflected_dual_solution,
            state->initial_dual_solution,
            state->primal_product,
            state->constraint_lower_bound,
            state->constraint_upper_bound,
            state->num_constraints,
            state->d_dual_step_size,
            state->d_inner_count,
            k_offset,
            reflection_coefficient);
    }
    else
    {
        compute_next_dual_solution_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
            state->current_dual_solution,
            state->initial_dual_solution,
            state->primal_product,
            state->constraint_lower_bound,
            state->constraint_upper_bound,
            state->num_constraints,
            state->d_dual_step_size,
            state->d_inner_count,
            k_offset,
            reflection_coefficient);
    }
}

static void perform_restart(pdhg_solver_state_t *state, const pdhg_parameters_t *params)
{
    compute_delta_solution_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
        state->initial_primal_solution,
        state->pdhg_primal_solution,
        state->delta_primal_solution,
        state->initial_dual_solution,
        state->pdhg_dual_solution,
        state->delta_dual_solution,
        state->num_variables,
        state->num_constraints);

    double primal_dist, dual_dist;
    CUBLAS_CHECK(
        cublasDnrm2_v2_64(state->blas_handle, state->num_variables, state->delta_primal_solution, 1, &primal_dist));
    CUBLAS_CHECK(
        cublasDnrm2_v2_64(state->blas_handle, state->num_constraints, state->delta_dual_solution, 1, &dual_dist));

    double ratio_infeas = state->relative_dual_residual / state->relative_primal_residual;

    if (primal_dist > 1e-16 && dual_dist > 1e-16 && primal_dist < 1e12 && dual_dist < 1e12 && ratio_infeas > 1e-8 &&
        ratio_infeas < 1e8)
    {
        double error = log(dual_dist) - log(primal_dist) - log(state->primal_weight);
        state->primal_weight_error_sum *= params->restart_params.i_smooth;
        state->primal_weight_error_sum += error;
        double delta_error = error - state->primal_weight_last_error;
        state->primal_weight *=
            exp(params->restart_params.k_p * error + params->restart_params.k_i * state->primal_weight_error_sum +
                params->restart_params.k_d * delta_error);
        state->primal_weight_last_error = error;
    }
    else
    {
        state->primal_weight = state->best_primal_weight;
        state->primal_weight_error_sum = 0.0;
        state->primal_weight_last_error = 0.0;
    }

    double primal_dual_residual_gap = fabs(log10(state->relative_dual_residual / state->relative_primal_residual));
    if (primal_dual_residual_gap < state->best_primal_dual_residual_gap)
    {
        state->best_primal_dual_residual_gap = primal_dual_residual_gap;
        state->best_primal_weight = state->primal_weight;
    }

    CUDA_CHECK(cudaMemcpy(state->initial_primal_solution,
                          state->pdhg_primal_solution,
                          state->num_variables * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->current_primal_solution,
                          state->pdhg_primal_solution,
                          state->num_variables * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->initial_dual_solution,
                          state->pdhg_dual_solution,
                          state->num_constraints * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->current_dual_solution,
                          state->pdhg_dual_solution,
                          state->num_constraints * sizeof(double),
                          cudaMemcpyDeviceToDevice));

    state->inner_count = 0;
    state->last_trial_fixed_point_error = INFINITY;
}

static void initialize_step_size_and_primal_weight(pdhg_solver_state_t *state, const pdhg_parameters_t *params)
{
    if (state->constraint_matrix->num_nonzeros == 0)
    {
        state->step_size = 1.0;
    }
    else
    {
        double max_sv = estimate_maximum_singular_value(state->sparse_handle,
                                                        state->blas_handle,
                                                        state->constraint_matrix,
                                                        state->constraint_matrix_t,
                                                        params->sv_max_iter,
                                                        params->sv_tol);
        state->step_size = 0.998 / max_sv;
    }

    if (params->bound_objective_rescaling)
    {
        state->primal_weight = 1.0;
    }
    else
    {
        state->primal_weight = (state->objective_vector_norm + 1.0) / (state->constraint_bound_norm + 1.0);
    }
    state->best_primal_weight = state->primal_weight;
}

static void compute_fixed_point_error(pdhg_solver_state_t *state)
{
    compute_delta_solution_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
        state->pdhg_primal_solution,
        state->reflected_primal_solution,
        state->delta_primal_solution,
        state->pdhg_dual_solution,
        state->reflected_dual_solution,
        state->delta_dual_solution,
        state->num_variables,
        state->num_constraints);

    cupdlpx_spmv_ATx(state->sparse_handle, state->spmv_ctx, state->delta_dual_solution, state->dual_product);

    double interaction, movement;

    double primal_norm = 0.0;
    double dual_norm = 0.0;
    double cross_term = 0.0;

    CUBLAS_CHECK(
        cublasDnrm2_v2_64(state->blas_handle, state->num_constraints, state->delta_dual_solution, 1, &dual_norm));
    CUBLAS_CHECK(
        cublasDnrm2_v2_64(state->blas_handle, state->num_variables, state->delta_primal_solution, 1, &primal_norm));
    movement = primal_norm * primal_norm * state->primal_weight + dual_norm * dual_norm / state->primal_weight;

    CUBLAS_CHECK(cublasDdot(state->blas_handle,
                            state->num_variables,
                            state->dual_product,
                            1,
                            state->delta_primal_solution,
                            1,
                            &cross_term));
    interaction = 2 * state->step_size * cross_term;

    state->fixed_point_error = sqrt(movement + interaction);
}

void pdhg_solver_state_free(pdhg_solver_state_t *state)
{
    if (state == NULL)
    {
        return;
    }

    if (state->spmv_ctx)
        cupdlpx_spmv_ctx_destroy(state->spmv_ctx);
    if (state->sparse_handle)
        CUSPARSE_CHECK(cusparseDestroy(state->sparse_handle));
    if (state->blas_handle)
        CUBLAS_CHECK(cublasDestroy(state->blas_handle));
    if (state->stream)
        CUDA_CHECK(cudaStreamDestroy(state->stream));

    if (state->variable_lower_bound)
        CUDA_CHECK(cudaFree(state->variable_lower_bound));
    if (state->variable_upper_bound)
        CUDA_CHECK(cudaFree(state->variable_upper_bound));
    if (state->objective_vector)
        CUDA_CHECK(cudaFree(state->objective_vector));
    if (state->constraint_matrix->row_ptr)
        CUDA_CHECK(cudaFree(state->constraint_matrix->row_ptr));
    if (state->constraint_matrix->col_ind)
        CUDA_CHECK(cudaFree(state->constraint_matrix->col_ind));
    if (state->constraint_matrix->row_ind)
        CUDA_CHECK(cudaFree(state->constraint_matrix->row_ind));
    if (state->constraint_matrix->val)
        CUDA_CHECK(cudaFree(state->constraint_matrix->val));
    if (state->constraint_matrix->transpose_map)
        CUDA_CHECK(cudaFree(state->constraint_matrix->transpose_map));
    if (state->constraint_matrix_t->row_ptr)
        CUDA_CHECK(cudaFree(state->constraint_matrix_t->row_ptr));
    if (state->constraint_matrix_t->col_ind)
        CUDA_CHECK(cudaFree(state->constraint_matrix_t->col_ind));
    if (state->constraint_matrix_t->row_ind)
        CUDA_CHECK(cudaFree(state->constraint_matrix_t->row_ind));
    if (state->constraint_matrix_t->val)
        CUDA_CHECK(cudaFree(state->constraint_matrix_t->val));
    if (state->constraint_matrix_t->transpose_map)
        CUDA_CHECK(cudaFree(state->constraint_matrix_t->transpose_map));
    if (state->constraint_lower_bound)
        CUDA_CHECK(cudaFree(state->constraint_lower_bound));
    if (state->constraint_upper_bound)
        CUDA_CHECK(cudaFree(state->constraint_upper_bound));
    if (state->constraint_lower_bound_finite_val)
        CUDA_CHECK(cudaFree(state->constraint_lower_bound_finite_val));
    if (state->constraint_upper_bound_finite_val)
        CUDA_CHECK(cudaFree(state->constraint_upper_bound_finite_val));
    if (state->variable_lower_bound_finite_val)
        CUDA_CHECK(cudaFree(state->variable_lower_bound_finite_val));
    if (state->variable_upper_bound_finite_val)
        CUDA_CHECK(cudaFree(state->variable_upper_bound_finite_val));
    if (state->initial_primal_solution)
        CUDA_CHECK(cudaFree(state->initial_primal_solution));
    if (state->current_primal_solution)
        CUDA_CHECK(cudaFree(state->current_primal_solution));
    if (state->pdhg_primal_solution)
        CUDA_CHECK(cudaFree(state->pdhg_primal_solution));
    if (state->reflected_primal_solution)
        CUDA_CHECK(cudaFree(state->reflected_primal_solution));
    if (state->dual_product)
        CUDA_CHECK(cudaFree(state->dual_product));
    if (state->initial_dual_solution)
        CUDA_CHECK(cudaFree(state->initial_dual_solution));
    if (state->current_dual_solution)
        CUDA_CHECK(cudaFree(state->current_dual_solution));
    if (state->pdhg_dual_solution)
        CUDA_CHECK(cudaFree(state->pdhg_dual_solution));
    if (state->reflected_dual_solution)
        CUDA_CHECK(cudaFree(state->reflected_dual_solution));
    if (state->primal_product)
        CUDA_CHECK(cudaFree(state->primal_product));
    if (state->constraint_rescaling)
        CUDA_CHECK(cudaFree(state->constraint_rescaling));
    if (state->variable_rescaling)
        CUDA_CHECK(cudaFree(state->variable_rescaling));
    if (state->primal_slack)
        CUDA_CHECK(cudaFree(state->primal_slack));
    if (state->dual_slack)
        CUDA_CHECK(cudaFree(state->dual_slack));
    if (state->primal_residual)
        CUDA_CHECK(cudaFree(state->primal_residual));
    if (state->dual_residual)
        CUDA_CHECK(cudaFree(state->dual_residual));
    if (state->delta_primal_solution)
        CUDA_CHECK(cudaFree(state->delta_primal_solution));
    if (state->delta_dual_solution)
        CUDA_CHECK(cudaFree(state->delta_dual_solution));
    if (state->ones_primal_d)
        CUDA_CHECK(cudaFree(state->ones_primal_d));
    if (state->ones_dual_d)
        CUDA_CHECK(cudaFree(state->ones_dual_d));
    if (state->d_primal_step_size)
        CUDA_CHECK(cudaFree(state->d_primal_step_size));
    if (state->d_dual_step_size)
        CUDA_CHECK(cudaFree(state->d_dual_step_size));
    if (state->d_inner_count)
        CUDA_CHECK(cudaFree(state->d_inner_count));

    if (state->constraint_matrix)
        free(state->constraint_matrix);
    if (state->constraint_matrix_t)
        free(state->constraint_matrix_t);

    free(state);
}

void rescale_info_free(rescale_info_t *info)
{
    if (info == NULL)
    {
        return;
    }

    CUDA_CHECK(cudaFree(info->con_rescale));
    CUDA_CHECK(cudaFree(info->var_rescale));

    free(info);
}

static cupdlpx_result_t *create_result_from_state(pdhg_solver_state_t *state, const lp_problem_t *original_problem)
{
    cupdlpx_result_t *results = (cupdlpx_result_t *)safe_calloc(1, sizeof(cupdlpx_result_t));

    // Compute reduced cost
    cupdlpx_spmv_ATx(state->sparse_handle, state->spmv_ctx, state->pdhg_dual_solution, state->dual_product);

    compute_and_rescale_reduced_cost_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream>>>(
        state->dual_slack,
        state->objective_vector,
        state->dual_product,
        state->variable_rescaling,
        state->objective_vector_rescaling,
        state->constraint_bound_rescaling,
        state->variable_lower_bound,
        state->variable_upper_bound,
        state->num_variables);

    rescale_solution_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
        state->pdhg_primal_solution,
        state->pdhg_dual_solution,
        state->variable_rescaling,
        state->constraint_rescaling,
        state->objective_vector_rescaling,
        state->constraint_bound_rescaling,
        state->num_variables,
        state->num_constraints);

    results->primal_solution = (double *)safe_malloc(state->num_variables * sizeof(double));
    results->dual_solution = (double *)safe_malloc(state->num_constraints * sizeof(double));
    results->reduced_cost = (double *)safe_malloc(state->num_variables * sizeof(double));

    CUDA_CHECK(cudaMemcpy(results->primal_solution,
                          state->pdhg_primal_solution,
                          state->num_variables * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(results->dual_solution,
                          state->pdhg_dual_solution,
                          state->num_constraints * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        results->reduced_cost, state->dual_slack, state->num_variables * sizeof(double), cudaMemcpyDeviceToHost));

    results->num_variables = original_problem->num_variables;
    results->num_constraints = original_problem->num_constraints;
    results->num_nonzeros = original_problem->constraint_matrix_num_nonzeros;
    results->total_count = state->total_count;
    results->rescaling_time_sec = state->rescaling_time_sec;
    results->cumulative_time_sec = state->cumulative_time_sec;
    results->relative_primal_residual = state->relative_primal_residual;
    results->relative_dual_residual = state->relative_dual_residual;
    results->primal_objective_value = state->primal_objective_value;
    results->dual_objective_value = state->dual_objective_value;
    results->objective_gap = state->objective_gap;
    results->relative_objective_gap = state->relative_objective_gap;
    results->max_primal_ray_infeasibility = state->max_primal_ray_infeasibility;
    results->max_dual_ray_infeasibility = state->max_dual_ray_infeasibility;
    results->primal_ray_linear_objective = state->primal_ray_linear_objective;
    results->dual_ray_objective = state->dual_ray_objective;
    results->termination_reason = state->termination_reason;
    results->feasibility_polishing_time = state->feasibility_polishing_time;
    results->feasibility_iteration = state->feasibility_iteration;
    // if (presolve_stats != NULL) {
    //     results->presolve_stats = *presolve_stats;
    // } else {
    //     memset(&(results->presolve_stats), 0, sizeof(PresolveStats));
    // }

    return results;
}
