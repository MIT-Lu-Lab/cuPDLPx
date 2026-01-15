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
#include "internal_types.h"
#include "preconditioner.h"
#include "presolve.h"
#include "solver.h"
#include "utils.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>

__global__ void compute_next_pdhg_primal_solution_kernel(
    const double *current_primal, double *reflected_primal,
    const double *dual_product, const double *objective, const double *var_lb,
    const double *var_ub, int n, double step_size);
__global__ void compute_next_pdhg_primal_solution_major_kernel(
    const double *current_primal, double *pdhg_primal, double *reflected_primal,
    const double *dual_product, const double *objective, const double *var_lb,
    const double *var_ub, int n, double step_size, double *dual_slack);
__global__ void compute_next_pdhg_dual_solution_kernel(
    const double *current_dual, double *reflected_dual,
    const double *primal_product, const double *const_lb,
    const double *const_ub, int n, double step_size);
__global__ void compute_next_pdhg_dual_solution_major_kernel(
    const double *current_dual, double *pdhg_dual, double *reflected_dual,
    const double *primal_product, const double *const_lb,
    const double *const_ub, int n, double step_size);
__global__ void
halpern_update_kernel(const double *initial_primal, double *current_primal,
                      const double *reflected_primal,
                      const double *initial_dual, double *current_dual,
                      const double *reflected_dual, int n_vars, int n_cons,
                      double weight, double reflection_coeff);
__global__ void rescale_solution_kernel(double *primal_solution,
                                        double *dual_solution,
                                        const double *variable_rescaling,
                                        const double *constraint_rescaling,
                                        const double objective_vector_rescaling,
                                        const double constraint_bound_rescaling,
                                        int n_vars, int n_cons);
__global__ void compute_delta_solution_kernel(
    const double *initial_primal, const double *pdhg_primal,
    double *delta_primal, const double *initial_dual, const double *pdhg_dual,
    double *delta_dual, int n_vars, int n_cons);
static void compute_next_pdhg_primal_solution(pdhg_solver_state_t *state);
static void compute_next_pdhg_dual_solution(pdhg_solver_state_t *state);
static void halpern_update(pdhg_solver_state_t *state,
                           double reflection_coefficient);
static void rescale_solution(pdhg_solver_state_t *state);
static cupdlpx_result_t *create_result_from_state(pdhg_solver_state_t *state, const lp_problem_t *original_problem);
static void perform_restart(pdhg_solver_state_t *state,
                            const pdhg_parameters_t *params);
static void
initialize_step_size_and_primal_weight(pdhg_solver_state_t *state,
                                       const pdhg_parameters_t *params);
static pdhg_solver_state_t *initialize_solver_state(const pdhg_parameters_t *params,
                                                    const lp_problem_t *original_problem,
                                                    const rescale_info_t *rescale_info);
static void compute_fixed_point_error(pdhg_solver_state_t *state);
void pdhg_solver_state_free(pdhg_solver_state_t *state);
void rescale_info_free(rescale_info_t *info);

static void perform_primal_restart(pdhg_solver_state_t *state);
static void perform_dual_restart(pdhg_solver_state_t *state);
void primal_feasibility_polish(const pdhg_parameters_t *params, pdhg_solver_state_t *state, const pdhg_solver_state_t *ori_state);
void dual_feasibility_polish(const pdhg_parameters_t *params, pdhg_solver_state_t *state, const pdhg_solver_state_t *ori_state);
void primal_feas_polish_state_free(pdhg_solver_state_t *state);
void dual_feas_polish_state_free(pdhg_solver_state_t *state);
void feasibility_polish(const pdhg_parameters_t *params, pdhg_solver_state_t *state);
static void compute_primal_fixed_point_error(pdhg_solver_state_t *state);
static void compute_dual_fixed_point_error(pdhg_solver_state_t *state);
static pdhg_solver_state_t *initialize_primal_feas_polish_state(
    const pdhg_solver_state_t *original_state);
static pdhg_solver_state_t *initialize_dual_feas_polish_state(
    const pdhg_solver_state_t *original_state);

cupdlpx_result_t *optimize(const pdhg_parameters_t *params,
                           const lp_problem_t *original_problem)
{
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
    rescale_info_t *rescale_info = rescale_problem(params, working_problem);
    pdhg_solver_state_t *state = initialize_solver_state(params, working_problem, rescale_info);

    rescale_info_free(rescale_info);
    initialize_step_size_and_primal_weight(state, params);
    clock_t start_time = clock();
    bool do_restart = false;
    while (state->total_count < params->termination_criteria.iteration_limit)
    {
        if ((state->is_this_major_iteration || state->total_count == 0) ||
            (state->total_count % get_print_frequency(state->total_count) == 0))
        {
            compute_residual(state, params->optimality_norm);
            if (state->is_this_major_iteration &&
                state->total_count < 3 * params->termination_evaluation_frequency)
            {
                compute_infeasibility_information(state);
            }

            state->cumulative_time_sec =
                (double)(clock() - start_time) / CLOCKS_PER_SEC;

            check_termination_criteria(state, &params->termination_criteria);
            display_iteration_stats(state, params->verbose);
            if (state->termination_reason != TERMINATION_REASON_UNSPECIFIED) {
                break;
            }
        }

        if ((state->is_this_major_iteration || state->total_count == 0))
        {
            do_restart =
                should_do_adaptive_restart(state, &params->restart_params,
                                           params->termination_evaluation_frequency);
            if (do_restart)
                perform_restart(state, params);
        }

        state->is_this_major_iteration =
            ((state->total_count + 1) % params->termination_evaluation_frequency) ==
            0;

        compute_next_pdhg_primal_solution(state);
        compute_next_pdhg_dual_solution(state);

        if (state->is_this_major_iteration || do_restart)
        {
            compute_fixed_point_error(state);
            if (do_restart)
            {
                state->initial_fixed_point_error = state->fixed_point_error;
                do_restart = false;
            }
        }
        halpern_update(state, params->reflection_coefficient);

        state->inner_count++;
        state->total_count++;
    }

    if (state->termination_reason == TERMINATION_REASON_UNSPECIFIED)
    {
        state->termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
        compute_residual(state, params->optimality_norm);
        display_iteration_stats(state, params->verbose);
    }

    if (params->feasibility_polishing &&
        state->termination_reason != TERMINATION_REASON_DUAL_INFEASIBLE &&
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

__global__ void compute_and_rescale_reduced_cost_kernel(
    double *reduced_cost,
    const double *objective,
    const double *dual_product,
    const double *variable_rescaling,
    const double objective_vector_rescaling,
    const double constraint_bound_rescaling,
    int n_vars)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        reduced_cost[i] = (objective[i] - dual_product[i]) * variable_rescaling[i] / objective_vector_rescaling;
    }
}

static pdhg_solver_state_t *
initialize_solver_state(const pdhg_parameters_t *params,
                        const lp_problem_t *working_problem,
                        const rescale_info_t *rescale_info)
{
    pdhg_solver_state_t *state =
        (pdhg_solver_state_t *)safe_calloc(1, sizeof(pdhg_solver_state_t));

    int n_vars = working_problem->num_variables;
    int n_cons = working_problem->num_constraints;
    size_t var_bytes = n_vars * sizeof(double);
    size_t con_bytes = n_cons * sizeof(double);

    state->num_variables = n_vars;
    state->num_constraints = n_cons;
    state->objective_constant = working_problem->objective_constant;

    state->constraint_matrix =
        (cu_sparse_matrix_csr_t *)safe_malloc(sizeof(cu_sparse_matrix_csr_t));
    state->constraint_matrix_t =
        (cu_sparse_matrix_csr_t *)safe_malloc(sizeof(cu_sparse_matrix_csr_t));

    state->constraint_matrix->num_rows = n_cons;
    state->constraint_matrix->num_cols = n_vars;
    state->constraint_matrix->num_nonzeros = working_problem->constraint_matrix_num_nonzeros;

    state->constraint_matrix_t->num_rows = n_vars;
    state->constraint_matrix_t->num_cols = n_cons;
    state->constraint_matrix_t->num_nonzeros = working_problem->constraint_matrix_num_nonzeros;

    state->termination_reason = TERMINATION_REASON_UNSPECIFIED;

    state->rescaling_time_sec = rescale_info->rescaling_time_sec;

#define ALLOC_AND_COPY(dest, src, bytes)  \
    CUDA_CHECK(cudaMalloc(&dest, bytes)); \
    CUDA_CHECK(cudaMemcpy(dest, src, bytes, cudaMemcpyHostToDevice));

    ALLOC_AND_COPY(state->constraint_matrix->row_ptr,
                   rescale_info->scaled_problem->constraint_matrix_row_pointers,
                   (n_cons + 1) * sizeof(int));
    ALLOC_AND_COPY(state->constraint_matrix->col_ind,
                   rescale_info->scaled_problem->constraint_matrix_col_indices,
                   rescale_info->scaled_problem->constraint_matrix_num_nonzeros *
                       sizeof(int));
    ALLOC_AND_COPY(state->constraint_matrix->val,
                   rescale_info->scaled_problem->constraint_matrix_values,
                   rescale_info->scaled_problem->constraint_matrix_num_nonzeros *
                       sizeof(double));

    CUDA_CHECK(cudaMalloc(&state->constraint_matrix_t->row_ptr,
                          (n_vars + 1) * sizeof(int)));
    CUDA_CHECK(
        cudaMalloc(&state->constraint_matrix_t->col_ind,
                   rescale_info->scaled_problem->constraint_matrix_num_nonzeros *
                       sizeof(int)));
    CUDA_CHECK(
        cudaMalloc(&state->constraint_matrix_t->val,
                   rescale_info->scaled_problem->constraint_matrix_num_nonzeros *
                       sizeof(double)));

    CUSPARSE_CHECK(cusparseCreate(&state->sparse_handle));
    CUBLAS_CHECK(cublasCreate(&state->blas_handle));
    CUBLAS_CHECK(
        cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_HOST));

    size_t buffer_size = 0;
    void *buffer = nullptr;
    CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(
        state->sparse_handle, state->constraint_matrix->num_rows,
        state->constraint_matrix->num_cols,
        state->constraint_matrix->num_nonzeros, state->constraint_matrix->val,
        state->constraint_matrix->row_ptr, state->constraint_matrix->col_ind,
        state->constraint_matrix_t->val, state->constraint_matrix_t->row_ptr,
        state->constraint_matrix_t->col_ind, CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG_DEFAULT, &buffer_size));
    CUDA_CHECK(cudaMalloc(&buffer, buffer_size));

    CUSPARSE_CHECK(cusparseCsr2cscEx2(
        state->sparse_handle, state->constraint_matrix->num_rows,
        state->constraint_matrix->num_cols,
        state->constraint_matrix->num_nonzeros, state->constraint_matrix->val,
        state->constraint_matrix->row_ptr, state->constraint_matrix->col_ind,
        state->constraint_matrix_t->val, state->constraint_matrix_t->row_ptr,
        state->constraint_matrix_t->col_ind, CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG_DEFAULT, buffer));

    CUDA_CHECK(cudaFree(buffer));

    ALLOC_AND_COPY(state->variable_lower_bound,
                   rescale_info->scaled_problem->variable_lower_bound, var_bytes);
    ALLOC_AND_COPY(state->variable_upper_bound,
                   rescale_info->scaled_problem->variable_upper_bound, var_bytes);
    ALLOC_AND_COPY(state->objective_vector,
                   rescale_info->scaled_problem->objective_vector, var_bytes);
    ALLOC_AND_COPY(state->constraint_lower_bound,
                   rescale_info->scaled_problem->constraint_lower_bound,
                   con_bytes);
    ALLOC_AND_COPY(state->constraint_upper_bound,
                   rescale_info->scaled_problem->constraint_upper_bound,
                   con_bytes);
    ALLOC_AND_COPY(state->constraint_rescaling, rescale_info->con_rescale,
                   con_bytes);
    ALLOC_AND_COPY(state->variable_rescaling, rescale_info->var_rescale,
                   var_bytes);

    state->constraint_bound_rescaling = rescale_info->con_bound_rescale;
    state->objective_vector_rescaling = rescale_info->obj_vec_rescale;

#define ALLOC_ZERO(dest, bytes)           \
    CUDA_CHECK(cudaMalloc(&dest, bytes)); \
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
        double *rescaled = (double *)safe_malloc(var_bytes);
        for (int i = 0; i < n_vars; ++i)
            rescaled[i] = working_problem->primal_start[i] *
                          rescale_info->var_rescale[i] *
                          rescale_info->con_bound_rescale;
        CUDA_CHECK(cudaMemcpy(state->initial_primal_solution, rescaled, var_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state->current_primal_solution, rescaled, var_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state->pdhg_primal_solution, rescaled, var_bytes,
                              cudaMemcpyHostToDevice));
        free(rescaled);
    }
    if (working_problem->dual_start)
    {
        double *rescaled = (double *)safe_malloc(con_bytes);
        for (int i = 0; i < n_cons; ++i)
            rescaled[i] = working_problem->dual_start[i] *
                          rescale_info->con_rescale[i] *
                          rescale_info->obj_vec_rescale;
        CUDA_CHECK(cudaMemcpy(state->initial_dual_solution, rescaled, con_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state->current_dual_solution, rescaled, con_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state->pdhg_dual_solution, rescaled, con_bytes,
                              cudaMemcpyHostToDevice));
        free(rescaled);
    }

    double *temp_host = (double *)safe_malloc(fmax(var_bytes, con_bytes));
    for (int i = 0; i < n_cons; ++i)
        temp_host[i] =
            isfinite(rescale_info->scaled_problem->constraint_lower_bound[i])
                ? rescale_info->scaled_problem->constraint_lower_bound[i]
                : 0.0;
    ALLOC_AND_COPY(state->constraint_lower_bound_finite_val, temp_host,
                   con_bytes);
    for (int i = 0; i < n_cons; ++i)
        temp_host[i] =
            isfinite(rescale_info->scaled_problem->constraint_upper_bound[i])
                ? rescale_info->scaled_problem->constraint_upper_bound[i]
                : 0.0;
    ALLOC_AND_COPY(state->constraint_upper_bound_finite_val, temp_host,
                   con_bytes);
    for (int i = 0; i < n_vars; ++i)
        temp_host[i] =
            isfinite(rescale_info->scaled_problem->variable_lower_bound[i])
                ? rescale_info->scaled_problem->variable_lower_bound[i]
                : 0.0;
    ALLOC_AND_COPY(state->variable_lower_bound_finite_val, temp_host, var_bytes);
    for (int i = 0; i < n_vars; ++i)
        temp_host[i] =
            isfinite(rescale_info->scaled_problem->variable_upper_bound[i])
                ? rescale_info->scaled_problem->variable_upper_bound[i]
                : 0.0;
    ALLOC_AND_COPY(state->variable_upper_bound_finite_val, temp_host, var_bytes);
    free(temp_host);

    double sum_of_squares = 0.0;
    double max_val = 0.0;
    double val = 0.0;

    for (int i = 0; i < n_vars; ++i)
    {
        if (params->optimality_norm == NORM_TYPE_L_INF) {
            val = fabs(working_problem->objective_vector[i]);
            if (val > max_val) max_val = val;
        } else {
            sum_of_squares += working_problem->objective_vector[i] * working_problem->objective_vector[i];
        }
    }

    if (params->optimality_norm == NORM_TYPE_L_INF) {
        state->objective_vector_norm = max_val;
    } else {
        state->objective_vector_norm = sqrt(sum_of_squares);
    }

    sum_of_squares = 0.0;
    max_val = 0.0;
    val = 0.0;

    for (int i = 0; i < n_cons; ++i)
    {
        double lower = working_problem->constraint_lower_bound[i];
        double upper = working_problem->constraint_upper_bound[i];

        if (params->optimality_norm == NORM_TYPE_L_INF) {
            if (isfinite(lower) && (lower != upper)) {
                val = fabs(lower);
                if (val > max_val) max_val = val;
            }
            if (isfinite(upper)) {
                val = fabs(upper);
                if (val > max_val) max_val = val;
            }
        } else {
            if (isfinite(lower) && (lower != upper)) {
                sum_of_squares += lower * lower;
            }
            if (isfinite(upper)) {
                sum_of_squares += upper * upper;
            }
        }
    }

    if (params->optimality_norm == NORM_TYPE_L_INF) {
        state->constraint_bound_norm = max_val;
    } else {
        state->constraint_bound_norm = sqrt(sum_of_squares);
    }

    state->num_blocks_primal =
        (state->num_variables + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    state->num_blocks_dual =
        (state->num_constraints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    state->num_blocks_primal_dual =
        (state->num_variables + state->num_constraints + THREADS_PER_BLOCK - 1) /
        THREADS_PER_BLOCK;

    state->best_primal_dual_residual_gap = INFINITY;
    state->last_trial_fixed_point_error = INFINITY;
    state->step_size = 0.0;
    state->is_this_major_iteration = false;

    size_t primal_spmv_buffer_size;
    size_t dual_spmv_buffer_size;

    CUSPARSE_CHECK(cusparseCreateCsr(
        &state->matA, state->num_constraints, state->num_variables,
        state->constraint_matrix->num_nonzeros, state->constraint_matrix->row_ptr,
        state->constraint_matrix->col_ind, state->constraint_matrix->val,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F));

    CUDA_CHECK(cudaGetLastError());

    CUSPARSE_CHECK(cusparseCreateCsr(
        &state->matAt, state->num_variables, state->num_constraints,
        state->constraint_matrix_t->num_nonzeros,
        state->constraint_matrix_t->row_ptr, state->constraint_matrix_t->col_ind,
        state->constraint_matrix_t->val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CUDA_CHECK(cudaGetLastError());

    CUSPARSE_CHECK(cusparseCreateDnVec(&state->vec_primal_sol,
                                       state->num_variables,
                                       state->pdhg_primal_solution, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&state->vec_dual_sol,
                                       state->num_constraints,
                                       state->pdhg_dual_solution, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&state->vec_primal_prod,
                                       state->num_constraints,
                                       state->primal_product, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&state->vec_dual_prod,
                                       state->num_variables, state->dual_product,
                                       CUDA_R_64F));
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
        state->matA, state->vec_primal_sol, &HOST_ZERO, state->vec_primal_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &primal_spmv_buffer_size));

    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
        state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &dual_spmv_buffer_size));
    CUDA_CHECK(cudaMalloc(&state->primal_spmv_buffer, primal_spmv_buffer_size));
    CUSPARSE_CHECK(cusparseSpMV_preprocess(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
        state->matA, state->vec_primal_sol, &HOST_ZERO, state->vec_primal_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->primal_spmv_buffer));

    CUDA_CHECK(cudaMalloc(&state->dual_spmv_buffer, dual_spmv_buffer_size));
    CUSPARSE_CHECK(cusparseSpMV_preprocess(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
        state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->dual_spmv_buffer));

    CUDA_CHECK(
        cudaMalloc(&state->ones_primal_d, state->num_variables * sizeof(double)));
    CUDA_CHECK(
        cudaMalloc(&state->ones_dual_d, state->num_constraints * sizeof(double)));

    double *ones_primal_h =
        (double *)safe_malloc(state->num_variables * sizeof(double));
    for (int i = 0; i < state->num_variables; ++i)
        ones_primal_h[i] = 1.0;
    CUDA_CHECK(cudaMemcpy(state->ones_primal_d, ones_primal_h,
                          state->num_variables * sizeof(double),
                          cudaMemcpyHostToDevice));
    free(ones_primal_h);

    double *ones_dual_h =
        (double *)safe_malloc(state->num_constraints * sizeof(double));
    for (int i = 0; i < state->num_constraints; ++i)
        ones_dual_h[i] = 1.0;
    CUDA_CHECK(cudaMemcpy(state->ones_dual_d, ones_dual_h,
                          state->num_constraints * sizeof(double),
                          cudaMemcpyHostToDevice));
    free(ones_dual_h);
    if (params->verbose) {
        printf("---------------------------------------------------------------------"
            "------------------\n");
        printf("%s | %s | %s | %s \n", "   runtime    ", "    objective     ",
            "  absolute residuals   ", "  relative residuals   ");
        printf("%s %s | %s %s | %s %s %s | %s %s %s \n", "  iter", "  time ",
            " pr obj ", "  du obj ", " pr res", " du res", "  gap  ", " pr res",
            " du res", "  gap  ");
        printf("---------------------------------------------------------------------"
            "------------------\n");
    }

    return state;
}

__global__ void compute_next_pdhg_primal_solution_kernel(
    const double *current_primal, double *reflected_primal,
    const double *dual_product, const double *objective, const double *var_lb,
    const double *var_ub, int n, double step_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double temp =
            current_primal[i] - step_size * (objective[i] - dual_product[i]);
        double temp_proj = fmax(var_lb[i], fmin(temp, var_ub[i]));
        reflected_primal[i] = 2.0 * temp_proj - current_primal[i];
    }
}

__global__ void compute_next_pdhg_primal_solution_major_kernel(
    const double *current_primal, double *pdhg_primal, double *reflected_primal,
    const double *dual_product, const double *objective, const double *var_lb,
    const double *var_ub, int n, double step_size, double *dual_slack)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double temp =
            current_primal[i] - step_size * (objective[i] - dual_product[i]);
        pdhg_primal[i] = fmax(var_lb[i], fmin(temp, var_ub[i]));
        dual_slack[i] = (pdhg_primal[i] - temp) / step_size;
        reflected_primal[i] = 2.0 * pdhg_primal[i] - current_primal[i];
    }
}

__global__ void compute_next_pdhg_dual_solution_kernel(
    const double *current_dual, double *reflected_dual,
    const double *primal_product, const double *const_lb,
    const double *const_ub, int n, double step_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double temp = current_dual[i] / step_size - primal_product[i];
        double temp_proj = fmax(-const_ub[i], fmin(temp, -const_lb[i]));
        reflected_dual[i] = 2.0 * (temp - temp_proj) * step_size - current_dual[i];
    }
}

__global__ void compute_next_pdhg_dual_solution_major_kernel(
    const double *current_dual, double *pdhg_dual, double *reflected_dual,
    const double *primal_product, const double *const_lb,
    const double *const_ub, int n, double step_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double temp = current_dual[i] / step_size - primal_product[i];
        double temp_proj = fmax(-const_ub[i], fmin(temp, -const_lb[i]));
        pdhg_dual[i] = (temp - temp_proj) * step_size;
        reflected_dual[i] = 2.0 * pdhg_dual[i] - current_dual[i];
    }
}

__global__ void
halpern_update_kernel(const double *initial_primal, double *current_primal,
                      const double *reflected_primal,
                      const double *initial_dual, double *current_dual,
                      const double *reflected_dual, int n_vars, int n_cons,
                      double weight, double reflection_coeff)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        double reflected = reflection_coeff * reflected_primal[i] +
                           (1.0 - reflection_coeff) * current_primal[i];
        current_primal[i] = weight * reflected + (1.0 - weight) * initial_primal[i];
    }
    else if (i < n_vars + n_cons)
    {
        int idx = i - n_vars;
        double reflected = reflection_coeff * reflected_dual[idx] +
                           (1.0 - reflection_coeff) * current_dual[idx];
        current_dual[idx] = weight * reflected + (1.0 - weight) * initial_dual[idx];
    }
}

__global__ void rescale_solution_kernel(double *primal_solution,
                                        double *dual_solution,
                                        const double *variable_rescaling,
                                        const double *constraint_rescaling,
                                        const double objective_vector_rescaling,
                                        const double constraint_bound_rescaling,
                                        int n_vars, int n_cons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        primal_solution[i] =
            primal_solution[i] / variable_rescaling[i] / constraint_bound_rescaling;
    }
    else if (i < n_vars + n_cons)
    {
        int idx = i - n_vars;
        dual_solution[idx] = dual_solution[idx] / constraint_rescaling[idx] /
                             objective_vector_rescaling;
    }
}

__global__ void compute_delta_solution_kernel(
    const double *initial_primal, const double *pdhg_primal,
    double *delta_primal, const double *initial_dual, const double *pdhg_dual,
    double *delta_dual, int n_vars, int n_cons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        delta_primal[i] = pdhg_primal[i] - initial_primal[i];
    }
    else if (i < n_vars + n_cons)
    {
        int idx = i - n_vars;
        delta_dual[idx] = pdhg_dual[idx] - initial_dual[idx];
    }
}

static void compute_next_pdhg_primal_solution(pdhg_solver_state_t *state)
{
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_sol,
                                          state->current_dual_solution));
    CUSPARSE_CHECK(
        cusparseDnVecSetValues(state->vec_dual_prod, state->dual_product));

    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
        state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->dual_spmv_buffer));

    double step = state->step_size / state->primal_weight;

    if (state->is_this_major_iteration ||
        ((state->total_count + 2) %
         get_print_frequency(state->total_count + 2)) == 0)
    {
        compute_next_pdhg_primal_solution_major_kernel<<<state->num_blocks_primal,
                                                         THREADS_PER_BLOCK>>>(
            state->current_primal_solution, state->pdhg_primal_solution,
            state->reflected_primal_solution, state->dual_product,
            state->objective_vector, state->variable_lower_bound,
            state->variable_upper_bound, state->num_variables, step,
            state->dual_slack);
    }
    else
    {
        compute_next_pdhg_primal_solution_kernel<<<state->num_blocks_primal,
                                                   THREADS_PER_BLOCK>>>(
            state->current_primal_solution, state->reflected_primal_solution,
            state->dual_product, state->objective_vector,
            state->variable_lower_bound, state->variable_upper_bound,
            state->num_variables, step);
    }
}

static void compute_next_pdhg_dual_solution(pdhg_solver_state_t *state)
{
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_sol,
                                          state->reflected_primal_solution));
    CUSPARSE_CHECK(
        cusparseDnVecSetValues(state->vec_primal_prod, state->primal_product));

    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
        state->matA, state->vec_primal_sol, &HOST_ZERO, state->vec_primal_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->primal_spmv_buffer));

    double step = state->step_size * state->primal_weight;

    if (state->is_this_major_iteration ||
        ((state->total_count + 2) %
         get_print_frequency(state->total_count + 2)) == 0)
    {
        compute_next_pdhg_dual_solution_major_kernel<<<state->num_blocks_dual,
                                                       THREADS_PER_BLOCK>>>(
            state->current_dual_solution, state->pdhg_dual_solution,
            state->reflected_dual_solution, state->primal_product,
            state->constraint_lower_bound, state->constraint_upper_bound,
            state->num_constraints, step);
    }
    else
    {
        compute_next_pdhg_dual_solution_kernel<<<state->num_blocks_dual,
                                                 THREADS_PER_BLOCK>>>(
            state->current_dual_solution, state->reflected_dual_solution,
            state->primal_product, state->constraint_lower_bound,
            state->constraint_upper_bound, state->num_constraints, step);
    }
}

static void halpern_update(pdhg_solver_state_t *state,
                           double reflection_coefficient)
{
    double weight = (double)(state->inner_count + 1) / (state->inner_count + 2);
    halpern_update_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK>>>(
        state->initial_primal_solution, state->current_primal_solution,
        state->reflected_primal_solution, state->initial_dual_solution,
        state->current_dual_solution, state->reflected_dual_solution,
        state->num_variables, state->num_constraints, weight,
        reflection_coefficient);
}

static void rescale_solution(pdhg_solver_state_t *state)
{
    rescale_solution_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK>>>(
        state->pdhg_primal_solution, state->pdhg_dual_solution,
        state->variable_rescaling, state->constraint_rescaling,
        state->objective_vector_rescaling, state->constraint_bound_rescaling,
        state->num_variables, state->num_constraints);
}

static void perform_restart(pdhg_solver_state_t *state,
                            const pdhg_parameters_t *params)
{
    compute_delta_solution_kernel<<<state->num_blocks_primal_dual,
                                    THREADS_PER_BLOCK>>>(
        state->initial_primal_solution, state->pdhg_primal_solution,
        state->delta_primal_solution, state->initial_dual_solution,
        state->pdhg_dual_solution, state->delta_dual_solution,
        state->num_variables, state->num_constraints);

    double primal_dist, dual_dist;
    CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_variables,
                                   state->delta_primal_solution, 1,
                                   &primal_dist));
    CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_constraints,
                                   state->delta_dual_solution, 1, &dual_dist));

    double ratio_infeas =
        state->relative_dual_residual / state->relative_primal_residual;

    if (primal_dist > 1e-16 && dual_dist > 1e-16 && primal_dist < 1e12 &&
        dual_dist < 1e12 && ratio_infeas > 1e-8 && ratio_infeas < 1e8)
    {
        double error =
            log(dual_dist) - log(primal_dist) - log(state->primal_weight);
        state->primal_weight_error_sum *= params->restart_params.i_smooth;
        state->primal_weight_error_sum += error;
        double delta_error = error - state->primal_weight_last_error;
        state->primal_weight *=
            exp(params->restart_params.k_p * error +
                params->restart_params.k_i * state->primal_weight_error_sum +
                params->restart_params.k_d * delta_error);
        state->primal_weight_last_error = error;
    }
    else
    {
        state->primal_weight = state->best_primal_weight;
        state->primal_weight_error_sum = 0.0;
        state->primal_weight_last_error = 0.0;
    }

    double primal_dual_residual_gap = abs(
        log10(state->relative_dual_residual / state->relative_primal_residual));
    if (primal_dual_residual_gap < state->best_primal_dual_residual_gap)
    {
        state->best_primal_dual_residual_gap = primal_dual_residual_gap;
        state->best_primal_weight = state->primal_weight;
    }

    CUDA_CHECK(cudaMemcpy(
        state->initial_primal_solution, state->pdhg_primal_solution,
        state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(
        state->current_primal_solution, state->pdhg_primal_solution,
        state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->initial_dual_solution, state->pdhg_dual_solution,
                          state->num_constraints * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->current_dual_solution, state->pdhg_dual_solution,
                          state->num_constraints * sizeof(double),
                          cudaMemcpyDeviceToDevice));

    state->inner_count = 0;
    state->last_trial_fixed_point_error = INFINITY;
}

static void
initialize_step_size_and_primal_weight(pdhg_solver_state_t *state,
                                       const pdhg_parameters_t *params)
{
    if (state->constraint_matrix->num_nonzeros == 0)
    {
        state->step_size = 1.0;
    }
    else
    {
        double max_sv = estimate_maximum_singular_value(
            state->sparse_handle, state->blas_handle, state->constraint_matrix,
            state->constraint_matrix_t, params->sv_max_iter, params->sv_tol);
        state->step_size = 0.998 / max_sv;
    }

    if (params->bound_objective_rescaling)
    {
        state->primal_weight = 1.0;
    }
    else
    {
        state->primal_weight = (state->objective_vector_norm + 1.0) /
                               (state->constraint_bound_norm + 1.0);
    }
    state->best_primal_weight = state->primal_weight;
}

static void compute_fixed_point_error(pdhg_solver_state_t *state)
{
    compute_delta_solution_kernel<<<state->num_blocks_primal_dual,
                                    THREADS_PER_BLOCK>>>(
        state->current_primal_solution, state->reflected_primal_solution,
        state->delta_primal_solution, state->current_dual_solution,
        state->reflected_dual_solution, state->delta_dual_solution,
        state->num_variables, state->num_constraints);

    CUSPARSE_CHECK(
        cusparseDnVecSetValues(state->vec_dual_sol, state->delta_dual_solution));
    CUSPARSE_CHECK(
        cusparseDnVecSetValues(state->vec_dual_prod, state->dual_product));

    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
        state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->dual_spmv_buffer));

    double interaction, movement;

    double primal_norm = 0.0;
    double dual_norm = 0.0;
    double cross_term = 0.0;

    CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_constraints,
                                   state->delta_dual_solution, 1, &dual_norm));
    CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_variables,
                                   state->delta_primal_solution, 1,
                                   &primal_norm));
    movement = primal_norm * primal_norm * state->primal_weight +
               dual_norm * dual_norm / state->primal_weight;

    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_variables,
                            state->dual_product, 1, state->delta_primal_solution,
                            1, &cross_term));
    interaction = 2 * state->step_size * cross_term;

    state->fixed_point_error = sqrt(movement + interaction);
}

void pdhg_solver_state_free(pdhg_solver_state_t *state)
{
    if (state == NULL)
    {
        return;
    }

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
    if (state->constraint_matrix->val)
        CUDA_CHECK(cudaFree(state->constraint_matrix->val));
    if (state->constraint_matrix_t->row_ptr)
        CUDA_CHECK(cudaFree(state->constraint_matrix_t->row_ptr));
    if (state->constraint_matrix_t->col_ind)
        CUDA_CHECK(cudaFree(state->constraint_matrix_t->col_ind));
    if (state->constraint_matrix_t->val)
        CUDA_CHECK(cudaFree(state->constraint_matrix_t->val));
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

    free(state);
}

void rescale_info_free(rescale_info_t *info)
{
    if (info == NULL)
    {
        return;
    }

    lp_problem_free(info->scaled_problem);
    free(info->con_rescale);
    free(info->var_rescale);

    free(info);
}

static cupdlpx_result_t *create_result_from_state(pdhg_solver_state_t *state, const lp_problem_t *original_problem)
{
    cupdlpx_result_t *results =
        (cupdlpx_result_t *)safe_calloc(1, sizeof(cupdlpx_result_t));

    // Compute reduced cost
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_sol,
                                          state->pdhg_dual_solution));
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_prod,
                                          state->dual_product));

    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
        state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->dual_spmv_buffer));

    compute_and_rescale_reduced_cost_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->dual_slack,
        state->objective_vector,
        state->dual_product,
        state->variable_rescaling,
        state->objective_vector_rescaling,
        state->constraint_bound_rescaling,
        state->num_variables);

    rescale_solution(state);

    results->primal_solution =
        (double *)safe_malloc(state->num_variables * sizeof(double));
    results->dual_solution =
        (double *)safe_malloc(state->num_constraints * sizeof(double));
    results->reduced_cost =
        (double *)safe_malloc(state->num_variables * sizeof(double));

    CUDA_CHECK(cudaMemcpy(results->primal_solution, state->pdhg_primal_solution,
                          state->num_variables * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(results->dual_solution, state->pdhg_dual_solution,
                          state->num_constraints * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(results->reduced_cost, state->dual_slack,
                          state->num_variables * sizeof(double),
                          cudaMemcpyDeviceToHost));

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

// Feasibility Polishing
void feasibility_polish(const pdhg_parameters_t *params, pdhg_solver_state_t *state)
{
    clock_t start_time = clock();
    if (state->relative_primal_residual < params->termination_criteria.eps_feas_polish_relative &&
        state->relative_dual_residual < params->termination_criteria.eps_feas_polish_relative)
    {
        printf("Skipping feasibility polishing as the solution is already sufficiently feasible.\n");
        return;
    }
    double original_primal_weight = 0.0;
    if (params->bound_objective_rescaling)
    {
        original_primal_weight = 1.0;
    }
    else
    {
        original_primal_weight = (state->objective_vector_norm + 1.0) / (state->constraint_bound_norm + 1.0);
    }

    // PRIMAL FEASIBILITY POLISHING
    pdhg_solver_state_t *primal_state = initialize_primal_feas_polish_state(state);
    primal_state->primal_weight = original_primal_weight;
    primal_state->best_primal_weight = original_primal_weight;
    primal_feasibility_polish(params, primal_state, state);

    if (primal_state->termination_reason == TERMINATION_REASON_FEAS_POLISH_SUCCESS)
    {
        CUDA_CHECK(cudaMemcpy(
            state->pdhg_primal_solution, primal_state->pdhg_primal_solution,
            state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
        state->absolute_primal_residual = primal_state->absolute_primal_residual;
        state->relative_primal_residual = primal_state->relative_primal_residual;
        state->primal_objective_value = primal_state->primal_objective_value;
    }
    state->feasibility_iteration += primal_state->total_count - 1;

    // DUAL FEASIBILITY POLISHING
    pdhg_solver_state_t *dual_state = initialize_dual_feas_polish_state(state);
    dual_state->primal_weight = original_primal_weight;
    dual_state->best_primal_weight = original_primal_weight;
    dual_feasibility_polish(params, dual_state, state);

    if (dual_state->termination_reason == TERMINATION_REASON_FEAS_POLISH_SUCCESS)
    {
        CUDA_CHECK(cudaMemcpy(
            state->pdhg_dual_solution, dual_state->pdhg_dual_solution,
            state->num_constraints * sizeof(double), cudaMemcpyDeviceToDevice));
        state->absolute_dual_residual = dual_state->absolute_dual_residual;
        state->relative_dual_residual = dual_state->relative_dual_residual;
        state->dual_objective_value = dual_state->dual_objective_value;
    }
    state->feasibility_iteration += dual_state->total_count - 1;

    state->objective_gap = fabs(state->primal_objective_value - state->dual_objective_value);
    state->relative_objective_gap = state->objective_gap / (1.0 + fabs(state->primal_objective_value) + fabs(state->dual_objective_value));

    // FINAL LOGGING
    pdhg_feas_polish_final_log(primal_state, dual_state, params->verbose);
    primal_feas_polish_state_free(primal_state);
    dual_feas_polish_state_free(dual_state);

    state->feasibility_polishing_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    return;
}

void primal_feasibility_polish(const pdhg_parameters_t *params, pdhg_solver_state_t *state, const pdhg_solver_state_t *ori_state)
{
    print_initial_feas_polish_info(true, params);
    clock_t start_time = clock();
    bool do_restart = false;
    while (state->termination_reason == TERMINATION_REASON_UNSPECIFIED)
    {
        if ((state->is_this_major_iteration || state->total_count == 0) || (state->total_count % get_print_frequency(state->total_count) == 0))
        {
            compute_primal_feas_polish_residual(state, ori_state, params->optimality_norm);

            state->cumulative_time_sec = (double)(clock() - start_time) / CLOCKS_PER_SEC;

            check_feas_polishing_termination_criteria(state, &params->termination_criteria, true);
            display_feas_polish_iteration_stats(state, params->verbose, true);
        }

        if ((state->is_this_major_iteration || state->total_count == 0))
        {
            do_restart = should_do_adaptive_restart(state, &params->restart_params, params->termination_evaluation_frequency);
            if (do_restart)
                perform_primal_restart(state);
        }

        state->is_this_major_iteration = ((state->total_count + 1) % params->termination_evaluation_frequency) == 0;

        compute_next_pdhg_primal_solution(state);
        compute_next_pdhg_dual_solution(state);

        if (state->is_this_major_iteration || do_restart)
        {
            compute_primal_fixed_point_error(state);
            if (do_restart)
            {
                state->initial_fixed_point_error = state->fixed_point_error;
                do_restart = false;
            }
        }
        halpern_update(state, params->reflection_coefficient);

        state->inner_count++;
        state->total_count++;
    }
    return;
}

void dual_feasibility_polish(const pdhg_parameters_t *params, pdhg_solver_state_t *state, const pdhg_solver_state_t *ori_state)
{
    print_initial_feas_polish_info(false, params);
    clock_t start_time = clock();
    bool do_restart = false;
    while (state->termination_reason == TERMINATION_REASON_UNSPECIFIED)
    {
        if ((state->is_this_major_iteration || state->total_count == 0) || (state->total_count % get_print_frequency(state->total_count) == 0))
        {
            compute_dual_feas_polish_residual(state, ori_state, params->optimality_norm);

            state->cumulative_time_sec = (double)(clock() - start_time) / CLOCKS_PER_SEC;

            check_feas_polishing_termination_criteria(state, &params->termination_criteria, false);
            display_feas_polish_iteration_stats(state, params->verbose, false);
        }

        if ((state->is_this_major_iteration || state->total_count == 0))
        {
            do_restart = should_do_adaptive_restart(state, &params->restart_params, params->termination_evaluation_frequency);
            if (do_restart)
                perform_dual_restart(state);
        }

        state->is_this_major_iteration = ((state->total_count + 1) % params->termination_evaluation_frequency) == 0;

        compute_next_pdhg_primal_solution(state);
        compute_next_pdhg_dual_solution(state);

        if (state->is_this_major_iteration || do_restart)
        {
            compute_dual_fixed_point_error(state);
            if (do_restart)
            {
                state->initial_fixed_point_error = state->fixed_point_error;
                do_restart = false;
            }
        }
        halpern_update(state, params->reflection_coefficient);

        state->inner_count++;
        state->total_count++;
    }
    return;
}

static pdhg_solver_state_t *initialize_primal_feas_polish_state(
    const pdhg_solver_state_t *original_state)
{
    pdhg_solver_state_t *primal_state = (pdhg_solver_state_t *)safe_malloc(sizeof(pdhg_solver_state_t));
    *primal_state = *original_state;
    int num_var = original_state->num_variables;
    int num_cons = original_state->num_constraints;

#define ALLOC_ZERO(dest, bytes)           \
    CUDA_CHECK(cudaMalloc(&dest, bytes)); \
    CUDA_CHECK(cudaMemset(dest, 0, bytes));

    // RESET PROBLEM TO FEASIBILITY PROBLEM
    ALLOC_ZERO(primal_state->objective_vector, num_var * sizeof(double));
    primal_state->objective_constant = 0.0;

#define ALLOC_AND_COPY_DEV(dest, src, bytes) \
    CUDA_CHECK(cudaMalloc(&dest, bytes));    \
    CUDA_CHECK(cudaMemcpy(dest, src, bytes, cudaMemcpyDeviceToDevice));

    // ALLOCATE AND COPY SOLUTION VECTORS
    ALLOC_AND_COPY_DEV(primal_state->initial_primal_solution, original_state->initial_primal_solution, num_var * sizeof(double));
    ALLOC_AND_COPY_DEV(primal_state->current_primal_solution, original_state->current_primal_solution, num_var * sizeof(double));
    ALLOC_AND_COPY_DEV(primal_state->pdhg_primal_solution, original_state->pdhg_primal_solution, num_var * sizeof(double));
    ALLOC_AND_COPY_DEV(primal_state->reflected_primal_solution, original_state->reflected_primal_solution, num_var * sizeof(double));
    ALLOC_AND_COPY_DEV(primal_state->primal_product, original_state->primal_product, num_cons * sizeof(double));

    // ALLOC ZERO FOR OTHERS
    ALLOC_ZERO(primal_state->initial_dual_solution, num_cons * sizeof(double));
    ALLOC_ZERO(primal_state->current_dual_solution, num_cons * sizeof(double));
    ALLOC_ZERO(primal_state->pdhg_dual_solution, num_cons * sizeof(double));
    ALLOC_ZERO(primal_state->reflected_dual_solution, num_cons * sizeof(double));
    ALLOC_ZERO(primal_state->dual_product, num_var * sizeof(double));

    ALLOC_ZERO(primal_state->dual_slack, num_var * sizeof(double));
    ALLOC_ZERO(primal_state->primal_slack, num_cons * sizeof(double));
    ALLOC_ZERO(primal_state->dual_residual, num_var * sizeof(double));
    ALLOC_ZERO(primal_state->primal_residual, num_cons * sizeof(double));
    ALLOC_ZERO(primal_state->delta_primal_solution, num_var * sizeof(double));
    ALLOC_ZERO(primal_state->delta_dual_solution, num_cons * sizeof(double));

    // RESET SCALAR
    primal_state->primal_weight_error_sum = 0.0;
    primal_state->primal_weight_last_error = 0.0;
    primal_state->best_primal_weight = 0.0;
    primal_state->fixed_point_error = INFINITY;
    primal_state->initial_fixed_point_error = INFINITY;
    primal_state->last_trial_fixed_point_error = INFINITY;
    primal_state->step_size = original_state->step_size;
    primal_state->primal_weight = original_state->primal_weight;
    primal_state->is_this_major_iteration = false;
    primal_state->total_count = 0;
    primal_state->inner_count = 0;
    primal_state->termination_reason = TERMINATION_REASON_UNSPECIFIED;
    primal_state->cumulative_time_sec = 0.0;
    primal_state->best_primal_dual_residual_gap = INFINITY;

    // IGNORE DUAL RESIDUAL AND OBJECTIVE GAP
    primal_state->relative_dual_residual = 0.0;
    primal_state->absolute_dual_residual = 0.0;
    primal_state->relative_objective_gap = 0.0;
    primal_state->objective_gap = 0.0;

    return primal_state;
}

void primal_feas_polish_state_free(pdhg_solver_state_t *state)
{
#define SAFE_CUDA_FREE(p)        \
    if ((p) != NULL)             \
    {                            \
        CUDA_CHECK(cudaFree(p)); \
        (p) = NULL;              \
    }

    if (!state)
        return;
    SAFE_CUDA_FREE(state->objective_vector);
    SAFE_CUDA_FREE(state->initial_primal_solution);
    SAFE_CUDA_FREE(state->current_primal_solution);
    SAFE_CUDA_FREE(state->pdhg_primal_solution);
    SAFE_CUDA_FREE(state->reflected_primal_solution);
    SAFE_CUDA_FREE(state->dual_product);
    SAFE_CUDA_FREE(state->initial_dual_solution);
    SAFE_CUDA_FREE(state->current_dual_solution);
    SAFE_CUDA_FREE(state->pdhg_dual_solution);
    SAFE_CUDA_FREE(state->reflected_dual_solution);
    SAFE_CUDA_FREE(state->primal_product);
    SAFE_CUDA_FREE(state->primal_slack);
    SAFE_CUDA_FREE(state->dual_slack);
    SAFE_CUDA_FREE(state->primal_residual);
    SAFE_CUDA_FREE(state->dual_residual);
    SAFE_CUDA_FREE(state->delta_primal_solution);
    SAFE_CUDA_FREE(state->delta_dual_solution);
    free(state);
}

__global__ void zero_finite_value_vectors_kernel(
    double *__restrict__ vec, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        if (isfinite(vec[idx]))
            vec[idx] = 0.0;
    }
}

static pdhg_solver_state_t *initialize_dual_feas_polish_state(
    const pdhg_solver_state_t *original_state)
{
    pdhg_solver_state_t *dual_state = (pdhg_solver_state_t *)safe_malloc(sizeof(pdhg_solver_state_t));
    *dual_state = *original_state;
    int num_var = original_state->num_variables;
    int num_cons = original_state->num_constraints;

#define ALLOC_AND_COPY_DEV(dest, src, bytes) \
    CUDA_CHECK(cudaMalloc(&dest, bytes));    \
    CUDA_CHECK(cudaMemcpy(dest, src, bytes, cudaMemcpyDeviceToDevice));

// RESET PROBLEM TO DUAL FEASIBILITY PROBLEM
#define SET_FINITE_TO_ZERO(vec, n)                                     \
    {                                                                  \
        int threads = 256;                                             \
        int blocks = (n + threads - 1) / threads;                      \
        zero_finite_value_vectors_kernel<<<blocks, threads>>>(vec, n); \
        CUDA_CHECK(cudaDeviceSynchronize());                           \
    }

    ALLOC_AND_COPY_DEV(dual_state->constraint_lower_bound, original_state->constraint_lower_bound, num_cons * sizeof(double));
    ALLOC_AND_COPY_DEV(dual_state->constraint_upper_bound, original_state->constraint_upper_bound, num_cons * sizeof(double));
    ALLOC_AND_COPY_DEV(dual_state->variable_lower_bound, original_state->variable_lower_bound, num_var * sizeof(double));
    ALLOC_AND_COPY_DEV(dual_state->variable_upper_bound, original_state->variable_upper_bound, num_var * sizeof(double));

    SET_FINITE_TO_ZERO(dual_state->constraint_lower_bound, num_cons);
    SET_FINITE_TO_ZERO(dual_state->constraint_upper_bound, num_cons);
    SET_FINITE_TO_ZERO(dual_state->variable_lower_bound, num_var);
    SET_FINITE_TO_ZERO(dual_state->variable_upper_bound, num_var);

#define ALLOC_ZERO(dest, bytes)           \
    CUDA_CHECK(cudaMalloc(&dest, bytes)); \
    CUDA_CHECK(cudaMemset(dest, 0, bytes));

    ALLOC_ZERO(dual_state->constraint_lower_bound_finite_val, num_cons * sizeof(double));
    ALLOC_ZERO(dual_state->constraint_upper_bound_finite_val, num_cons * sizeof(double));
    ALLOC_ZERO(dual_state->variable_lower_bound_finite_val, num_var * sizeof(double));
    ALLOC_ZERO(dual_state->variable_upper_bound_finite_val, num_var * sizeof(double));

    // ALLOCATE AND COPY SOLUTION VECTORS
    ALLOC_AND_COPY_DEV(dual_state->initial_dual_solution, original_state->initial_dual_solution, num_cons * sizeof(double));
    ALLOC_AND_COPY_DEV(dual_state->current_dual_solution, original_state->current_dual_solution, num_cons * sizeof(double));
    ALLOC_AND_COPY_DEV(dual_state->pdhg_dual_solution, original_state->pdhg_dual_solution, num_cons * sizeof(double));
    ALLOC_AND_COPY_DEV(dual_state->reflected_dual_solution, original_state->reflected_dual_solution, num_cons * sizeof(double));
    ALLOC_AND_COPY_DEV(dual_state->dual_product, original_state->dual_product, num_var * sizeof(double));
    ALLOC_AND_COPY_DEV(dual_state->dual_slack, original_state->dual_slack, num_var * sizeof(double));

    // ALLOC ZERO FOR OTHERS
    ALLOC_ZERO(dual_state->initial_primal_solution, num_var * sizeof(double));
    ALLOC_ZERO(dual_state->current_primal_solution, num_var * sizeof(double));
    ALLOC_ZERO(dual_state->pdhg_primal_solution, num_var * sizeof(double));
    ALLOC_ZERO(dual_state->reflected_primal_solution, num_var * sizeof(double));
    ALLOC_ZERO(dual_state->primal_product, num_cons * sizeof(double));
    ALLOC_ZERO(dual_state->primal_slack, num_cons * sizeof(double));
    ALLOC_ZERO(dual_state->dual_residual, num_var * sizeof(double));
    ALLOC_ZERO(dual_state->primal_residual, num_cons * sizeof(double));
    ALLOC_ZERO(dual_state->delta_primal_solution, num_var * sizeof(double));
    ALLOC_ZERO(dual_state->delta_dual_solution, num_cons * sizeof(double));

    // RESET SCALAR
    dual_state->primal_weight_error_sum = 0.0;
    dual_state->primal_weight_last_error = 0.0;
    dual_state->best_primal_weight = 0.0;
    dual_state->fixed_point_error = INFINITY;
    dual_state->initial_fixed_point_error = INFINITY;
    dual_state->last_trial_fixed_point_error = INFINITY;
    dual_state->step_size = original_state->step_size;
    dual_state->primal_weight = original_state->primal_weight;
    dual_state->is_this_major_iteration = false;
    dual_state->total_count = 0;
    dual_state->inner_count = 0;
    dual_state->termination_reason = TERMINATION_REASON_UNSPECIFIED;
    dual_state->cumulative_time_sec = 0.0;
    dual_state->best_primal_dual_residual_gap = INFINITY;

    // IGNORE PRIMAL RESIDUAL AND OBJECTIVE GAP
    dual_state->relative_primal_residual = 0.0;
    dual_state->absolute_primal_residual = 0.0;
    dual_state->relative_objective_gap = 0.0;
    dual_state->objective_gap = 0.0;
    return dual_state;
}

void dual_feas_polish_state_free(pdhg_solver_state_t *state)
{
#define SAFE_CUDA_FREE(p)        \
    if ((p) != NULL)             \
    {                            \
        CUDA_CHECK(cudaFree(p)); \
        (p) = NULL;              \
    }

    if (!state)
        return;
    SAFE_CUDA_FREE(state->constraint_lower_bound);
    SAFE_CUDA_FREE(state->constraint_upper_bound);
    SAFE_CUDA_FREE(state->variable_lower_bound);
    SAFE_CUDA_FREE(state->variable_upper_bound);
    SAFE_CUDA_FREE(state->constraint_lower_bound_finite_val);
    SAFE_CUDA_FREE(state->constraint_upper_bound_finite_val);
    SAFE_CUDA_FREE(state->variable_lower_bound_finite_val);
    SAFE_CUDA_FREE(state->variable_upper_bound_finite_val);

    SAFE_CUDA_FREE(state->initial_primal_solution);
    SAFE_CUDA_FREE(state->current_primal_solution);
    SAFE_CUDA_FREE(state->pdhg_primal_solution);
    SAFE_CUDA_FREE(state->reflected_primal_solution);

    SAFE_CUDA_FREE(state->dual_product);
    SAFE_CUDA_FREE(state->initial_dual_solution);
    SAFE_CUDA_FREE(state->current_dual_solution);
    SAFE_CUDA_FREE(state->pdhg_dual_solution);
    SAFE_CUDA_FREE(state->reflected_dual_solution);
    SAFE_CUDA_FREE(state->primal_product);

    SAFE_CUDA_FREE(state->primal_slack);
    SAFE_CUDA_FREE(state->dual_slack);
    SAFE_CUDA_FREE(state->primal_residual);
    SAFE_CUDA_FREE(state->dual_residual);
    SAFE_CUDA_FREE(state->delta_primal_solution);
    SAFE_CUDA_FREE(state->delta_dual_solution);
    free(state);
}

static void perform_primal_restart(pdhg_solver_state_t *state)
{
    CUDA_CHECK(cudaMemcpy(state->initial_primal_solution, state->pdhg_primal_solution, state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->current_primal_solution, state->pdhg_primal_solution, state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
    state->inner_count = 0;
    state->last_trial_fixed_point_error = INFINITY;
}

static void perform_dual_restart(pdhg_solver_state_t *state)
{
    CUDA_CHECK(cudaMemcpy(state->initial_dual_solution, state->pdhg_dual_solution, state->num_constraints * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->current_dual_solution, state->pdhg_dual_solution, state->num_constraints * sizeof(double), cudaMemcpyDeviceToDevice));
    state->inner_count = 0;
    state->last_trial_fixed_point_error = INFINITY;
}

__global__ void compute_delta_primal_solution_kernel(
    const double *initial_primal, const double *pdhg_primal, double *delta_primal,
    int n_vars)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        delta_primal[i] = pdhg_primal[i] - initial_primal[i];
    }
}

__global__ void compute_delta_dual_solution_kernel(
    const double *initial_dual, const double *pdhg_dual, double *delta_dual,
    int n_cons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_cons)
    {
        delta_dual[i] = pdhg_dual[i] - initial_dual[i];
    }
}

static void compute_primal_fixed_point_error(pdhg_solver_state_t *state)
{
    compute_delta_primal_solution_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->current_primal_solution,
        state->reflected_primal_solution,
        state->delta_primal_solution,
        state->num_variables);
    double primal_norm = 0.0;
    CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle,
                                   state->num_variables,
                                   state->delta_primal_solution,
                                   1,
                                   &primal_norm));
    state->fixed_point_error = primal_norm * primal_norm * state->primal_weight;
}

static void compute_dual_fixed_point_error(pdhg_solver_state_t *state)
{
    compute_delta_dual_solution_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
        state->current_dual_solution,
        state->reflected_dual_solution,
        state->delta_dual_solution,
        state->num_constraints);
    double dual_norm = 0.0;
    CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle,
                                   state->num_constraints,
                                   state->delta_dual_solution,
                                   1,
                                   &dual_norm));
    state->fixed_point_error = dual_norm * dual_norm / state->primal_weight;
}
