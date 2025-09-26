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

#include "utils.h"
#include "version.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <random>

std::mt19937 gen(1);
std::normal_distribution<double> dist(0.0, 1.0);

const double HOST_ONE = 1.0;
const double HOST_ZERO = 0.0;

void *safe_malloc(size_t size)
{
    void *ptr = malloc(size);
    if (ptr == NULL)
    {
        perror("Fatal error: malloc failed");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void *safe_calloc(size_t num, size_t size)
{
    void *ptr = calloc(num, size);
    if (ptr == NULL)
    {
        perror("Fatal error: calloc failed");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void *safe_realloc(void *ptr, size_t new_size)
{
    if (new_size == 0) {
        free(ptr);
        return NULL;
    }
    void *tmp = realloc(ptr, new_size);
    if (!tmp) {
        perror("Fatal error: realloc failed");
        exit(EXIT_FAILURE);
    }
    return tmp;
}

double estimate_maximum_singular_value(
    cusparseHandle_t sparse_handle,
    cublasHandle_t blas_handle,
    const cu_sparse_matrix_csr_t *A,
    const cu_sparse_matrix_csr_t *AT,
    int max_iterations,
    double tolerance)
{
    int m = A->num_rows;
    int n = A->num_cols;
    double *eigenvector_d, *next_eigenvector_d, *dual_product_d;

    CUDA_CHECK(cudaMalloc(&eigenvector_d, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&next_eigenvector_d, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dual_product_d, n * sizeof(double)));

    double *eigenvector_h = (double *)safe_malloc(m * sizeof(double));
    for (int i = 0; i < m; ++i)
    {
        eigenvector_h[i] = dist(gen);
    }

    CUDA_CHECK(cudaMemcpy(eigenvector_d, eigenvector_h, m * sizeof(double), cudaMemcpyHostToDevice));
    free(eigenvector_h);

    double sigma_max_sq = 1.0;
    const double one = 1.0;
    const double zero = 0.0;

    cusparseSpMatDescr_t matA, matAT;
    CUSPARSE_CHECK(cusparseCreateCsr(&matA, A->num_rows, A->num_cols, A->num_nonzeros, A->row_ptr, A->col_ind, A->val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateCsr(&matAT, AT->num_rows, AT->num_cols, AT->num_nonzeros, AT->row_ptr, AT->col_ind, AT->val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    cusparseDnVecDescr_t vecEigen, vecNextEigen, vecDual;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecEigen, m, eigenvector_d, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecNextEigen, m, next_eigenvector_d, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecDual, n, dual_product_d, CUDA_R_64F));

    void *dBufferAT = NULL;
    void *dBufferA = NULL;
    size_t bufferSizeAT = 0, bufferSizeA = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matAT, vecNextEigen, &zero, vecDual, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &bufferSizeAT));
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vecDual, &zero, vecEigen, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &bufferSizeA));

    CUDA_CHECK(cudaMalloc(&dBufferAT, bufferSizeAT));
    CUDA_CHECK(cudaMalloc(&dBufferA, bufferSizeA));

    for (int i = 0; i < max_iterations; ++i)
    {

        CUDA_CHECK(cudaMemcpy(next_eigenvector_d, eigenvector_d, m * sizeof(double), cudaMemcpyDeviceToDevice));
        double eigenvector_norm;
        CUBLAS_CHECK(cublasDnrm2_v2_64(blas_handle, m, next_eigenvector_d, 1, &eigenvector_norm));

        double inv_eigenvector_norm = 1.0 / eigenvector_norm;
        CUBLAS_CHECK(cublasDscal(blas_handle, m, &inv_eigenvector_norm, next_eigenvector_d, 1));

        CUSPARSE_CHECK(cusparseSpMV(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matAT, vecNextEigen, &zero, vecDual, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, dBufferAT));

        CUSPARSE_CHECK(cusparseSpMV(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vecDual, &zero, vecEigen, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, dBufferA));

        CUBLAS_CHECK(cublasDdot(blas_handle, m, next_eigenvector_d, 1, eigenvector_d, 1, &sigma_max_sq));

        double neg_sigma_sq = -sigma_max_sq;
        CUBLAS_CHECK(cublasDscal(blas_handle, m, &neg_sigma_sq, next_eigenvector_d, 1));
        CUBLAS_CHECK(cublasDaxpy(blas_handle, m, &one, eigenvector_d, 1, next_eigenvector_d, 1));

        double residual_norm;
        CUBLAS_CHECK(cublasDnrm2_v2_64(blas_handle, m, next_eigenvector_d, 1, &residual_norm));

        if (residual_norm < tolerance)
            break;
    }

    CUDA_CHECK(cudaFree(dBufferAT));
    CUDA_CHECK(cudaFree(dBufferA));
    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    CUSPARSE_CHECK(cusparseDestroySpMat(matAT));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecEigen));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecNextEigen));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecDual));
    CUDA_CHECK(cudaFree(eigenvector_d));
    CUDA_CHECK(cudaFree(next_eigenvector_d));
    CUDA_CHECK(cudaFree(dual_product_d));

    return sqrt(sigma_max_sq);
}

void compute_interaction_and_movement(pdhg_solver_state_t *state, double *interaction, double *movement)
{
    double dual_norm, primal_norm, cross_term;

    CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle,
                                   state->num_constraints,
                                   state->delta_dual_solution,
                                   1,
                                   &dual_norm));
    CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle,
                                   state->num_variables,
                                   state->delta_primal_solution,
                                   1,
                                   &primal_norm));
    *movement = 0.5 * (primal_norm * primal_norm * state->primal_weight + dual_norm * dual_norm / state->primal_weight);

    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_variables, state->dual_product, 1, state->delta_primal_solution, 1, &cross_term));
    *interaction = fabs(cross_term);
}

const char *termination_reason_to_string(termination_reason_t reason)
{
    switch (reason)
    {
    case TERMINATION_REASON_OPTIMAL:
        return "OPTIMAL";
    case TERMINATION_REASON_PRIMAL_INFEASIBLE:
        return "PRIMAL_INFEASIBLE";
    case TERMINATION_REASON_DUAL_INFEASIBLE:
        return "DUAL_INFEASIBLE";
    case TERMINATION_REASON_TIME_LIMIT:
        return "TIME_LIMIT";
    case TERMINATION_REASON_ITERATION_LIMIT:
        return "ITERATION_LIMIT";
    case TERMINATION_REASON_UNSPECIFIED:
    default:
        return "UNSPECIFIED";
    }
}

bool optimality_criteria_met(const pdhg_solver_state_t *state, double rel_opt_tol, double rel_feas_tol)
{
    return state->relative_dual_residual < rel_feas_tol &&
           state->relative_primal_residual < rel_feas_tol &&
           state->relative_objective_gap < rel_opt_tol;
}

bool primal_infeasibility_criteria_met(const pdhg_solver_state_t *state, double eps)
{
    if (state->dual_ray_objective <= 0.0)
    {
        return false;
    }
    return state->max_dual_ray_infeasibility / state->dual_ray_objective <= eps;
}

bool dual_infeasibility_criteria_met(const pdhg_solver_state_t *state, double eps)
{
    if (state->primal_ray_linear_objective >= 0.0)
    {
        return false;
    }
    return state->max_primal_ray_infeasibility / (-state->primal_ray_linear_objective) <= eps;
}

void check_termination_criteria(
    pdhg_solver_state_t *solver_state,
    const termination_criteria_t *criteria)
{
    if (optimality_criteria_met(solver_state, criteria->eps_optimal_relative, criteria->eps_feasible_relative))
    {
        solver_state->termination_reason = TERMINATION_REASON_OPTIMAL;
        return;
    }
    if (primal_infeasibility_criteria_met(solver_state, criteria->eps_infeasible))
    {
        solver_state->termination_reason = TERMINATION_REASON_PRIMAL_INFEASIBLE;
        return;
    }
    if (dual_infeasibility_criteria_met(solver_state, criteria->eps_infeasible))
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

bool should_do_adaptive_restart(
    pdhg_solver_state_t *solver_state,
    const restart_parameters_t *restart_params,
    int termination_evaluation_frequency)
{
    bool do_restart = false;
    if (solver_state->total_count == termination_evaluation_frequency)
    {
        do_restart = true;
    }
    else if (solver_state->total_count > termination_evaluation_frequency)
    {
        if (solver_state->fixed_point_error <= restart_params->sufficient_reduction_for_restart * solver_state->initial_fixed_point_error)
        {
            do_restart = true;
        }
        if (solver_state->fixed_point_error <= restart_params->necessary_reduction_for_restart * solver_state->initial_fixed_point_error)
        {
            if (solver_state->fixed_point_error > solver_state->last_trial_fixed_point_error)
            {
                do_restart = true;
            }
        }
        if (solver_state->inner_count >= restart_params->artificial_restart_threshold * solver_state->total_count)
        {
            do_restart = true;
        }
    }
    solver_state->last_trial_fixed_point_error = solver_state->fixed_point_error;
    return do_restart;
}

void print_initial_info(const pdhg_parameters_t *params, const lp_problem_t *problem)
{
    if (!params->verbose)
    {
        return;
    }
    printf("---------------------------------------------------------------------------------------\n");
    printf("                                    cuPDLPx v%s                                    \n", CUPDLPX_VERSION);
    printf("                        A GPU-Accelerated First-Order LP Solver                        \n");
    printf("               (c) Haihao Lu, Massachusetts Institute of Technology, 2025              \n");
    printf("---------------------------------------------------------------------------------------\n");

    printf("problem:\n");
    printf("  variables     : %d\n", problem->num_variables);
    printf("  constraints   : %d\n", problem->num_constraints);
    printf("  nnz(A)        : %d\n", problem->constraint_matrix_num_nonzeros);

    printf("settings:\n");
    printf("  iter_limit         : %d\n", params->termination_criteria.iteration_limit);
    printf("  time_limit         : %.2f sec\n", params->termination_criteria.time_sec_limit);
    printf("  eps_opt            : %.1e\n", params->termination_criteria.eps_optimal_relative);
    printf("  eps_feas           : %.1e\n", params->termination_criteria.eps_feasible_relative);
    printf("  eps_infeas_detect  : %.1e\n", params->termination_criteria.eps_infeasible);

    printf("---------------------------------------------------------------------------------------\n");
    printf("%s | %s | %s | %s \n",
           "   runtime    ", "    objective     ", "  absolute residuals   ", "  relative residuals   ");
    printf("%s %s | %s %s | %s %s %s | %s %s %s \n",
           "  iter", "  time ", " pr obj ", "  du obj ", " pr res", " du res", "  gap  ", " pr res", " du res", "  gap  ");
    printf("---------------------------------------------------------------------------------------\n");
}

void pdhg_final_log(const pdhg_solver_state_t *state, bool verbose, termination_reason_t reason)
{
    if (verbose)
    {
        printf("---------------------------------------------------------------------------------------\n");
    }
    printf("Solution Summary\n");
    printf("  Status        : %s\n", termination_reason_to_string(reason));
    printf("  Iterations    : %d\n", state->total_count - 1);
    printf("  Solve time    : %.3g sec\n", state->cumulative_time_sec);
    printf("  Primal obj    : %.10g\n", state->primal_objective_value);
    printf("  Dual obj      : %.10g\n", state->dual_objective_value);
    printf("  Primal infeas : %.3e\n", state->relative_primal_residual);
    printf("  Dual infeas   : %.3e\n", state->relative_dual_residual);
}

void display_iteration_stats(const pdhg_solver_state_t *state, bool verbose)
{
    if (!verbose)
    {
        return;
    }
    if (state->total_count % get_print_frequency(state->total_count) == 0)
    {
        printf("%6d %.1e | %8.1e  %8.1e | %.1e %.1e %.1e | %.1e %.1e %.1e \n",
            state->total_count,
            state->cumulative_time_sec,
            state->primal_objective_value,
            state->dual_objective_value,
            state->absolute_primal_residual,
            state->absolute_dual_residual,
            state->objective_gap,
            state->relative_primal_residual,
            state->relative_dual_residual,
            state->relative_objective_gap);
    }

}

int get_print_frequency(int iter)
{
    int step = 10;
    long long threshold = 1000;

    while (iter >= threshold) {
        step *= 10;
        threshold *= 10;
    }
    return step;
}

__global__ void compute_residual_kernel(
    double *primal_residual,
    const double *primal_product,
    const double *constraint_lower_bound,
    const double *constraint_upper_bound,
    const double *dual_solution,
    double *dual_residual,
    const double *dual_product,
    const double *dual_slack,
    const double *objective_vector,
    const double *constraint_rescaling,
    const double *variable_rescaling,
    double *dual_obj_contribution,
    const double *const_lb_finite,
    const double *const_ub_finite,
    int num_constraints,
    int num_variables)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_constraints)
    {

        double clamped_val = fmax(constraint_lower_bound[i], fmin(primal_product[i], constraint_upper_bound[i]));
        primal_residual[i] = (primal_product[i] - clamped_val) * constraint_rescaling[i];

        dual_obj_contribution[i] = fmax(dual_solution[i], 0.0) * const_lb_finite[i] + fmin(dual_solution[i], 0.0) * const_ub_finite[i];
    }
    else if (i < num_constraints + num_variables)
    {
        int idx = i - num_constraints;
        dual_residual[idx] = (objective_vector[idx] - dual_product[idx] - dual_slack[idx]) * variable_rescaling[idx];
    }
}

__global__ void primal_infeasibility_project_kernel(
    double *primal_ray_estimate,
    const double *variable_lower_bound,
    const double *variable_upper_bound,
    int num_variables)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_variables)
    {
        if (isfinite(variable_lower_bound[i]))
        {
            primal_ray_estimate[i] = fmax(primal_ray_estimate[i], 0.0);
        }
        if (isfinite(variable_upper_bound[i]))
        {
            primal_ray_estimate[i] = fmin(primal_ray_estimate[i], 0.0);
        }
    }
}

__global__ void dual_infeasibility_project_kernel(
    double *dual_ray_estimate,
    const double *constraint_lower_bound,
    const double *constraint_upper_bound,
    int num_constraints)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_constraints)
    {
        if (!isfinite(constraint_lower_bound[i]))
        {
            dual_ray_estimate[i] = fmin(dual_ray_estimate[i], 0.0);
        }
        if (!isfinite(constraint_upper_bound[i]))
        {
            dual_ray_estimate[i] = fmax(dual_ray_estimate[i], 0.0);
        }
    }
}

__global__ void compute_primal_infeasibility_kernel(
    const double *primal_product,
    const double *const_lb,
    const double *const_ub,
    int num_constraints,
    double *primal_infeasibility,
    const double *constraint_rescaling)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_constraints)
    {
        double pp_val = primal_product[i];
        primal_infeasibility[i] = (fmax(0.0, -pp_val) * isfinite(const_lb[i]) + fmax(0.0, pp_val) * isfinite(const_ub[i])) * constraint_rescaling[i];
    }
}

__global__ void compute_dual_infeasibility_kernel(
    const double *dual_product,
    const double *var_lb,
    const double *var_ub,
    int num_variables,
    double *dual_infeasibility,
    const double *variable_rescaling)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_variables)
    {
        double dp_val = -dual_product[i];
        dual_infeasibility[i] = (fmax(0.0, dp_val) * !isfinite(var_lb[i]) - fmin(0.0, dp_val) * !isfinite(var_ub[i])) * variable_rescaling[i];
    }
}

__global__ void dual_solution_dual_objective_contribution_kernel(
    const double *constraint_lower_bound_finite_val,
    const double *constraint_upper_bound_finite_val,
    const double *dual_solution,
    int num_constraints,
    double *dual_objective_dual_solution_contribution_array)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_constraints)
    {
        dual_objective_dual_solution_contribution_array[i] =
            fmax(dual_solution[i], 0.0) * constraint_lower_bound_finite_val[i] +
            fmin(dual_solution[i], 0.0) * constraint_upper_bound_finite_val[i];
    }
}

__global__ void dual_objective_dual_slack_contribution_array_kernel(
    const double *dual_slack,
    double *dual_objective_dual_slack_contribution_array,
    const double *variable_lower_bound_finite_val,
    const double *variable_upper_bound_finite_val,
    int num_variables)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_variables)
    {
        dual_objective_dual_slack_contribution_array[i] =
            fmax(-dual_slack[i], 0.0) * variable_lower_bound_finite_val[i] +
            fmin(-dual_slack[i], 0.0) * variable_upper_bound_finite_val[i];
    }
}

static double get_vector_inf_norm(cublasHandle_t handle, int n, const double *x_d)
{
    if (n <= 0)
        return 0.0;
    int index;

    cublasIdamax(handle, n, x_d, 1, &index);
    double max_val;

    CUDA_CHECK(cudaMemcpy(&max_val, x_d + (index - 1), sizeof(double), cudaMemcpyDeviceToHost));
    return fabs(max_val);
}

static double get_vector_sum(cublasHandle_t handle, int n, double *ones_d, const double *x_d)
{
    if (n <= 0)
        return 0.0;

    double sum;
    CUBLAS_CHECK(cublasDdot(handle, n, x_d, 1, ones_d, 1, &sum));
    return sum;
}

void compute_residual(pdhg_solver_state_t *state)
{
    cusparseDnVecSetValues(state->vec_primal_sol, state->pdhg_primal_solution);
    cusparseDnVecSetValues(state->vec_dual_sol, state->pdhg_dual_solution);
    cusparseDnVecSetValues(state->vec_primal_prod, state->primal_product);
    cusparseDnVecSetValues(state->vec_dual_prod, state->dual_product);

    CUSPARSE_CHECK(cusparseSpMV(state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE, state->matA, state->vec_primal_sol, &HOST_ZERO, state->vec_primal_prod, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->primal_spmv_buffer));

    CUSPARSE_CHECK(cusparseSpMV(state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE, state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->dual_spmv_buffer));

    compute_residual_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK>>>(
        state->primal_residual, state->primal_product, state->constraint_lower_bound,
        state->constraint_upper_bound, state->pdhg_dual_solution, state->dual_residual,
        state->dual_product, state->dual_slack, state->objective_vector,
        state->constraint_rescaling, state->variable_rescaling, state->primal_slack,
        state->constraint_lower_bound_finite_val, state->constraint_upper_bound_finite_val,
        state->num_constraints, state->num_variables);

    CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_constraints, state->primal_residual, 1, &state->absolute_primal_residual));
    state->absolute_primal_residual /= state->constraint_bound_rescaling;
    CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_variables, state->dual_residual, 1, &state->absolute_dual_residual));
    state->absolute_dual_residual /= state->objective_vector_rescaling;

    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_variables, state->objective_vector, 1, state->pdhg_primal_solution, 1, &state->primal_objective_value));
    state->primal_objective_value = state->primal_objective_value / (state->constraint_bound_rescaling * state->objective_vector_rescaling) + state->objective_constant;

    double base_dual_objective;
    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_variables, state->dual_slack, 1, state->pdhg_primal_solution, 1, &base_dual_objective));
    double dual_slack_sum = get_vector_sum(state->blas_handle, state->num_constraints, state->ones_dual_d, state->primal_slack);
    state->dual_objective_value = (base_dual_objective + dual_slack_sum) / (state->constraint_bound_rescaling * state->objective_vector_rescaling) + state->objective_constant;

    state->relative_primal_residual = state->absolute_primal_residual / (1.0 + state->constraint_bound_norm);
    state->relative_dual_residual = state->absolute_dual_residual / (1.0 + state->objective_vector_norm);
    state->relative_objective_gap = fabs(state->primal_objective_value - state->dual_objective_value) /
                                    (1.0 + fabs(state->primal_objective_value) + fabs(state->dual_objective_value));
}

void compute_infeasibility_information(pdhg_solver_state_t *state)
{
    primal_infeasibility_project_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(state->delta_primal_solution, state->variable_lower_bound, state->variable_upper_bound, state->num_variables);
    dual_infeasibility_project_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(state->delta_dual_solution, state->constraint_lower_bound, state->constraint_upper_bound, state->num_constraints);

    double primal_ray_inf_norm = get_vector_inf_norm(state->blas_handle, state->num_variables, state->delta_primal_solution);
    if (primal_ray_inf_norm > 0.0)
    {
        double scale = 1.0 / primal_ray_inf_norm;
        cublasDscal(state->blas_handle, state->num_variables, &scale, state->delta_primal_solution, 1);
    }
    double dual_ray_inf_norm = get_vector_inf_norm(state->blas_handle, state->num_constraints, state->delta_dual_solution);

    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_sol, state->delta_primal_solution));
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_sol, state->delta_dual_solution));
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_prod, state->primal_product));
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_prod, state->dual_product));

    CUSPARSE_CHECK(cusparseSpMV(state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE, state->matA, state->vec_primal_sol, &HOST_ZERO, state->vec_primal_prod, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->primal_spmv_buffer));

    CUSPARSE_CHECK(cusparseSpMV(state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE, state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->dual_spmv_buffer));

    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_variables, state->objective_vector, 1, state->delta_primal_solution, 1, &state->primal_ray_linear_objective));
    state->primal_ray_linear_objective /= (state->constraint_bound_rescaling * state->objective_vector_rescaling);

    dual_solution_dual_objective_contribution_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
        state->constraint_lower_bound_finite_val,
        state->constraint_upper_bound_finite_val,
        state->delta_dual_solution,
        state->num_constraints,
        state->primal_slack);

    dual_objective_dual_slack_contribution_array_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->dual_product,
        state->dual_slack,
        state->variable_lower_bound_finite_val,
        state->variable_upper_bound_finite_val,
        state->num_variables);

    double sum_primal_slack = get_vector_sum(state->blas_handle, state->num_constraints, state->ones_dual_d, state->primal_slack);
    double sum_dual_slack = get_vector_sum(state->blas_handle, state->num_variables, state->ones_primal_d, state->dual_slack);
    state->dual_ray_objective = (sum_primal_slack + sum_dual_slack) / (state->constraint_bound_rescaling * state->objective_vector_rescaling);

    compute_primal_infeasibility_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(state->primal_product, state->constraint_lower_bound, state->constraint_upper_bound, state->num_constraints, state->primal_slack, state->constraint_rescaling);
    compute_dual_infeasibility_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(state->dual_product, state->variable_lower_bound, state->variable_upper_bound, state->num_variables, state->dual_slack, state->variable_rescaling);

    state->max_primal_ray_infeasibility = get_vector_inf_norm(state->blas_handle, state->num_constraints, state->primal_slack);
    double dual_slack_norm = get_vector_inf_norm(state->blas_handle, state->num_variables, state->dual_slack);
    state->max_dual_ray_infeasibility = dual_slack_norm;

    double scaling_factor = fmax(dual_ray_inf_norm, dual_slack_norm);
    if (scaling_factor > 0.0)
    {
        state->max_dual_ray_infeasibility /= scaling_factor;
        state->dual_ray_objective /= scaling_factor;
    }
    else
    {
        state->max_dual_ray_infeasibility = 0.0;
        state->dual_ray_objective = 0.0;
    }
}