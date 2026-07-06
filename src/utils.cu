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
#include <math.h>
#include <random>

#ifndef CUPDLPX_VERSION
#define CUPDLPX_VERSION "unknown"
#endif

std::mt19937 gen(1);
std::normal_distribution<double> dist(0.0, 1.0);

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
    if (new_size == 0)
    {
        free(ptr);
        return NULL;
    }
    void *tmp = realloc(ptr, new_size);
    if (!tmp)
    {
        perror("Fatal error: realloc failed");
        exit(EXIT_FAILURE);
    }
    return tmp;
}

double estimate_maximum_singular_value(cusparseHandle_t sparse_handle,
                                       cublasHandle_t blas_handle,
                                       const cu_sparse_matrix_csr_t *A,
                                       const cu_sparse_matrix_csr_t *AT,
                                       int max_iterations,
                                       double tolerance)
{
    const int m = A->num_rows;
    const int n = A->num_cols;
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

    cusparseSpMatDescr_t matA, matAT;
    CUSPARSE_CHECK(cusparseCreateCsr(&matA,
                                     A->num_rows,
                                     A->num_cols,
                                     A->num_nonzeros,
                                     A->row_ptr,
                                     A->col_ind,
                                     A->val,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateCsr(&matAT,
                                     AT->num_rows,
                                     AT->num_cols,
                                     AT->num_nonzeros,
                                     AT->row_ptr,
                                     AT->col_ind,
                                     AT->val,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_64F));

    cusparseDnVecDescr_t vecEigen, vecNextEigen, vecDual;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecEigen, m, eigenvector_d, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecNextEigen, m, next_eigenvector_d, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecDual, n, dual_product_d, CUDA_R_64F));

    void *descrAT = NULL;
    void *descrA = NULL;
    void *planAT = NULL;
    void *planA = NULL;

    void *dBufferAT = NULL;
    void *dBufferA = NULL;
    size_t bufferSizeAT = 0, bufferSizeA = 0;
    cupdlpx_spmv_buffer_size(sparse_handle, matAT, vecNextEigen, vecDual, &bufferSizeAT);
    cupdlpx_spmv_buffer_size(sparse_handle, matA, vecDual, vecEigen, &bufferSizeA);

    CUDA_CHECK(cudaMalloc(&dBufferAT, bufferSizeAT));
    CUDA_CHECK(cudaMalloc(&dBufferA, bufferSizeA));

    cupdlpx_spmv_prepare(sparse_handle, matAT, vecNextEigen, vecDual, dBufferAT, &descrAT, &planAT);
    cupdlpx_spmv_prepare(sparse_handle, matA, vecDual, vecEigen, dBufferA, &descrA, &planA);

    for (int i = 0; i < max_iterations; ++i)
    {
        CUDA_CHECK(cudaMemcpy(next_eigenvector_d, eigenvector_d, m * sizeof(double), cudaMemcpyDeviceToDevice));
        double eigenvector_norm;
        CUBLAS_CHECK(cublasDnrm2_v2_64(blas_handle, m, next_eigenvector_d, 1, &eigenvector_norm));

        double inv_eigenvector_norm = 1.0 / eigenvector_norm;
        CUBLAS_CHECK(cublasDscal(blas_handle, m, &inv_eigenvector_norm, next_eigenvector_d, 1));

        cupdlpx_spmv_execute(sparse_handle, matAT, vecNextEigen, vecDual, dBufferAT, planAT);
        cupdlpx_spmv_execute(sparse_handle, matA, vecDual, vecEigen, dBufferA, planA);

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
    cupdlpx_spmv_release(descrAT, planAT);
    cupdlpx_spmv_release(descrA, planA);
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

    CUBLAS_CHECK(
        cublasDnrm2_v2_64(state->blas_handle, state->num_constraints, state->delta_dual_solution, 1, &dual_norm));
    CUBLAS_CHECK(
        cublasDnrm2_v2_64(state->blas_handle, state->num_variables, state->delta_primal_solution, 1, &primal_norm));
    *movement = 0.5 * (primal_norm * primal_norm * state->primal_weight + dual_norm * dual_norm / state->primal_weight);

    CUBLAS_CHECK(cublasDdot(state->blas_handle,
                            state->num_variables,
                            state->dual_product,
                            1,
                            state->delta_primal_solution,
                            1,
                            &cross_term));
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
        case TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED:
            return "INFEASIBLE_OR_UNBOUNDED";
        case TERMINATION_REASON_TIME_LIMIT:
            return "TIME_LIMIT";
        case TERMINATION_REASON_ITERATION_LIMIT:
            return "ITERATION_LIMIT";
        case TERMINATION_REASON_UNSPECIFIED:
            return "UNSPECIFIED";
        case TERMINATION_REASON_FEAS_POLISH_SUCCESS:
            return "FEAS_POLISH_SUCCESS";
        default:
            return "UNKNOWN";
    }
}

bool optimality_criteria_met(const pdhg_solver_state_t *state, double rel_opt_tol, double rel_feas_tol)
{
    return state->relative_dual_residual < rel_feas_tol && state->relative_primal_residual < rel_feas_tol &&
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

void check_termination_criteria(pdhg_solver_state_t *solver_state, const termination_criteria_t *criteria)
{
    solver_state->cumulative_time_sec = (double)(clock() - solver_state->start_time) / CLOCKS_PER_SEC;
    if (optimality_criteria_met(solver_state, criteria->eps_optimal_relative, criteria->eps_feasible_relative))
    {
        solver_state->termination_reason = TERMINATION_REASON_OPTIMAL;
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

bool should_do_adaptive_restart(pdhg_solver_state_t *solver_state,
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
        if (solver_state->fixed_point_error <=
            restart_params->sufficient_reduction_for_restart * solver_state->initial_fixed_point_error)
        {
            do_restart = true;
        }
        if (solver_state->fixed_point_error <=
            restart_params->necessary_reduction_for_restart * solver_state->initial_fixed_point_error)
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

void set_default_parameters(pdhg_parameters_t *params)
{
    params->l_inf_ruiz_iterations = 10;
    params->has_pock_chambolle_alpha = true;
    params->pock_chambolle_alpha = 1.0;
    params->bound_objective_rescaling = true;
    params->verbose = true;
    params->termination_evaluation_frequency = 200;
    params->feasibility_polishing = false;
    params->reflection_coefficient = 1.0;
    params->sv_max_iter = 5000;
    params->sv_tol = 1e-4;

    params->termination_criteria.eps_optimal_relative = 1e-4;
    params->termination_criteria.eps_feasible_relative = 1e-4;
    params->termination_criteria.time_sec_limit = 3600.0;
    params->termination_criteria.iteration_limit = INT32_MAX;
    params->termination_criteria.eps_feas_polish_relative = 1e-6;

    params->restart_params.artificial_restart_threshold = 0.36;
    params->restart_params.sufficient_reduction_for_restart = 0.2;
    params->restart_params.necessary_reduction_for_restart = 0.5;
    params->restart_params.k_p = 0.99;
    params->restart_params.k_i = 0.01;
    params->restart_params.k_d = 0.0;
    params->restart_params.i_smooth = 0.3;

    params->optimality_norm = NORM_TYPE_L2;
    params->presolve = true;
    params->matrix_zero_tol = 1e-9;
}

void filter_constraint_matrix_entries(lp_problem_t *out, const lp_problem_t *in, const pdhg_parameters_t *params)
{
    if (out == NULL || in == NULL)
    {
        fprintf(stderr, "Error: problem pointer is NULL.\n");
        exit(EXIT_FAILURE);
    }

    const int num_rows = in->num_constraints;
    const int nnz = in->constraint_matrix_num_nonzeros;

    if (num_rows == 0 || nnz == 0)
    {
        return;
    }

    const int *row_ptr = in->constraint_matrix_row_pointers;
    const int *col_ind = in->constraint_matrix_col_indices;
    const double *vals = in->constraint_matrix_values;

    if (!row_ptr || !col_ind || !vals)
    {
        fprintf(stderr, "Error: constraint matrix data is not initialized.\n");
        exit(EXIT_FAILURE);
    }

    int filtered_nnz = 0;
    for (int i = 0; i < num_rows; ++i)
    {
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k)
        {
            if (fabs(vals[k]) > params->matrix_zero_tol)
            {
                ++filtered_nnz;
            }
        }
    }

    if (filtered_nnz == nnz)
    {
        return;
    }

    if (params->verbose)
    {
        const int dropped = nnz - filtered_nnz;
        printf("Dropped %d near-zero %s (|value| <= %.1e) from the constraint matrix\n",
               dropped,
               (dropped == 1 ? "entry" : "entries"),
               params->matrix_zero_tol);
    }

    const size_t alloc_nnz = (filtered_nnz > 0) ? (size_t)filtered_nnz : 1;
    int *new_row_ptr = (int *)safe_malloc((size_t)(num_rows + 1) * sizeof(int));
    int *new_col_ind = (int *)safe_malloc(alloc_nnz * sizeof(int));
    double *new_vals = (double *)safe_malloc(alloc_nnz * sizeof(double));

    int pos = 0;
    new_row_ptr[0] = 0;
    for (int i = 0; i < num_rows; ++i)
    {
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k)
        {
            double value = vals[k];
            if (fabs(value) <= params->matrix_zero_tol)
            {
                continue;
            }
            new_col_ind[pos] = col_ind[k];
            new_vals[pos] = value;
            ++pos;
        }
        new_row_ptr[i + 1] = pos;
    }

    out->constraint_matrix_row_pointers = new_row_ptr;
    out->constraint_matrix_col_indices = new_col_ind;
    out->constraint_matrix_values = new_vals;
    out->constraint_matrix_num_nonzeros = filtered_nnz;
}

lp_problem_t preprocess_problem(const lp_problem_t *original, const pdhg_parameters_t *params)
{
    lp_problem_t working = *original;
    if (original->objective_sense == OBJECTIVE_SENSE_MAXIMIZE)
    {
        double *negated = (double *)safe_malloc((size_t)original->num_variables * sizeof(double));
        for (int i = 0; i < original->num_variables; ++i)
        {
            negated[i] = -original->objective_vector[i];
        }
        working.objective_vector = negated;
        working.objective_constant = -original->objective_constant;
        working.objective_sense = OBJECTIVE_SENSE_MINIMIZE;
    }
    filter_constraint_matrix_entries(&working, original, params);
    return working;
}

void free_preprocessed_problem(const lp_problem_t *preprocessed, const lp_problem_t *original)
{
    if (preprocessed->objective_vector != original->objective_vector)
        free(preprocessed->objective_vector);
    if (preprocessed->constraint_matrix_values != original->constraint_matrix_values)
    {
        free(preprocessed->constraint_matrix_row_pointers);
        free(preprocessed->constraint_matrix_col_indices);
        free(preprocessed->constraint_matrix_values);
    }
}

void restore_original_objective_sense(cupdlpx_result_t *result, objective_sense_t sense)
{
    if (result == NULL || sense != OBJECTIVE_SENSE_MAXIMIZE)
        return;
    if (result->dual_solution != NULL)
        for (int i = 0; i < result->num_constraints; ++i)
            result->dual_solution[i] = -result->dual_solution[i];
    if (result->reduced_cost != NULL)
        for (int i = 0; i < result->num_variables; ++i)
            result->reduced_cost[i] = -result->reduced_cost[i];
    result->primal_objective_value = -result->primal_objective_value;
    result->dual_objective_value = -result->dual_objective_value;
    result->primal_ray_linear_objective = -result->primal_ray_linear_objective;
    result->dual_ray_objective = -result->dual_ray_objective;
}

#define PRINT_DIFF_INT(name, current, default_val)                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        if ((current) != (default_val))                                                                                \
        {                                                                                                              \
            printf("  %-18s : %d\n", name, current);                                                                   \
        }                                                                                                              \
    } while (0)

#define PRINT_DIFF_DBL(name, current, default_val)                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        if (fabs((current) - (default_val)) > 1e-9)                                                                    \
        {                                                                                                              \
            printf("  %-18s : %.1e\n", name, (double)(current));                                                       \
        }                                                                                                              \
    } while (0)

#define PRINT_DIFF_BOOL(name, current, default_val)                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        if ((current) != (default_val))                                                                                \
        {                                                                                                              \
            printf("  %-18s : %s\n", name, (current) ? "on" : "off");                                                  \
        }                                                                                                              \
    } while (0)

void print_initial_info(const pdhg_parameters_t *params, const lp_problem_t *problem)
{
    pdhg_parameters_t default_params;
    set_default_parameters(&default_params);
    if (!params->verbose)
    {
        return;
    }
    printf("---------------------------------------------------------------------"
           "------------------\n");
    printf("                                    cuPDLPx v%s                      "
           "              \n",
           CUPDLPX_VERSION);
    printf("                        A GPU-Accelerated First-Order LP Solver      "
           "                  \n");
    printf("               (c) Haihao Lu, Massachusetts Institute of Technology, "
           "2025              \n");
    printf("---------------------------------------------------------------------"
           "------------------\n");

    printf("Problem: %d rows, %d columns, %d nonzeros\n",
           problem->num_constraints,
           problem->num_variables,
           problem->constraint_matrix_num_nonzeros);

    printf("Settings:\n");
    printf("  iter_limit         : %d\n", params->termination_criteria.iteration_limit);
    printf("  time_limit         : %.2f sec\n", params->termination_criteria.time_sec_limit);
    printf("  eps_opt            : %.1e\n", params->termination_criteria.eps_optimal_relative);
    printf("  eps_feas           : %.1e\n", params->termination_criteria.eps_feasible_relative);
    printf("  spmv_backend       : %s (auto)\n", cupdlpx_use_spmvop_by_default() ? "cusparseSpMVOp" : "cusparseSpMV");
    if (params->optimality_norm != default_params.optimality_norm)
    {
        printf("  optimality_norm    : %s\n", params->optimality_norm == NORM_TYPE_L_INF ? "L_inf" : "L2");
    }

    PRINT_DIFF_INT("l_inf_ruiz_iter", params->l_inf_ruiz_iterations, default_params.l_inf_ruiz_iterations);
    PRINT_DIFF_DBL("pock_chambolle_alpha", params->pock_chambolle_alpha, default_params.pock_chambolle_alpha);
    PRINT_DIFF_BOOL(
        "has_pock_chambolle_alpha", params->has_pock_chambolle_alpha, default_params.has_pock_chambolle_alpha);
    PRINT_DIFF_BOOL("bound_obj_rescaling", params->bound_objective_rescaling, default_params.bound_objective_rescaling);
    PRINT_DIFF_INT("sv_max_iter", params->sv_max_iter, default_params.sv_max_iter);
    PRINT_DIFF_DBL("sv_tol", params->sv_tol, default_params.sv_tol);
    PRINT_DIFF_INT(
        "evaluation_freq", params->termination_evaluation_frequency, default_params.termination_evaluation_frequency);
    PRINT_DIFF_BOOL("feasibility_polishing", params->feasibility_polishing, default_params.feasibility_polishing);
    PRINT_DIFF_DBL("eps_feas_polish_relative",
                   params->termination_criteria.eps_feas_polish_relative,
                   default_params.termination_criteria.eps_feas_polish_relative);
    PRINT_DIFF_BOOL("presolve", params->presolve, default_params.presolve);
    PRINT_DIFF_DBL("matrix_zero_tol", params->matrix_zero_tol, default_params.matrix_zero_tol);
}

#undef PRINT_DIFF_INT
#undef PRINT_DIFF_DBL
#undef PRINT_DIFF_BOOL

void pdhg_final_log(const cupdlpx_result_t *result, const pdhg_parameters_t *params)
{
    if (params->verbose)
    {
        printf("-------------------------------------------------------------------"
               "--------------------\n");
        printf("Solution Summary\n");
        printf("  Status                 : %s\n", termination_reason_to_string(result->termination_reason));
        if (params->presolve)
        {
            printf("  Presolve time          : %.3g sec\n", result->presolve_time);
        }
        printf("  Precondition time      : %.5g sec\n", result->rescaling_time_sec);
        printf("  Solve time             : %.3g sec\n", result->cumulative_time_sec);
        printf("  Iterations             : %d\n", result->total_count);
        printf("  Primal objective       : %.10g\n", result->primal_objective_value);
        printf("  Dual objective         : %.10g\n", result->dual_objective_value);
        printf("  Objective gap          : %.3e\n", result->relative_objective_gap);
        printf("  Primal infeas          : %.3e\n", result->relative_primal_residual);
        printf("  Dual infeas            : %.3e\n", result->relative_dual_residual);
    }

    // if (stats != NULL && stats->n_rows_original > 0) {
    //     printf("\nPresolve Summary\n");
    //     printf("  [Dimensions]\n");
    //     printf("  Original           : %d rows, %d cols, %d nnz\n",
    //            stats->n_rows_original, stats->n_cols_original, stats->nnz_original);
    //     printf("  Reduced            : %d rows, %d cols, %d nnz\n",
    //            stats->n_rows_reduced, stats->n_cols_reduced, stats->nnz_reduced);

    //     printf("  [Reduction Details (NNZ Removed)]\n");
    //     printf("  Trivial            : %d\n", stats->nnz_removed_trivial);
    //     printf("  Fast               : %d\n", stats->nnz_removed_fast);
    //     printf("  Primal Propagation : %d\n", stats->nnz_removed_primal_propagation);
    //     printf("  Parallel Rows      : %d\n", stats->nnz_removed_parallel_rows);
    //     printf("  Parallel Cols      : %d\n", stats->nnz_removed_parallel_cols);

    //     printf("  [Timing]\n");
    //     printf("  Total Presolve     : %.3g sec\n", stats->presolve_total_time);
    //     printf("  Init               : %.3g sec\n", stats->ps_time_init);
    //     printf("  Fast               : %.3g sec\n", stats->ps_time_fast);
    //     printf("  Medium             : %.3g sec\n", stats->ps_time_medium);
    //     printf("  Primal Propagation : %.3g sec\n", stats->ps_time_primal_propagation);
    //     printf("  Parallel Rows      : %.3g sec\n", stats->ps_time_parallel_rows);
    //     printf("  Parallel Cols      : %.3g sec\n", stats->ps_time_parallel_cols);
    //     printf("  Postsolve          : %.3g sec\n", stats->ps_time_post_solve);
    // }
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
               state->original_objective_sign * state->primal_objective_value,
               state->original_objective_sign * state->dual_objective_value,
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

    while (iter >= threshold)
    {
        step *= 10;
        threshold *= 10;
    }
    return step;
}

__global__ void compute_residual_kernel(double *__restrict__ primal_residual,
                                        const double *__restrict__ primal_product,
                                        const double *__restrict__ constraint_lower_bound,
                                        const double *__restrict__ constraint_upper_bound,
                                        const double *__restrict__ dual_solution,
                                        double *__restrict__ dual_residual,
                                        const double *__restrict__ dual_product,
                                        const double *__restrict__ dual_slack,
                                        const double *__restrict__ objective_vector,
                                        const double *__restrict__ constraint_rescaling,
                                        const double *__restrict__ variable_rescaling,
                                        double *__restrict__ dual_obj_contribution,
                                        const double *__restrict__ const_lb_finite,
                                        const double *__restrict__ const_ub_finite,
                                        int num_constraints,
                                        int num_variables)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_constraints)
    {

        double clamped_val = fmax(constraint_lower_bound[i], fmin(primal_product[i], constraint_upper_bound[i]));
        primal_residual[i] = (primal_product[i] - clamped_val) * constraint_rescaling[i];

        dual_obj_contribution[i] =
            fmax(dual_solution[i], 0.0) * const_lb_finite[i] + fmin(dual_solution[i], 0.0) * const_ub_finite[i];
    }
    else if (i < num_constraints + num_variables)
    {
        int idx = i - num_constraints;
        dual_residual[idx] = (objective_vector[idx] - dual_product[idx] - dual_slack[idx]) * variable_rescaling[idx];
    }
}

__global__ void primal_infeasibility_project_kernel(double *__restrict__ primal_ray_estimate,
                                                    const double *__restrict__ variable_lower_bound,
                                                    const double *__restrict__ variable_upper_bound,
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

__global__ void dual_infeasibility_project_kernel(double *__restrict__ dual_ray_estimate,
                                                  const double *__restrict__ constraint_lower_bound,
                                                  const double *__restrict__ constraint_upper_bound,
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

__global__ void compute_primal_infeasibility_kernel(const double *__restrict__ primal_product,
                                                    const double *__restrict__ const_lb,
                                                    const double *__restrict__ const_ub,
                                                    int num_constraints,
                                                    double *__restrict__ primal_infeasibility,
                                                    const double *__restrict__ constraint_rescaling)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_constraints)
    {
        double pp_val = primal_product[i];
        primal_infeasibility[i] =
            (fmax(0.0, -pp_val) * isfinite(const_lb[i]) + fmax(0.0, pp_val) * isfinite(const_ub[i])) *
            constraint_rescaling[i];
    }
}

__global__ void compute_dual_infeasibility_kernel(const double *dual_product,
                                                  const double *__restrict__ var_lb,
                                                  const double *__restrict__ var_ub,
                                                  int num_variables,
                                                  double *__restrict__ dual_infeasibility,
                                                  const double *__restrict__ variable_rescaling)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_variables)
    {
        double dp_val = -dual_product[i];
        dual_infeasibility[i] = (fmax(0.0, dp_val) * !isfinite(var_lb[i]) - fmin(0.0, dp_val) * !isfinite(var_ub[i])) *
            variable_rescaling[i];
    }
}

__global__ void
dual_solution_dual_objective_contribution_kernel(const double *__restrict__ constraint_lower_bound_finite_val,
                                                 const double *__restrict__ constraint_upper_bound_finite_val,
                                                 const double *__restrict__ dual_solution,
                                                 int num_constraints,
                                                 double *__restrict__ dual_objective_dual_solution_contribution_array)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_constraints)
    {
        dual_objective_dual_solution_contribution_array[i] =
            fmax(dual_solution[i], 0.0) * constraint_lower_bound_finite_val[i] +
            fmin(dual_solution[i], 0.0) * constraint_upper_bound_finite_val[i];
    }
}

__global__ void
dual_objective_dual_slack_contribution_array_kernel(const double *__restrict__ dual_slack,
                                                    double *__restrict__ dual_objective_dual_slack_contribution_array,
                                                    const double *__restrict__ variable_lower_bound_finite_val,
                                                    const double *__restrict__ variable_upper_bound_finite_val,
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

void compute_residual(pdhg_solver_state_t *state, norm_type_t optimality_norm)
{
    cupdlpx_spmv_Ax(state->sparse_handle, state->spmv_ctx, state->pdhg_primal_solution, state->primal_product);
    cupdlpx_spmv_ATx(state->sparse_handle, state->spmv_ctx, state->pdhg_dual_solution, state->dual_product);

    compute_residual_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
        state->primal_residual,
        state->primal_product,
        state->constraint_lower_bound,
        state->constraint_upper_bound,
        state->pdhg_dual_solution,
        state->dual_residual,
        state->dual_product,
        state->dual_slack,
        state->objective_vector,
        state->constraint_rescaling,
        state->variable_rescaling,
        state->primal_slack,
        state->constraint_lower_bound_finite_val,
        state->constraint_upper_bound_finite_val,
        state->num_constraints,
        state->num_variables);

    if (optimality_norm == NORM_TYPE_L_INF)
    {
        state->absolute_primal_residual =
            get_vector_inf_norm(state->blas_handle, state->num_constraints, state->primal_residual);
    }
    else
    {
        CUBLAS_CHECK(cublasDnrm2_v2_64(
            state->blas_handle, state->num_constraints, state->primal_residual, 1, &state->absolute_primal_residual));
    }
    state->absolute_primal_residual /= state->constraint_bound_rescaling;

    if (optimality_norm == NORM_TYPE_L_INF)
    {
        state->absolute_dual_residual =
            get_vector_inf_norm(state->blas_handle, state->num_variables, state->dual_residual);
    }
    else
    {
        CUBLAS_CHECK(cublasDnrm2_v2_64(
            state->blas_handle, state->num_variables, state->dual_residual, 1, &state->absolute_dual_residual));
    }

    state->absolute_dual_residual /= state->objective_vector_rescaling;

    CUBLAS_CHECK(cublasDdot(state->blas_handle,
                            state->num_variables,
                            state->objective_vector,
                            1,
                            state->pdhg_primal_solution,
                            1,
                            &state->primal_objective_value));
    state->primal_objective_value =
        state->primal_objective_value / (state->constraint_bound_rescaling * state->objective_vector_rescaling) +
        state->objective_constant;

    double base_dual_objective;
    CUBLAS_CHECK(cublasDdot(state->blas_handle,
                            state->num_variables,
                            state->dual_slack,
                            1,
                            state->pdhg_primal_solution,
                            1,
                            &base_dual_objective));
    double dual_slack_sum =
        get_vector_sum(state->blas_handle, state->num_constraints, state->ones_dual_d, state->primal_slack);
    state->dual_objective_value = (base_dual_objective + dual_slack_sum) /
            (state->constraint_bound_rescaling * state->objective_vector_rescaling) +
        state->objective_constant;

    state->relative_primal_residual = state->absolute_primal_residual / (1.0 + state->constraint_bound_norm);

    state->relative_dual_residual = state->absolute_dual_residual / (1.0 + state->objective_vector_norm);

    state->objective_gap = fabs(state->primal_objective_value - state->dual_objective_value);

    state->relative_objective_gap =
        state->objective_gap / (1.0 + fabs(state->primal_objective_value) + fabs(state->dual_objective_value));
}

void compute_infeasibility_information(pdhg_solver_state_t *state)
{
    primal_infeasibility_project_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream>>>(
        state->delta_primal_solution, state->variable_lower_bound, state->variable_upper_bound, state->num_variables);
    dual_infeasibility_project_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
        state->delta_dual_solution,
        state->constraint_lower_bound,
        state->constraint_upper_bound,
        state->num_constraints);

    double primal_ray_inf_norm =
        get_vector_inf_norm(state->blas_handle, state->num_variables, state->delta_primal_solution);
    if (primal_ray_inf_norm > 0.0)
    {
        double scale = 1.0 / primal_ray_inf_norm;
        cublasDscal(state->blas_handle, state->num_variables, &scale, state->delta_primal_solution, 1);
    }
    double dual_ray_inf_norm =
        get_vector_inf_norm(state->blas_handle, state->num_constraints, state->delta_dual_solution);

    cupdlpx_spmv_Ax(state->sparse_handle, state->spmv_ctx, state->delta_primal_solution, state->primal_product);
    cupdlpx_spmv_ATx(state->sparse_handle, state->spmv_ctx, state->delta_dual_solution, state->dual_product);

    CUBLAS_CHECK(cublasDdot(state->blas_handle,
                            state->num_variables,
                            state->objective_vector,
                            1,
                            state->delta_primal_solution,
                            1,
                            &state->primal_ray_linear_objective));
    state->primal_ray_linear_objective /= (state->constraint_bound_rescaling * state->objective_vector_rescaling);

    dual_solution_dual_objective_contribution_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
        state->constraint_lower_bound_finite_val,
        state->constraint_upper_bound_finite_val,
        state->delta_dual_solution,
        state->num_constraints,
        state->primal_slack);

    dual_objective_dual_slack_contribution_array_kernel<<<state->num_blocks_primal,
                                                          THREADS_PER_BLOCK,
                                                          0,
                                                          state->stream>>>(state->dual_product,
                                                                           state->dual_slack,
                                                                           state->variable_lower_bound_finite_val,
                                                                           state->variable_upper_bound_finite_val,
                                                                           state->num_variables);

    double sum_primal_slack =
        get_vector_sum(state->blas_handle, state->num_constraints, state->ones_dual_d, state->primal_slack);
    double sum_dual_slack =
        get_vector_sum(state->blas_handle, state->num_variables, state->ones_primal_d, state->dual_slack);
    state->dual_ray_objective =
        (sum_primal_slack + sum_dual_slack) / (state->constraint_bound_rescaling * state->objective_vector_rescaling);

    compute_primal_infeasibility_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
        state->primal_product,
        state->constraint_lower_bound,
        state->constraint_upper_bound,
        state->num_constraints,
        state->primal_slack,
        state->constraint_rescaling);
    compute_dual_infeasibility_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream>>>(
        state->dual_product,
        state->variable_lower_bound,
        state->variable_upper_bound,
        state->num_variables,
        state->dual_slack,
        state->variable_rescaling);

    state->max_primal_ray_infeasibility =
        get_vector_inf_norm(state->blas_handle, state->num_constraints, state->primal_slack);
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

// helper function to allocate and fill or copy an array
void fill_or_copy(double **dst, int n, const double *src, double fill_val)
{
    *dst = (double *)safe_malloc((size_t)n * sizeof(double));
    if (src)
        memcpy(*dst, src, (size_t)n * sizeof(double));
    else
        for (int i = 0; i < n; ++i)
            (*dst)[i] = fill_val;
}

// convert dense → CSR
int dense_to_csr(const matrix_desc_t *desc, int **row_ptr, int **col_ind, double **vals, int *nnz_out)
{
    int m = desc->m, n = desc->n;

    // count nnz
    int nnz = 0;
    for (int i = 0; i < m * n; ++i)
    {
        if (fabs(desc->data.dense.A[i]) > 0.0)
            ++nnz;
    }

    // allocate
    *row_ptr = (int *)safe_malloc((size_t)(m + 1) * sizeof(int));
    *col_ind = (int *)safe_malloc((size_t)nnz * sizeof(int));
    *vals = (double *)safe_malloc((size_t)nnz * sizeof(double));

    // fill
    int nz = 0;
    for (int i = 0; i < m; ++i)
    {
        (*row_ptr)[i] = nz;
        for (int j = 0; j < n; ++j)
        {
            double v = desc->data.dense.A[i * n + j];
            if (fabs(v) > 0.0)
            {
                (*col_ind)[nz] = j;
                (*vals)[nz] = v;
                ++nz;
            }
        }
    }
    (*row_ptr)[m] = nz;
    *nnz_out = nz;
    return 0;
}

// convert CSC → CSR
int csc_to_csr(const matrix_desc_t *desc, int **row_ptr, int **col_ind, double **vals)
{
    const int m = desc->m, n = desc->n;
    const int *col_ptr = desc->data.csc.col_ptr;
    const int *row_ind = desc->data.csc.row_ind;
    const double *v = desc->data.csc.vals;

    // count entries per row
    *row_ptr = (int *)safe_malloc((size_t)(m + 1) * sizeof(int));
    for (int i = 0; i <= m; ++i)
        (*row_ptr)[i] = 0;

    for (int j = 0; j < n; ++j)
    {
        for (int k = col_ptr[j]; k < col_ptr[j + 1]; ++k)
        {
            int ri = row_ind[k];
            if (ri < 0 || ri >= m)
            {
                fprintf(stderr, "[interface] CSC: row index out of range\n");
                return -1;
            }
            ++((*row_ptr)[ri + 1]);
        }
    }

    // exclusive scan
    for (int i = 0; i < m; ++i)
        (*row_ptr)[i + 1] += (*row_ptr)[i];

    // allocate
    *col_ind = (int *)safe_malloc((size_t)desc->data.csc.nnz * sizeof(int));
    *vals = (double *)safe_malloc((size_t)desc->data.csc.nnz * sizeof(double));

    // next position to fill in each row
    int *next = (int *)safe_malloc((size_t)m * sizeof(int));
    for (int i = 0; i < m; ++i)
        next[i] = (*row_ptr)[i];

    // fill column indices and values
    for (int j = 0; j < n; ++j)
    {
        for (int k = col_ptr[j]; k < col_ptr[j + 1]; ++k)
        {
            int ri = row_ind[k];
            double val = v[k];
            int pos = next[ri]++;
            (*col_ind)[pos] = j;
            (*vals)[pos] = val;
        }
    }

    free(next);
    return 0;
}

// convert COO → CSR
int coo_to_csr(const matrix_desc_t *desc, int **row_ptr, int **col_ind, double **vals)
{
    const int m = desc->m, n = desc->n;
    const int nnz = desc->data.coo.nnz;
    const int *r = desc->data.coo.row_ind;
    const int *c = desc->data.coo.col_ind;
    const double *v = desc->data.coo.vals;

    *row_ptr = (int *)safe_malloc((size_t)(m + 1) * sizeof(int));
    *col_ind = (int *)safe_malloc((size_t)nnz * sizeof(int));
    *vals = (double *)safe_malloc((size_t)nnz * sizeof(double));

    // count entries per row
    for (int i = 0; i <= m; ++i)
        (*row_ptr)[i] = 0;

    for (int k = 0; k < nnz; ++k)
    {
        int ri = r[k];
        if (ri < 0 || ri >= m)
        {
            fprintf(stderr, "[interface] COO: row index out of range\n");
            return -1;
        }
        ++((*row_ptr)[ri + 1]);
    }

    // exclusive scan
    for (int i = 0; i < m; ++i)
        (*row_ptr)[i + 1] += (*row_ptr)[i];

    // next position to fill in each row
    int *next = (int *)safe_malloc((size_t)m * sizeof(int));
    for (int i = 0; i < m; ++i)
        next[i] = (*row_ptr)[i];

    // fill column indices and values
    for (int k = 0; k < nnz; ++k)
    {
        int ri = r[k], cj = c[k];
        if (cj < 0 || cj >= n)
        {
            fprintf(stderr, "[interface] COO: col index out of range\n");
            free(next);
            return -1;
        }
        int pos = next[ri]++;
        (*col_ind)[pos] = cj;
        (*vals)[pos] = v[k];
    }

    free(next);
    return 0;
}

void check_feas_polishing_termination_criteria(pdhg_solver_state_t *solver_state,
                                               const pdhg_solver_state_t *ori_solver_state,
                                               const termination_criteria_t *criteria,
                                               bool is_primal_polish)
{
    solver_state->cumulative_time_sec = (double)(clock() - solver_state->start_time) / CLOCKS_PER_SEC;
    if (is_primal_polish)
    {
        if (solver_state->relative_primal_residual <= criteria->eps_feas_polish_relative)
        {
            solver_state->termination_reason = TERMINATION_REASON_FEAS_POLISH_SUCCESS;
            return;
        }
    }
    else
    {
        if (solver_state->relative_dual_residual <= criteria->eps_feas_polish_relative)
        {
            solver_state->termination_reason = TERMINATION_REASON_FEAS_POLISH_SUCCESS;
            return;
        }
    }
    if (solver_state->total_count >= criteria->iteration_limit)
    {
        solver_state->termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
        return;
    }
    double total_time_sec = (double)(clock() - ori_solver_state->start_time) / CLOCKS_PER_SEC;
    if (total_time_sec >= criteria->time_sec_limit)
    {
        solver_state->termination_reason = TERMINATION_REASON_TIME_LIMIT;
        return;
    }
}

void print_initial_feas_polish_info(bool is_primal_polish, const pdhg_parameters_t *params)
{
    if (!params->verbose)
    {
        return;
    }
    printf("---------------------------------------------------------------------------------------\n");
    printf("Starting %s Feasibility Polishing Phase with relative tolerance %.2e\n",
           is_primal_polish ? "Primal" : "Dual",
           params->termination_criteria.eps_feas_polish_relative);
    printf("---------------------------------------------------------------------------------------\n");
    if (is_primal_polish)
        printf("%s %s |  %s  | %s | %s \n", "  iter", "  time ", "pr obj", " abs pr res ", " rel pr res ");
    else
        printf("%s %s |  %s  | %s | %s \n", "  iter", "  time ", "du obj", " abs du res ", " rel du res ");
    printf("---------------------------------------------------------------------------------------\n");
}

void pdhg_feas_polish_final_log(const pdhg_solver_state_t *primal_state,
                                const pdhg_solver_state_t *dual_state,
                                bool verbose)
{
    if (!verbose)
    {
        return;
    }
    printf("---------------------------------------------------------------------------------------\n");
    printf("Feasibility Polishing Summary\n");
    printf("  Primal Status        : %s\n", termination_reason_to_string(primal_state->termination_reason));
    printf("  Primal Iterations    : %d\n", primal_state->total_count);
    printf("  Primal Time Usage    : %.3g sec\n", primal_state->cumulative_time_sec);
    printf("  Dual Status          : %s\n", termination_reason_to_string(dual_state->termination_reason));
    printf("  Dual Iterations      : %d\n", dual_state->total_count);
    printf("  Dual Time Usage      : %.3g sec\n", dual_state->cumulative_time_sec);
    printf("  Primal Residual      : %.3e\n", primal_state->relative_primal_residual);
    printf("  Dual Residual        : %.3e\n", dual_state->relative_dual_residual);
    printf("  Primal Dual Gap      : %.3e\n",
           fabs(primal_state->primal_objective_value - dual_state->dual_objective_value) /
               (1.0 + fabs(primal_state->primal_objective_value) + fabs(dual_state->dual_objective_value)));
}

void display_feas_polish_iteration_stats(const pdhg_solver_state_t *state, bool verbose, bool is_primal_polish)
{
    if (!verbose)
    {
        return;
    }
    if (state->total_count % get_print_frequency(state->total_count) == 0)
    {
        if (is_primal_polish)
        {
            printf("%6d %.1e | %8.1e |    %.1e   |   %.1e   \n",
                   state->total_count,
                   state->cumulative_time_sec,
                   state->original_objective_sign * state->primal_objective_value,
                   state->absolute_primal_residual,
                   state->relative_primal_residual);
        }
        else
        {
            printf("%6d %.1e | %8.1e |    %.1e   |   %.1e   \n",
                   state->total_count,
                   state->cumulative_time_sec,
                   state->original_objective_sign * state->dual_objective_value,
                   state->absolute_dual_residual,
                   state->relative_dual_residual);
        }
    }
}

__global__ void compute_primal_feas_polish_residual_kernel(double *__restrict__ primal_residual,
                                                           const double *__restrict__ primal_product,
                                                           const double *__restrict__ constraint_lower_bound,
                                                           const double *__restrict__ constraint_upper_bound,
                                                           const double *__restrict__ constraint_rescaling,
                                                           int num_constraints)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_constraints)
    {

        double clamped_val = fmax(constraint_lower_bound[i], fmin(primal_product[i], constraint_upper_bound[i]));
        primal_residual[i] = (primal_product[i] - clamped_val) * constraint_rescaling[i];
    }
}

__global__ void compute_dual_feas_polish_residual_kernel(double *__restrict__ dual_residual,
                                                         const double *__restrict__ dual_solution,
                                                         const double *__restrict__ dual_product,
                                                         const double *__restrict__ dual_slack,
                                                         const double *__restrict__ objective_vector,
                                                         const double *__restrict__ variable_rescaling,
                                                         double *__restrict__ dual_obj_contribution,
                                                         const double *__restrict__ const_lb_finite,
                                                         const double *__restrict__ const_ub_finite,
                                                         int num_variables,
                                                         int num_constraints)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_variables)
    {
        dual_residual[i] = (objective_vector[i] - dual_product[i] - dual_slack[i]) * variable_rescaling[i];
    }
    else if (i < num_constraints + num_variables)
    {
        int idx = i - num_variables;
        dual_obj_contribution[idx] =
            fmax(dual_solution[idx], 0.0) * const_lb_finite[idx] + fmin(dual_solution[idx], 0.0) * const_ub_finite[idx];
    }
}

void compute_primal_feas_polish_residual(pdhg_solver_state_t *state,
                                         const pdhg_solver_state_t *ori_state,
                                         norm_type_t optimality_norm)
{
    cupdlpx_spmv_Ax(state->sparse_handle, state->spmv_ctx, state->pdhg_primal_solution, state->primal_product);

    compute_primal_feas_polish_residual_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
        state->primal_residual,
        state->primal_product,
        state->constraint_lower_bound,
        state->constraint_upper_bound,
        state->constraint_rescaling,
        state->num_constraints);

    if (optimality_norm == NORM_TYPE_L_INF)
    {
        state->absolute_primal_residual =
            get_vector_inf_norm(state->blas_handle, state->num_constraints, state->primal_residual);
    }
    else
    {
        CUBLAS_CHECK(cublasDnrm2_v2_64(
            state->blas_handle, state->num_constraints, state->primal_residual, 1, &state->absolute_primal_residual));
    }

    state->absolute_primal_residual /= state->constraint_bound_rescaling;

    state->relative_primal_residual = state->absolute_primal_residual / (1.0 + state->constraint_bound_norm);

    CUBLAS_CHECK(cublasDdot(state->blas_handle,
                            state->num_variables,
                            ori_state->objective_vector,
                            1,
                            state->pdhg_primal_solution,
                            1,
                            &state->primal_objective_value));
    state->primal_objective_value =
        state->primal_objective_value / (state->constraint_bound_rescaling * state->objective_vector_rescaling) +
        state->objective_constant;
}

void compute_dual_feas_polish_residual(pdhg_solver_state_t *state,
                                       const pdhg_solver_state_t *ori_state,
                                       norm_type_t optimality_norm)
{
    cupdlpx_spmv_ATx(state->sparse_handle, state->spmv_ctx, state->pdhg_dual_solution, state->dual_product);

    compute_dual_feas_polish_residual_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK>>>(
        state->dual_residual,
        state->pdhg_dual_solution,
        state->dual_product,
        state->dual_slack,
        state->objective_vector,
        state->variable_rescaling,
        state->primal_slack,
        ori_state->constraint_lower_bound_finite_val,
        ori_state->constraint_upper_bound_finite_val,
        state->num_variables,
        state->num_constraints);

    if (optimality_norm == NORM_TYPE_L_INF)
    {
        state->absolute_dual_residual =
            get_vector_inf_norm(state->blas_handle, state->num_variables, state->dual_residual);
    }
    else
    {
        CUBLAS_CHECK(cublasDnrm2_v2_64(
            state->blas_handle, state->num_variables, state->dual_residual, 1, &state->absolute_dual_residual));
    }

    state->absolute_dual_residual /= state->objective_vector_rescaling;

    state->relative_dual_residual = state->absolute_dual_residual / (1.0 + state->objective_vector_norm);

    double base_dual_objective;
    CUBLAS_CHECK(cublasDdot(state->blas_handle,
                            state->num_variables,
                            state->dual_slack,
                            1,
                            ori_state->pdhg_primal_solution,
                            1,
                            &base_dual_objective));
    double dual_slack_sum =
        get_vector_sum(state->blas_handle, state->num_constraints, state->ones_dual_d, state->primal_slack);
    state->dual_objective_value = (base_dual_objective + dual_slack_sum) /
            (state->constraint_bound_rescaling * state->objective_vector_rescaling) +
        state->objective_constant;
}
