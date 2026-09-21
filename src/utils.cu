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

#include <string.h>

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

        if (residual_norm < tolerance * fmin(1.0, sigma_max_sq))
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
    params->geometric_mean_iterations = 12;
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
    params->termination_criteria.eps_infeasible_relative = 1e-8;
    params->termination_criteria.infeasible_ray_objective_margin = 1e-8;

    params->restart_params.artificial_restart_threshold = 0.36;
    params->restart_params.sufficient_reduction_for_restart = 0.2;
    params->restart_params.necessary_reduction_for_restart = 0.5;
    params->restart_params.k_p = 0.99;
    params->restart_params.k_i = 0.01;
    params->restart_params.k_d = 0.0;
    params->restart_params.i_smooth = 0.3;

    params->optimality_norm = NORM_TYPE_L2;
    params->presolve = true;
    params->infeasibility_detection = true;
    params->matrix_zero_tol = 1e-9;
    params->infinite_bound = 1e20;
}

#define MATRIX_LARGE_VALUE 1e15
#define OBJECTIVE_LARGE_VALUE 1e20

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
    int num_large = 0;
    double max_large = 0.0;
    for (int i = 0; i < num_rows; ++i)
    {
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k)
        {
            const double abs_value = fabs(vals[k]);
            if (abs_value > params->matrix_zero_tol)
            {
                ++filtered_nnz;
            }
            if (abs_value >= MATRIX_LARGE_VALUE)
            {
                ++num_large;
                if (abs_value > max_large)
                    max_large = abs_value;
            }
        }
    }

    if (num_large > 0)
    {
        fprintf(stderr,
                "WARNING: %d constraint matrix %s |value| >= %.1e (largest %.3e); the problem is badly scaled.\n",
                num_large,
                (num_large == 1 ? "entry has" : "entries have"),
                MATRIX_LARGE_VALUE,
                max_large);
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

/* Bounds at or beyond +/-infinite_bound stand for an infinite bound. */
static void replace_large_bounds_with_infinity(
    const double *in_lower, const double *in_upper, double **lower, double **upper, int n, double infinite_bound)
{
    bool any = false;
    for (int i = 0; i < n && !any; ++i)
        any = in_lower[i] <= -infinite_bound || in_upper[i] >= infinite_bound;
    if (!any)
        return;

    double *new_lower = (double *)safe_malloc((size_t)n * sizeof(double));
    double *new_upper = (double *)safe_malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; ++i)
    {
        new_lower[i] = (in_lower[i] <= -infinite_bound) ? -INFINITY : in_lower[i];
        new_upper[i] = (in_upper[i] >= infinite_bound) ? INFINITY : in_upper[i];
    }
    *lower = new_lower;
    *upper = new_upper;
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
    replace_large_bounds_with_infinity(original->variable_lower_bound,
                                       original->variable_upper_bound,
                                       &working.variable_lower_bound,
                                       &working.variable_upper_bound,
                                       original->num_variables,
                                       params->infinite_bound);
    replace_large_bounds_with_infinity(original->constraint_lower_bound,
                                       original->constraint_upper_bound,
                                       &working.constraint_lower_bound,
                                       &working.constraint_upper_bound,
                                       original->num_constraints,
                                       params->infinite_bound);

    int num_large_obj = 0;
    double max_large_obj = 0.0;
    for (int i = 0; i < original->num_variables; ++i)
    {
        const double abs_cost = fabs(original->objective_vector[i]);
        if (abs_cost >= OBJECTIVE_LARGE_VALUE)
        {
            ++num_large_obj;
            if (abs_cost > max_large_obj)
                max_large_obj = abs_cost;
        }
    }
    if (num_large_obj > 0)
    {
        fprintf(stderr,
                "WARNING: %d objective %s |value| >= %.1e (largest %.3e); the problem is badly scaled.\n",
                num_large_obj,
                (num_large_obj == 1 ? "coefficient has" : "coefficients have"),
                OBJECTIVE_LARGE_VALUE,
                max_large_obj);
    }

    filter_constraint_matrix_entries(&working, original, params);
    return working;
}

void free_preprocessed_problem(const lp_problem_t *preprocessed, const lp_problem_t *original)
{
    if (preprocessed->objective_vector != original->objective_vector)
        free(preprocessed->objective_vector);
    if (preprocessed->variable_lower_bound != original->variable_lower_bound)
    {
        free(preprocessed->variable_lower_bound);
        free(preprocessed->variable_upper_bound);
    }
    if (preprocessed->constraint_lower_bound != original->constraint_lower_bound)
    {
        free(preprocessed->constraint_lower_bound);
        free(preprocessed->constraint_upper_bound);
    }
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

    PRINT_DIFF_INT("geo_mean_iter", params->geometric_mean_iterations, default_params.geometric_mean_iterations);
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
    PRINT_DIFF_DBL("eps_infeasible_relative",
                   params->termination_criteria.eps_infeasible_relative,
                   default_params.termination_criteria.eps_infeasible_relative);
    PRINT_DIFF_DBL("infeasible_ray_objective_margin",
                   params->termination_criteria.infeasible_ray_objective_margin,
                   default_params.termination_criteria.infeasible_ray_objective_margin);
    PRINT_DIFF_BOOL("presolve", params->presolve, default_params.presolve);
    PRINT_DIFF_BOOL("infeasibility_detection", params->infeasibility_detection, default_params.infeasibility_detection);
    PRINT_DIFF_DBL("matrix_zero_tol", params->matrix_zero_tol, default_params.matrix_zero_tol);
    PRINT_DIFF_DBL("infinite_bound", params->infinite_bound, default_params.infinite_bound);
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
