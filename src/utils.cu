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

__global__ void elementwise_mask_kernel(double *__restrict__ v, const bool *__restrict__ mask, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        v[i] = mask[i] ? v[i] : 0.0;
}

__global__ void warm_start_fill_kernel(double *__restrict__ v, const bool *__restrict__ mask, double amplitude, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    if (mask[i] && v[i] == 0.0)
    {
        /* Deterministic per-index noise (splitmix64 hash to [-0.5, 0.5)) preserves
           reproducibility without RNG state or a host-to-device copy. */
        unsigned long long z = (unsigned long long)i + 0x9E3779B97F4A7C15ULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        double u = (double)(z >> 11) * (1.0 / 9007199254740992.0); /* [0, 1) */
        v[i] = amplitude * (u - 0.5);
    }
}

/* Masks are applied elementwise per run, so a cached context stays valid across mask
   changes; it must be recreated only if the matrix itself changes. */
struct sv_estimator_ctx
{
    cusparseHandle_t sparse_handle;
    cublasHandle_t blas_handle;
    int num_rows, num_cols;
    double *eigenvector_d;
    double *next_eigenvector_d;
    double *dual_product_d;
    cusparseSpMatDescr_t matA, matAT;
    cusparseDnVecDescr_t vecEigen, vecNextEigen, vecDual;
    void *descrAT, *descrA;
    void *planAT, *planA;
    void *dBufferAT, *dBufferA;
    bool have_warm_start; /* eigenvector_d contains the final vector from the previous estimate */
};

sv_estimator_ctx_t *sv_estimator_create(cusparseHandle_t sparse_handle,
                                        cublasHandle_t blas_handle,
                                        const cu_sparse_matrix_csr_t *A,
                                        const cu_sparse_matrix_csr_t *AT)
{
    sv_estimator_ctx_t *ctx = (sv_estimator_ctx_t *)safe_calloc(1, sizeof(sv_estimator_ctx_t));
    ctx->sparse_handle = sparse_handle;
    ctx->blas_handle = blas_handle;
    const int m = A->num_rows;
    const int n = A->num_cols;
    ctx->num_rows = m;
    ctx->num_cols = n;
    CUDA_CHECK(cudaMalloc(&ctx->eigenvector_d, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx->next_eigenvector_d, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx->dual_product_d, n * sizeof(double)));
    CUSPARSE_CHECK(cusparseCreateCsr(&ctx->matA,
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
    CUSPARSE_CHECK(cusparseCreateCsr(&ctx->matAT,
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
    CUSPARSE_CHECK(cusparseCreateDnVec(&ctx->vecEigen, m, ctx->eigenvector_d, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&ctx->vecNextEigen, m, ctx->next_eigenvector_d, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&ctx->vecDual, n, ctx->dual_product_d, CUDA_R_64F));
    size_t bufferSizeAT = 0, bufferSizeA = 0;
    cupdlpx_spmv_buffer_size(sparse_handle, ctx->matAT, ctx->vecNextEigen, ctx->vecDual, &bufferSizeAT);
    cupdlpx_spmv_buffer_size(sparse_handle, ctx->matA, ctx->vecDual, ctx->vecEigen, &bufferSizeA);
    CUDA_CHECK(cudaMalloc(&ctx->dBufferAT, bufferSizeAT));
    CUDA_CHECK(cudaMalloc(&ctx->dBufferA, bufferSizeA));
    cupdlpx_spmv_prepare(
        sparse_handle, ctx->matAT, ctx->vecNextEigen, ctx->vecDual, ctx->dBufferAT, &ctx->descrAT, &ctx->planAT);
    cupdlpx_spmv_prepare(
        sparse_handle, ctx->matA, ctx->vecDual, ctx->vecEigen, ctx->dBufferA, &ctx->descrA, &ctx->planA);
    return ctx;
}

void sv_estimator_free(sv_estimator_ctx_t *ctx)
{
    if (!ctx)
        return;
    cupdlpx_spmv_release(ctx->descrAT, ctx->planAT);
    cupdlpx_spmv_release(ctx->descrA, ctx->planA);
    CUDA_CHECK(cudaFree(ctx->dBufferAT));
    CUDA_CHECK(cudaFree(ctx->dBufferA));
    CUSPARSE_CHECK(cusparseDestroySpMat(ctx->matA));
    CUSPARSE_CHECK(cusparseDestroySpMat(ctx->matAT));
    CUSPARSE_CHECK(cusparseDestroyDnVec(ctx->vecEigen));
    CUSPARSE_CHECK(cusparseDestroyDnVec(ctx->vecNextEigen));
    CUSPARSE_CHECK(cusparseDestroyDnVec(ctx->vecDual));
    CUDA_CHECK(cudaFree(ctx->eigenvector_d));
    CUDA_CHECK(cudaFree(ctx->next_eigenvector_d));
    CUDA_CHECK(cudaFree(ctx->dual_product_d));
    free(ctx);
}

sv_estimator_result_t sv_estimator_run(sv_estimator_ctx_t *ctx, const sv_estimator_opts_t *opts)
{
    cusparseHandle_t sparse_handle = ctx->sparse_handle;
    cublasHandle_t blas_handle = ctx->blas_handle;
    const int m = ctx->num_rows;
    const int n = ctx->num_cols;
    const bool *d_row_mask = opts->d_row_mask;
    const bool *d_col_mask = opts->d_col_mask;
    double *eigenvector_d = ctx->eigenvector_d;
    double *next_eigenvector_d = ctx->next_eigenvector_d;

    sv_estimator_result_t result = {};
    result.status = SV_ESTIMATOR_MAX_ITER;
    double max_singular_value_squared = 1.0;
    const double one = 1.0;

    cudaStream_t handle_stream = 0;
    CUSPARSE_CHECK(cusparseGetStream(sparse_handle, &handle_stream));
    const int row_blocks = (m + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const int col_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    bool random_start = !ctx->have_warm_start;
    if (!random_start && d_row_mask)
    {
        elementwise_mask_kernel<<<row_blocks, THREADS_PER_BLOCK, 0, handle_stream>>>(eigenvector_d, d_row_mask, m);
        /* Rows added to the mask after the previous estimate have zero entries in the
           warm-start vector and could be omitted, causing the masked singular value to be
           underestimated. Initialize these entries with small deterministic perturbations.
           A masked warm vector that vanished (mask disjoint from the previous eigenvector's
           support) carries no direction at all: start over from a random vector. */
        double warm_norm = 0.0;
        CUBLAS_CHECK(cublasDnrm2_v2_64(blas_handle, m, eigenvector_d, 1, &warm_norm));
        if (isfinite(warm_norm) && warm_norm > 0.0)
        {
            double amplitude = 1e-3 * warm_norm / sqrt((double)m);
            warm_start_fill_kernel<<<row_blocks, THREADS_PER_BLOCK, 0, handle_stream>>>(
                eigenvector_d, d_row_mask, amplitude, m);
        }
        else
        {
            random_start = true;
        }
    }
    if (random_start)
    {
        double *eigenvector_h = (double *)safe_malloc(m * sizeof(double));
        for (int i = 0; i < m; ++i)
        {
            eigenvector_h[i] = dist(gen);
        }
        CUDA_CHECK(cudaMemcpy(eigenvector_d, eigenvector_h, m * sizeof(double), cudaMemcpyHostToDevice));
        free(eigenvector_h);
        if (d_row_mask)
        {
            elementwise_mask_kernel<<<row_blocks, THREADS_PER_BLOCK, 0, handle_stream>>>(eigenvector_d, d_row_mask, m);
        }
    }

    for (int i = 0; i < opts->max_iterations; ++i)
    {
        result.iterations = i + 1;
        CUDA_CHECK(cudaMemcpy(next_eigenvector_d, eigenvector_d, m * sizeof(double), cudaMemcpyDeviceToDevice));
        double eigenvector_norm;
        CUBLAS_CHECK(cublasDnrm2_v2_64(blas_handle, m, next_eigenvector_d, 1, &eigenvector_norm));
        if (!isfinite(eigenvector_norm) || eigenvector_norm <= 0.0)
        {
            result.status = SV_ESTIMATOR_DEGENERATE;
            break;
        }

        double inv_eigenvector_norm = 1.0 / eigenvector_norm;
        CUBLAS_CHECK(cublasDscal(blas_handle, m, &inv_eigenvector_norm, next_eigenvector_d, 1));

        cupdlpx_spmv_execute(sparse_handle, ctx->matAT, ctx->vecNextEigen, ctx->vecDual, ctx->dBufferAT, ctx->planAT);
        if (d_col_mask)
            elementwise_mask_kernel<<<col_blocks, THREADS_PER_BLOCK, 0, handle_stream>>>(
                ctx->dual_product_d, d_col_mask, n);
        cupdlpx_spmv_execute(sparse_handle, ctx->matA, ctx->vecDual, ctx->vecEigen, ctx->dBufferA, ctx->planA);
        if (d_row_mask)
            elementwise_mask_kernel<<<row_blocks, THREADS_PER_BLOCK, 0, handle_stream>>>(eigenvector_d, d_row_mask, m);

        CUBLAS_CHECK(cublasDdot(blas_handle, m, next_eigenvector_d, 1, eigenvector_d, 1, &max_singular_value_squared));
        if (!isfinite(max_singular_value_squared) || max_singular_value_squared <= 0.0)
        {
            result.status = SV_ESTIMATOR_DEGENERATE;
            break;
        }

        if (opts->abort_singular_value_threshold > 0.0 &&
            sqrt(max_singular_value_squared) >= opts->abort_singular_value_threshold)
        {
            result.status = SV_ESTIMATOR_ABORTED;
            break;
        }

        double negative_max_singular_value_squared = -max_singular_value_squared;
        CUBLAS_CHECK(cublasDscal(blas_handle, m, &negative_max_singular_value_squared, next_eigenvector_d, 1));
        CUBLAS_CHECK(cublasDaxpy(blas_handle, m, &one, eigenvector_d, 1, next_eigenvector_d, 1));

        double residual_norm;
        CUBLAS_CHECK(cublasDnrm2_v2_64(blas_handle, m, next_eigenvector_d, 1, &residual_norm));
        if (!isfinite(residual_norm))
        {
            result.status = SV_ESTIMATOR_DEGENERATE;
            break;
        }

        /* Use an absolute test above max_singular_value_squared = 1 and a relative test below it;
           a purely absolute test is vacuously satisfied for operators with norm far below 1. */
        if (residual_norm < opts->tolerance * fmin(1.0, max_singular_value_squared))
        {
            result.status = SV_ESTIMATOR_CONVERGED;
            break;
        }
    }

    if (result.status != SV_ESTIMATOR_DEGENERATE)
    {
        result.max_singular_value = sqrt(max_singular_value_squared);
    }
    ctx->have_warm_start = result.status != SV_ESTIMATOR_DEGENERATE;
    return result;
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

bool should_do_adaptive_restart(pdhg_solver_state_t *solver_state,
                                const restart_parameters_t *restart_params,
                                int termination_evaluation_frequency)
{
    const char *reason = NULL;
    if (solver_state->total_count == termination_evaluation_frequency)
    {
        /* inner_count == total_count at the first check, so the long-inner-loop criterion holds */
        reason = "long inner loop";
    }
    else if (solver_state->total_count > termination_evaluation_frequency)
    {
        if (solver_state->fixed_point_error <=
            restart_params->sufficient_reduction_for_restart * solver_state->initial_fixed_point_error)
        {
            reason = "sufficient decay";
        }
        else if (solver_state->fixed_point_error <=
                     restart_params->necessary_reduction_for_restart * solver_state->initial_fixed_point_error &&
                 solver_state->fixed_point_error > solver_state->last_trial_fixed_point_error)
        {
            reason = "necessary decay + no local progress";
        }
        else if (solver_state->inner_count >= restart_params->artificial_restart_threshold * solver_state->total_count)
        {
            reason = "long inner loop";
        }
    }
    solver_state->last_trial_fixed_point_error = solver_state->fixed_point_error;
    if (reason)
    {
        solver_state->last_restart_reason = reason;
    }
    return reason != NULL;
}

void set_default_parameters(pdhg_parameters_t *params)
{
    params->geometric_mean_iterations = 12;
    params->l_inf_ruiz_iterations = 10;
    params->has_pock_chambolle_alpha = true;
    params->pock_chambolle_alpha = 1.0;
    params->bound_objective_rescaling = true;
    params->verbose = true;
    params->debug = false;
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
    params->termination_criteria.eps_infeasible_relative = 1e-10;

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
    params->infinite_bound = 1e20;
    params->active_set_boost = true;
    params->asb_activation_tol = 1e-4;
    params->asb_window_iter = 10000;
    params->asb_safety_factor = 0.9;
    params->asb_max_reverts = 2;
    params->asb_min_raise_ratio = 1.1;
    params->asb_reestimate_change_ratio = 0.01;
    params->asb_constraint_tol = 1e-8;
    params->asb_variable_tol = 1e-8;
    params->asb_divergence_ceiling_ratio = 0.7;
    params->asb_divergence_margin = 0.05;
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

/* Width of the iteration table for the given options; the banner uses it too. */
static int iteration_table_width(const pdhg_parameters_t *params)
{
    const bool asb_columns = params->debug && params->active_set_boost;
    return 88 + (params->debug ? 10 : 0) + (asb_columns ? 30 : 0);
}

static void print_rule(int width)
{
    for (int i = 0; i < width; ++i)
        putchar('-');
    putchar('\n');
}

static void print_centered(const char *text, int width)
{
    int pad = (width - (int)strlen(text)) / 2;
    printf("%*s%s\n", pad > 0 ? pad : 0, "", text);
}

void print_initial_info(const pdhg_parameters_t *params, const lp_problem_t *problem)
{
    pdhg_parameters_t default_params;
    set_default_parameters(&default_params);
    if (!params->verbose)
    {
        return;
    }
    const int width = iteration_table_width(params);
    char version_line[64];
    snprintf(version_line, sizeof(version_line), "cuPDLPx v%s", CUPDLPX_VERSION);
    print_rule(width);
    print_centered(version_line, width);
    print_centered("A GPU-Accelerated First-Order LP Solver", width);
    print_centered("(c) Haihao Lu, Massachusetts Institute of Technology, 2025", width);
    print_rule(width);

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
    PRINT_DIFF_BOOL("presolve", params->presolve, default_params.presolve);
    PRINT_DIFF_BOOL("debug", params->debug, default_params.debug);
    PRINT_DIFF_DBL("matrix_zero_tol", params->matrix_zero_tol, default_params.matrix_zero_tol);
    PRINT_DIFF_DBL("infinite_bound", params->infinite_bound, default_params.infinite_bound);
    PRINT_DIFF_BOOL("active_set_boost", params->active_set_boost, default_params.active_set_boost);
    PRINT_DIFF_DBL("asb_activation_tol", params->asb_activation_tol, default_params.asb_activation_tol);
    PRINT_DIFF_INT("asb_window_iter", params->asb_window_iter, default_params.asb_window_iter);
    PRINT_DIFF_DBL("asb_safety_factor", params->asb_safety_factor, default_params.asb_safety_factor);
    PRINT_DIFF_INT("asb_max_reverts", params->asb_max_reverts, default_params.asb_max_reverts);
    PRINT_DIFF_DBL("asb_min_raise_ratio", params->asb_min_raise_ratio, default_params.asb_min_raise_ratio);
    PRINT_DIFF_DBL(
        "asb_reestimate_change_ratio", params->asb_reestimate_change_ratio, default_params.asb_reestimate_change_ratio);
    PRINT_DIFF_DBL("asb_constraint_tol", params->asb_constraint_tol, default_params.asb_constraint_tol);
    PRINT_DIFF_DBL("asb_variable_tol", params->asb_variable_tol, default_params.asb_variable_tol);
    PRINT_DIFF_DBL("asb_divergence_ceiling_ratio",
                   params->asb_divergence_ceiling_ratio,
                   default_params.asb_divergence_ceiling_ratio);
    PRINT_DIFF_DBL("asb_divergence_margin", params->asb_divergence_margin, default_params.asb_divergence_margin);
}

#undef PRINT_DIFF_INT
#undef PRINT_DIFF_DBL
#undef PRINT_DIFF_BOOL

void pdhg_final_log(const cupdlpx_result_t *result, const pdhg_parameters_t *params)
{
    if (params->verbose)
    {
        print_rule(iteration_table_width(params));
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
        if (params->active_set_boost)
        {
            printf("  Active set boost       : %d raises, %d reverts, %d PI iterations\n",
                   result->asb_raise_count,
                   result->asb_revert_count,
                   result->asb_pi_iterations);
        }
    }
}

void display_iteration_header(const pdhg_parameters_t *params)
{
    if (!params->verbose)
    {
        return;
    }
    const bool asb_columns = params->debug && params->active_set_boost;
    const int width = iteration_table_width(params);
    printf("\n*: restart triggered\n");
    print_rule(width);
    printf(" %s | %s | %s | %s",
           "   runtime    ",
           "    objective     ",
           "  absolute residuals   ",
           "  relative residuals   ");
    if (params->debug)
    {
        printf(" | %s", " primal");
    }
    if (asb_columns)
    {
        printf(" | %s", "      active-set boost     ");
    }
    printf(" \n");
    printf(" %s %s | %s %s | %s %s %s | %s %s %s",
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
    if (params->debug)
    {
        printf(" | %s", " weight");
    }
    if (asb_columns)
    {
        printf(" | %s %s %s", "  step ", "free var", "active con");
    }
    printf(" \n");
    print_rule(width);
}

void display_iteration_stats(pdhg_solver_state_t *state, const pdhg_parameters_t *params)
{
    if (!params->verbose)
    {
        return;
    }
    if (state->total_count % get_print_frequency(state->total_count) == 0)
    {
        /* a leading star marks that at least one restart happened since the previous row */
        char restart_marker = state->restart_count > state->logged_restart_count ? '*' : ' ';
        state->logged_restart_count = state->restart_count;
        printf("%c%6d %.1e | %8.1e  %8.1e | %.1e %.1e %.1e | %.1e %.1e %.1e",
               restart_marker,
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
        if (params->debug)
        {
            printf(" | %.1e", state->primal_weight);
        }
        if (params->debug && params->active_set_boost)
        {
            long free_variables = state->asb_free_variables;
            long binding_constraints = state->asb_binding_constraints;
            if (state->total_count == 0)
            {
                free_variables = state->num_variables;
                binding_constraints = state->num_constraints;
            }
            printf(" | %.1e %8ld %10ld", state->step_size, free_variables, binding_constraints);
        }
        printf(" \n");
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
                                                                           state->infeasibility_dual_scratch,
                                                                           state->variable_lower_bound_finite_val,
                                                                           state->variable_upper_bound_finite_val,
                                                                           state->num_variables);

    double sum_primal_slack =
        get_vector_sum(state->blas_handle, state->num_constraints, state->ones_dual_d, state->primal_slack);
    double sum_dual_slack = get_vector_sum(
        state->blas_handle, state->num_variables, state->ones_primal_d, state->infeasibility_dual_scratch);
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
        state->infeasibility_dual_scratch,
        state->variable_rescaling);

    state->max_primal_ray_infeasibility =
        get_vector_inf_norm(state->blas_handle, state->num_constraints, state->primal_slack) /
        state->constraint_bound_rescaling;
    double dual_scratch_norm =
        get_vector_inf_norm(state->blas_handle, state->num_variables, state->infeasibility_dual_scratch);
    state->max_dual_ray_infeasibility = dual_scratch_norm / state->objective_vector_rescaling;

    double scaling_factor = fmax(dual_ray_inf_norm, dual_scratch_norm);
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
