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

#include "preconditioner.h"
#include "utils.h"
#include <time.h>
#include <stdio.h>
#include <math.h>

#include <cuda_runtime.h>

#define SCALING_EPSILON 1e-12

__global__ void invert_vec_kernel(const double* __restrict__ x, double* __restrict__ invx, int n);
__global__ void scale_variables_kernel(double* __restrict__ c,
                                       double* __restrict__ var_lb,
                                       double* __restrict__ var_ub,
                                       double* __restrict__ var_lb_finite,
                                       double* __restrict__ var_ub_finite,
                                       const double* __restrict__ D,
                                       const double* __restrict__ invD,
                                       int n);
__global__ void scale_constraints_kernel(double* __restrict__ con_lb,
                                         double* __restrict__ con_ub,
                                         double* __restrict__ con_lb_finite,
                                         double* __restrict__ con_ub_finite,
                                         const double* __restrict__ invE,
                                         int m);
__global__ void csr_scale_nnz_kernel(const int* __restrict__ row_ids,
                                     const int* __restrict__ col_ind,
                                     double* __restrict__ vals,
                                     const double* __restrict__ invD,
                                     const double* __restrict__ invE,
                                     int nnz);
static void scale_problem(pdhg_solver_state_t *state, const double *con_rescale, const double *var_rescale);
static void ruiz_rescaling(pdhg_solver_state_t *state, int num_iters, double *cum_con_rescale, double *cum_var_rescale);
static void pock_chambolle_rescaling(pdhg_solver_state_t *state, double alpha, double *cum_con_rescale, double *cum_var_rescale);
static void bound_objective_rescaling(pdhg_solver_state_t *state, rescale_info_t *rescale_info);

__global__ void invert_vec_kernel(const double* __restrict__ x, double* __restrict__ invx, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    double v = x[t];
    if (fabs(v) < SCALING_EPSILON) v = (v < 0 ? -SCALING_EPSILON : SCALING_EPSILON);
    invx[t] = 1.0 / v;
}

__global__ void scale_variables_kernel(double* __restrict__ c,
                                       double* __restrict__ var_lb,
                                       double* __restrict__ var_ub,
                                       double* __restrict__ var_lb_finite,
                                       double* __restrict__ var_ub_finite,
                                       const double* __restrict__ D,
                                       const double* __restrict__ invD,
                                       int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    double dj = D[j];
    double inv_dj = invD[j];
    c[j]      *= inv_dj;
    var_lb[j] *= dj;
    var_ub[j] *= dj;
    var_lb_finite[j] *= dj;
    var_ub_finite[j] *= dj;
}

__global__ void scale_constraints_kernel(double* __restrict__ con_lb,
                                         double* __restrict__ con_ub,
                                         double* __restrict__ con_lb_finite,
                                         double* __restrict__ con_ub_finite,
                                         const double* __restrict__ invE,
                                         int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    double inv_ei = invE[i];
    con_lb[i] *= inv_ei;
    con_ub[i] *= inv_ei;
    con_lb_finite[i] *= inv_ei;
    con_ub_finite[i] *= inv_ei;
}

__global__ void csr_scale_nnz_kernel(const int* __restrict__ row_ids,
                                     const int* __restrict__ col_ind,
                                     double* __restrict__ vals,
                                     const double* __restrict__ invD,
                                     const double* __restrict__ invE,
                                     int nnz)
{
    for (int k = blockIdx.x * blockDim.x + threadIdx.x;
         k < nnz; 
         k += gridDim.x * blockDim.x)
    {
        int i = row_ids[k];
        int j = col_ind[k];
        vals[k] *= invD[j] * invE[i];
    }
}

static void scale_problem(
    pdhg_solver_state_t *state,
    const double *constraint_rescaling,
    const double *variable_rescaling)
{
    const double *E = constraint_rescaling;
    const double *D = variable_rescaling;

    int n_vars = state->num_variables;
    int n_cons = state->num_constraints;

    double *invE=nullptr, *invD=nullptr;
    CUDA_CHECK(cudaMalloc(&invE, n_cons*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&invD, n_vars*sizeof(double)));
    invert_vec_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(E, invE, n_cons);
    invert_vec_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(D, invD, n_vars);

    scale_variables_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->objective_vector,
        state->variable_lower_bound,
        state->variable_upper_bound,
        state->variable_lower_bound_finite_val,
        state->variable_upper_bound_finite_val,
        D,
        invD,
        n_vars);

    scale_constraints_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
        state->constraint_lower_bound,
        state->constraint_upper_bound,
        state->constraint_lower_bound_finite_val,
        state->constraint_upper_bound_finite_val,
        invE,
        n_cons);

    csr_scale_nnz_kernel<<<state->num_blocks_nnz, THREADS_PER_BLOCK>>>(
        state->constraint_matrix->row_ind,
        state->constraint_matrix->col_ind,
        state->constraint_matrix->val,
        invD,
        invE,
        state->constraint_matrix->num_nonzeros);

    CUDA_CHECK(cudaFree(invE));
    CUDA_CHECK(cudaFree(invD));
}

static void ruiz_rescaling(
    pdhg_solver_state_t *state,
    int num_iterations,
    double *cum_constraint_rescaling,
    double *cum_variable_rescaling)
{
    int n_cons = state->num_constraints;
    int n_vars = state->num_variables;
    double *con_rescale = (double*)safe_malloc(n_cons * sizeof(double));
    double *var_rescale = (double*)safe_malloc(n_vars * sizeof(double));

    for (int iter = 0; iter < num_iterations; ++iter)
    {
        for (int i = 0; i < n_vars; ++i)
            var_rescale[i] = 0.0;
        for (int i = 0; i < n_cons; ++i)
            con_rescale[i] = 0.0;

        for (int row = 0; row < n_cons; ++row)
        {
            for (int nz_idx = state->constraint_matrix_row_pointers[row];
                 nz_idx < state->constraint_matrix_row_pointers[row + 1]; ++nz_idx)
            {
                int col = state->constraint_matrix_col_indices[nz_idx];
                if (col < 0 || col >= n_vars)
                {
                    fprintf(stderr, "Error: Invalid column index %d at nz_idx %d for row %d. Must be in [0, %d).\n",
                            col, nz_idx, row, n_vars);
                }
                double val = fabs(state->constraint_matrix_values[nz_idx]);
                if (val > var_rescale[col])
                    var_rescale[col] = val;
                if (val > con_rescale[row])
                    con_rescale[row] = val;
            }
        }

        for (int i = 0; i < n_vars; ++i)
            var_rescale[i] = (var_rescale[i] < SCALING_EPSILON) ? 1.0 : sqrt(var_rescale[i]);
        for (int i = 0; i < n_cons; ++i)
            con_rescale[i] = (con_rescale[i] < SCALING_EPSILON) ? 1.0 : sqrt(con_rescale[i]);

        scale_problem(state, con_rescale, var_rescale);
        for (int i = 0; i < n_vars; ++i)
            cum_variable_rescaling[i] *= var_rescale[i];
        for (int i = 0; i < n_cons; ++i)
            cum_constraint_rescaling[i] *= con_rescale[i];
    }
    free(con_rescale);
    free(var_rescale);
}

static void pock_chambolle_rescaling(
    pdhg_solver_state_t *state,
    const double alpha,
    double *cum_constraint_rescaling,
    double *cum_variable_rescaling)
{
    int num_cons = state->num_constraints;
    int num_vars = state->num_variables;
    double *con_rescale = (double*)safe_calloc(num_cons, sizeof(double));
    double *var_rescale = (double*)safe_calloc(num_vars, sizeof(double));

    for (int row = 0; row < num_cons; ++row)
    {
        for (int nz_idx = state->constraint_matrix_row_pointers[row];
             nz_idx < state->constraint_matrix_row_pointers[row + 1]; ++nz_idx)
        {
            int col = state->constraint_matrix_col_indices[nz_idx];
            double val = fabs(state->constraint_matrix_values[nz_idx]);
            var_rescale[col] += pow(val, 2.0 - alpha);
            con_rescale[row] += pow(val, alpha);
        }
    }

    for (int i = 0; i < num_vars; ++i)
        var_rescale[i] = (var_rescale[i] < SCALING_EPSILON) ? 1.0 : sqrt(var_rescale[i]);
    for (int i = 0; i < num_cons; ++i)
        con_rescale[i] = (con_rescale[i] < SCALING_EPSILON) ? 1.0 : sqrt(con_rescale[i]);

    scale_problem(state, con_rescale, var_rescale);
    for (int i = 0; i < num_vars; ++i)
        cum_variable_rescaling[i] *= var_rescale[i];
    for (int i = 0; i < num_cons; ++i)
        cum_constraint_rescaling[i] *= con_rescale[i];

    free(con_rescale);
    free(var_rescale);
}

static void bound_objective_rescaling(
    pdhg_solver_state_t *state,
    rescale_info_t *rescale_info
)
{

    int n_cons = state->num_constraints;
    int n_vars = state->num_variables;

    double bound_norm_sq = 0.0;
    for (int i = 0; i < n_cons; ++i)
    {
        if (isfinite(state->constraint_lower_bound[i]) && (state->constraint_lower_bound[i] != state->constraint_upper_bound[i]))
        {
            bound_norm_sq += state->constraint_lower_bound[i] * state->constraint_lower_bound[i];
        }
        if (isfinite(state->constraint_upper_bound[i]))
        {
            bound_norm_sq += state->constraint_upper_bound[i] * state->constraint_upper_bound[i];
        }
    }

    double obj_norm_sq = 0.0;
    for (int i = 0; i < n_vars; ++i)
    {
         obj_norm_sq += state->objective_vector[i] * state->objective_vector[i];
    }

    rescale_info->con_bound_rescale = 1.0 / (sqrt(bound_norm_sq) + 1.0);
    rescale_info->obj_vec_rescale = 1.0 / (sqrt(obj_norm_sq) + 1.0);

    for (int i = 0; i < n_cons; ++i)
    {
        state->constraint_lower_bound[i] *= rescale_info->con_bound_rescale;
        state->constraint_upper_bound[i] *= rescale_info->con_bound_rescale;
    }
    for (int i = 0; i < n_vars; ++i)
    {
        state->variable_lower_bound[i] *= rescale_info->con_bound_rescale;
        state->variable_upper_bound[i] *= rescale_info->con_bound_rescale;
        state->objective_vector[i] *= rescale_info->obj_vec_rescale;
    }
}

rescale_info_t *rescale_problem(
    const pdhg_parameters_t *params,
    pdhg_solver_state_t *state)
{
    int n_vars = state->num_variables;
    int n_cons = state->num_constraints;

    clock_t start_rescaling = clock();
    rescale_info_t *rescale_info = (rescale_info_t *)safe_calloc(1, sizeof(rescale_info_t));

    int n_vars = state->num_variables;
    int n_cons = state->num_constraints;

    rescale_info->con_rescale = (double *)safe_malloc(n_cons * sizeof(double));
    rescale_info->var_rescale = (double *)safe_malloc(n_vars * sizeof(double));
    for (int i = 0; i < n_cons; ++i)
        rescale_info->con_rescale[i] = 1.0;
    for (int i = 0; i < n_vars; ++i)
        rescale_info->var_rescale[i] = 1.0;

    if (params->l_inf_ruiz_iterations > 0)
    {
        ruiz_rescaling(state, params->l_inf_ruiz_iterations, rescale_info->con_rescale, rescale_info->var_rescale);
    }

    if (params->has_pock_chambolle_alpha)
    {
        pock_chambolle_rescaling(state, params->pock_chambolle_alpha, rescale_info->con_rescale, rescale_info->var_rescale);
    }
    
    rescale_info->con_bound_rescale = 1.0;
    rescale_info->obj_vec_rescale = 1.0;
    if (params->bound_objective_rescaling)
    {
        bound_objective_rescaling(state, rescale_info);
    }

    rescale_info->rescaling_time_sec = (double)(clock() - start_rescaling) / CLOCKS_PER_SEC;
    
    return rescale_info;
}