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
#include <cublas_v2.h>
#include <cub/device/device_reduce.cuh>

#define SCALING_EPSILON 1e-12

__global__ void scale_variables_kernel(double *__restrict__ objective_vector,
                                       double *__restrict__ variable_lower_bound,
                                       double *__restrict__ variable_upper_bound,
                                       double *__restrict__ initial_primal_solution,
                                       const double *__restrict__ variable_rescaling,
                                       const double *__restrict__ inverse_variable_rescaling,
                                       int num_variables);
__global__ void scale_constraints_kernel(double *__restrict__ constraint_lower_bound,
                                         double *__restrict__ constraint_upper_bound,
                                         double *__restrict__ initial_dual_solution,
                                         const double *__restrict__ constraint_rescaling,
                                         const double *__restrict__ inverse_constraint_rescaling,
                                         int num_constraints);
__global__ void scale_csr_nnz_kernel(const int *__restrict__ constraint_row_ind,
                                     const int *__restrict__ constraint_col_ind,
                                     double *__restrict__ constraint_matrix_val,
                                     double *__restrict__ constraint_matrix_transpose_val,
                                     const int *__restrict__ constraint_to_transpose_position,
                                     const double *__restrict__ inverse_variable_rescaling,
                                     const double *__restrict__ inverse_constraint_rescaling,
                                     int nnz);
__global__ void compute_csr_row_absmax_kernel(const int *__restrict__ row_ptr,
                                              const double *__restrict__ matrix_vals,
                                              int num_rows,
                                              double *__restrict__ row_absmax_values);
__global__ void compute_csr_row_powsum_kernel(const int *__restrict__ row_ptr,
                                              const double *__restrict__ matrix_vals,
                                              int num_rows,
                                              double degree,
                                              double *__restrict__ row_powsum_values);
__global__ void clamp_sqrt_and_accum_kernel(double *__restrict__ scaling_factors,
                                            double *__restrict__ inverse_scaling_factors,
                                            double *__restrict__ cumulative_rescaling,
                                            int num_variables);
__global__ void compute_bound_contrib_kernel(
    const double *__restrict__ constraint_lower_bound,
    const double *__restrict__ constraint_upper_bound,
    int num_constraints,
    double *__restrict__ contrib);
__global__ void scale_bounds_kernel(
    double *__restrict__ constraint_lower_bound,
    double *__restrict__ constraint_upper_bound,
    double *__restrict__ initial_dual_solution,
    int num_constraints,
    double constraint_scale,
    double objective_scale);
__global__ void scale_objective_kernel(
    double *__restrict__ objective_vector,
    double *__restrict__ variable_lower_bound,
    double *__restrict__ variable_upper_bound,
    double *__restrict__ initial_primal_solution,
    int num_variables,
    double constraint_scale,
    double objective_scale);
__global__ void fill_ones_kernel(double *__restrict__ x, int num_variables);
static void scale_problem(pdhg_solver_state_t *state, double *constraint_rescaling, double *variable_rescaling, double *inverse_constraint_rescaling, double *inverse_variable_rescaling);
static void ruiz_rescaling(pdhg_solver_state_t *state, int num_iters, rescale_info_t *rescale_info,
                           double *constraint_rescaling, double *variable_rescaling, double *inverse_constraint_rescaling, double *inverse_variable_rescaling);
static void pock_chambolle_rescaling(pdhg_solver_state_t *state, const double alpha, rescale_info_t *rescale_info,
                                     double *constraint_rescaling, double *variable_rescaling, double *inverse_constraint_rescaling, double *inverse_variable_rescaling);
static void bound_objective_rescaling(pdhg_solver_state_t *state, rescale_info_t *rescale_info);

static void scale_problem(
    pdhg_solver_state_t *state,
    double *constraint_rescaling,
    double *variable_rescaling,
    double *inverse_constraint_rescaling,
    double *inverse_variable_rescaling)
{
    int num_variables = state->num_variables;
    int num_constraints = state->num_constraints;

    scale_variables_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->objective_vector,
        state->variable_lower_bound,
        state->variable_upper_bound,
        state->initial_primal_solution,
        variable_rescaling,
        inverse_variable_rescaling,
        num_variables);

    scale_constraints_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
        state->constraint_lower_bound,
        state->constraint_upper_bound,
        state->initial_dual_solution,
        constraint_rescaling,
        inverse_constraint_rescaling,
        num_constraints);

    scale_csr_nnz_kernel<<<state->num_blocks_nnz, THREADS_PER_BLOCK>>>(
        state->constraint_matrix->row_ind,
        state->constraint_matrix->col_ind,
        state->constraint_matrix->val,
        state->constraint_matrix_t->val,
        state->constraint_matrix->transpose_map,
        inverse_variable_rescaling,
        inverse_constraint_rescaling,
        state->constraint_matrix->num_nonzeros);
}

static void ruiz_rescaling(
    pdhg_solver_state_t *state,
    int num_iterations,
    rescale_info_t *rescale_info,
    double *constraint_rescaling,
    double *variable_rescaling,
    double *inverse_constraint_rescaling,
    double *inverse_variable_rescaling)
{
    const int num_constraints = state->num_constraints;
    const int num_variables = state->num_variables;

    for (int iter = 0; iter < num_iterations; ++iter)
    {
        compute_csr_row_absmax_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
            state->constraint_matrix->row_ptr,
            state->constraint_matrix->val,
            num_constraints,
            constraint_rescaling);
        clamp_sqrt_and_accum_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
            constraint_rescaling,
            inverse_constraint_rescaling,
            rescale_info->con_rescale,
            num_constraints);

        compute_csr_row_absmax_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
            state->constraint_matrix_t->row_ptr,
            state->constraint_matrix_t->val,
            num_variables,
            variable_rescaling);
        clamp_sqrt_and_accum_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
            variable_rescaling,
            inverse_variable_rescaling,
            rescale_info->var_rescale,
            num_variables);

        scale_problem(state, constraint_rescaling, variable_rescaling, inverse_constraint_rescaling, inverse_variable_rescaling);
    }
}

static void pock_chambolle_rescaling(
    pdhg_solver_state_t *state,
    const double alpha,
    rescale_info_t *rescale_info,
    double *constraint_rescaling,
    double *variable_rescaling,
    double *inverse_constraint_rescaling,
    double *inverse_variable_rescaling)
{
    const int num_constraints = state->num_constraints;
    const int num_variables = state->num_variables;

    compute_csr_row_powsum_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
        state->constraint_matrix->row_ptr,
        state->constraint_matrix->val,
        num_constraints,
        alpha,
        constraint_rescaling);
    clamp_sqrt_and_accum_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
        constraint_rescaling,
        inverse_constraint_rescaling,
        rescale_info->con_rescale,
        num_constraints);

    compute_csr_row_powsum_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->constraint_matrix_t->row_ptr,
        state->constraint_matrix_t->val,
        num_variables,
        2.0 - alpha,
        variable_rescaling);
    clamp_sqrt_and_accum_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        variable_rescaling,
        inverse_variable_rescaling,
        rescale_info->var_rescale,
        num_variables);

    scale_problem(state, constraint_rescaling, variable_rescaling, inverse_constraint_rescaling, inverse_variable_rescaling);
}

static void bound_objective_rescaling(
    pdhg_solver_state_t *state,
    rescale_info_t *rescale_info)
{
    const int num_constraints = state->num_constraints;
    const int num_variables = state->num_variables;

    double *contrib_d = nullptr;
    CUDA_CHECK(cudaMalloc(&contrib_d, num_constraints * sizeof(double)));
    compute_bound_contrib_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
        state->constraint_lower_bound,
        state->constraint_upper_bound,
        num_constraints,
        contrib_d);

    double *bnd_norm_sq_d = nullptr;
    CUDA_CHECK(cudaMalloc(&bnd_norm_sq_d, sizeof(double)));
    void *temp_storage = nullptr;
    size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceReduce::Sum(temp_storage, temp_bytes, contrib_d, bnd_norm_sq_d, num_constraints));
    CUDA_CHECK(cudaMalloc(&temp_storage, temp_bytes));
    CUDA_CHECK(cub::DeviceReduce::Sum(temp_storage, temp_bytes, contrib_d, bnd_norm_sq_d, num_constraints));
    CUDA_CHECK(cudaFree(contrib_d));
    CUDA_CHECK(cudaFree(temp_storage));

    double bnd_norm_sq_h = 0.0;
    CUDA_CHECK(cudaMemcpy(&bnd_norm_sq_h, bnd_norm_sq_d, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(bnd_norm_sq_d));
    double bnd_norm = sqrt(bnd_norm_sq_h);

    double obj_norm = 0.0;
    CUBLAS_CHECK(cublasDnrm2(state->blas_handle,
                             state->num_variables,
                             state->objective_vector, 1,
                             &obj_norm));

    double constraint_scale = 1.0 / (bnd_norm + 1.0);
    double objective_scale = 1.0 / (obj_norm + 1.0);

    scale_bounds_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
        state->constraint_lower_bound,
        state->constraint_upper_bound,
        state->initial_dual_solution,
        num_constraints,
        constraint_scale,
        objective_scale);

    scale_objective_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->objective_vector,
        state->variable_lower_bound,
        state->variable_upper_bound,
        state->initial_primal_solution,
        num_variables,
        constraint_scale,
        objective_scale);

    rescale_info->con_bound_rescale = constraint_scale;
    rescale_info->obj_vec_rescale = objective_scale;
}

rescale_info_t *rescale_problem(
    const pdhg_parameters_t *params,
    pdhg_solver_state_t *state)
{
    if (params->verbose) {
        printf("[Precondition] Start\n");
    }

    int num_variables = state->num_variables;
    int num_constraints = state->num_constraints;

    clock_t start_rescaling = clock();
    rescale_info_t *rescale_info = (rescale_info_t *)safe_calloc(1, sizeof(rescale_info_t));
    CUDA_CHECK(cudaMalloc(&rescale_info->con_rescale, num_constraints * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&rescale_info->var_rescale, num_variables * sizeof(double)));
    fill_ones_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
        rescale_info->con_rescale, num_constraints);
    fill_ones_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        rescale_info->var_rescale, num_variables);

    double *constraint_rescaling = NULL, *variable_rescaling = NULL, *inverse_constraint_rescaling = NULL, *inverse_variable_rescaling = NULL;
    CUDA_CHECK(cudaMalloc(&constraint_rescaling, num_constraints * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&variable_rescaling, num_variables * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&inverse_constraint_rescaling, num_constraints * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&inverse_variable_rescaling, num_variables * sizeof(double)));

    if (params->l_inf_ruiz_iterations > 0)
    {
        if (params->verbose) {
            printf("[Precondition] Ruiz scaling (%d iterations)\n", params->l_inf_ruiz_iterations);
        }
        ruiz_rescaling(state, params->l_inf_ruiz_iterations, rescale_info, constraint_rescaling, variable_rescaling, inverse_constraint_rescaling, inverse_variable_rescaling);
    }
    if (params->has_pock_chambolle_alpha)
    {
        if (params->verbose) {
            printf("[Precondition] Pock-Chambolle scaling (alpha=%.4f)\n", params->pock_chambolle_alpha);
        }
        pock_chambolle_rescaling(state, params->pock_chambolle_alpha, rescale_info, constraint_rescaling, variable_rescaling, inverse_constraint_rescaling, inverse_variable_rescaling);
    }

    rescale_info->con_bound_rescale = 1.0;
    rescale_info->obj_vec_rescale = 1.0;
    if (params->bound_objective_rescaling)
    {
        if (params->verbose) {
            printf("[Precondition] Bound-objective scaling\n");
        }
        bound_objective_rescaling(state, rescale_info);
    }

    rescale_info->rescaling_time_sec = (double)(clock() - start_rescaling) / CLOCKS_PER_SEC;

    CUDA_CHECK(cudaFree(constraint_rescaling));
    CUDA_CHECK(cudaFree(variable_rescaling));
    CUDA_CHECK(cudaFree(inverse_constraint_rescaling));
    CUDA_CHECK(cudaFree(inverse_variable_rescaling));

    return rescale_info;
}

__global__ void scale_variables_kernel(double *__restrict__ objective_vector,
                                       double *__restrict__ variable_lower_bound,
                                       double *__restrict__ variable_upper_bound,
                                       double *__restrict__ initial_primal_solution,
                                       const double *__restrict__ variable_rescaling,
                                       const double *__restrict__ inverse_variable_rescaling,
                                       int num_variables)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_variables)
        return;
    double dj = variable_rescaling[j];
    double inv_dj = inverse_variable_rescaling[j];
    objective_vector[j] *= inv_dj;
    variable_lower_bound[j] *= dj;
    variable_upper_bound[j] *= dj;
    initial_primal_solution[j] *= dj;
}

__global__ void scale_constraints_kernel(double *__restrict__ constraint_lower_bound,
                                         double *__restrict__ constraint_upper_bound,
                                         double *__restrict__ initial_dual_solution,
                                         const double *__restrict__ constraint_rescaling,
                                         const double *__restrict__ inverse_constraint_rescaling,
                                         int num_constraints)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_constraints)
        return;
    double inv_ei = inverse_constraint_rescaling[i];
    double ei = constraint_rescaling[i];
    constraint_lower_bound[i] *= inv_ei;
    constraint_upper_bound[i] *= inv_ei;
    initial_dual_solution[i] *= ei;
}

__global__ void scale_csr_nnz_kernel(const int *__restrict__ constraint_row_ind,
                                     const int *__restrict__ constraint_col_ind,
                                     double *__restrict__ constraint_matrix_val,
                                     double *__restrict__ constraint_matrix_transpose_val,
                                     const int *__restrict__ constraint_to_transpose_position,
                                     const double *__restrict__ inverse_variable_rescaling,
                                     const double *__restrict__ inverse_constraint_rescaling,
                                     int nnz)
{
    for (int k = blockIdx.x * blockDim.x + threadIdx.x;
         k < nnz;
         k += gridDim.x * blockDim.x)
    {
        int i = constraint_row_ind[k];
        int j = constraint_col_ind[k];
        double scale = inverse_variable_rescaling[j] * inverse_constraint_rescaling[i];
        constraint_matrix_val[k] *= scale;
        constraint_matrix_transpose_val[constraint_to_transpose_position[k]] *= scale;
    }
}

__global__ void compute_csr_row_absmax_kernel(const int *__restrict__ row_ptr,
                                              const double *__restrict__ matrix_vals,
                                              int num_rows,
                                              double *__restrict__ row_absmax_values)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rows)
        return;
    int s = row_ptr[i], e = row_ptr[i + 1];
    double m = 0.0;
    for (int k = s; k < e; ++k)
    {
        double v = fabs(matrix_vals[k]);
        if (!isfinite(v))
            v = 0.0;
        if (v > m)
            m = v;
    }
    row_absmax_values[i] = m;
}

__device__ __forceinline__ double pow_fast(double v, double p)
{
    if (p == 2.0)
        return v * v;
    if (p == 1.0)
        return v;
    if (p == 0.5)
        return sqrt(v);
    return pow(v, p);
}

__global__ void compute_csr_row_powsum_kernel(const int *__restrict__ row_ptr,
                                              const double *__restrict__ matrix_vals,
                                              int num_rows,
                                              double degree,
                                              double *__restrict__ row_powsum_values)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rows)
        return;
    int s = row_ptr[i], e = row_ptr[i + 1];
    double acc = 0.0;
    for (int k = s; k < e; ++k)
    {
        double v = fabs(matrix_vals[k]);
        if (!isfinite(v))
            v = 0.0;
        acc += pow_fast(v, degree);
    }
    row_powsum_values[i] = acc;
}

__global__ void clamp_sqrt_and_accum_kernel(double *__restrict__ scaling_factors,
                                            double *__restrict__ inverse_scaling_factors,
                                            double *__restrict__ cumulative_rescaling,
                                            int num_variables)
{
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < num_variables; t += blockDim.x * gridDim.x)
    {
        double v = scaling_factors[t];
        double s = (v < SCALING_EPSILON) ? 1.0 : sqrt(v);
        cumulative_rescaling[t] *= s;
        scaling_factors[t] = s;
        inverse_scaling_factors[t] = 1.0 / s;
    }
}

__global__ void compute_bound_contrib_kernel(
    const double *__restrict__ constraint_lower_bound,
    const double *__restrict__ constraint_upper_bound,
    int num_constraints,
    double *__restrict__ contrib)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_constraints) return;

    double Li = constraint_lower_bound[i];
    double Ui = constraint_upper_bound[i];
    bool fL = isfinite(Li);
    bool fU = isfinite(Ui);

    double acc = 0.0;

    // follow the existing semantics
    if (fL && (!fU || fabs(Li - Ui) > SCALING_EPSILON))
        acc += Li * Li;
    if (fU)
        acc += Ui * Ui;

    contrib[i] = acc;
}

__global__ void scale_bounds_kernel(
    double *__restrict__ constraint_lower_bound,
    double *__restrict__ constraint_upper_bound,
    double *__restrict__ initial_dual_solution,
    int num_constraints,
    double constraint_scale,
    double objective_scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_constraints)
        return;
    constraint_lower_bound[i] *= constraint_scale;
    constraint_upper_bound[i] *= constraint_scale;
    initial_dual_solution[i] *= objective_scale;
}

__global__ void scale_objective_kernel(
    double *__restrict__ objective_vector,
    double *__restrict__ variable_lower_bound,
    double *__restrict__ variable_upper_bound,
    double *__restrict__ initial_primal_solution,
    int num_variables,
    double constraint_scale,
    double objective_scale)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_variables)
        return;
    variable_lower_bound[j] *= constraint_scale;
    variable_upper_bound[j] *= constraint_scale;
    objective_vector[j] *= objective_scale;
    initial_primal_solution[j] *= constraint_scale;
}

__global__ void fill_ones_kernel(double *__restrict__ x, int num_variables)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_variables)
        x[i] = 1.0;
}
