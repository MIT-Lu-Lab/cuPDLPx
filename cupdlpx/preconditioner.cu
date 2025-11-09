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
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define SCALING_EPSILON 1e-12

__global__ void scale_variables_kernel(double* __restrict__ c,
                                       double* __restrict__ var_lb,
                                       double* __restrict__ var_ub,
                                       double* __restrict__ var_lb_finite,
                                       double* __restrict__ var_ub_finite,
                                       double* __restrict__ primal_start,
                                       const double* __restrict__ D,
                                       const double* __restrict__ invD,
                                       int n);
__global__ void scale_constraints_kernel(double* __restrict__ con_lb,
                                         double* __restrict__ con_ub,
                                         double* __restrict__ con_lb_finite,
                                         double* __restrict__ con_ub_finite,
                                         double* __restrict__ dual_start,
                                         const double* __restrict__ E,
                                         const double* __restrict__ invE,
                                         int m);
__global__ void csr_scale_nnz_kernel(const int* __restrict__ row_ids,
                                     const int* __restrict__ col_ind,
                                     double* __restrict__ vals,
                                     const double* __restrict__ invD,
                                     const double* __restrict__ invE,
                                     int nnz);
__global__ void csr_row_absmax_kernel(const int* __restrict__ row_ptr,
                                      const double* __restrict__ vals,
                                      int num_rows,
                                      double* __restrict__ out_max);
__global__ void csr_col_absmax_atomic_kernel(const int* __restrict__ col_ind,
                                             const double* __restrict__ vals,
                                             int nnz,
                                             unsigned long long* __restrict__ out_max_bits);
__global__ void u64bits_to_double(const unsigned long long* __restrict__ in_bits,
                                  double* __restrict__ out_val,
                                  int n);
__global__ void csr_row_powsum_kernel(const int* __restrict__ row_ptr,
                                          const double* __restrict__ vals,
                                          int num_rows,
                                          double degree,
                                          double* __restrict__ out_sum);
__global__ void csr_col_powsum_atomic_kernel(const int* __restrict__ col_ind,
                                                const double* __restrict__ vals,
                                                int nnz,
                                                double degree,
                                                double* __restrict__ out_sum);
__global__ void clamp_sqrt_and_accum(double* __restrict__ x,
                                     double* __restrict__ inv_x,
                                     double* __restrict__ cum, 
                                     int n);
static void scale_problem(pdhg_solver_state_t *state, double *E, double *D, double *invE, double *invD);
static void ruiz_rescaling(pdhg_solver_state_t *state, int num_iters, rescale_info_t *rescale_info,
                           double *E, double *D, double *invE, double *invD);
static void pock_chambolle_rescaling(pdhg_solver_state_t *state, double alpha, rescale_info_t *rescale_info,
                                     double *E, double *D, double *invE, double *invD);
static void bound_objective_rescaling(pdhg_solver_state_t *state, rescale_info_t *rescale_info,
                                     double *E, double *D, double *invE, double *invD);

__global__ void scale_variables_kernel(double* __restrict__ c,
                                       double* __restrict__ var_lb,
                                       double* __restrict__ var_ub,
                                       double* __restrict__ var_lb_finite,
                                       double* __restrict__ var_ub_finite,
                                       double* __restrict__ primal_start,
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
    primal_start[j] *= dj;
}

__global__ void scale_constraints_kernel(double* __restrict__ con_lb,
                                         double* __restrict__ con_ub,
                                         double* __restrict__ con_lb_finite,
                                         double* __restrict__ con_ub_finite,
                                         double* __restrict__ dual_start,
                                         const double* __restrict__ E,
                                         const double* __restrict__ invE,
                                         int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    double inv_ei = invE[i];
    double ei = E[i];
    con_lb[i] *= inv_ei;
    con_ub[i] *= inv_ei;
    con_lb_finite[i] *= inv_ei;
    con_ub_finite[i] *= inv_ei;
    dual_start[i] *= ei;
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

__global__ void csr_row_absmax_kernel(const int* __restrict__ row_ptr,
                                      const double* __restrict__ vals,
                                      int num_rows,
                                      double* __restrict__ out_max)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rows) return;
    int s = row_ptr[i], e = row_ptr[i + 1];
    double m = 0.0;
    for (int k = s; k < e; ++k) {
        double v = fabs(vals[k]);
        if (!isfinite(v)) v = 0.0;
        if (v > m) m = v;
    }
    out_max[i] = m;
}

__global__ void csr_col_absmax_atomic_kernel(const int* __restrict__ col_ind,
                                             const double* __restrict__ vals,
                                             int nnz,
                                             unsigned long long* __restrict__ out_max_bits)
{
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < nnz; k += gridDim.x * blockDim.x) {
        int j = col_ind[k];
        double v = fabs(vals[k]);
        if (!isfinite(v)) v = 0.0;
        unsigned long long bits = __double_as_longlong(v);
        atomicMax(&out_max_bits[j], bits);
    }
}

__global__ void u64bits_to_double(const unsigned long long* __restrict__ in_bits,
                                  double* __restrict__ out_val,
                                  int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        out_val[i] = __longlong_as_double(in_bits[i]);
    }
}

__device__ __forceinline__ double pow_fast(double v, double p) {
    if (p == 2.0)   return v * v;
    if (p == 1.0)   return v;
    if (p == 0.5)   return sqrt(v);
    return pow(v, p);
}

__global__ void csr_row_powsum_kernel(const int* __restrict__ row_ptr,
                                       const double* __restrict__ vals,
                                       int num_rows,
                                       double degree,
                                       double* __restrict__ out_sum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rows) return;
    int s = row_ptr[i], e = row_ptr[i + 1];
    double acc = 0.0;
    for (int k = s; k < e; ++k) {
        double v = fabs(vals[k]);
        if (!isfinite(v)) v = 0.0;
        acc += pow_fast(v, degree);
    }
    out_sum[i] = acc;
}

__global__ void csr_col_powsum_atomic_kernel(const int* __restrict__ col_ind,
                                                const double* __restrict__ vals,
                                                int nnz,
                                                double degree,
                                                double* __restrict__ out_sum)
{
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < nnz; k += gridDim.x * blockDim.x) {
        int j = col_ind[k];
        double v = fabs(vals[k]);
        if (!isfinite(v)) v = 0.0;
        double t = pow_fast(v, degree);
        atomicAdd(&out_sum[j], t);
    }
}

__global__ void clamp_sqrt_and_accum(double* __restrict__ x, 
                                     double* __restrict__ inv_x,
                                     double* __restrict__ cum, 
                                     int n) 
{
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += blockDim.x * gridDim.x)
    {
        double v = x[t]; 
        double s = (v < SCALING_EPSILON) ? 1.0 : sqrt(v); 
        cum[t] *= s; 
        x[t] = s;
        inv_x[t] = 1.0 / s;
    }
}

__global__ void reduce_bound_norm_sq_atomic(
    const double* __restrict__ L,
    const double* __restrict__ U,
    int m,
    double* __restrict__ out_sum)
{
    double acc = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += blockDim.x * gridDim.x) {
        double Li = L[i], Ui = U[i];
        bool fL = isfinite(Li), fU = isfinite(Ui);
        if (fL && (!fU || fabs(Li - Ui) > SCALING_EPSILON)) acc += Li * Li;
        if (fU)                                 acc += Ui * Ui;
    }
    atomicAdd(out_sum, acc);
}

static void scale_problem(
    pdhg_solver_state_t *state,
    double *E,
    double *D,
    double *invE,
    double *invD)
{
    int n_vars = state->num_variables;
    int n_cons = state->num_constraints;

    scale_variables_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->objective_vector,
        state->variable_lower_bound,
        state->variable_upper_bound,
        state->variable_lower_bound_finite_val,
        state->variable_upper_bound_finite_val,
        state->initial_primal_solution,
        D,
        invD,
        n_vars);

    scale_constraints_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
        state->constraint_lower_bound,
        state->constraint_upper_bound,
        state->constraint_lower_bound_finite_val,
        state->constraint_upper_bound_finite_val,
        state->initial_dual_solution,
        E,
        invE,
        n_cons);

    csr_scale_nnz_kernel<<<state->num_blocks_nnz, THREADS_PER_BLOCK>>>(
        state->constraint_matrix->row_ind,
        state->constraint_matrix->col_ind,
        state->constraint_matrix->val,
        invD,
        invE,
        state->constraint_matrix->num_nonzeros);
}

static void ruiz_rescaling(
    pdhg_solver_state_t *state,
    int num_iterations,
    rescale_info_t *rescale_info,
    double *E,
    double *D,
    double *invE,
    double *invD)
{
    const int n_cons = state->num_constraints;
    const int n_vars = state->num_variables;
    const int nnz    = state->constraint_matrix->num_nonzeros;

    unsigned long long *D_bits=nullptr;
    CUDA_CHECK(cudaMalloc(&D_bits, n_vars*sizeof(unsigned long long)));

    for (int iter = 0; iter < num_iterations; ++iter)
    {
        csr_row_absmax_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
            state->constraint_matrix->row_ptr,
            state->constraint_matrix->val,
            n_cons,
            E);
        clamp_sqrt_and_accum<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
            E,
            invE,
            rescale_info->con_rescale,
            n_cons);

        CUDA_CHECK(cudaMemset(D_bits, 0, n_vars*sizeof(unsigned long long)));
        csr_col_absmax_atomic_kernel<<<state->num_blocks_nnz, THREADS_PER_BLOCK>>>(
            state->constraint_matrix->col_ind,
            state->constraint_matrix->val,
            nnz,
            D_bits);
        u64bits_to_double<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(D_bits, D, n_vars); 
        clamp_sqrt_and_accum<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
            D,
            invD,
            rescale_info->var_rescale,
            n_vars);

        scale_problem(state, E, D, invE, invD);
    }

    CUDA_CHECK(cudaFree(D_bits));
}

static void pock_chambolle_rescaling(
    pdhg_solver_state_t *state,
    const double alpha,
    rescale_info_t *rescale_info,
    double *E,
    double *D,
    double *invE,
    double *invD)
{
    const int n_cons = state->num_constraints;
    const int n_vars = state->num_variables;
    const int nnz    = state->constraint_matrix->num_nonzeros;

    csr_row_powsum_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
        state->constraint_matrix->row_ptr,
        state->constraint_matrix->val,
        n_cons,
        alpha,
        E);
    clamp_sqrt_and_accum<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
        E,
        invE,
        rescale_info->con_rescale,
        n_cons);
        
    CUDA_CHECK(cudaMemset(D, 0, n_vars*sizeof(double)));
    csr_col_powsum_atomic_kernel<<<state->num_blocks_nnz, THREADS_PER_BLOCK>>>(
        state->constraint_matrix->col_ind,
        state->constraint_matrix->val,
        nnz,
        2.0 - alpha,
        D);
    clamp_sqrt_and_accum<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        D,
        invD,
        rescale_info->var_rescale,
        n_vars);

    scale_problem(state, E, D, invE, invD);

}

static void bound_objective_rescaling(
    pdhg_solver_state_t *state,
    rescale_info_t *rescale_info,
    double *E,
    double *D,
    double *invE,
    double *invD
    )
{
    const int n_cons = state->num_constraints;
    const int n_vars = state->num_variables;

    double *bnd_norm_sq_cuda = nullptr;
    CUDA_CHECK(cudaMalloc(&bnd_norm_sq_cuda, sizeof(double)));
    CUDA_CHECK(cudaMemset(bnd_norm_sq_cuda, 0, sizeof(double)));
    reduce_bound_norm_sq_atomic<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
        state->constraint_lower_bound,
        state->constraint_upper_bound,
        n_cons,
        bnd_norm_sq_cuda);
    
    double bnd_norm_sq = 0.0;
    CUDA_CHECK(cudaMemcpy(&bnd_norm_sq, bnd_norm_sq_cuda, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(bnd_norm_sq_cuda));
    double bnd_norm = sqrt(bnd_norm_sq);

    double obj_norm = 0.0;
    CUBLAS_CHECK(cublasDnrm2(state->blas_handle,
                             state->num_variables,
                             state->objective_vector, 1,
                             &obj_norm));

    const double E_const = bnd_norm + 1.0;
    const double D_const = obj_norm + 1.0;
    {
        std::vector<double> h1(n_cons,E_const), h2(n_vars,D_const);
        CUDA_CHECK(cudaMemcpy(E, h1.data(), n_cons*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(D, h2.data(), n_vars*sizeof(double), cudaMemcpyHostToDevice));
        std::vector<double> h3(n_cons,1/E_const), h4(n_vars,1/D_const);
        CUDA_CHECK(cudaMemcpy(invE, h3.data(), n_cons*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(invD, h4.data(), n_vars*sizeof(double), cudaMemcpyHostToDevice));
    }

    CUBLAS_CHECK(cublasDscal(state->blas_handle,
                             n_cons,
                             &E_const,
                             rescale_info->con_rescale,
                             1));
    CUBLAS_CHECK(cublasDscal(state->blas_handle,
                             n_vars,
                             &D_const,
                             rescale_info->var_rescale,
                             1));

    scale_problem(state, E, D, invE, invD);
}

rescale_info_t *rescale_problem(
    const pdhg_parameters_t *params,
    pdhg_solver_state_t *state)
{
    int n_vars = state->num_variables;
    int n_cons = state->num_constraints;

    clock_t start_rescaling = clock();
    rescale_info_t *rescale_info = (rescale_info_t *)safe_calloc(1, sizeof(rescale_info_t));
    rescale_info->con_rescale = nullptr;
    rescale_info->var_rescale = nullptr;
    CUDA_CHECK(cudaMalloc(&rescale_info->con_rescale, n_cons*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&rescale_info->var_rescale, n_vars*sizeof(double)));
    {
        std::vector<double> h1(n_cons,1.0), h2(n_vars,1.0);
        CUDA_CHECK(cudaMemcpy(rescale_info->con_rescale, h1.data(), 
                              n_cons*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(rescale_info->var_rescale, h2.data(), 
                              n_vars*sizeof(double), cudaMemcpyHostToDevice));
    }

    double *E=nullptr, *D=nullptr, *invE=nullptr, *invD=nullptr;
    CUDA_CHECK(cudaMalloc(&E, n_cons*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&D, n_vars*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&invE, n_cons*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&invD, n_vars*sizeof(double)));

    if (params->l_inf_ruiz_iterations > 0)
    {
        ruiz_rescaling(state, params->l_inf_ruiz_iterations, rescale_info, E, D, invE, invD);
    }
    if (params->has_pock_chambolle_alpha)
    {
        pock_chambolle_rescaling(state, params->pock_chambolle_alpha, rescale_info, E, D, invE, invD);
    }
    if (params->bound_objective_rescaling)
    {
        bound_objective_rescaling(state, rescale_info, E, D, invE, invD);
    }

    rescale_info->rescaling_time_sec = (double)(clock() - start_rescaling) / CLOCKS_PER_SEC;

    CUDA_CHECK(cudaFree(E));
    CUDA_CHECK(cudaFree(D));
    CUDA_CHECK(cudaFree(invE));
    CUDA_CHECK(cudaFree(invD));
    
    return rescale_info;
}