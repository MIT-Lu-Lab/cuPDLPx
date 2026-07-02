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

#pragma once

// Include cuda_to_hip.h first: it pulls in the CUDA runtime/cuBLAS/cuSPARSE
// headers on CUDA builds and their HIP equivalents on ROCm builds.
#include "cuda_to_hip.h"
#include "cusparse_compat.h"
#include "internal_types.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef __cplusplus
extern "C"
{
#endif

#define CUDA_CHECK(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess)                                                                                        \
        {                                                                                                              \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorName(err));                   \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define CUBLAS_CHECK(call)                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        cublasStatus_t status = call;                                                                                  \
        if (status != CUBLAS_STATUS_SUCCESS)                                                                           \
        {                                                                                                              \
            fprintf(stderr, "cuBLAS Error at %s:%d: %s\n", __FILE__, __LINE__, cublasGetStatusName(status));           \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define CUSPARSE_CHECK(call)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        cusparseStatus_t status = call;                                                                                \
        if (status != CUSPARSE_STATUS_SUCCESS)                                                                         \
        {                                                                                                              \
            fprintf(stderr, "cuSPARSE Error at %s:%d: %s\n", __FILE__, __LINE__, cusparseGetErrorName(status));        \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define THREADS_PER_BLOCK 256

    void *safe_malloc(size_t size);

    void *safe_calloc(size_t num, size_t size);

    void *safe_realloc(void *ptr, size_t new_size);

    double estimate_maximum_singular_value(cusparseHandle_t sparse_handle,
                                           cublasHandle_t blas_handle,
                                           const cu_sparse_matrix_csr_t *A,
                                           const cu_sparse_matrix_csr_t *AT,
                                           int max_iterations,
                                           double tolerance);

    bool cupdlpx_use_spmvop_by_default(void);

    void cupdlpx_spmv_buffer_size(cusparseHandle_t sparse_handle,
                                  cusparseSpMatDescr_t mat,
                                  cusparseDnVecDescr_t vec_x,
                                  cusparseDnVecDescr_t vec_y,
                                  size_t *buffer_size);

    void cupdlpx_spmv_prepare(cusparseHandle_t sparse_handle,
                              cusparseSpMatDescr_t mat,
                              cusparseDnVecDescr_t vec_x,
                              cusparseDnVecDescr_t vec_y,
                              void *buffer,
                              void **descr,
                              void **plan);

    void cupdlpx_spmv_release(void *descr, void *plan);

    void cupdlpx_spmv_execute(cusparseHandle_t sparse_handle,
                              cusparseSpMatDescr_t mat,
                              cusparseDnVecDescr_t vec_x,
                              cusparseDnVecDescr_t vec_y,
                              void *buffer,
                              void *plan);

    void *cupdlpx_spmv_ctx_create(cusparseHandle_t sparse_handle,
                                  const cu_sparse_matrix_csr_t *A,
                                  const cu_sparse_matrix_csr_t *AT,
                                  const double *ax_x_init,
                                  double *ax_y_init,
                                  const double *atx_x_init,
                                  double *atx_y_init);

    void cupdlpx_spmv_ctx_destroy(void *ctx);

    void cupdlpx_spmv_Ax(cusparseHandle_t sparse_handle, void *ctx, const double *x, double *y);

    void cupdlpx_spmv_ATx(cusparseHandle_t sparse_handle, void *ctx, const double *x, double *y);

    void compute_interaction_and_movement(pdhg_solver_state_t *solver_state, double *interaction, double *movement);

    bool should_do_adaptive_restart(pdhg_solver_state_t *solver_state,
                                    const restart_parameters_t *restart_params,
                                    int termination_evaluation_frequency);

    void check_termination_criteria(pdhg_solver_state_t *solver_state, const termination_criteria_t *criteria);

    void print_initial_info(const pdhg_parameters_t *params, lp_problem_t *problem);

    void pdhg_final_log(const cupdlpx_result_t *result, const pdhg_parameters_t *params);

    void display_iteration_stats(const pdhg_solver_state_t *solver_state, bool verbose);

    const char *termination_reason_to_string(termination_reason_t reason);

    int get_print_frequency(int iter);

    void compute_residual(pdhg_solver_state_t *state, norm_type_t optimality_norm);

    void compute_infeasibility_information(pdhg_solver_state_t *state);

    void fill_or_copy(double **dest, int n, const double *src, double fill_value);

    int dense_to_csr(const matrix_desc_t *desc, int **row_ptr, int **col_ind, double **vals, int *nnz_out);

    int csc_to_csr(const matrix_desc_t *desc, int **row_ptr, int **col_ind, double **vals);

    int coo_to_csr(const matrix_desc_t *desc, int **row_ptr, int **col_ind, double **vals);

    void check_feas_polishing_termination_criteria(pdhg_solver_state_t *solver_state,
                                                   const pdhg_solver_state_t *ori_solver_state,
                                                   const termination_criteria_t *criteria,
                                                   bool is_primal_polish);

    void print_initial_feas_polish_info(bool is_primal_polish, const pdhg_parameters_t *params);

    void display_feas_polish_iteration_stats(const pdhg_solver_state_t *state, bool verbose, bool is_primal_polish);

    void pdhg_feas_polish_final_log(const pdhg_solver_state_t *primal_state,
                                    const pdhg_solver_state_t *dual_state,
                                    bool verbose);

    void compute_primal_feas_polish_residual(pdhg_solver_state_t *state,
                                             const pdhg_solver_state_t *ori_state,
                                             norm_type_t optimality_norm);

    void compute_dual_feas_polish_residual(pdhg_solver_state_t *state,
                                           const pdhg_solver_state_t *ori_state,
                                           norm_type_t optimality_norm);

    void set_default_parameters(pdhg_parameters_t *params);

#ifdef __cplusplus
}

#endif
