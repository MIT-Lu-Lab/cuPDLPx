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

static const double HOST_ONE = 1.0;
static const double HOST_ZERO = 0.0;

typedef struct
{
    cusparseSpMatDescr_t matA;
    cusparseSpMatDescr_t matAT;
    cusparseDnVecDescr_t vec_ax_x;
    cusparseDnVecDescr_t vec_ax_y;
    cusparseDnVecDescr_t vec_atx_x;
    cusparseDnVecDescr_t vec_atx_y;
    void *ax_buffer;
    void *atx_buffer;
    void *ax_descr;
    void *ax_plan;
    void *atx_descr;
    void *atx_plan;
} cupdlpx_spmv_ctx_t;

bool cupdlpx_use_spmvop_by_default(void)
{
    return CUPDLPX_HAS_SPMVOP;
}

void cupdlpx_spmv_buffer_size(cusparseHandle_t sparse_handle,
                              cusparseSpMatDescr_t mat,
                              cusparseDnVecDescr_t vec_x,
                              cusparseDnVecDescr_t vec_y,
                              size_t *buffer_size)
{
#if CUPDLPX_HAS_SPMVOP
    CUSPARSE_CHECK(cusparseSpMVOp_bufferSize(sparse_handle,
                                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             mat,
                                             vec_x,
                                             vec_y,
                                             vec_y,
                                             CUDA_R_64F,
                                             CUSPARSE_SPMVOP_ALG2,
                                             buffer_size));
#else
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(sparse_handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &HOST_ONE,
                                           mat,
                                           vec_x,
                                           &HOST_ZERO,
                                           vec_y,
                                           CUDA_R_64F,
                                           CUSPARSE_SPMV_CSR_ALG2,
                                           buffer_size));
#endif
}

void cupdlpx_spmv_prepare(cusparseHandle_t sparse_handle,
                          cusparseSpMatDescr_t mat,
                          cusparseDnVecDescr_t vec_x,
                          cusparseDnVecDescr_t vec_y,
                          void *buffer,
                          void **descr,
                          void **plan)
{
#if CUPDLPX_HAS_SPMVOP
    cusparseSpMVOpDescr_t local_descr = NULL;
    cusparseSpMVOpPlan_t local_plan = NULL;
    CUSPARSE_CHECK(cusparseSpMVOp_createDescr(sparse_handle,
                                              &local_descr,
                                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              mat,
                                              vec_x,
                                              vec_y,
                                              vec_y,
                                              CUDA_R_64F,
                                              CUSPARSE_SPMVOP_ALG2,
                                              buffer));
    CUSPARSE_CHECK(cusparseSpMVOp_createPlan(sparse_handle, local_descr, &local_plan, NULL, 0));
    *descr = (void *)local_descr;
    *plan = (void *)local_plan;
#else
    (void)descr;
    (void)plan;
    CUSPARSE_CHECK(cusparseSpMV_preprocess(sparse_handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &HOST_ONE,
                                           mat,
                                           vec_x,
                                           &HOST_ZERO,
                                           vec_y,
                                           CUDA_R_64F,
                                           CUSPARSE_SPMV_CSR_ALG2,
                                           buffer));
#endif
}

void cupdlpx_spmv_release(void *descr, void *plan)
{
#if CUPDLPX_HAS_SPMVOP
    if (descr)
    {
        CUSPARSE_CHECK(cusparseSpMVOp_destroyDescr((cusparseSpMVOpDescr_t)descr));
    }
    if (plan)
    {
        CUSPARSE_CHECK(cusparseSpMVOp_destroyPlan((cusparseSpMVOpPlan_t)plan));
    }
#else
    (void)descr;
    (void)plan;
#endif
}

void cupdlpx_spmv_execute(cusparseHandle_t sparse_handle,
                          cusparseSpMatDescr_t mat,
                          cusparseDnVecDescr_t vec_x,
                          cusparseDnVecDescr_t vec_y,
                          void *buffer,
                          void *plan)
{
#if CUPDLPX_HAS_SPMVOP
    (void)mat;
    (void)buffer;
    CUSPARSE_CHECK(
        cusparseSpMVOp(sparse_handle, (cusparseSpMVOpPlan_t)plan, &HOST_ONE, &HOST_ZERO, vec_x, vec_y, vec_y));
#else
    (void)plan;
    CUSPARSE_CHECK(cusparseSpMV(sparse_handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &HOST_ONE,
                                mat,
                                vec_x,
                                &HOST_ZERO,
                                vec_y,
                                CUDA_R_64F,
                                CUSPARSE_SPMV_CSR_ALG2,
                                buffer));
#endif
}

void *cupdlpx_spmv_ctx_create(cusparseHandle_t sparse_handle,
                              const cu_sparse_matrix_csr_t *A,
                              const cu_sparse_matrix_csr_t *AT,
                              const double *ax_x_init,
                              double *ax_y_init,
                              const double *atx_x_init,
                              double *atx_y_init)
{
    cupdlpx_spmv_ctx_t *ctx = (cupdlpx_spmv_ctx_t *)safe_calloc(1, sizeof(cupdlpx_spmv_ctx_t));
    size_t ax_buffer_size = 0;
    size_t atx_buffer_size = 0;

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

    CUSPARSE_CHECK(cusparseCreateDnVec(&ctx->vec_ax_x, A->num_cols, (void *)ax_x_init, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&ctx->vec_ax_y, A->num_rows, ax_y_init, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&ctx->vec_atx_x, AT->num_cols, (void *)atx_x_init, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&ctx->vec_atx_y, AT->num_rows, atx_y_init, CUDA_R_64F));

    cupdlpx_spmv_buffer_size(sparse_handle, ctx->matA, ctx->vec_ax_x, ctx->vec_ax_y, &ax_buffer_size);
    cupdlpx_spmv_buffer_size(sparse_handle, ctx->matAT, ctx->vec_atx_x, ctx->vec_atx_y, &atx_buffer_size);
    CUDA_CHECK(cudaMalloc(&ctx->ax_buffer, ax_buffer_size));
    CUDA_CHECK(cudaMalloc(&ctx->atx_buffer, atx_buffer_size));

    cupdlpx_spmv_prepare(
        sparse_handle, ctx->matA, ctx->vec_ax_x, ctx->vec_ax_y, ctx->ax_buffer, &ctx->ax_descr, &ctx->ax_plan);
    cupdlpx_spmv_prepare(
        sparse_handle, ctx->matAT, ctx->vec_atx_x, ctx->vec_atx_y, ctx->atx_buffer, &ctx->atx_descr, &ctx->atx_plan);

    return (void *)ctx;
}

void cupdlpx_spmv_ctx_destroy(void *ctx_void)
{
    cupdlpx_spmv_ctx_t *ctx = (cupdlpx_spmv_ctx_t *)ctx_void;
    if (ctx == NULL)
    {
        return;
    }

    cupdlpx_spmv_release(ctx->ax_descr, ctx->ax_plan);
    cupdlpx_spmv_release(ctx->atx_descr, ctx->atx_plan);

    if (ctx->ax_buffer)
    {
        CUDA_CHECK(cudaFree(ctx->ax_buffer));
    }
    if (ctx->atx_buffer)
    {
        CUDA_CHECK(cudaFree(ctx->atx_buffer));
    }

    if (ctx->vec_ax_x)
    {
        CUSPARSE_CHECK(cusparseDestroyDnVec(ctx->vec_ax_x));
    }
    if (ctx->vec_ax_y)
    {
        CUSPARSE_CHECK(cusparseDestroyDnVec(ctx->vec_ax_y));
    }
    if (ctx->vec_atx_x)
    {
        CUSPARSE_CHECK(cusparseDestroyDnVec(ctx->vec_atx_x));
    }
    if (ctx->vec_atx_y)
    {
        CUSPARSE_CHECK(cusparseDestroyDnVec(ctx->vec_atx_y));
    }
    if (ctx->matA)
    {
        CUSPARSE_CHECK(cusparseDestroySpMat(ctx->matA));
    }
    if (ctx->matAT)
    {
        CUSPARSE_CHECK(cusparseDestroySpMat(ctx->matAT));
    }

    free(ctx);
}

void cupdlpx_spmv_Ax(cusparseHandle_t sparse_handle, void *ctx_void, const double *x, double *y)
{
    cupdlpx_spmv_ctx_t *ctx = (cupdlpx_spmv_ctx_t *)ctx_void;
    CUSPARSE_CHECK(cusparseDnVecSetValues(ctx->vec_ax_x, (void *)x));
    CUSPARSE_CHECK(cusparseDnVecSetValues(ctx->vec_ax_y, y));
    cupdlpx_spmv_execute(sparse_handle, ctx->matA, ctx->vec_ax_x, ctx->vec_ax_y, ctx->ax_buffer, ctx->ax_plan);
}

void cupdlpx_spmv_ATx(cusparseHandle_t sparse_handle, void *ctx_void, const double *x, double *y)
{
    cupdlpx_spmv_ctx_t *ctx = (cupdlpx_spmv_ctx_t *)ctx_void;
    CUSPARSE_CHECK(cusparseDnVecSetValues(ctx->vec_atx_x, (void *)x));
    CUSPARSE_CHECK(cusparseDnVecSetValues(ctx->vec_atx_y, y));
    cupdlpx_spmv_execute(sparse_handle, ctx->matAT, ctx->vec_atx_x, ctx->vec_atx_y, ctx->atx_buffer, ctx->atx_plan);
}
