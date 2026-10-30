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

// Detection of primal and dual infeasibility from ray certificates.

#include "infeasibility.h"
#include "reduce.h"
#include "termination.h"
#include "utils.h"

#include <math.h>

static const double HOST_ONE = 1.0;
static const double HOST_ZERO = 0.0;

// Statistics of one candidate ray in original units, and whether each side
// passed its objective margin. Lives for one compute_infeasibility_host call.
typedef struct
{
    double max_primal_ray_infeasibility;
    double primal_ray_linear_objective;
    double max_dual_ray_infeasibility;
    double dual_ray_objective;
    // the objective terms with absolute values; the margin is measured against them
    double primal_objective_abs;
    double dual_objective_abs;
    bool primal_passes_margin;
    bool dual_passes_margin;
} ray_candidate_stats_t;

// out[i] = |x[i]|; out may be x.
__global__ void abs_kernel(double *out, const double *x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = fabs(x[i]);
}

// The candidates are the columns of dense n x K and m x K matrices, so one SpMM
// applies A or A^T to all of them. |A| and |A^T| reuse the CSR indices of A and
// A^T.
struct infeas_ctx
{
    double *abs_val_A;
    double *abs_val_AT;
    cusparseSpMatDescr_t matA, matAT, matAbsA, matAbsAT;
    // column-major, one column per candidate; first round, then second round
    double *D;      // n x K: primal candidates, then |D|
    double *Y;      // m x K: dual candidates, then |Y|
    double *prod_m; // m x K: A D, then |A||D|
    double *prod_n; // n x K: A^T Y, then |A^T||Y|
    cusparseDnMatDescr_t dnD, dnY, dnProdM, dnProdN;
    void *buf_A, *buf_AT, *buf_absA, *buf_absAT;
};

// The direction to - from; from == NULL means to itself.
typedef struct
{
    const double *to;
    const double *from;
} direction_t;

// Wrapped in a struct so that the array is passed to a kernel by value.
typedef struct
{
    direction_t candidate[INFEAS_NUM_CANDIDATES];
} candidate_directions_t;

__device__ __forceinline__ double direction_entry(const direction_t &d, int i)
{
    return d.from ? d.to[i] - d.from[i] : d.to[i];
}

// Sign conditions of a primal ray: (A d)_i >= 0 under a finite lower bound and
// <= 0 under a finite upper bound.
__host__ __device__ __forceinline__ double primal_ray_row_violation(double ad, bool lb_finite, bool ub_finite)
{
    return (lb_finite ? fmax(-ad, 0.0) : 0.0) + (ub_finite ? fmax(ad, 0.0) : 0.0);
}

// Sign conditions of a dual ray on r = -A^T y: r_j > 0 needs a finite lower bound
// on x_j and r_j < 0 a finite upper bound.
__host__ __device__ __forceinline__ double dual_ray_reduced_cost_violation(double r, bool lb_finite, bool ub_finite)
{
    return (lb_finite ? 0.0 : fmax(r, 0.0)) + (ub_finite ? 0.0 : fmax(-r, 0.0));
}

// Dual objective term of an entry v (y_i or r_j): v^+ times the finite lower
// bound plus v^- times the finite upper bound.
__host__ __device__ __forceinline__ double bound_term(double v, double lower_finite, double upper_finite)
{
    return fmax(v, 0.0) * lower_finite + fmin(v, 0.0) * upper_finite;
}

// The same with each summand in absolute value.
__host__ __device__ __forceinline__ double bound_term_abs(double v, double lower_finite, double upper_finite)
{
    return fabs(fmax(v, 0.0) * lower_finite) + fabs(fmin(v, 0.0) * upper_finite);
}

// Forms candidate blockIdx.y and projects it onto the sign conditions.
__global__ void form_primal_candidates_kernel(candidate_directions_t directions,
                                              const double *__restrict__ var_lb,
                                              const double *__restrict__ var_ub,
                                              double *__restrict__ D,
                                              int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y;
    if (j >= n)
        return;
    double v = direction_entry(directions.candidate[c], j);
    if (isfinite(var_lb[j]))
        v = fmax(v, 0.0);
    if (isfinite(var_ub[j]))
        v = fmin(v, 0.0);
    D[(size_t)c * n + j] = v;
}

__global__ void form_dual_candidates_kernel(candidate_directions_t directions,
                                            const double *__restrict__ con_lb,
                                            const double *__restrict__ con_ub,
                                            double *__restrict__ Y,
                                            int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y;
    if (i >= m)
        return;
    double v = direction_entry(directions.candidate[c], i);
    if (!isfinite(con_lb[i]))
        v = fmin(v, 0.0);
    if (!isfinite(con_ub[i]))
        v = fmax(v, 0.0);
    Y[(size_t)c * m + i] = v;
}

// First-round scalars of candidate c. Violations are reduced in the rescaled
// space, which decides the certificate, and in original units, which are reported.
void launch_candidate_reductions(const pdhg_solver_state_t *state, size_t *required_bytes, int c)
{
    infeas_ctx_t *ctx = state->infeas_ctx;
    const int n = state->num_variables, m = state->num_constraints;
    infeas_candidate_scalars_t *out = &state->scalars_d->infeas[c];
    // NULL while planning before the context exists; a size query never reads them
    const double *d = ctx ? ctx->D + (size_t)c * n : NULL; // the primal ray
    const double *y = ctx ? ctx->Y + (size_t)c * m : NULL; // the dual ray
    const double *Ad = ctx ? ctx->prod_m + (size_t)c * m : NULL;
    const double *ATy = ctx ? ctx->prod_n + (size_t)c * n : NULL;
    const double *objective = state->objective_vector;
    const double *variable_rescaling = state->variable_rescaling;
    const double *constraint_rescaling = state->constraint_rescaling;
    const double bound_rescaling = state->constraint_bound_rescaling;
    const double objective_rescaling = state->objective_vector_rescaling;
    const double *variable_lower = state->variable_lower_bound, *variable_upper = state->variable_upper_bound;
    const double *variable_lower_finite = state->variable_lower_bound_finite_val;
    const double *variable_upper_finite = state->variable_upper_bound_finite_val;
    const double *constraint_lower = state->constraint_lower_bound, *constraint_upper = state->constraint_upper_bound;
    const double *constraint_lower_finite = state->constraint_lower_bound_finite_val;
    const double *constraint_upper_finite = state->constraint_upper_bound_finite_val;

    // primal ray d
    auto d_abs_orig = DEVICE_FN(j) { return fabs(d[j]) / (variable_rescaling[j] * bound_rescaling); };
    auto objective_term = DEVICE_FN(j) { return objective[j] * d[j]; };
    auto objective_term_abs = DEVICE_FN(j) { return fabs(objective[j] * d[j]); };
    auto row_violation = DEVICE_FN(i)
    { return primal_ray_row_violation(Ad[i], isfinite(constraint_lower[i]), isfinite(constraint_upper[i])); };
    auto row_violation_orig = DEVICE_FN(i) { return row_violation(i) * constraint_rescaling[i]; };

    reduce_max(state, required_bytes, n, d_abs_orig, &out->primal_ray.max_abs);
    reduce_sum(state, required_bytes, n, objective_term, &out->primal_ray.obj);
    reduce_sum(state, required_bytes, n, objective_term_abs, &out->primal_ray.obj_abs);
    reduce_max(state, required_bytes, m, row_violation, &out->primal_ray.viol_max_scaled);
    reduce_max(state, required_bytes, m, row_violation_orig, &out->primal_ray.viol_max_orig);

    // dual ray y and its reduced cost r = -A^T y
    auto y_abs_orig = DEVICE_FN(i) { return fabs(y[i]) / (constraint_rescaling[i] * objective_rescaling); };
    auto con_bounds_term = DEVICE_FN(i)
    { return bound_term(y[i], constraint_lower_finite[i], constraint_upper_finite[i]); };
    auto con_bounds_term_abs = DEVICE_FN(i)
    { return bound_term_abs(y[i], constraint_lower_finite[i], constraint_upper_finite[i]); };
    auto var_bounds_term =
        DEVICE_FN(j) { return bound_term(-ATy[j], variable_lower_finite[j], variable_upper_finite[j]); };
    auto var_bounds_term_abs = DEVICE_FN(j)
    { return bound_term_abs(-ATy[j], variable_lower_finite[j], variable_upper_finite[j]); };
    auto reduced_cost_violation = DEVICE_FN(j)
    { return dual_ray_reduced_cost_violation(-ATy[j], isfinite(variable_lower[j]), isfinite(variable_upper[j])); };
    auto reduced_cost_violation_orig = DEVICE_FN(j) { return reduced_cost_violation(j) * variable_rescaling[j]; };

    reduce_max(state, required_bytes, m, y_abs_orig, &out->dual_ray.max_abs);
    reduce_sum(state, required_bytes, m, con_bounds_term, &out->dual_ray.obj_con_bounds);
    reduce_sum(state, required_bytes, n, var_bounds_term, &out->dual_ray.obj_var_bounds);
    reduce_sum(state, required_bytes, m, con_bounds_term_abs, &out->dual_ray.obj_con_bounds_abs);
    reduce_sum(state, required_bytes, n, var_bounds_term_abs, &out->dual_ray.obj_var_bounds_abs);
    reduce_max(state, required_bytes, n, reduced_cost_violation, &out->dual_ray.viol_max_scaled);
    reduce_max(state, required_bytes, n, reduced_cost_violation_orig, &out->dual_ray.viol_max_orig);
}

static void spmm_buffer(
    cusparseHandle_t handle, cusparseSpMatDescr_t mat, cusparseDnMatDescr_t in, cusparseDnMatDescr_t out, void **buffer)
{
    size_t size = 0;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &HOST_ONE,
                                           mat,
                                           in,
                                           &HOST_ZERO,
                                           out,
                                           CUDA_R_64F,
                                           CUSPARSE_SPMM_ALG_DEFAULT,
                                           &size));
    CUDA_CHECK(cudaMalloc(buffer, size > 0 ? size : 16));
}

// One-time analysis of the matrix for the repeated products.
static void spmm_preprocess(
    cusparseHandle_t handle, cusparseSpMatDescr_t mat, cusparseDnMatDescr_t in, cusparseDnMatDescr_t out, void *buffer)
{
    CUSPARSE_CHECK(cusparseSpMM_preprocess(handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &HOST_ONE,
                                           mat,
                                           in,
                                           &HOST_ZERO,
                                           out,
                                           CUDA_R_64F,
                                           CUSPARSE_SPMM_ALG_DEFAULT,
                                           buffer));
}

static void
spmm(cusparseHandle_t handle, cusparseSpMatDescr_t mat, cusparseDnMatDescr_t in, cusparseDnMatDescr_t out, void *buffer)
{
    CUSPARSE_CHECK(cusparseSpMM(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &HOST_ONE,
                                mat,
                                in,
                                &HOST_ZERO,
                                out,
                                CUDA_R_64F,
                                CUSPARSE_SPMM_ALG_DEFAULT,
                                buffer));
}

// A CSR descriptor on the indices of M with the given values.
static cusparseSpMatDescr_t create_csr_descr(const cu_sparse_matrix_csr_t *M, double *values)
{
    cusparseSpMatDescr_t descr;
    CUSPARSE_CHECK(cusparseCreateCsr(&descr,
                                     M->num_rows,
                                     M->num_cols,
                                     M->num_nonzeros,
                                     M->row_ptr,
                                     M->col_ind,
                                     values,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_64F));
    return descr;
}

// Allocates a column-major rows x K matrix and creates its descriptor.
static double *create_dense(int rows, cusparseDnMatDescr_t *descr)
{
    const int K = INFEAS_NUM_CANDIDATES;
    double *values = NULL;
    CUDA_CHECK(cudaMalloc(&values, (size_t)rows * K * sizeof(double)));
    CUSPARSE_CHECK(cusparseCreateDnMat(descr, rows, K, rows, values, CUDA_R_64F, CUSPARSE_ORDER_COL));
    return values;
}

infeas_ctx_t *infeasibility_context_create(pdhg_solver_state_t *state)
{
    infeas_ctx_t *ctx = (infeas_ctx_t *)safe_calloc(1, sizeof(infeas_ctx_t));
    const int n = state->num_variables, m = state->num_constraints;
    const cu_sparse_matrix_csr_t *A = state->constraint_matrix, *AT = state->constraint_matrix_t;
    const int nnz = A->num_nonzeros;

    CUDA_CHECK(cudaMalloc(&ctx->abs_val_A, (size_t)nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx->abs_val_AT, (size_t)nnz * sizeof(double)));
    abs_kernel<<<state->num_blocks_nnz, THREADS_PER_BLOCK, 0, state->stream>>>(ctx->abs_val_A, A->val, nnz);
    abs_kernel<<<state->num_blocks_nnz, THREADS_PER_BLOCK, 0, state->stream>>>(ctx->abs_val_AT, AT->val, nnz);

    ctx->matA = create_csr_descr(A, A->val);
    ctx->matAT = create_csr_descr(AT, AT->val);
    ctx->matAbsA = create_csr_descr(A, ctx->abs_val_A);
    ctx->matAbsAT = create_csr_descr(AT, ctx->abs_val_AT);

    ctx->D = create_dense(n, &ctx->dnD);
    ctx->prod_n = create_dense(n, &ctx->dnProdN);
    ctx->Y = create_dense(m, &ctx->dnY);
    ctx->prod_m = create_dense(m, &ctx->dnProdM);

    spmm_buffer(state->sparse_handle, ctx->matA, ctx->dnD, ctx->dnProdM, &ctx->buf_A);
    spmm_buffer(state->sparse_handle, ctx->matAT, ctx->dnY, ctx->dnProdN, &ctx->buf_AT);
    spmm_buffer(state->sparse_handle, ctx->matAbsA, ctx->dnD, ctx->dnProdM, &ctx->buf_absA);
    spmm_buffer(state->sparse_handle, ctx->matAbsAT, ctx->dnY, ctx->dnProdN, &ctx->buf_absAT);
    spmm_preprocess(state->sparse_handle, ctx->matA, ctx->dnD, ctx->dnProdM, ctx->buf_A);
    spmm_preprocess(state->sparse_handle, ctx->matAT, ctx->dnY, ctx->dnProdN, ctx->buf_AT);
    spmm_preprocess(state->sparse_handle, ctx->matAbsA, ctx->dnD, ctx->dnProdM, ctx->buf_absA);
    spmm_preprocess(state->sparse_handle, ctx->matAbsAT, ctx->dnY, ctx->dnProdN, ctx->buf_absAT);
    return ctx;
}

void infeasibility_context_destroy(infeas_ctx_t *ctx)
{
    if (!ctx)
        return;
    cusparseSpMatDescr_t mats[] = {ctx->matA, ctx->matAT, ctx->matAbsA, ctx->matAbsAT};
    for (size_t i = 0; i < sizeof(mats) / sizeof(mats[0]); i++)
        if (mats[i])
            CUSPARSE_CHECK(cusparseDestroySpMat(mats[i]));
    cusparseDnMatDescr_t dns[] = {ctx->dnD, ctx->dnY, ctx->dnProdM, ctx->dnProdN};
    for (size_t i = 0; i < sizeof(dns) / sizeof(dns[0]); i++)
        if (dns[i])
            CUSPARSE_CHECK(cusparseDestroyDnMat(dns[i]));
    void *bufs[] = {ctx->abs_val_A,
                    ctx->abs_val_AT,
                    ctx->D,
                    ctx->Y,
                    ctx->prod_m,
                    ctx->prod_n,
                    ctx->buf_A,
                    ctx->buf_AT,
                    ctx->buf_absA,
                    ctx->buf_absAT};
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++)
        if (bufs[i])
            CUDA_CHECK(cudaFree(bufs[i]));
    free(ctx);
}

// On an infeasible problem the PDHG iterates diverge along a direction whose
// primal or dual part is an infeasibility certificate. Three candidates are
// tried: T(z) - z, T(z), and T(z) - z^0 (the displacement since the last
// restart). After projection onto the sign conditions, a primal ray d
// certifies dual infeasibility and a dual ray y primal infeasibility when
//   -c^T d   >= delta |c|^T |d|           and  viol(A d)   / max_i (|A||d|)_i   <= eps
//   Omega(y) >= delta sum |Omega terms|   and  viol(A^T y) / max_j (|A^T||y|)_j <= eps
// with delta = infeasible_ray_objective_margin and eps = eps_infeasible_relative.
// The ratio is taken in the rescaled space, where rows have comparable scales;
// in original units a large row can hide a violated one.
void compute_infeasibility_device(pdhg_solver_state_t *state)
{
    infeas_ctx_t *ctx = state->infeas_ctx;
    const int n = state->num_variables, m = state->num_constraints, K = INFEAS_NUM_CANDIDATES;

    // reflected = 2 T(z) - z, so reflected - pdhg = T(z) - z
    const candidate_directions_t primal = {{
        {state->reflected_primal_solution, state->pdhg_primal_solution}, // T(z) - z
        {state->pdhg_primal_solution, NULL},                             // T(z)
        {state->pdhg_primal_solution, state->initial_primal_solution},   // T(z) - z^0
    }};
    const candidate_directions_t dual = {{
        {state->reflected_dual_solution, state->pdhg_dual_solution},
        {state->pdhg_dual_solution, NULL},
        {state->pdhg_dual_solution, state->initial_dual_solution},
    }};
    form_primal_candidates_kernel<<<dim3(state->num_blocks_primal, K), THREADS_PER_BLOCK, 0, state->stream>>>(
        primal, state->variable_lower_bound, state->variable_upper_bound, ctx->D, n);
    form_dual_candidates_kernel<<<dim3(state->num_blocks_dual, K), THREADS_PER_BLOCK, 0, state->stream>>>(
        dual, state->constraint_lower_bound, state->constraint_upper_bound, ctx->Y, m);

    spmm(state->sparse_handle, ctx->matA, ctx->dnD, ctx->dnProdM, ctx->buf_A);
    spmm(state->sparse_handle, ctx->matAT, ctx->dnY, ctx->dnProdN, ctx->buf_AT);

    // everything that needs D, Y or the products, before the second round overwrites them
    for (int c = 0; c < K; c++)
    {
        launch_candidate_reductions(state, NULL, c);
    }
}

// a / b, or 0 when b is 0: a ray of zero norm has no statistics.
static double divide_or_zero(double a, double b)
{
    return b > 0.0 ? a / b : 0.0;
}

// violation / product_abs_max. A zero product certifies only a zero violation.
static double certificate_error(double violation, double product_abs_max)
{
    if (product_abs_max > 0.0)
        return violation / product_abs_max;
    return violation > 0.0 ? INFINITY : 0.0;
}

// Ray statistics in original units and the objective margin test of every candidate.
static void
evaluate_objective_margins(const pdhg_solver_state_t *state, double objective_margin, ray_candidate_stats_t *stats)
{
    const double bound_rescaling = state->constraint_bound_rescaling;
    const double objective_rescaling = state->objective_vector_rescaling;
    const double objective_scale = bound_rescaling * objective_rescaling;

    for (int c = 0; c < INFEAS_NUM_CANDIDATES; c++)
    {
        const primal_ray_scalars_t *primal = &state->scalars_h->infeas[c].primal_ray;
        const dual_ray_scalars_t *dual = &state->scalars_h->infeas[c].dual_ray;
        ray_candidate_stats_t *out = &stats[c];

        out->primal_ray_linear_objective = divide_or_zero(primal->obj / objective_scale, primal->max_abs);
        out->primal_objective_abs = divide_or_zero(primal->obj_abs / objective_scale, primal->max_abs);
        out->max_primal_ray_infeasibility = divide_or_zero(primal->viol_max_orig / bound_rescaling, primal->max_abs);
        out->dual_ray_objective =
            divide_or_zero((dual->obj_con_bounds + dual->obj_var_bounds) / objective_scale, dual->max_abs);
        out->dual_objective_abs =
            divide_or_zero((dual->obj_con_bounds_abs + dual->obj_var_bounds_abs) / objective_scale, dual->max_abs);
        out->max_dual_ray_infeasibility = divide_or_zero(dual->viol_max_orig / objective_rescaling, dual->max_abs);

        out->primal_passes_margin = out->primal_ray_linear_objective < 0.0 && out->primal_objective_abs > 0.0 &&
            -out->primal_ray_linear_objective >= objective_margin * out->primal_objective_abs;
        out->dual_passes_margin = out->dual_ray_objective > 0.0 && out->dual_objective_abs > 0.0 &&
            out->dual_ray_objective >= objective_margin * out->dual_objective_abs;
    }
}

void launch_primal_abs_reductions(const pdhg_solver_state_t *state, size_t *required_bytes)
{
    const int m = state->num_constraints;
    for (int c = 0; c < INFEAS_NUM_CANDIDATES; c++)
    {
        const double *p = state->infeas_ctx ? state->infeas_ctx->prod_m + (size_t)c * m : NULL;
        reduce_max(
            state,
            required_bytes,
            m,
            DEVICE_FN(i) { return p[i]; },
            &state->scalars_d->infeas[c].primal_ray.product_abs_max);
    }
}

void launch_dual_abs_reductions(const pdhg_solver_state_t *state, size_t *required_bytes)
{
    const int n = state->num_variables;
    for (int c = 0; c < INFEAS_NUM_CANDIDATES; c++)
    {
        const double *p = state->infeas_ctx ? state->infeas_ctx->prod_n + (size_t)c * n : NULL;
        reduce_max(
            state, required_bytes, n, DEVICE_FN(j) { return p[j]; }, &state->scalars_d->infeas[c].dual_ray.product_abs_max);
    }
}

void plan_infeasibility_reductions(const pdhg_solver_state_t *state, size_t *required_bytes)
{
    for (int c = 0; c < INFEAS_NUM_CANDIDATES; c++)
        launch_candidate_reductions(state, required_bytes, c);
    launch_primal_abs_reductions(state, required_bytes);
    launch_dual_abs_reductions(state, required_bytes);
}

// Second round for the sides that need it: D becomes |D| in place and prod_m
// becomes |A||D|; likewise on the dual side.
void launch_abs_products(pdhg_solver_state_t *state, bool primal_side, bool dual_side)
{
    infeas_ctx_t *ctx = state->infeas_ctx;
    const int n = state->num_variables, m = state->num_constraints, K = INFEAS_NUM_CANDIDATES;

    if (primal_side)
    {
        abs_kernel<<<state->num_blocks_primal * K, THREADS_PER_BLOCK, 0, state->stream>>>(ctx->D, ctx->D, n * K);
        spmm(state->sparse_handle, ctx->matAbsA, ctx->dnD, ctx->dnProdM, ctx->buf_absA);
        launch_primal_abs_reductions(state, NULL);
    }
    if (dual_side)
    {
        abs_kernel<<<state->num_blocks_dual * K, THREADS_PER_BLOCK, 0, state->stream>>>(ctx->Y, ctx->Y, m * K);
        spmm(state->sparse_handle, ctx->matAbsAT, ctx->dnY, ctx->dnProdN, ctx->buf_absAT);
        launch_dual_abs_reductions(state, NULL);
    }
}

// Keeps the candidate with the smallest certificate error per side. A side with
// no candidate past the margin reports candidate 0 with an infinite error.
static void select_best_candidates(pdhg_solver_state_t *state, const ray_candidate_stats_t *stats)
{
    int best_primal = 0, best_dual = 0;
    double best_primal_error = INFINITY, best_dual_error = INFINITY;
    for (int c = 0; c < INFEAS_NUM_CANDIDATES; c++)
    {
        const primal_ray_scalars_t *primal = &state->scalars_h->infeas[c].primal_ray;
        const dual_ray_scalars_t *dual = &state->scalars_h->infeas[c].dual_ray;
        double primal_error = stats[c].primal_passes_margin
            ? certificate_error(primal->viol_max_scaled, primal->product_abs_max)
            : INFINITY;
        double dual_error =
            stats[c].dual_passes_margin ? certificate_error(dual->viol_max_scaled, dual->product_abs_max) : INFINITY;
        if (primal_error < best_primal_error)
        {
            best_primal_error = primal_error;
            best_primal = c;
        }
        if (dual_error < best_dual_error)
        {
            best_dual_error = dual_error;
            best_dual = c;
        }
    }
    state->max_primal_ray_infeasibility = stats[best_primal].max_primal_ray_infeasibility;
    state->primal_ray_linear_objective = stats[best_primal].primal_ray_linear_objective;
    state->primal_ray_certificate_error = best_primal_error;
    state->max_dual_ray_infeasibility = stats[best_dual].max_dual_ray_infeasibility;
    state->dual_ray_objective = stats[best_dual].dual_ray_objective;
    state->dual_ray_certificate_error = best_dual_error;
}

// Runs the second round here, with one more synchronization, only when some
// candidate passes its objective margin.
void compute_infeasibility_host(pdhg_solver_state_t *state, double objective_margin)
{
    ray_candidate_stats_t stats[INFEAS_NUM_CANDIDATES];
    evaluate_objective_margins(state, objective_margin, stats);

    bool primal_side = false, dual_side = false;
    for (int c = 0; c < INFEAS_NUM_CANDIDATES; c++)
    {
        primal_side = primal_side || stats[c].primal_passes_margin;
        dual_side = dual_side || stats[c].dual_passes_margin;
    }
    if (primal_side || dual_side)
    {
        launch_abs_products(state, primal_side, dual_side);
        fetch_device_scalars(state);
    }
    select_best_candidates(state, stats);
}
