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

// Statistics of one candidate ray, in the original problem space.
typedef struct
{
    double max_primal_ray_infeasibility;
    double primal_ray_linear_objective;
    double max_dual_ray_infeasibility;
    double dual_ray_objective;
    // the objective terms with absolute values; the margin is measured against them
    double primal_objective_abs;
    double dual_objective_abs;
} ray_candidate_stats_t;

__global__ void abs_kernel(double *__restrict__ out, const double *__restrict__ x, int n)
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
    // from the margin test, read when the best candidate is selected
    ray_candidate_stats_t stats[INFEAS_NUM_CANDIDATES];
    bool need_primal[INFEAS_NUM_CANDIDATES];
    bool need_dual[INFEAS_NUM_CANDIDATES];
};

// Candidate c is x[c] + beta[c] * y[c], or x[c] alone when y[c] is NULL.
typedef struct
{
    const double *x[INFEAS_NUM_CANDIDATES];
    const double *y[INFEAS_NUM_CANDIDATES];
    double beta[INFEAS_NUM_CANDIDATES];
} candidate_sources_t;

__device__ __forceinline__ double candidate_entry(const candidate_sources_t &src, int c, int i)
{
    return src.y[c] ? src.x[c][i] + src.beta[c] * src.y[c][i] : src.x[c][i];
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

// Dual objective terms of an entry v (y_i or r_j): v^+ times the finite lower
// bound and v^- times the finite upper bound.
__host__ __device__ __forceinline__ double lower_bound_term(double v, double lower_bound_finite)
{
    return fmax(v, 0.0) * lower_bound_finite;
}

__host__ __device__ __forceinline__ double upper_bound_term(double v, double upper_bound_finite)
{
    return fmin(v, 0.0) * upper_bound_finite;
}

// Forms candidate blockIdx.y and projects it onto the sign conditions.
__global__ void form_primal_candidates_kernel(candidate_sources_t src,
                                              const double *__restrict__ var_lb,
                                              const double *__restrict__ var_ub,
                                              double *__restrict__ D,
                                              int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y;
    if (j >= n)
        return;
    double v = candidate_entry(src, c, j);
    if (isfinite(var_lb[j]))
        v = fmax(v, 0.0);
    if (isfinite(var_ub[j]))
        v = fmin(v, 0.0);
    D[(size_t)c * n + j] = v;
}

__global__ void form_dual_candidates_kernel(candidate_sources_t src,
                                            const double *__restrict__ con_lb,
                                            const double *__restrict__ con_ub,
                                            double *__restrict__ Y,
                                            int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y;
    if (i >= m)
        return;
    double v = candidate_entry(src, c, i);
    if (!isfinite(con_lb[i]))
        v = fmin(v, 0.0);
    if (!isfinite(con_ub[i]))
        v = fmax(v, 0.0);
    Y[(size_t)c * m + i] = v;
}

__global__ void abs_columns_kernel(double *__restrict__ X, int rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows)
        X[(size_t)blockIdx.y * rows + i] = fabs(X[(size_t)blockIdx.y * rows + i]);
}

// First-round scalars of candidate c. Violations are reduced in the rescaled
// space, which decides the certificate, and in original units, which are reported.
static void launch_candidate_reductions(pdhg_solver_state_t *state, int c)
{
    infeas_ctx_t *ctx = state->infeas_ctx;
    const int n = state->num_variables, m = state->num_constraints;
    infeas_candidate_scalars_t *out = &state->scalars_d->infeas[c];
    const double *d = ctx->D + (size_t)c * n; // the primal ray
    const double *y = ctx->Y + (size_t)c * m; // the dual ray
    const double *Ad = ctx->prod_m + (size_t)c * m;
    const double *ATy = ctx->prod_n + (size_t)c * n;
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
    auto row_violation = [=] __host__ __device__(int i)
    { return primal_ray_row_violation(Ad[i], isfinite(constraint_lower[i]), isfinite(constraint_upper[i])); };
    reduce_max(
        state,
        n,
        [=] __host__ __device__(int j) { return fabs(d[j]) / (variable_rescaling[j] * bound_rescaling); },
        &out->primal_ray.max_abs);
    reduce_sum(state, n, [=] __host__ __device__(int j) { return objective[j] * d[j]; }, &out->primal_ray.obj);
    reduce_sum(
        state, n, [=] __host__ __device__(int j) { return fabs(objective[j] * d[j]); }, &out->primal_ray.obj_abs);
    reduce_max(state, m, row_violation, &out->primal_ray.viol_max_scaled);
    reduce_max(
        state,
        m,
        [=] __host__ __device__(int i) { return row_violation(i) * constraint_rescaling[i]; },
        &out->primal_ray.viol_max_orig);

    // dual ray y and its reduced cost r = -A^T y
    auto reduced_cost = [=] __host__ __device__(int j) { return -ATy[j]; };
    auto y_lower_term = [=] __host__ __device__(int i) { return lower_bound_term(y[i], constraint_lower_finite[i]); };
    auto y_upper_term = [=] __host__ __device__(int i) { return upper_bound_term(y[i], constraint_upper_finite[i]); };
    auto r_lower_term = [=] __host__ __device__(int j)
    { return lower_bound_term(reduced_cost(j), variable_lower_finite[j]); };
    auto r_upper_term = [=] __host__ __device__(int j)
    { return upper_bound_term(reduced_cost(j), variable_upper_finite[j]); };
    auto reduced_cost_violation = [=] __host__ __device__(int j)
    {
        return dual_ray_reduced_cost_violation(
            reduced_cost(j), isfinite(variable_lower[j]), isfinite(variable_upper[j]));
    };
    reduce_max(
        state,
        m,
        [=] __host__ __device__(int i) { return fabs(y[i]) / (constraint_rescaling[i] * objective_rescaling); },
        &out->dual_ray.max_abs);
    reduce_sum(
        state,
        m,
        [=] __host__ __device__(int i) { return y_lower_term(i) + y_upper_term(i); },
        &out->dual_ray.obj_con_bounds);
    reduce_sum(
        state,
        n,
        [=] __host__ __device__(int j) { return r_lower_term(j) + r_upper_term(j); },
        &out->dual_ray.obj_var_bounds);
    reduce_sum(
        state,
        m,
        [=] __host__ __device__(int i) { return fabs(y_lower_term(i)) + fabs(y_upper_term(i)); },
        &out->dual_ray.obj_con_bounds_abs);
    reduce_sum(
        state,
        n,
        [=] __host__ __device__(int j) { return fabs(r_lower_term(j)) + fabs(r_upper_term(j)); },
        &out->dual_ray.obj_var_bounds_abs);
    reduce_max(state, n, reduced_cost_violation, &out->dual_ray.viol_max_scaled);
    reduce_max(
        state,
        n,
        [=] __host__ __device__(int j) { return reduced_cost_violation(j) * variable_rescaling[j]; },
        &out->dual_ray.viol_max_orig);
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

infeas_ctx_t *infeasibility_context_create(pdhg_solver_state_t *state)
{
    infeas_ctx_t *ctx = (infeas_ctx_t *)safe_calloc(1, sizeof(infeas_ctx_t));
    const int n = state->num_variables, m = state->num_constraints, K = INFEAS_NUM_CANDIDATES;
    const cu_sparse_matrix_csr_t *A = state->constraint_matrix, *AT = state->constraint_matrix_t;
    const int nnz = A->num_nonzeros;

    CUDA_CHECK(cudaMalloc(&ctx->abs_val_A, (size_t)nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx->abs_val_AT, (size_t)nnz * sizeof(double)));
    abs_kernel<<<state->num_blocks_nnz, THREADS_PER_BLOCK, 0, state->stream>>>(ctx->abs_val_A, A->val, nnz);
    abs_kernel<<<state->num_blocks_nnz, THREADS_PER_BLOCK, 0, state->stream>>>(ctx->abs_val_AT, AT->val, nnz);

#define CREATE_CSR(descr, M, vals)                                                                                     \
    CUSPARSE_CHECK(cusparseCreateCsr(&(descr),                                                                         \
                                     (M)->num_rows,                                                                    \
                                     (M)->num_cols,                                                                    \
                                     (M)->num_nonzeros,                                                                \
                                     (M)->row_ptr,                                                                     \
                                     (M)->col_ind,                                                                     \
                                     (vals),                                                                           \
                                     CUSPARSE_INDEX_32I,                                                               \
                                     CUSPARSE_INDEX_32I,                                                               \
                                     CUSPARSE_INDEX_BASE_ZERO,                                                         \
                                     CUDA_R_64F))
    CREATE_CSR(ctx->matA, A, A->val);
    CREATE_CSR(ctx->matAT, AT, AT->val);
    CREATE_CSR(ctx->matAbsA, A, ctx->abs_val_A);
    CREATE_CSR(ctx->matAbsAT, AT, ctx->abs_val_AT);
#undef CREATE_CSR

#define CREATE_DN(ptr, descr, rows)                                                                                    \
    CUDA_CHECK(cudaMalloc(&(ptr), (size_t)(rows) * K * sizeof(double)));                                               \
    CUSPARSE_CHECK(cusparseCreateDnMat(&(descr), (rows), K, (rows), (ptr), CUDA_R_64F, CUSPARSE_ORDER_COL))
    CREATE_DN(ctx->D, ctx->dnD, n);
    CREATE_DN(ctx->prod_n, ctx->dnProdN, n);
    CREATE_DN(ctx->Y, ctx->dnY, m);
    CREATE_DN(ctx->prod_m, ctx->dnProdM, m);
#undef CREATE_DN

    spmm_buffer(state->sparse_handle, ctx->matA, ctx->dnD, ctx->dnProdM, &ctx->buf_A);
    spmm_buffer(state->sparse_handle, ctx->matAT, ctx->dnY, ctx->dnProdN, &ctx->buf_AT);
    spmm_buffer(state->sparse_handle, ctx->matAbsA, ctx->dnD, ctx->dnProdM, &ctx->buf_absA);
    spmm_buffer(state->sparse_handle, ctx->matAbsAT, ctx->dnY, ctx->dnProdN, &ctx->buf_absAT);
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
// with delta = infeasible_objective_margin and eps = eps_infeasible_relative.
// The ratio is taken in the rescaled space, where rows have comparable scales;
// in original units a large row can hide a violated one.
void compute_infeasibility_device(pdhg_solver_state_t *state)
{
    infeas_ctx_t *ctx = state->infeas_ctx;
    const int n = state->num_variables, m = state->num_constraints, K = INFEAS_NUM_CANDIDATES;

    candidate_sources_t ps, ds;
    // T(z) - z = reflected - pdhg, since reflected = 2 T(z) - z
    ps.x[0] = state->reflected_primal_solution, ps.y[0] = state->pdhg_primal_solution, ps.beta[0] = -1.0;
    ds.x[0] = state->reflected_dual_solution, ds.y[0] = state->pdhg_dual_solution, ds.beta[0] = -1.0;
    // T(z)
    ps.x[1] = state->pdhg_primal_solution, ps.y[1] = NULL, ps.beta[1] = 0.0;
    ds.x[1] = state->pdhg_dual_solution, ds.y[1] = NULL, ds.beta[1] = 0.0;
    // T(z) - z^0
    ps.x[2] = state->pdhg_primal_solution, ps.y[2] = state->initial_primal_solution, ps.beta[2] = -1.0;
    ds.x[2] = state->pdhg_dual_solution, ds.y[2] = state->initial_dual_solution, ds.beta[2] = -1.0;
    form_primal_candidates_kernel<<<dim3(state->num_blocks_primal, K), THREADS_PER_BLOCK, 0, state->stream>>>(
        ps, state->variable_lower_bound, state->variable_upper_bound, ctx->D, n);
    form_dual_candidates_kernel<<<dim3(state->num_blocks_dual, K), THREADS_PER_BLOCK, 0, state->stream>>>(
        ds, state->constraint_lower_bound, state->constraint_upper_bound, ctx->Y, m);

    spmm(state->sparse_handle, ctx->matA, ctx->dnD, ctx->dnProdM, ctx->buf_A);
    spmm(state->sparse_handle, ctx->matAT, ctx->dnY, ctx->dnProdN, ctx->buf_AT);

    // everything that needs D, Y or the products, before the second round overwrites them
    for (int c = 0; c < K; c++)
    {
        launch_candidate_reductions(state, c);
    }
}

// Ray statistics in original units and the objective margin test. Returns
// whether any candidate passed, that is, whether the second round is needed.
static bool evaluate_objective_margins(pdhg_solver_state_t *state, double objective_margin)
{
    infeas_ctx_t *ctx = state->infeas_ctx;
    const double bc = state->constraint_bound_rescaling, oc = state->objective_vector_rescaling;

    bool any = false;
    for (int c = 0; c < INFEAS_NUM_CANDIDATES; c++)
    {
        const infeas_candidate_scalars_t *v = &state->scalars_h->infeas[c];
        ray_candidate_stats_t *s = &ctx->stats[c];
        double primal_norm = v->primal_ray.max_abs, dual_norm = v->dual_ray.max_abs;
        s->primal_ray_linear_objective = primal_norm > 0.0 ? v->primal_ray.obj / (bc * oc) / primal_norm : 0.0;
        s->primal_objective_abs = primal_norm > 0.0 ? v->primal_ray.obj_abs / (bc * oc) / primal_norm : 0.0;
        s->max_primal_ray_infeasibility = primal_norm > 0.0 ? v->primal_ray.viol_max_orig / bc / primal_norm : 0.0;
        s->dual_ray_objective =
            dual_norm > 0.0 ? (v->dual_ray.obj_con_bounds + v->dual_ray.obj_var_bounds) / (bc * oc) / dual_norm : 0.0;
        s->dual_objective_abs = dual_norm > 0.0
            ? (v->dual_ray.obj_con_bounds_abs + v->dual_ray.obj_var_bounds_abs) / (bc * oc) / dual_norm
            : 0.0;
        s->max_dual_ray_infeasibility = dual_norm > 0.0 ? v->dual_ray.viol_max_orig / oc / dual_norm : 0.0;
        ctx->need_primal[c] = s->primal_ray_linear_objective < 0.0 && s->primal_objective_abs > 0.0 &&
            -s->primal_ray_linear_objective >= objective_margin * s->primal_objective_abs;
        ctx->need_dual[c] = s->dual_ray_objective > 0.0 && s->dual_objective_abs > 0.0 &&
            s->dual_ray_objective >= objective_margin * s->dual_objective_abs;
        any = any || ctx->need_primal[c] || ctx->need_dual[c];
    }
    return any;
}

// Second round, for each side on which some candidate passed the margin: D
// becomes |D| in place and prod_m becomes |A||D|; likewise on the dual side.
static void launch_abs_products(pdhg_solver_state_t *state)
{
    infeas_ctx_t *ctx = state->infeas_ctx;
    const int n = state->num_variables, m = state->num_constraints, K = INFEAS_NUM_CANDIDATES;

    bool any_primal = false, any_dual = false;
    for (int c = 0; c < K; c++)
    {
        any_primal = any_primal || ctx->need_primal[c];
        any_dual = any_dual || ctx->need_dual[c];
    }
    if (any_primal)
    {
        abs_columns_kernel<<<dim3(state->num_blocks_primal, K), THREADS_PER_BLOCK, 0, state->stream>>>(ctx->D, n);
        spmm(state->sparse_handle, ctx->matAbsA, ctx->dnD, ctx->dnProdM, ctx->buf_absA);
        for (int c = 0; c < K; c++)
        {
            const double *p = ctx->prod_m + (size_t)c * m;
            reduce_max(
                state,
                m,
                [=] __host__ __device__(int i) { return p[i]; },
                &state->scalars_d->infeas[c].primal_ray.product_abs_max);
        }
    }
    if (any_dual)
    {
        abs_columns_kernel<<<dim3(state->num_blocks_dual, K), THREADS_PER_BLOCK, 0, state->stream>>>(ctx->Y, m);
        spmm(state->sparse_handle, ctx->matAbsAT, ctx->dnY, ctx->dnProdN, ctx->buf_absAT);
        for (int c = 0; c < K; c++)
        {
            const double *p = ctx->prod_n + (size_t)c * n;
            reduce_max(
                state,
                n,
                [=] __host__ __device__(int j) { return p[j]; },
                &state->scalars_d->infeas[c].dual_ray.product_abs_max);
        }
    }
}

// Keeps the candidate with the smallest certificate error per side. A side with
// no candidate past the margin reports candidate 0 with an infinite error.
static void select_best_candidates(pdhg_solver_state_t *state)
{
    infeas_ctx_t *ctx = state->infeas_ctx;
    int best_p = 0, best_d = 0;
    double best_perr = INFINITY, best_derr = INFINITY;
    for (int c = 0; c < INFEAS_NUM_CANDIDATES; c++)
    {
        const infeas_candidate_scalars_t *v = &state->scalars_h->infeas[c];
        double perr = INFINITY, derr = INFINITY;
        if (ctx->need_primal[c])
        {
            double viol = v->primal_ray.viol_max_scaled, act = v->primal_ray.product_abs_max;
            perr = act > 0.0 ? viol / act : (viol > 0.0 ? INFINITY : 0.0);
        }
        if (ctx->need_dual[c])
        {
            double viol = v->dual_ray.viol_max_scaled, act = v->dual_ray.product_abs_max;
            derr = act > 0.0 ? viol / act : (viol > 0.0 ? INFINITY : 0.0);
        }
        if (perr < best_perr)
        {
            best_perr = perr;
            best_p = c;
        }
        if (derr < best_derr)
        {
            best_derr = derr;
            best_d = c;
        }
    }
    state->max_primal_ray_infeasibility = ctx->stats[best_p].max_primal_ray_infeasibility;
    state->primal_ray_linear_objective = ctx->stats[best_p].primal_ray_linear_objective;
    state->primal_ray_certificate_error = best_perr;
    state->max_dual_ray_infeasibility = ctx->stats[best_d].max_dual_ray_infeasibility;
    state->dual_ray_objective = ctx->stats[best_d].dual_ray_objective;
    state->dual_ray_certificate_error = best_derr;
}

// Runs the second round here, with one more synchronization, only when some
// candidate passes its objective margin.
void compute_infeasibility_host(pdhg_solver_state_t *state, double objective_margin)
{
    if (evaluate_objective_margins(state, objective_margin))
    {
        launch_abs_products(state);
        fetch_device_scalars(state);
    }
    select_best_candidates(state);
}
