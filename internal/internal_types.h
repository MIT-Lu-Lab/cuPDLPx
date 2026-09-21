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

#include "cupdlpx_types.h"

// Include cuda_to_hip.h first to map CUDA -> HIP symbols when building for ROCm.
#include "cuda_to_hip.h"
#include "cusparse_compat.h"

#include <stdbool.h>
#include <time.h>

// Scalars of a termination check, one reduction each, in the rescaled space
// except viol_max_orig. Name parts: res = residual, sq = sum of squares,
// max = infinity norm, dot = inner product.

#define INFEAS_NUM_CANDIDATES 3 // candidate rays of one infeasibility check

// One infeasibility candidate: a primal ray d, which certifies dual
// infeasibility, and a dual ray y with reduced cost r = -A^T y, which certifies
// primal infeasibility. viol_max_scaled / product_abs_max is the certificate
// error; viol_max_orig is only reported.
typedef struct
{
    double max_abs;         // ||d||_inf in the original space; normalizes the reported statistics
    double obj;             // c^T d, the objective term
    double obj_abs;         // |c|^T |d|, the same with absolute values
    double viol_max_scaled; // max violation of the sign conditions on A d, rescaled space
    double product_abs_max; // max_i (|A||d|)_i, scaled like the violation above (second round)
    double viol_max_orig;   // the violation in original units, reported only
} primal_ray_scalars_t;

typedef struct
{
    double max_abs;            // ||y||_inf in the original space
    double obj_con_bounds;     // l_c^T y^+ - u_c^T y^-, the constraint-bound part of the dual objective term
    double obj_var_bounds;     // l_v^T r^+ - u_v^T r^-, the variable-bound part
    double obj_con_bounds_abs; // |l_c|^T y^+ + |u_c|^T y^-, the constraint-bound part with absolute values
    double obj_var_bounds_abs; // |l_v|^T r^+ + |u_v|^T r^-, the variable-bound part with absolute values
    double viol_max_scaled;    // max violation of the sign conditions on r, rescaled space
    double product_abs_max;    // max_j (|A^T||y|)_j, scaled like the violation above (second round)
    double viol_max_orig;      // the violation in original units, reported only
} dual_ray_scalars_t;

typedef struct
{
    primal_ray_scalars_t primal_ray;
    dual_ray_scalars_t dual_ray;
} infeas_candidate_scalars_t;

typedef struct
{
    double fixed_point_dual_sq;   // ||y - T(z)_y||^2, dual part of the fixed point error
    double fixed_point_primal_sq; // ||x - T(z)_x||^2, primal part
    double fixed_point_cross;     // (A^T dy)^T dx, cross term of the same error
    double primal_res_sq;         // sum of squares of the primal residual
    double primal_res_max;        // infinity norm of the primal residual
    double primal_slack_sum;      // sum of the primal slacks, a dual objective term
    double dual_res_sq;           // sum of squares of the dual residual
    double dual_res_max;          // infinity norm of the dual residual
    double objective_dot;         // c^T x, the primal objective
    double dual_slack_dot;        // (dual slack)^T x, the other dual objective term
    double restart_primal_sq;     // ||x - x_anchor||^2, the primal weight's PID input
    double restart_dual_sq;       // ||y - y_anchor||^2, the same on the dual side
    infeas_candidate_scalars_t infeas[INFEAS_NUM_CANDIDATES];
} device_scalars_t;

// Private state of the infeasibility detection, defined in infeasibility.cu.
typedef struct infeas_ctx infeas_ctx_t;

typedef struct
{
    int num_rows;
    int num_cols;
    int num_nonzeros;
    int *row_ptr;
    int *col_ind;
    int *row_ind;
    double *val;
    int *transpose_map;
} cu_sparse_matrix_csr_t;

typedef struct
{
    int num_variables;
    int num_constraints;
    double *variable_lower_bound;
    double *variable_upper_bound;
    double *objective_vector;
    double objective_constant;
    double original_objective_sign;
    cu_sparse_matrix_csr_t *constraint_matrix;
    cu_sparse_matrix_csr_t *constraint_matrix_t;
    double *constraint_lower_bound;
    double *constraint_upper_bound;
    int num_blocks_primal;
    int num_blocks_dual;
    int num_blocks_primal_dual;
    int num_blocks_nnz;
    double objective_vector_norm;
    double constraint_bound_norm;
    double *constraint_lower_bound_finite_val;
    double *constraint_upper_bound_finite_val;
    double *variable_lower_bound_finite_val;
    double *variable_upper_bound_finite_val;

    double *initial_primal_solution;
    double *current_primal_solution;
    double *pdhg_primal_solution;
    double *reflected_primal_solution;
    double *dual_product;
    double *initial_dual_solution;
    double *current_dual_solution;
    double *pdhg_dual_solution;
    double *reflected_dual_solution;
    double *primal_product;
    double step_size;
    double *d_primal_step_size;
    double *d_dual_step_size;
    double primal_weight;
    int total_count;
    bool is_this_major_iteration;
    double primal_weight_error_sum;
    double primal_weight_last_error;
    double best_primal_weight;
    double best_primal_dual_residual_gap;

    double *constraint_rescaling;
    double *variable_rescaling;
    double constraint_bound_rescaling;
    double objective_vector_rescaling;
    double *primal_slack;
    double *dual_slack;
    double rescaling_time_sec;
    clock_t start_time;
    double cumulative_time_sec;

    double *primal_residual;
    double absolute_primal_residual;
    double relative_primal_residual;
    double *dual_residual;
    double absolute_dual_residual;
    double relative_dual_residual;
    double primal_objective_value;
    double dual_objective_value;
    double objective_gap;
    double relative_objective_gap;
    double max_primal_ray_infeasibility;
    double max_dual_ray_infeasibility;
    double primal_ray_linear_objective;
    double dual_ray_objective;
    double primal_ray_certificate_error;
    double dual_ray_certificate_error;
    termination_reason_t termination_reason;

    double *delta_primal_solution;
    double *delta_dual_solution;
    double fixed_point_error;
    double initial_fixed_point_error;
    double last_trial_fixed_point_error;
    int inner_count;
    int *d_inner_count;

    cusparseHandle_t sparse_handle;
    cublasHandle_t blas_handle;
    void *spmv_ctx;
    infeas_ctx_t *infeas_ctx; // detection buffers, created before the main loop when detection is on

    // Both norms of each residual are kept so that the norm type is a host-side choice.
    device_scalars_t *scalars_d;
    device_scalars_t *scalars_h; // pinned, so that the copy is asynchronous
    void *reduce_temp_d;         // temporary storage of the reductions
    size_t reduce_temp_bytes;

    double *delta_dual_product; // A^T delta_dual_solution, scratch of the fixed point error

    double feasibility_polishing_time;
    int feasibility_iteration;

    cudaStream_t stream;
} pdhg_solver_state_t;

typedef struct
{
    double *con_rescale;
    double *var_rescale;
    double con_bound_rescale;
    double obj_vec_rescale;
    double rescaling_time_sec;
} rescale_info_t;
