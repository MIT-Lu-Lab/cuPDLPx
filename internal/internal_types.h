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

/* Power-iteration estimator of the maximum singular value. The context owns the
   device workspace and SpMV plans for one (A, AT) pair and is defined in utils.cu;
   after a non-degenerate run it warm-starts the next run from its final eigenvector. */
typedef struct sv_estimator_ctx sv_estimator_ctx_t;

/* Per-run options; zero-initialize and set what you need. */
typedef struct
{
    int max_iterations;
    double tolerance;
    const bool *d_row_mask;                /* NULL = unmasked */
    const bool *d_col_mask;                /* NULL = unmasked */
    double abort_singular_value_threshold; /* > 0: stop once the running Rayleigh lower
                                              bound reaches it; 0 = never */
} sv_estimator_opts_t;

typedef enum
{
    SV_ESTIMATOR_CONVERGED = 0,  /* residual test passed */
    SV_ESTIMATOR_ABORTED = 1,    /* abort threshold crossed; the estimate is a lower bound */
    SV_ESTIMATOR_MAX_ITER = 2,   /* max_iterations exhausted without convergence */
    SV_ESTIMATOR_DEGENERATE = 3, /* zero or non-finite iterate; max_singular_value is 0.0 */
} sv_estimator_status_t;

typedef struct
{
    sv_estimator_status_t status;
    double max_singular_value;
    int iterations; /* power iterations actually run */
} sv_estimator_result_t;

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
    double base_step_size;
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
    double *infeasibility_dual_scratch; /* n-vector scratch for the ray certificate; dual_slack must survive */
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
    termination_reason_t termination_reason;

    double *delta_primal_solution;
    double *delta_dual_solution;
    double fixed_point_error;
    double initial_fixed_point_error;
    double last_trial_fixed_point_error;
    int inner_count;
    int restart_count;               /* restarts performed, ASB reverts included */
    const char *last_restart_reason; /* criterion that fired for the most recent adaptive restart */
    int logged_restart_count;        /* restart_count at the last printed iteration row */
    int *d_inner_count;

    /* active-set step controller */
    sv_estimator_ctx_t *asb_sv_ctx; /* cached power-iteration workspace + SpMV plans */
    int asb_phase;
    double asb_sv;
    double asb_step_ceiling; /* post-divergence cap on the target step (INFINITY = none) */
    double asb_anchor_primal_weight;
    /* primal-weight PID state captured with the anchor; a revert restores it too */
    double asb_anchor_pw_error_sum;
    double asb_anchor_pw_last_error;
    double asb_anchor_best_pw;
    double asb_anchor_best_pd_residual_gap;
    int asb_revert_count;
    int asb_raise_count;
    int asb_pi_early_exit_count;
    int asb_pi_iterations;
    int asb_sv_failed_count;
    bool asb_no_raise_certified;     /* a NO_RAISE abort covers the unchanged union */
    long asb_changes_since_estimate; /* mask entries added or removed since the last sv estimate */
    long asb_free_variables;         /* variables in the mask: not confidently clamped within the window */
    long asb_binding_constraints;    /* constraints in the mask: binding within the window */
    int *d_asb_var_last_free;
    int *d_asb_row_last_binding;
    bool *d_asb_col_mask;
    bool *d_asb_row_mask;
    double *d_asb_primal_anchor;
    double *d_asb_dual_anchor;
    double *d_asb_dual_slack_anchor;
    double *d_asb_dual_projection_input;
    int *d_asb_count;

    cusparseHandle_t sparse_handle;
    cublasHandle_t blas_handle;
    void *spmv_ctx;

    double *ones_primal_d;
    double *ones_dual_d;

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
