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

#include <stdlib.h>
#include <stdbool.h>
#include <cusparse.h>
#include <cublas_v2.h>

typedef struct
{
	int num_rows;
	int num_cols;
	int num_nonzeros;
	int *row_ptr;
	int *col_ind;
	double *val;
} cu_sparse_matrix_csr_t;

typedef struct
{
	int num_variables;
	int num_constraints;
	double *variable_lower_bound;
	double *variable_upper_bound;
	double *objective_vector;
	double objective_constant;

	int *constraint_matrix_row_pointers;
	int *constraint_matrix_col_indices;
	double *constraint_matrix_values;
	int constraint_matrix_num_nonzeros;

	double *constraint_lower_bound;
	double *constraint_upper_bound;
} lp_problem_t;

typedef enum
{
	TERMINATION_REASON_UNSPECIFIED,
	TERMINATION_REASON_OPTIMAL,
	TERMINATION_REASON_PRIMAL_INFEASIBLE,
	TERMINATION_REASON_DUAL_INFEASIBLE,
	TERMINATION_REASON_TIME_LIMIT,
	TERMINATION_REASON_ITERATION_LIMIT
} termination_reason_t;

typedef struct
{
	lp_problem_t *scaled_problem;
	double *con_rescale;
	double *var_rescale;
	double con_bound_rescale;
	double obj_vec_rescale;
	double rescaling_time_sec;
} rescale_info_t;

typedef struct
{
	double artificial_restart_threshold;
	double sufficient_reduction_for_restart;
	double necessary_reduction_for_restart;
	double k_p;
	double k_i;
	double k_d;
	double i_smooth;
} restart_parameters_t;

typedef struct
{
	double eps_optimal_relative;
	double eps_feasible_relative;
	double eps_infeasible;
	double time_sec_limit;
	int iteration_limit;
} termination_criteria_t;

typedef struct
{
	int l_inf_ruiz_iterations;
	bool has_pock_chambolle_alpha;
	double pock_chambolle_alpha;
	bool bound_objective_rescaling;
	bool verbose;
	int termination_evaluation_frequency;
	termination_criteria_t termination_criteria;
	restart_parameters_t restart_params;
	double reflection_coefficient;
} pdhg_parameters_t;

typedef struct
{
	int num_variables;
	int num_constraints;
	double *variable_lower_bound;
	double *variable_upper_bound;
	double *objective_vector;
	double objective_constant;
	cu_sparse_matrix_csr_t *constraint_matrix;
	cu_sparse_matrix_csr_t *constraint_matrix_t;
	double *constraint_lower_bound;
	double *constraint_upper_bound;
	int num_blocks_primal;
	int num_blocks_dual;
	int num_blocks_primal_dual;
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

	cusparseHandle_t sparse_handle;
	cublasHandle_t blas_handle;
	size_t spmv_buffer_size;
	size_t primal_spmv_buffer_size;
	size_t dual_spmv_buffer_size;
	void *primal_spmv_buffer;
	void *dual_spmv_buffer;
	void *spmv_buffer;

	cusparseSpMatDescr_t matA;
	cusparseSpMatDescr_t matAt;
	cusparseDnVecDescr_t vec_primal_sol;
	cusparseDnVecDescr_t vec_dual_sol;
	cusparseDnVecDescr_t vec_primal_prod;
	cusparseDnVecDescr_t vec_dual_prod;

	double *ones_primal_d;
	double *ones_dual_d;
} pdhg_solver_state_t;

typedef struct
{
	int num_variables;
	int num_constraints;

	double *primal_solution;
	double *dual_solution;
	
	int total_count;
	double rescaling_time_sec;
	double cumulative_time_sec;

	double absolute_primal_residual;
	double relative_primal_residual;
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

} cupdlpx_result_t;

// matrix formats
typedef enum {
    matrix_dense = 0,
    matrix_csr   = 1,
    matrix_csc   = 2,
    matrix_coo   = 3
} matrix_format_t;

// matrix descriptor
typedef struct {
    int m; // num_constraints
    int n; // num_variables
    matrix_format_t fmt;

    // treat abs(x) < zero_tolerance as zero
    double zero_tolerance;

    union {
        struct { // Dense (row-major)
            const double* A; // m*n
        } dense;

        struct { // CSR
            int nnz;
            const int*    row_ptr; 
            const int*    col_ind;
            const double* vals;
        } csr;

        struct { // CSC
            int nnz;
            const int*    col_ptr;
            const int*    row_ind;
            const double* vals;
        } csc;

        struct { // COO
            int nnz;
            const int*    row_ind;
            const int*    col_ind; 
            const double* vals; 
        } coo;
    } data;
} matrix_desc_t;
