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

#include <stdbool.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C"
{
#endif

	typedef enum
	{
		TERMINATION_REASON_UNSPECIFIED,
		TERMINATION_REASON_OPTIMAL,
		TERMINATION_REASON_PRIMAL_INFEASIBLE,
		TERMINATION_REASON_DUAL_INFEASIBLE,
		TERMINATION_REASON_TIME_LIMIT,
		TERMINATION_REASON_ITERATION_LIMIT,
		TERMINATION_REASON_FEAS_POLISH_SUCCESS
	} termination_reason_t;

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

		double *primal_start;
		double *dual_start;
	} lp_problem_t;

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
		double eps_feas_polish_relative;
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
		bool feasibility_polishing;
	} pdhg_parameters_t;

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
		double feasibility_polishing_time;
		int feasibility_iteration;
	} cupdlpx_result_t;

	// matrix formats
	typedef enum
	{
		matrix_dense = 0,
		matrix_csr = 1,
		matrix_csc = 2,
		matrix_coo = 3
	} matrix_format_t;

	// matrix descriptor
	typedef struct
	{
		int m; // num_constraints
		int n; // num_variables
		matrix_format_t fmt;

		// treat abs(x) < zero_tolerance as zero
		double zero_tolerance;

		union
		{
			struct
			{					 // Dense (row-major)
				const double *A; // m*n
			} dense;

			struct
			{ // CSR
				int nnz;
				const int *row_ptr;
				const int *col_ind;
				const double *vals;
			} csr;

			struct
			{ // CSC
				int nnz;
				const int *col_ptr;
				const int *row_ind;
				const double *vals;
			} csc;

			struct
			{ // COO
				int nnz;
				const int *row_ind;
				const int *col_ind;
				const double *vals;
			} coo;
		} data;
	} matrix_desc_t;

#ifdef __cplusplus
} // extern "C"
#endif