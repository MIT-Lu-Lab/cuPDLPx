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

#include "interface.h"
#include "solver.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// helper function to allocate and fill or copy an array
static void fill_or_copy(double** dst, int n, const double* src, double fill_val) {
    *dst = (double*)safe_malloc((size_t)n * sizeof(double));
    if (src) memcpy(*dst, src, (size_t)n * sizeof(double));
    else     for (int i = 0; i < n; ++i) (*dst)[i] = fill_val;
}

// convert dense → CSR
static void dense_to_csr(const matrix_desc_t* desc,
                         int** row_ptr, int** col_ind, double** vals, int* nnz_out) {
    int m = desc->m, n = desc->n;
    double tol = (desc->zero_tolerance > 0) ? desc->zero_tolerance : 1e-12;

    // count nnz
    int nnz = 0;
    for (int i = 0; i < m * n; ++i) {
        if (fabs(desc->data.dense.A[i]) > tol) ++nnz;
    }

    *row_ptr = (int*)safe_malloc((size_t)(m + 1) * sizeof(int));
    *col_ind = (int*)safe_malloc((size_t)nnz * sizeof(int));
    *vals    = (double*)safe_malloc((size_t)nnz * sizeof(double));

    int nz = 0;
    for (int i = 0; i < m; ++i) {
        (*row_ptr)[i] = nz;
        for (int j = 0; j < n; ++j) {
            double v = desc->data.dense.A[i * n + j];
            if (fabs(v) > tol) {
                (*col_ind)[nz] = j;
                (*vals)[nz] = v;
                ++nz;
            }
        }
    }
    (*row_ptr)[m] = nz;
    *nnz_out = nz;
}

// create an lp_problem_t from a matrix
lp_problem_t* make_problem_from_matrix(
    const matrix_desc_t* A_desc,
    const double* objective_c,
    const double* objective_constant,
    const double* var_lb,
    const double* var_ub,
    const double* con_lb,
    const double* con_ub
) {
    lp_problem_t* prob = (lp_problem_t*)safe_malloc(sizeof(lp_problem_t));

    prob->num_variables   = A_desc->n;
    prob->num_constraints = A_desc->m;

    // handle matrix by format → convert to CSR if needed
    switch (A_desc->fmt) {
        case matrix_dense:
            dense_to_csr(A_desc,
                         &prob->constraint_matrix_row_pointers,
                         &prob->constraint_matrix_col_indices,
                         &prob->constraint_matrix_values,
                         &prob->constraint_matrix_num_nonzeros);
            break;

        case matrix_csr:
            prob->constraint_matrix_num_nonzeros = A_desc->data.csr.nnz;
            prob->constraint_matrix_row_pointers = (int*)safe_malloc((size_t)(A_desc->m + 1) * sizeof(int));
            prob->constraint_matrix_col_indices = (int*)safe_malloc((size_t)A_desc->data.csr.nnz * sizeof(int));
            prob->constraint_matrix_values = (double*)safe_malloc((size_t)A_desc->data.csr.nnz * sizeof(double));
            memcpy(prob->constraint_matrix_row_pointers, A_desc->data.csr.row_ptr, (size_t)(A_desc->m + 1) * sizeof(int));
            memcpy(prob->constraint_matrix_col_indices, A_desc->data.csr.col_ind, (size_t)A_desc->data.csr.nnz * sizeof(int));
            memcpy(prob->constraint_matrix_values, A_desc->data.csr.vals, (size_t)A_desc->data.csr.nnz * sizeof(double));
            break;

        // TODO: other formats
        default:
            fprintf(stderr, "[interface] make_problem_from_matrix: unsupported matrix format %d.\n", A_desc->fmt);
            free(prob);
            return NULL;
    }

    // default fill values
    prob->objective_constant = objective_constant ? *objective_constant : 0.0;
    fill_or_copy(&prob->objective_vector, prob->num_variables, objective_c, 0.0);
    fill_or_copy(&prob->variable_lower_bound, prob->num_variables, var_lb, 0.0);
    fill_or_copy(&prob->variable_upper_bound, prob->num_variables, var_ub, INFINITY);
    fill_or_copy(&prob->constraint_lower_bound, prob->num_constraints, con_lb, -INFINITY);
    fill_or_copy(&prob->constraint_upper_bound, prob->num_constraints, con_ub, INFINITY);

    return prob;
}

lp_solution_t solve_lp_problem(
    const lp_problem_t* prob,
    const pdhg_parameters_t* params
) {
    // initialize output
    lp_solution_t out = {0};

    // argument checks
    if (!prob) {
        fprintf(stderr, "[interface] solve_lp_problem: invalid arguments.\n");
        return out;
    }

    // prepare parameters: use defaults if not provided 
    pdhg_parameters_t local_params;
    if (params) {
        local_params = *params;
    } else {
        set_default_parameters(&local_params);
    }

    // call optimizer
    pdhg_solver_state_t* state = optimize(&local_params, prob);
    if (!state) {
        fprintf(stderr, "[interface] optimize returned NULL.\n");
        return out;
    }

    // prepare output
    out.n = prob->num_variables;
    out.m = prob->num_constraints;
    out.x = (double*)safe_malloc((size_t)out.n * sizeof(double));
    out.y = (double*)safe_malloc((size_t)out.m * sizeof(double));

    // copy solution from GPU to CPU
    cudaError_t e1 = cudaMemcpy(out.x, state->pdhg_primal_solution,
                                (size_t)out.n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaError_t e2 = cudaMemcpy(out.y, state->pdhg_dual_solution,
                                (size_t)out.m * sizeof(double), cudaMemcpyDeviceToHost);
    if (e1 != cudaSuccess || e2 != cudaSuccess) {
        fprintf(stderr, "[interface] cudaMemcpy failed: %s / %s\n",
                cudaGetErrorName(e1), cudaGetErrorName(e2));
        free(out.x); out.x = NULL;
        free(out.y); out.y = NULL;
        pdhg_solver_state_free(state);
        return out;
    }

    out.primal_obj = state->primal_objective_value;
    out.dual_obj = state->dual_objective_value;
    out.reason = state->termination_reason;

    pdhg_solver_state_free(state);
    return out;
}

// free lp solution
void lp_solution_free(lp_solution_t* sol) {
    if (!sol) return;
    if (sol->x) free(sol->x);
    if (sol->y) free(sol->y);
    sol->n = 0;
    sol->m = 0;
    sol->primal_obj = 0.0;
    sol->dual_obj   = 0.0;
    sol->reason = TERMINATION_REASON_UNSPECIFIED;
}