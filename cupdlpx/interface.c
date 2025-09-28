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

    // allocate
    *row_ptr = (int*)safe_malloc((size_t)(m + 1) * sizeof(int));
    *col_ind = (int*)safe_malloc((size_t)nnz * sizeof(int));
    *vals    = (double*)safe_malloc((size_t)nnz * sizeof(double));

    // fill
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

// convert CSC → CSR
static int csc_to_csr(const matrix_desc_t* desc,
                      int** row_ptr, int** col_ind, double** vals, int* nnz_out) {
    const int m = desc->m, n = desc->n;
    const int *col_ptr = desc->data.csc.col_ptr;
    const int *row_ind = desc->data.csc.row_ind; 
    const double *v    = desc->data.csc.vals;

    const double tol = (desc->zero_tolerance > 0) ? desc->zero_tolerance : 0.0;

    // count entries per row
    *row_ptr = (int*)safe_malloc((size_t)(m + 1) * sizeof(int));
    for (int i = 0; i <= m; ++i) (*row_ptr)[i] = 0;

    // count nnz
    int eff_nnz = 0;
    for (int j = 0; j < n; ++j) {
        for (int k = col_ptr[j]; k < col_ptr[j + 1]; ++k) {
            int ri = row_ind[k];
            if (ri < 0 || ri >= m) { fprintf(stderr, "[interface] CSC: row index out of range\n"); return -1; }
            double val = v[k];
            if (tol > 0 && fabs(val) <= tol) continue;
            ++((*row_ptr)[ri + 1]);
            ++eff_nnz;
        }
    }

    // exclusive scan
    for (int i = 0; i < m; ++i) (*row_ptr)[i + 1] += (*row_ptr)[i];

    // allocate
    *col_ind = (int*)safe_malloc((size_t)eff_nnz * sizeof(int));
    *vals    = (double*)safe_malloc((size_t)eff_nnz * sizeof(double));

    // next position to fill in each row
    int *next = (int*)safe_malloc((size_t)m * sizeof(int));
    for (int i = 0; i < m; ++i) next[i] = (*row_ptr)[i];

    // fill column indices and values
    for (int j = 0; j < n; ++j) {
        for (int k = col_ptr[j]; k < col_ptr[j + 1]; ++k) {
            int ri = row_ind[k];
            double val = v[k];
            if (tol > 0 && fabs(val) <= tol) continue;
            int pos = next[ri]++;
            (*col_ind)[pos] = j;
            (*vals)[pos]    = val;
        }
    }

    free(next);
    *nnz_out = eff_nnz;
    return 0;
}

// convert COO → CSR
static int coo_to_csr(const matrix_desc_t* desc,
                      int** row_ptr, int** col_ind, double** vals, int* nnz_out) {
    const int m = desc->m, n = desc->n;
    const int nnz_in = desc->data.coo.nnz;
    const int *r = desc->data.coo.row_ind;
    const int *c = desc->data.coo.col_ind;
    const double *v = desc->data.coo.vals;
    const double tol = (desc->zero_tolerance > 0) ? desc->zero_tolerance : 0.0;

    // count nnz
    int nnz = 0;
    if (tol > 0) {
        for (int k = 0; k < nnz_in; ++k)
            if (fabs(v[k]) > tol) ++nnz;
    } else {
        nnz = nnz_in;
    }

    *row_ptr = (int*)safe_malloc((size_t)(m + 1) * sizeof(int));
    *col_ind = (int*)safe_malloc((size_t)nnz * sizeof(int));
    *vals    = (double*)safe_malloc((size_t)nnz * sizeof(double));

    // count entries per row
    for (int i = 0; i <= m; ++i) (*row_ptr)[i] = 0;
    if (tol > 0) {
        for (int k = 0; k < nnz_in; ++k)
            if (fabs(v[k]) > tol) {
                int ri = r[k];
                if (ri < 0 || ri >= m) { fprintf(stderr, "[interface] COO: row index out of range\n"); return -1; }
                ++((*row_ptr)[ri + 1]);
            }
    } else {
        for (int k = 0; k < nnz_in; ++k) {
            int ri = r[k];
            if (ri < 0 || ri >= m) { fprintf(stderr, "[interface] COO: row index out of range\n"); return -1; }
            ++((*row_ptr)[ri + 1]);
        }
    }

    // exclusive scan
    for (int i = 0; i < m; ++i) (*row_ptr)[i + 1] += (*row_ptr)[i];

    // next position to fill in each row
    int *next = (int*)safe_malloc((size_t)m * sizeof(int));
    for (int i = 0; i < m; ++i) next[i] = (*row_ptr)[i];

    // fill column indices and values
    if (tol > 0) {
        for (int k = 0; k < nnz_in; ++k) {
            if (fabs(v[k]) <= tol) continue;
            int ri = r[k], cj = c[k];
            if (cj < 0 || cj >= n) { fprintf(stderr, "[interface] COO: col index out of range\n"); free(next); return -1; }
            int pos = next[ri]++;
            (*col_ind)[pos] = cj;
            (*vals)[pos]    = v[k];
        }
    } else {
        for (int k = 0; k < nnz_in; ++k) {
            int ri = r[k], cj = c[k];
            if (cj < 0 || cj >= n) { fprintf(stderr, "[interface] COO: col index out of range\n"); free(next); return -1; }
            int pos = next[ri]++;
            (*col_ind)[pos] = cj;
            (*vals)[pos]    = v[k];
        }
    }

    free(next);
    *nnz_out = nnz;
    return 0;
}

// create an lp_problem_t from a matrix
lp_problem_t* create_lp_problem(
    const matrix_desc_t* A_desc,
    const double* objective_c,
    const double* objective_constant,
    const double* var_lb,
    const double* var_ub,
    const double* con_lb,
    const double* con_ub
) {
    lp_problem_t* prob = (lp_problem_t*)safe_malloc(sizeof(lp_problem_t));
    prob->initial_primal = NULL;
    prob->initial_dual = NULL;

    prob->num_variables   = A_desc->n;
    prob->num_constraints = A_desc->m;

    // handle matrix by format
    switch (A_desc->fmt) {
        case matrix_dense:
            dense_to_csr(A_desc,
                         &prob->constraint_matrix_row_pointers,
                         &prob->constraint_matrix_col_indices,
                         &prob->constraint_matrix_values,
                         &prob->constraint_matrix_num_nonzeros);
            break;

        case matrix_csc: {
            int *row_ptr=NULL, *col_ind=NULL; double *vals=NULL; int nnz=0;
            if (csc_to_csr(A_desc, &row_ptr, &col_ind, &vals, &nnz) != 0) {
                fprintf(stderr, "[interface] CSC->CSR failed.\n");
                free(prob);
                return NULL;
            }
            prob->constraint_matrix_num_nonzeros = nnz;
            prob->constraint_matrix_row_pointers = row_ptr;
            prob->constraint_matrix_col_indices  = col_ind;
            prob->constraint_matrix_values       = vals;
            break;
        }

        case matrix_coo: {
            int *row_ptr=NULL, *col_ind=NULL; double *vals=NULL; int nnz=0;
            if (coo_to_csr(A_desc, &row_ptr, &col_ind, &vals, &nnz) != 0) {
                fprintf(stderr, "[interface] COO->CSR failed.\n");
                free(prob);
                return NULL;
            }
            prob->constraint_matrix_num_nonzeros = nnz;
            prob->constraint_matrix_row_pointers = row_ptr;
            prob->constraint_matrix_col_indices  = col_ind;
            prob->constraint_matrix_values       = vals;
           break;
        }

        case matrix_csr:
            prob->constraint_matrix_num_nonzeros = A_desc->data.csr.nnz;
            prob->constraint_matrix_row_pointers = (int*)safe_malloc((size_t)(A_desc->m + 1) * sizeof(int));
            prob->constraint_matrix_col_indices = (int*)safe_malloc((size_t)A_desc->data.csr.nnz * sizeof(int));
            prob->constraint_matrix_values = (double*)safe_malloc((size_t)A_desc->data.csr.nnz * sizeof(double));
            memcpy(prob->constraint_matrix_row_pointers, A_desc->data.csr.row_ptr, (size_t)(A_desc->m + 1) * sizeof(int));
            memcpy(prob->constraint_matrix_col_indices, A_desc->data.csr.col_ind, (size_t)A_desc->data.csr.nnz * sizeof(int));
            memcpy(prob->constraint_matrix_values, A_desc->data.csr.vals, (size_t)A_desc->data.csr.nnz * sizeof(double));
            break;
            
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

void lp_problem_set_initial_solution(lp_problem_t* prob, const double* primal, const double* dual)
{
    if (!prob) return;

    int n = prob->num_variables;
    int m = prob->num_constraints;

    // Free previous if any
    if (prob->initial_primal) { free(prob->initial_primal); prob->initial_primal = NULL; }
    if (prob->initial_dual)   { free(prob->initial_dual);   prob->initial_dual = NULL; }

    if (primal) {
        prob->initial_primal = (double*)safe_malloc(n * sizeof(double));
        memcpy(prob->initial_primal, primal, n * sizeof(double));
    }
    if (dual) {
        prob->initial_dual = (double*)safe_malloc(m * sizeof(double));
        memcpy(prob->initial_dual, dual, m * sizeof(double));
    }
}

cupdlpx_result_t* solve_lp_problem(
    const lp_problem_t* prob,
    const pdhg_parameters_t* params
) {
    // argument checks
    if (!prob) {
        fprintf(stderr, "[interface] solve_lp_problem: invalid arguments.\n");
        return NULL;
    }
    
    // prepare parameters: use defaults if not provided 
    pdhg_parameters_t local_params;
    if (params) {
        local_params = *params;
    } else {
        set_default_parameters(&local_params);
    }

    // call optimizer
    cupdlpx_result_t* res = optimize(&local_params, prob);
    if (!res) {
        fprintf(stderr, "[interface] optimize returned NULL.\n");
        return NULL;
    }
    
    return res;
}
