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

#include "cupdlpx.h"
#include "solver.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// create an lp_problem_t from a matrix
lp_problem_t *create_lp_problem(const double *objective_c,
                                const matrix_desc_t *A_desc,
                                const double *con_lb, const double *con_ub,
                                const double *var_lb, const double *var_ub,
                                const double *objective_constant)
{
    lp_problem_t *prob = (lp_problem_t *)safe_malloc(sizeof(lp_problem_t));
    prob->primal_start = NULL;
    prob->dual_start = NULL;

    prob->num_variables = A_desc->n;
    prob->num_constraints = A_desc->m;

    // handle matrix by format
    switch (A_desc->fmt)
    {
    case matrix_dense:
        dense_to_csr(A_desc, &prob->constraint_matrix_row_pointers,
                     &prob->constraint_matrix_col_indices,
                     &prob->constraint_matrix_values,
                     &prob->constraint_matrix_num_nonzeros);
        break;

    case matrix_csc:
    {
        int *row_ptr = NULL, *col_ind = NULL;
        double *vals = NULL;
        int nnz = 0;
        if (csc_to_csr(A_desc, &row_ptr, &col_ind, &vals, &nnz) != 0)
        {
            fprintf(stderr, "[interface] CSC->CSR failed.\n");
            free(prob);
            return NULL;
        }
        prob->constraint_matrix_num_nonzeros = nnz;
        prob->constraint_matrix_row_pointers = row_ptr;
        prob->constraint_matrix_col_indices = col_ind;
        prob->constraint_matrix_values = vals;
        break;
    }

    case matrix_coo:
    {
        int *row_ptr = NULL, *col_ind = NULL;
        double *vals = NULL;
        int nnz = 0;
        if (coo_to_csr(A_desc, &row_ptr, &col_ind, &vals, &nnz) != 0)
        {
            fprintf(stderr, "[interface] COO->CSR failed.\n");
            free(prob);
            return NULL;
        }
        prob->constraint_matrix_num_nonzeros = nnz;
        prob->constraint_matrix_row_pointers = row_ptr;
        prob->constraint_matrix_col_indices = col_ind;
        prob->constraint_matrix_values = vals;
        break;
    }

    case matrix_csr:
        prob->constraint_matrix_num_nonzeros = A_desc->data.csr.nnz;
        prob->constraint_matrix_row_pointers =
            (int *)safe_malloc((size_t)(A_desc->m + 1) * sizeof(int));
        prob->constraint_matrix_col_indices =
            (int *)safe_malloc((size_t)A_desc->data.csr.nnz * sizeof(int));
        prob->constraint_matrix_values =
            (double *)safe_malloc((size_t)A_desc->data.csr.nnz * sizeof(double));
        memcpy(prob->constraint_matrix_row_pointers, A_desc->data.csr.row_ptr,
               (size_t)(A_desc->m + 1) * sizeof(int));
        memcpy(prob->constraint_matrix_col_indices, A_desc->data.csr.col_ind,
               (size_t)A_desc->data.csr.nnz * sizeof(int));
        memcpy(prob->constraint_matrix_values, A_desc->data.csr.vals,
               (size_t)A_desc->data.csr.nnz * sizeof(double));
        break;

    default:
        fprintf(
            stderr,
            "[interface] make_problem_from_matrix: unsupported matrix format %d.\n",
            A_desc->fmt);
        free(prob);
        return NULL;
    }

    // default fill values
    prob->objective_constant = objective_constant ? *objective_constant : 0.0;
    fill_or_copy(&prob->objective_vector, prob->num_variables, objective_c, 0.0);
    fill_or_copy(&prob->variable_lower_bound, prob->num_variables, var_lb, 0.0);
    fill_or_copy(&prob->variable_upper_bound, prob->num_variables, var_ub,
                 INFINITY);
    fill_or_copy(&prob->constraint_lower_bound, prob->num_constraints, con_lb,
                 -INFINITY);
    fill_or_copy(&prob->constraint_upper_bound, prob->num_constraints, con_ub,
                 INFINITY);

    return prob;
}

void cupdlpx_result_free(cupdlpx_result_t *results)
{
    if (results == NULL)
    {
        return;
    }

    free(results->primal_solution);
    free(results->dual_solution);
    free(results);
}

void lp_problem_free(lp_problem_t *prob)
{
    if (!prob)
        return;
    free(prob->constraint_matrix_row_pointers);
    free(prob->constraint_matrix_col_indices);
    free(prob->constraint_matrix_values);
    free(prob->variable_lower_bound);
    free(prob->variable_upper_bound);
    free(prob->objective_vector);
    free(prob->constraint_lower_bound);
    free(prob->constraint_upper_bound);
    free(prob->primal_start);
    free(prob->dual_start);
    memset(prob, 0, sizeof(*prob));
    free(prob);
}

void set_start_values(lp_problem_t *prob, const double *primal,
                      const double *dual)
{
    if (!prob)
        return;

    int n = prob->num_variables;
    int m = prob->num_constraints;

    // Free previous if any
    if (prob->primal_start)
    {
        free(prob->primal_start);
        prob->primal_start = NULL;
    }
    if (prob->dual_start)
    {
        free(prob->dual_start);
        prob->dual_start = NULL;
    }

    if (primal)
    {
        prob->primal_start = (double *)safe_malloc(n * sizeof(double));
        memcpy(prob->primal_start, primal, n * sizeof(double));
    }
    if (dual)
    {
        prob->dual_start = (double *)safe_malloc(m * sizeof(double));
        memcpy(prob->dual_start, dual, m * sizeof(double));
    }
}

cupdlpx_result_t *solve_lp_problem(const lp_problem_t *prob,
                                   const pdhg_parameters_t *params)
{
    // argument checks
    if (!prob)
    {
        fprintf(stderr, "[interface] solve_lp_problem: invalid arguments.\n");
        return NULL;
    }

    // prepare parameters: use defaults if not provided
    pdhg_parameters_t local_params;
    if (params)
    {
        local_params = *params;
    }
    else
    {
        set_default_parameters(&local_params);
    }

    // call optimizer
    cupdlpx_result_t *res = optimize(&local_params, prob);
    if (!res)
    {
        fprintf(stderr, "[interface] optimize returned NULL.\n");
        return NULL;
    }

    return res;
}
