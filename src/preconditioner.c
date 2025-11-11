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

#include "preconditioner.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define SCALING_EPSILON 1e-12

static lp_problem_t *deepcopy_problem(const lp_problem_t *prob);
static void scale_problem(lp_problem_t *problem, const double *con_rescale,
                          const double *var_rescale);
static void ruiz_rescaling(lp_problem_t *problem, int num_iters,
                           double *cum_con_rescale, double *cum_var_rescale);
static void pock_chambolle_rescaling(lp_problem_t *problem, double alpha,
                                     double *cum_con_rescale,
                                     double *cum_var_rescale);

static lp_problem_t *deepcopy_problem(const lp_problem_t *prob)
{
    lp_problem_t *new_prob = (lp_problem_t *)safe_malloc(sizeof(lp_problem_t));

    new_prob->num_variables = prob->num_variables;
    new_prob->num_constraints = prob->num_constraints;
    new_prob->constraint_matrix_num_nonzeros =
        prob->constraint_matrix_num_nonzeros;
    new_prob->objective_constant = prob->objective_constant;

    size_t var_bytes = prob->num_variables * sizeof(double);
    size_t con_bytes = prob->num_constraints * sizeof(double);
    size_t nnz_bytes_val = prob->constraint_matrix_num_nonzeros * sizeof(double);
    size_t nnz_bytes_col = prob->constraint_matrix_num_nonzeros * sizeof(int);
    size_t row_ptr_bytes = (prob->num_constraints + 1) * sizeof(int);

    new_prob->variable_lower_bound = safe_malloc(var_bytes);
    new_prob->variable_upper_bound = safe_malloc(var_bytes);
    new_prob->objective_vector = safe_malloc(var_bytes);
    new_prob->constraint_lower_bound = safe_malloc(con_bytes);
    new_prob->constraint_upper_bound = safe_malloc(con_bytes);
    new_prob->constraint_matrix_row_pointers = safe_malloc(row_ptr_bytes);
    new_prob->constraint_matrix_col_indices = safe_malloc(nnz_bytes_col);
    new_prob->constraint_matrix_values = safe_malloc(nnz_bytes_val);

    memcpy(new_prob->variable_lower_bound, prob->variable_lower_bound, var_bytes);
    memcpy(new_prob->variable_upper_bound, prob->variable_upper_bound, var_bytes);
    memcpy(new_prob->objective_vector, prob->objective_vector, var_bytes);
    memcpy(new_prob->constraint_lower_bound, prob->constraint_lower_bound,
           con_bytes);
    memcpy(new_prob->constraint_upper_bound, prob->constraint_upper_bound,
           con_bytes);
    memcpy(new_prob->constraint_matrix_row_pointers,
           prob->constraint_matrix_row_pointers, row_ptr_bytes);
    memcpy(new_prob->constraint_matrix_col_indices,
           prob->constraint_matrix_col_indices, nnz_bytes_col);
    memcpy(new_prob->constraint_matrix_values, prob->constraint_matrix_values,
           nnz_bytes_val);

    if (prob->primal_start)
    {
        new_prob->primal_start = safe_malloc(var_bytes);
        memcpy(new_prob->primal_start, prob->primal_start, var_bytes);
    }
    else
    {
        new_prob->primal_start = NULL;
    }
    if (prob->dual_start)
    {
        new_prob->dual_start = safe_malloc(con_bytes);
        memcpy(new_prob->dual_start, prob->dual_start, con_bytes);
    }
    else
    {
        new_prob->dual_start = NULL;
    }

    return new_prob;
}

static void scale_problem(lp_problem_t *problem,
                          const double *constraint_rescaling,
                          const double *variable_rescaling)
{
    for (int i = 0; i < problem->num_variables; ++i)
    {
        problem->objective_vector[i] /= variable_rescaling[i];
        problem->variable_upper_bound[i] *= variable_rescaling[i];
        problem->variable_lower_bound[i] *= variable_rescaling[i];
    }
    for (int i = 0; i < problem->num_constraints; ++i)
    {
        problem->constraint_lower_bound[i] /= constraint_rescaling[i];
        problem->constraint_upper_bound[i] /= constraint_rescaling[i];
    }

    for (int row = 0; row < problem->num_constraints; ++row)
    {
        for (int nz_idx = problem->constraint_matrix_row_pointers[row];
             nz_idx < problem->constraint_matrix_row_pointers[row + 1]; ++nz_idx)
        {
            int col = problem->constraint_matrix_col_indices[nz_idx];
            problem->constraint_matrix_values[nz_idx] /=
                (constraint_rescaling[row] * variable_rescaling[col]);
        }
    }
}

static void ruiz_rescaling(lp_problem_t *problem, int num_iterations,
                           double *cum_constraint_rescaling,
                           double *cum_variable_rescaling)
{
    int num_cons = problem->num_constraints;
    int num_vars = problem->num_variables;
    double *con_rescale = safe_malloc(num_cons * sizeof(double));
    double *var_rescale = safe_malloc(num_vars * sizeof(double));

    for (int iter = 0; iter < num_iterations; ++iter)
    {
        for (int i = 0; i < num_vars; ++i)
            var_rescale[i] = 0.0;
        for (int i = 0; i < num_cons; ++i)
            con_rescale[i] = 0.0;

        for (int row = 0; row < num_cons; ++row)
        {
            for (int nz_idx = problem->constraint_matrix_row_pointers[row];
                 nz_idx < problem->constraint_matrix_row_pointers[row + 1];
                 ++nz_idx)
            {
                int col = problem->constraint_matrix_col_indices[nz_idx];
                if (col < 0 || col >= num_vars)
                {
                    fprintf(stderr,
                            "Error: Invalid column index %d at nz_idx %d for row %d. "
                            "Must be in [0, %d).\n",
                            col, nz_idx, row, num_vars);
                }
                double val = fabs(problem->constraint_matrix_values[nz_idx]);
                if (val > var_rescale[col])
                    var_rescale[col] = val;
                if (val > con_rescale[row])
                    con_rescale[row] = val;
            }
        }

        for (int i = 0; i < num_vars; ++i)
            var_rescale[i] =
                (var_rescale[i] < SCALING_EPSILON) ? 1.0 : sqrt(var_rescale[i]);
        for (int i = 0; i < num_cons; ++i)
            con_rescale[i] =
                (con_rescale[i] < SCALING_EPSILON) ? 1.0 : sqrt(con_rescale[i]);

        scale_problem(problem, con_rescale, var_rescale);
        for (int i = 0; i < num_vars; ++i)
            cum_variable_rescaling[i] *= var_rescale[i];
        for (int i = 0; i < num_cons; ++i)
            cum_constraint_rescaling[i] *= con_rescale[i];
    }
    free(con_rescale);
    free(var_rescale);
}

static void pock_chambolle_rescaling(lp_problem_t *problem, double alpha,
                                     double *cum_constraint_rescaling,
                                     double *cum_variable_rescaling)
{
    int num_cons = problem->num_constraints;
    int num_vars = problem->num_variables;
    double *con_rescale = safe_calloc(num_cons, sizeof(double));
    double *var_rescale = safe_calloc(num_vars, sizeof(double));

    for (int row = 0; row < num_cons; ++row)
    {
        for (int nz_idx = problem->constraint_matrix_row_pointers[row];
             nz_idx < problem->constraint_matrix_row_pointers[row + 1]; ++nz_idx)
        {
            int col = problem->constraint_matrix_col_indices[nz_idx];
            double val = fabs(problem->constraint_matrix_values[nz_idx]);
            var_rescale[col] += pow(val, 2.0 - alpha);
            con_rescale[row] += pow(val, alpha);
        }
    }

    for (int i = 0; i < num_vars; ++i)
        var_rescale[i] =
            (var_rescale[i] < SCALING_EPSILON) ? 1.0 : sqrt(var_rescale[i]);
    for (int i = 0; i < num_cons; ++i)
        con_rescale[i] =
            (con_rescale[i] < SCALING_EPSILON) ? 1.0 : sqrt(con_rescale[i]);

    scale_problem(problem, con_rescale, var_rescale);
    for (int i = 0; i < num_vars; ++i)
        cum_variable_rescaling[i] *= var_rescale[i];
    for (int i = 0; i < num_cons; ++i)
        cum_constraint_rescaling[i] *= con_rescale[i];

    free(con_rescale);
    free(var_rescale);
}

rescale_info_t *rescale_problem(const pdhg_parameters_t *params,
                                const lp_problem_t *original_problem)
{
    clock_t start_rescaling = clock();
    rescale_info_t *rescale_info =
        (rescale_info_t *)safe_calloc(1, sizeof(rescale_info_t));
    rescale_info->scaled_problem = deepcopy_problem(original_problem);
    if (rescale_info->scaled_problem == NULL)
    {
        fprintf(stderr,
                "Failed to create a copy of the problem. Aborting rescale.\n");
        return NULL;
    }
    int num_cons = original_problem->num_constraints;
    int num_vars = original_problem->num_variables;

    rescale_info->con_rescale = safe_malloc(num_cons * sizeof(double));
    rescale_info->var_rescale = safe_malloc(num_vars * sizeof(double));
    for (int i = 0; i < num_cons; ++i)
        rescale_info->con_rescale[i] = 1.0;
    for (int i = 0; i < num_vars; ++i)
        rescale_info->var_rescale[i] = 1.0;
    if (params->l_inf_ruiz_iterations > 0)
    {
        ruiz_rescaling(rescale_info->scaled_problem, params->l_inf_ruiz_iterations,
                       rescale_info->con_rescale, rescale_info->var_rescale);
    }
    if (params->has_pock_chambolle_alpha)
    {
        pock_chambolle_rescaling(
            rescale_info->scaled_problem, params->pock_chambolle_alpha,
            rescale_info->con_rescale, rescale_info->var_rescale);
    }
    if (params->bound_objective_rescaling)
    {
        double bound_norm_sq = 0.0;
        for (int i = 0; i < num_cons; ++i)
        {
            if (isfinite(rescale_info->scaled_problem->constraint_lower_bound[i]) &&
                (rescale_info->scaled_problem->constraint_lower_bound[i] !=
                 rescale_info->scaled_problem->constraint_upper_bound[i]))
            {
                bound_norm_sq +=
                    rescale_info->scaled_problem->constraint_lower_bound[i] *
                    rescale_info->scaled_problem->constraint_lower_bound[i];
            }
            if (isfinite(rescale_info->scaled_problem->constraint_upper_bound[i]))
            {
                bound_norm_sq +=
                    rescale_info->scaled_problem->constraint_upper_bound[i] *
                    rescale_info->scaled_problem->constraint_upper_bound[i];
            }
        }

        double obj_norm_sq = 0.0;
        for (int i = 0; i < num_vars; ++i)
        {
            obj_norm_sq += rescale_info->scaled_problem->objective_vector[i] *
                           rescale_info->scaled_problem->objective_vector[i];
        }

        rescale_info->con_bound_rescale = 1.0 / (sqrt(bound_norm_sq) + 1.0);
        rescale_info->obj_vec_rescale = 1.0 / (sqrt(obj_norm_sq) + 1.0);

        for (int i = 0; i < num_cons; ++i)
        {
            rescale_info->scaled_problem->constraint_lower_bound[i] *=
                rescale_info->con_bound_rescale;
            rescale_info->scaled_problem->constraint_upper_bound[i] *=
                rescale_info->con_bound_rescale;
        }
        for (int i = 0; i < num_vars; ++i)
        {
            rescale_info->scaled_problem->variable_lower_bound[i] *=
                rescale_info->con_bound_rescale;
            rescale_info->scaled_problem->variable_upper_bound[i] *=
                rescale_info->con_bound_rescale;
            rescale_info->scaled_problem->objective_vector[i] *=
                rescale_info->obj_vec_rescale;
        }
    }
    else
    {
        rescale_info->con_bound_rescale = 1.0;
        rescale_info->obj_vec_rescale = 1.0;
    }
    rescale_info->rescaling_time_sec =
        (double)(clock() - start_rescaling) / CLOCKS_PER_SEC;
    return rescale_info;
}