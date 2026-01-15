#include "presolve.h"
#include "PSLP_sol.h"
#include "cupdlpx.h"
#include "utils.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef PSLP_VERSION
#define PSLP_VERSION "unknown"
#endif

const char *get_presolve_status_str(enum PresolveStatus_ status)
{
    switch (status)
    {
    case UNCHANGED:
        return "UNCHANGED";
    case REDUCED:
        return "REDUCED";
    case INFEASIBLE:
        return "INFEASIBLE";
    case UNBNDORINFEAS:
        return "INFEASIBLE_OR_UNBOUNDED";
    default:
        return "UNKNOWN_STATUS";
    }
}

lp_problem_t *convert_pslp_to_cupdlpx(PresolvedProblem *reduced_prob)
{

    lp_problem_t *cupdlpx_prob = (lp_problem_t *)safe_malloc(sizeof(lp_problem_t));
    // TODO: handle warmstart here
    cupdlpx_prob->primal_start = NULL;
    cupdlpx_prob->dual_start = NULL;

    cupdlpx_prob->objective_constant = reduced_prob->obj_offset;
    cupdlpx_prob->objective_vector = reduced_prob->c;

    cupdlpx_prob->constraint_lower_bound = reduced_prob->lhs;
    cupdlpx_prob->constraint_upper_bound = reduced_prob->rhs;
    cupdlpx_prob->variable_lower_bound = reduced_prob->lbs;
    cupdlpx_prob->variable_upper_bound = reduced_prob->ubs;

    cupdlpx_prob->constraint_matrix_num_nonzeros = reduced_prob->nnz;
    cupdlpx_prob->constraint_matrix_row_pointers = reduced_prob->Ap;
    cupdlpx_prob->constraint_matrix_col_indices = reduced_prob->Ai;
    cupdlpx_prob->constraint_matrix_values = reduced_prob->Ax;

    cupdlpx_prob->num_variables = reduced_prob->n;
    cupdlpx_prob->num_constraints = reduced_prob->m;

    return cupdlpx_prob;
}

cupdlpx_presolve_info_t *pslp_presolve(const lp_problem_t *original_prob, const pdhg_parameters_t *params)
{
    if (original_prob->primal_start || original_prob->dual_start)
    {
        printf("Warning: Warm-starting is currently not supported when presolve is enabled.\n"
               "The provided initial solutions will be ignored.\n");
    }
    if (params->verbose)
    {
        printf("\nRunning presolver (PSLP %s)...\n", PSLP_VERSION);
    }
    clock_t start_time = clock();

    cupdlpx_presolve_info_t *info = (cupdlpx_presolve_info_t *)safe_calloc(1, sizeof(cupdlpx_presolve_info_t));

    // 1. Init Settings
    info->settings = default_settings();
    info->settings->verbose = false;

    // 2. Init Presolver
    info->presolver = new_presolver(
        original_prob->constraint_matrix_values,
        original_prob->constraint_matrix_col_indices,
        original_prob->constraint_matrix_row_pointers,
        original_prob->num_constraints,
        original_prob->num_variables,
        original_prob->constraint_matrix_num_nonzeros,
        original_prob->constraint_lower_bound,
        original_prob->constraint_upper_bound,
        original_prob->variable_lower_bound,
        original_prob->variable_upper_bound,
        original_prob->objective_vector,
        info->settings);

    // 3. Run Presolve
    PresolveStatus status = run_presolver(info->presolver);
    info->presolve_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;

    info->presolve_status = status;
    if (params->verbose)
    {
        printf("  %-15s : %s\n", "status", get_presolve_status_str(status));
        printf("  %-15s : %.3g sec\n", "presolve time", info->presolve_time);
        printf("  %-15s : %d rows, %d columns, %d nonzeros\n",
                "reduced problem",
                info->presolver->reduced_prob->m,
                info->presolver->reduced_prob->n,
                info->presolver->reduced_prob->nnz);
    }

    if (status & INFEASIBLE || status & UNBNDORINFEAS || info->presolver->reduced_prob->n == 0)
    {
        info->problem_solved_during_presolve = true;
        info->reduced_problem = NULL;
    }
    else
    {
        info->problem_solved_during_presolve = false;
        info->reduced_problem = convert_pslp_to_cupdlpx(info->presolver->reduced_prob);
    }
    return info;
}

cupdlpx_result_t *create_result_from_presolve(const cupdlpx_presolve_info_t *info, const lp_problem_t *original_prob)
{

    cupdlpx_result_t *result = (cupdlpx_result_t *)safe_calloc(1, sizeof(cupdlpx_result_t));
    result->num_variables = original_prob->num_variables;
    result->num_constraints = original_prob->num_constraints;
    result->num_nonzeros = original_prob->constraint_matrix_num_nonzeros;
    result->num_reduced_variables = info->presolver->reduced_prob->n;
    result->num_reduced_constraints = info->presolver->reduced_prob->m;
    result->num_reduced_nonzeros = info->presolver->reduced_prob->nnz;
    result->presolve_status = info->presolve_status;
    result->presolve_time = info->presolve_time;

    if (info->presolve_status == INFEASIBLE)
    {
        result->termination_reason = TERMINATION_REASON_PRIMAL_INFEASIBLE;
    }
    else if (info->presolve_status == UNBNDORINFEAS)
    {
        result->termination_reason = TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED;
    }
    else if (info->presolver->reduced_prob->n == 0)
    {
        result->termination_reason = TERMINATION_REASON_OPTIMAL;
        pslp_postsolve(info, result, original_prob);
        return result;
    }
    else
    {
        result->termination_reason = TERMINATION_REASON_UNSPECIFIED;
    }
    // result->presolve_stats = *(info->presolver->stats);
    if (result->num_variables > 0)
    {
        result->primal_solution = (double *)safe_calloc(result->num_variables, sizeof(double));
        result->reduced_cost = (double *)safe_calloc(result->num_variables, sizeof(double));
    }
    if (result->num_constraints > 0)
    {
        result->dual_solution = (double *)safe_calloc(result->num_constraints, sizeof(double));
    }
    return result;
}

void pslp_postsolve(const cupdlpx_presolve_info_t *info,
                    cupdlpx_result_t *result,
                    const lp_problem_t *original_prob)
{
    postsolve(info->presolver,
              result->primal_solution,
              result->dual_solution,
              result->reduced_cost);

    result->num_reduced_variables = info->presolver->reduced_prob->n;
    result->num_reduced_constraints = info->presolver->reduced_prob->m;
    result->num_reduced_nonzeros = info->presolver->reduced_prob->nnz;
    result->presolve_status = info->presolve_status;

    result->primal_solution = (double *)safe_malloc(original_prob->num_variables * sizeof(double));
    result->dual_solution = (double *)safe_malloc(original_prob->num_constraints * sizeof(double));
    result->reduced_cost = (double *)safe_malloc(original_prob->num_variables * sizeof(double));

    memcpy(result->primal_solution, info->presolver->sol->x, original_prob->num_variables * sizeof(double));
    memcpy(result->dual_solution, info->presolver->sol->y, original_prob->num_constraints * sizeof(double));
    memcpy(result->reduced_cost, info->presolver->sol->z, original_prob->num_variables * sizeof(double));
    result->presolve_time = info->presolve_time;
    if (info->presolver->reduced_prob->n == 0)
    {
        double obj = 0.0;
        for (int i = 0; i < original_prob->num_variables; i++)
        {
            obj += original_prob->objective_vector[i] * result->primal_solution[i];
        }
        obj += original_prob->objective_constant;
        result->primal_objective_value = obj;
        result->dual_objective_value = obj;
    }
    // if (info->presolver->stats != NULL) {
    //     result->presolve_stats = *(info->presolver->stats);
    // }
    return;
}

void cupdlpx_presolve_info_free(cupdlpx_presolve_info_t *info)
{
    if (!info)
        return;
    // if (info->reduced_problem) lp_problem_free(info->reduced_problem);
    if (info->presolver)
        free_presolver(info->presolver);
    if (info->settings)
        free_settings(info->settings);
    free(info);
}