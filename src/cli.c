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
#include "mps_parser.h"
#include "presolve.h"
#include "solver.h"
#include "utils.h"
#include <getopt.h>
#include <libgen.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *get_output_path(const char *output_dir, const char *instance_name, const char *suffix)
{
    size_t path_len = strlen(output_dir) + strlen(instance_name) + strlen(suffix) + 2;
    char *full_path = safe_malloc(path_len * sizeof(char));
    snprintf(full_path, path_len, "%s/%s%s", output_dir, instance_name, suffix);
    return full_path;
}

char *extract_instance_name(const char *filename)
{
    char *filename_copy = strdup(filename);
    if (filename_copy == NULL)
    {
        perror("Memory allocation failed");
        return NULL;
    }

    char *base = basename(filename_copy);
    char *dot = strchr(base, '.');
    if (dot)
    {
        *dot = '\0';
    }

    char *instance_name = strdup(base);
    free(filename_copy);
    return instance_name;
}

void save_solution(const double *data, int size, const char *output_dir, const char *instance_name, const char *suffix)
{
    char *file_path = get_output_path(output_dir, instance_name, suffix);
    if (file_path == NULL || data == NULL)
    {
        return;
    }

    FILE *outfile = fopen(file_path, "w");
    if (outfile == NULL)
    {
        perror("Error opening solution file");
        free(file_path);
        return;
    }

    for (int i = 0; i < size; ++i)
    {
        fprintf(outfile, "%.10g\n", data[i]);
    }

    fclose(outfile);
    free(file_path);
}

void save_solver_summary(const cupdlpx_result_t *result, const char *output_dir, const char *instance_name)
{
    char *file_path = get_output_path(output_dir, instance_name, "_summary.txt");
    if (file_path == NULL)
    {
        return;
    }

    FILE *outfile = fopen(file_path, "w");
    if (outfile == NULL)
    {
        perror("Error opening summary file");
        free(file_path);
        return;
    }
    fprintf(outfile, "Termination Reason: %s\n", termination_reason_to_string(result->termination_reason));
    if (result->presolve_time > 0.0)
    {
        fprintf(outfile, "Presolve Status: %s\n", get_presolve_status_str(result->presolve_status));
        fprintf(outfile, "Presolve Time (sec): %e\n", result->presolve_time);
        fprintf(outfile, "Reduced Rows: %d\n", result->num_reduced_constraints);
        fprintf(outfile, "Reduced Columns: %d\n", result->num_reduced_variables);
        fprintf(outfile, "Reduced Nonzeros: %d\n", result->num_reduced_nonzeros);

        // if (result->presolve_stats.n_cols_original > 0) {
        //     fprintf(outfile, "NNZ Removed Trivial: %d\n", result->presolve_stats.nnz_removed_trivial);
        //     fprintf(outfile, "NNZ Removed Fast: %d\n", result->presolve_stats.nnz_removed_fast);
        //     fprintf(outfile, "NNZ Removed Primal Propagation: %d\n", result->presolve_stats.nnz_removed_primal_propagation);
        //     fprintf(outfile, "NNZ Removed Parallel Rows: %d\n", result->presolve_stats.nnz_removed_parallel_rows);
        //     fprintf(outfile, "NNZ Removed Parallel Cols: %d\n", result->presolve_stats.nnz_removed_parallel_cols);

        //     fprintf(outfile, "Presolve Time Init (sec): %e\n", result->presolve_stats.time_init);
        //     fprintf(outfile, "Presolve Time Run (sec): %e\n", result->presolve_stats.time_presolve);
        //     fprintf(outfile, "Presolve Time Fast (sec): %e\n", result->presolve_stats.time_fast_reductions);
        //     fprintf(outfile, "Presolve Time Medium (sec): %e\n", result->presolve_stats.time_medium_reductions);
        //     fprintf(outfile, "Presolve Time Primal Proppagation (sec): %e\n", result->presolve_stats.time_primal_propagation);
        //     fprintf(outfile, "Presolve Time Parallel Rows (sec): %e\n", result->presolve_stats.time_parallel_rows);
        //     fprintf(outfile, "Presolve Time Parallel Cols (sec): %e\n", result->presolve_stats.time_parallel_cols);
        //     fprintf(outfile, "Postsolve Time (sec): %e\n", result->presolve_stats.time_postsolve);
        // }
    }
    fprintf(outfile, "Precondition time (sec): %e\n", result->rescaling_time_sec);
    fprintf(outfile, "Runtime (sec): %e\n", result->cumulative_time_sec);
    fprintf(outfile, "Iterations Count: %d\n", result->total_count);
    fprintf(outfile, "Primal Objective Value: %e\n", result->primal_objective_value);
    fprintf(outfile, "Dual Objective Value: %e\n", result->dual_objective_value);
    fprintf(outfile, "Relative Primal Residual: %e\n", result->relative_primal_residual);
    fprintf(outfile, "Relative Dual Residual: %e\n", result->relative_dual_residual);
    fprintf(outfile, "Absolute Objective Gap: %e\n", result->objective_gap);
    fprintf(outfile, "Relative Objective Gap: %e\n", result->relative_objective_gap);
    fprintf(outfile, "Rows: %d\n", result->num_constraints);
    fprintf(outfile, "Columns: %d\n", result->num_variables);
    fprintf(outfile, "Nonzeros: %d\n", result->num_nonzeros);
    if (result->feasibility_polishing_time > 0.0)
    {
        fprintf(outfile, "Feasibility Polishing Time (sec): %e\n", result->feasibility_polishing_time);
        fprintf(outfile, "Feasibility Polishing Iteration Count: %d\n", result->feasibility_iteration);
    }
    fclose(outfile);
    free(file_path);
}

void print_usage(const char *prog_name)
{
    fprintf(stderr, "Usage: %s [OPTIONS] <mps_file> <output_dir>\n\n", prog_name);

    fprintf(stderr, "Arguments:\n");
    fprintf(stderr,
            "  <mps_file>               Path to the input problem in MPS "
            "format (.mps or .mps.gz).\n");
    fprintf(stderr,
            "  <output_dir>             Directory where output files "
            "will be saved. It will contain:\n");
    fprintf(stderr, "                             - <basename>_summary.txt\n");
    fprintf(stderr, "                             - <basename>_primal_solution.txt\n");
    fprintf(stderr, "                             - <basename>_dual_solution.txt\n\n");

    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -h, --help                          Display this help message.\n");
    fprintf(stderr,
            "  -v, --verbose                       "
            "Enable verbose logging (enabled by default; kept for compatibility).\n");
    fprintf(stderr,
            "  -q, --quiet                         "
            "Disable verbose logging.\n");
    fprintf(stderr,
            "      --time_limit <seconds>          "
            "Time limit in seconds (default: 3600.0).\n");
    fprintf(stderr, "      --iter_limit <iterations>       Iteration limit (default: %d).\n", INT32_MAX);
    fprintf(stderr,
            "      --eps_opt <tolerance>           "
            "Relative optimality tolerance (default: 1e-4).\n");
    fprintf(stderr,
            "      --eps_feas <tolerance>          "
            "Relative feasibility tolerance (default: 1e-4).\n");
    fprintf(stderr,
            "      --l_inf_ruiz_iter <int>         "
            "Iterations for L-inf Ruiz rescaling (default: 10).\n");
    fprintf(stderr,
            "      --no_pock_chambolle             "
            "Disable Pock-Chambolle rescaling (default: enabled).\n");
    fprintf(stderr,
            "      --pock_chambolle_alpha <float>  "
            "Value for Pock-Chambolle alpha (default: 1.0).\n");
    fprintf(stderr,
            "      --no_bound_obj_rescaling        "
            "Disable bound objective rescaling (default: enabled).\n");
    fprintf(stderr,
            "      --eval_freq <int>               "
            "Termination evaluation frequency (default: 200).\n");
    fprintf(stderr,
            "      --sv_max_iter <int>             "
            "Max iterations for singular value estimation (default: 5000).\n");
    fprintf(stderr,
            "      --sv_tol <float>                "
            "Tolerance for singular value estimation (default: 1e-4).\n");
    fprintf(stderr,
            "  -f  --feasibility_polishing         "
            "Enable feasibility use feasibility polishing (default: false).\n");
    fprintf(stderr,
            "      --eps_feas_polish <tolerance>   Relative feasibility "
            "polish tolerance (default: 1e-6).\n");
    fprintf(stderr,
            "      --opt_norm <norm_type>          "
            "Norm for optimality criteria: l2 or linf (default: l2).\n");
    fprintf(stderr,
            "      --no_presolve                   "
            "Disable presolve (default: enabled).\n");
    fprintf(stderr,
            "      --matrix_zero_tol <tolerance>.  "
            "Zero tolerance in constraint matrix.\n");
    fprintf(stderr,
            "      --infinite_bound <value>.       "
            "Bounds at or beyond this are treated as infinite (default: 1e20).\n");
}

int main(int argc, char *argv[])
{
    pdhg_parameters_t params;
    set_default_parameters(&params);

    static struct option long_options[] = {{"help", no_argument, 0, 'h'},
                                           {"verbose", no_argument, 0, 'v'},
                                           {"quiet", no_argument, 0, 'q'},
                                           {"time_limit", required_argument, 0, 1001},
                                           {"iter_limit", required_argument, 0, 1002},
                                           {"eps_opt", required_argument, 0, 1003},
                                           {"eps_feas", required_argument, 0, 1004},
                                           {"eps_feas_polish", required_argument, 0, 1006},
                                           {"feasibility_polishing", no_argument, 0, 'f'},
                                           {"l_inf_ruiz_iter", required_argument, 0, 1007},
                                           {"pock_chambolle_alpha", required_argument, 0, 1008},
                                           {"no_pock_chambolle", no_argument, 0, 1009},
                                           {"no_bound_obj_rescaling", no_argument, 0, 1010},
                                           {"sv_max_iter", required_argument, 0, 1011},
                                           {"sv_tol", required_argument, 0, 1012},
                                           {"eval_freq", required_argument, 0, 1013},
                                           {"opt_norm", required_argument, 0, 1014},
                                           {"no_presolve", no_argument, 0, 1015},
                                           {"matrix_zero_tol", required_argument, 0, 1016},
                                           {"infinite_bound", required_argument, 0, 1017},
                                           {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "hvqfp", long_options, NULL)) != -1)
    {
        switch (opt)
        {
            case 'h':
                print_usage(argv[0]);
                return 0;
            case 'v':
                params.verbose = true;
                break;
            case 'q':
                params.verbose = false;
                break;
            case 1001: // --time_limit
                params.termination_criteria.time_sec_limit = atof(optarg);
                break;
            case 1002: // --iter_limit
                params.termination_criteria.iteration_limit = atoi(optarg);
                break;
            case 1003: // --eps_optimal
                params.termination_criteria.eps_optimal_relative = atof(optarg);
                break;
            case 1004: // --eps_feas
                params.termination_criteria.eps_feasible_relative = atof(optarg);
                break;
            case 1006: // --eps_feas_polish_relative
                params.termination_criteria.eps_feas_polish_relative = atof(optarg);
                break;
            case 'f': // --feasibility_polishing
                params.feasibility_polishing = true;
                break;
            case 1007: // --l_inf_ruiz_iter
                params.l_inf_ruiz_iterations = atoi(optarg);
                break;
            case 1008: // --pock_chambolle_alpha
                params.pock_chambolle_alpha = atof(optarg);
                break;
            case 1009: // --no_pock_chambolle
                params.has_pock_chambolle_alpha = false;
                break;
            case 1010: // --no_bound_obj_rescaling
                params.bound_objective_rescaling = false;
                break;
            case 1011: // --sv_max_iter
                params.sv_max_iter = atoi(optarg);
                break;
            case 1012: // --sv_tol
                params.sv_tol = atof(optarg);
                break;
            case 1013: // --eval_freq
                params.termination_evaluation_frequency = atoi(optarg);
                break;
            case 1014: // --opt_norm
            {
                const char *norm_str = optarg;
                if (strcmp(norm_str, "l2") == 0)
                {
                    params.optimality_norm = NORM_TYPE_L2;
                }
                else if (strcmp(norm_str, "linf") == 0)
                {
                    params.optimality_norm = NORM_TYPE_L_INF;
                }
                else
                {
                    fprintf(stderr, "Error: opt_norm must be 'l2' or 'linf'\n");
                    return 1;
                }
            }
            break;
            case 1015: // --no_presolve
                params.presolve = false;
                break;
            case 1016: // --matrix_zero_tol
                params.matrix_zero_tol = atof(optarg);
                break;
            case 1017: // --infinite_bound
                params.infinite_bound = atof(optarg);
                break;
            case '?': // Unknown option
                return 1;
        }
    }

    if (argc - optind != 2)
    {
        fprintf(stderr, "Error: You must specify an input file and an output directory.\n\n");
        print_usage(argv[0]);
        return 1;
    }

    const char *filename = argv[optind];
    const char *output_dir = argv[optind + 1];

    char *instance_name = extract_instance_name(filename);
    if (instance_name == NULL)
    {
        return 1;
    }

    lp_problem_t *problem = read_mps_file(filename);

    if (problem == NULL)
    {
        fprintf(stderr, "Failed to read or parse the file.\n");
        free(instance_name);
        return 1;
    }

    cupdlpx_result_t *result = solve_lp_problem(problem, &params);

    if (result == NULL)
    {
        fprintf(stderr, "Solver failed.\n");
    }
    else
    {
        save_solver_summary(result, output_dir, instance_name);
        save_solution(
            result->primal_solution, problem->num_variables, output_dir, instance_name, "_primal_solution.txt");
        save_solution(result->dual_solution, problem->num_constraints, output_dir, instance_name, "_dual_solution.txt");
        cupdlpx_result_free(result);
    }

    lp_problem_free(problem);
    free(instance_name);

    return 0;
}
