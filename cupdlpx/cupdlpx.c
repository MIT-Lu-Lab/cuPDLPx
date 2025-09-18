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

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <getopt.h>
#include <string.h>
#include <libgen.h>
#include "io.h"
#include "struct.h"
#include "solver.h"

const char *termination_reason_tToString(termination_reason_t reason)
{
    switch (reason)
    {
    case TERMINATION_REASON_OPTIMAL:
        return "OPTIMAL";
    case TERMINATION_REASON_PRIMAL_INFEASIBLE:
        return "PRIMAL_INFEASIBLE";
    case TERMINATION_REASON_DUAL_INFEASIBLE:
        return "DUAL_INFEASIBLE";
    case TERMINATION_REASON_TIME_LIMIT:
        return "TIME_LIMIT";
    case TERMINATION_REASON_ITERATION_LIMIT:
        return "ITERATION_LIMIT";
    case TERMINATION_REASON_UNSPECIFIED:
    default:
        return "UNSPECIFIED";
    }
}

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

void save_solution(const double *data_d, int size, const char *output_dir,
                   const char *instance_name, const char *suffix)
{
    char *file_path = get_output_path(output_dir, instance_name, suffix);
    if (file_path == NULL)
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

    double *solution_host = (double *)safe_malloc(size * sizeof(double));
    CUDA_CHECK(cudaMemcpy(solution_host, data_d, size * sizeof(double), cudaMemcpyDeviceToHost));

    for (int i = 0; i < size; ++i)
    {
        fprintf(outfile, "%.10g\n", solution_host[i]);
    }

    fclose(outfile);
    free(file_path);
    free(solution_host);
}

void save_solver_summary(const pdhg_solver_state_t *solver_state, const char *output_dir, const char *instance_name)
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
    fprintf(outfile, "Termination Reason: %s\n", termination_reason_tToString(solver_state->termination_reason));
    fprintf(outfile, "Runtime (sec): %e\n", solver_state->cumulative_time_sec);
    fprintf(outfile, "Iterations Count: %d\n", solver_state->total_count);
    fprintf(outfile, "Primal Objective Value: %e\n", solver_state->primal_objective_value);
    fprintf(outfile, "Dual Objective Value: %e\n", solver_state->dual_objective_value);
    fprintf(outfile, "Absolute Primal Residual: %e\n", solver_state->absolute_primal_residual);
    fprintf(outfile, "Relative Primal Residual: %e\n", solver_state->relative_primal_residual);
    fprintf(outfile, "Absolute Dual Residual: %e\n", solver_state->absolute_dual_residual);
    fprintf(outfile, "Relative Dual Residual: %e\n", solver_state->relative_dual_residual);
    fprintf(outfile, "Absolute Objective Gap: %e\n", solver_state->objective_gap);
    fprintf(outfile, "Relative Objective Gap: %e\n", solver_state->relative_objective_gap);
    fclose(outfile);
    free(file_path);
}

void print_usage(const char *prog_name)
{
    fprintf(stderr, "Usage: %s [OPTIONS] <mps_file> <output_dir>\n\n", prog_name);

    fprintf(stderr, "Arguments:\n");
    fprintf(stderr, "  <mps_file>               Path to the input problem in MPS format (.mps or .mps.gz).\n");
    fprintf(stderr, "  <output_dir>             Directory where output files will be saved. It will contain:\n");
    fprintf(stderr, "                             - <basename>_summary.txt\n");
    fprintf(stderr, "                             - <basename>_primal_solution.txt\n");
    fprintf(stderr, "                             - <basename>_dual_solution.txt\n\n");

    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -h, --help                          Display this help message.\n");
    fprintf(stderr, "  -v, --verbose                       Enable verbose logging (default: false).\n");
    fprintf(stderr, "      --time_limit <seconds>          Time limit in seconds (default: 3600.0).\n");
    fprintf(stderr, "      --iter_limit <iterations>       Iteration limit (default: %d).\n", INT32_MAX);
    fprintf(stderr, "      --eps_opt <tolerance>           Relative optimality tolerance (default: 1e-4).\n");
    fprintf(stderr, "      --eps_feas <tolerance>          Relative feasibility tolerance (default: 1e-4).\n");
    fprintf(stderr, "      --eps_infeas_detect <tolerance> Infeasibility detection tolerance (default: 1e-10).\n");
}

int main(int argc, char *argv[])
{
    pdhg_parameters_t params;
    set_default_parameters(&params);

    static struct option long_options[] = {
        {"help", no_argument, 0, 'h'},
        {"verbose", no_argument, 0, 'v'},
        {"time_limit", required_argument, 0, 1001},
        {"iter_limit", required_argument, 0, 1002},
        {"eps_opt", required_argument, 0, 1003},
        {"eps_feas", required_argument, 0, 1004},
        {"eps_infeas_detect", required_argument, 0, 1005},
        {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "hv", long_options, NULL)) != -1)
    {
        switch (opt)
        {
        case 'h':
            print_usage(argv[0]);
            return 0;
        case 'v':
            params.verbose = true;
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
        case 1005: // --eps_infeas_detect
            params.termination_criteria.eps_infeasible = atof(optarg);
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

    pdhg_solver_state_t *solver_state = optimize(&params, problem);

    if (solver_state == NULL)
    {
        fprintf(stderr, "Solver failed.\n");
    }
    else
    {
        save_solver_summary(solver_state, output_dir, instance_name);
        save_solution(solver_state->pdhg_primal_solution, problem->num_variables, output_dir,
                      instance_name, "_primal_solution.txt");
        save_solution(solver_state->pdhg_dual_solution, problem->num_constraints, output_dir,
                      instance_name, "_dual_solution.txt");
        pdhg_solver_state_free(solver_state);
    }

    lp_problem_free(problem);
    free(instance_name);

    return 0;
}