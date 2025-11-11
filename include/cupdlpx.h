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

#ifdef __cplusplus
extern "C"
{
#endif

    // create an lp_problem_t from a matrix descriptor
    lp_problem_t *create_lp_problem(
        const double *objective_c,
        const matrix_desc_t *A_desc,
        const double *con_lb,
        const double *con_ub,
        const double *var_lb,
        const double *var_ub,
        const double *objective_constant);

    // Set up initial primal and dual solution for an lp_problem_t
    void set_start_values(lp_problem_t *prob, const double *primal, const double *dual);

    // solve the LP problem using PDHG
    cupdlpx_result_t *solve_lp_problem(
        const lp_problem_t *prob,
        const pdhg_parameters_t *params);

    // parameter
    void set_default_parameters(pdhg_parameters_t *params);

    void cupdlpx_result_free(cupdlpx_result_t *results);

    void lp_problem_free(lp_problem_t *prob);

#ifdef __cplusplus
} // extern "C"
#endif