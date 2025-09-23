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

#include "struct.h"
#include "utils.h"
#include "io.h"

#ifdef __cplusplus
extern "C" {
#endif

// create an lp_problem_t from a matrix descriptor
lp_problem_t* make_problem_from_matrix(
    const matrix_desc_t* A_desc,
    const double* objective_c,
    const double* objective_constant,
    const double* var_lb,
    const double* var_ub,
    const double* con_lb,
    const double* con_ub
);

// solve the LP problem using PDHG
lp_solution_t solve_lp_problem(
    const lp_problem_t* prob,
    const pdhg_parameters_t* params
);

// free lp solution
void lp_solution_free(lp_solution_t* sol);


#ifdef __cplusplus
} // extern "C"
#endif