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

#include "cupdlpx/interface.h"
#include <stdio.h>
#include <math.h> 

static const char* term_to_str(termination_reason_t r) {
    switch (r) {
        case TERMINATION_REASON_OPTIMAL:           return "OPTIMAL";
        case TERMINATION_REASON_PRIMAL_INFEASIBLE: return "PRIMAL_INFEASIBLE";
        case TERMINATION_REASON_DUAL_INFEASIBLE:   return "DUAL_INFEASIBLE";
        case TERMINATION_REASON_TIME_LIMIT:        return "TIME_LIMIT";
        case TERMINATION_REASON_ITERATION_LIMIT:   return "ITERATION_LIMIT";
        default:                                   return "UNSPECIFIED";
    }
}

static void print_vec(const char* name, const double* v, int n) {
    printf("%s:", name);
    for (int i = 0; i < n; ++i) printf(" % .6g", v[i]);
    printf("\n");
}

int main() {
    // Example: min c^T x
    // s.t. l <= A x <= u, x >= 0

    int m = 3; // number of constraints
    int n = 2; // number of variables

    // A as a dense matrix
    double A[3][2] = {
        {1.0, 2.0},
        {0.0, 1.0},
        {3.0, 2.0}
    };

    // describe A using matrix_desc_t
    matrix_desc_t A_desc;
    A_desc.m = m;
    A_desc.n = n;
    A_desc.fmt = matrix_dense;
    A_desc.zero_tolerance = 0.0;
    A_desc.data.dense.A = &A[0][0];

    // c: objective coefficients
    double c[2] = {1.0, 1.0};

    // l&u: constraintbounds
    double l[3] = {5.0, -INFINITY, -INFINITY};  // lower bounds
    double u[3] = {5.0, 2.0, 8.0}; // upper bounds

    printf("Objective c = [");
    for (int j = 0; j < n; j++) {
        printf(" %g", c[j]);
    }
    printf(" ]\n");

    printf("Matrix A:\n");
    for (int i = 0; i < m; i++) {
        printf("  [");
        for (int j = 0; j < n; j++) {
            printf(" %g", A[i][j]);
        }
        printf(" ]\n");
    }

    printf("Constraint bounds (l <= A x <= b):\n");
    for (int i = 0; i < m; i++) {
        printf("  row %d: l = %g, u = %g\n", i, l[i], u[i]);
    }

     lp_problem_t* prob = make_problem_from_matrix(
        &A_desc,        // A
        c,              // c
        NULL,           // objective_constant
        NULL,           // var_lb
        NULL,           // var_ub
        l,              // con_lb
        u               // con_ub
    );
    if (!prob) {
        fprintf(stderr, "[test] make_problem_from_matrix failed.\n");
        return 1;
    }

    // solve problem
    lp_solution_t sol = solve_lp_problem(prob, NULL);

    // free problem
    lp_problem_free(prob);

    // check solution
    if (!sol.x || !sol.y) {
        fprintf(stderr, "[test] solve_lp_problem failed (x/y null). reason=%s\n",
                term_to_str(sol.reason));
        lp_solution_free(&sol);
        return 2;
    }

    // print solution
    printf("Solution:\n");
    printf("Termination: %s\n", term_to_str(sol.reason));
    printf("Primal obj: %.10g\n", sol.primal_obj);
    printf("Dual   obj: %.10g\n", sol.dual_obj);
    print_vec("x", sol.x, sol.n);
    print_vec("y", sol.y, sol.m);

    // free solution
    lp_solution_free(&sol);

    return 0;
}
