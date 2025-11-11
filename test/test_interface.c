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
#include <stdio.h>
#include <math.h> 

static void print_vec(const char* name, const double* v, int n) {
    printf("%s:", name);
    for (int i = 0; i < n; ++i) printf(" % .6g", v[i]);
    printf("\n");
}

static void run_once(const char* tag,
                     const matrix_desc_t* A_desc,
                     const double* c, const double* l, const double* u)
{
    printf("\n=== %s ===\n", tag);
    
    // build problem
    lp_problem_t* prob = create_lp_problem(
        c,       // objective_c
        A_desc,  // constraint matrix A
        l,       // constraint lower bound con_lb
        u,       // constraint upper bound con_ub
        NULL,    // variable lower bound var_lb (defaults to 0)
        NULL,    // variable upper bound var_ub (defaults to +inf)
        NULL     // objective constant c0 (defaults to 0.0)
    );
    if (!prob) {
        fprintf(stderr, "[test] create_lp_problem failed for %s.\n", tag);
        return;
    }

    // solve
    cupdlpx_result_t* res = solve_lp_problem(prob, NULL);
    lp_problem_free(prob);
    if (!res) {
        fprintf(stderr, "[test] solve_lp_problem failed for %s.\n", tag);
        return;
    }

    // print results
    print_vec("x", res->primal_solution, res->num_variables);
    print_vec("y", res->dual_solution, res->num_constraints);
    
    // free
    cupdlpx_result_free(res);
}

static void test_warm_start(const char* tag,
                                      const matrix_desc_t* A_desc,
                                      const double* c, const double* l, const double* u)
{
    printf("\n=== %s (with initial solution) ===\n", tag);

    int n = A_desc->n;
    int m = A_desc->m;

    lp_problem_t* prob = create_lp_problem(
        c, A_desc, l, u, NULL, NULL, NULL
    );
    if (!prob) {
        fprintf(stderr, "[test] create_lp_problem failed for %s.\n", tag);
        return;
    }

    // Allocate and set initial solutions (e.g., zeros)
    double* primal = (double*)malloc(n * sizeof(double));
    double* dual   = (double*)malloc(m * sizeof(double));
    for (int i = 0; i < n; ++i) primal[i] = 1.0;
    for (int i = 0; i < m; ++i) dual[i] = 1.0;

    set_start_values(prob, primal, dual);

    cupdlpx_result_t* res = solve_lp_problem(prob, NULL);
    if (!res) {
        fprintf(stderr, "[test] solve_lp_problem failed for %s.\n", tag);
        lp_problem_free(prob);
        return;
    }

    print_vec("x", res->primal_solution, res->num_variables);
    print_vec("y", res->dual_solution, res->num_constraints);

    
    free(primal);
    free(dual);
    cupdlpx_result_free(res);
    lp_problem_free(prob);
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
    matrix_desc_t A_dense;
    A_dense.m = m;
    A_dense.n = n;
    A_dense.fmt = matrix_dense;
    A_dense.zero_tolerance = 0.0;
    A_dense.data.dense.A = &A[0][0];

    // A as a CSR matrix
    static int csr_row_ptr[4] = {0, 2, 3, 5};
    static int csr_col_ind[5] = {0, 1, 1, 0, 1};
    static double csr_vals[5] = {1, 2, 1, 3, 2};

    // describe A using matrix_desc_t
    matrix_desc_t A_csr;
    A_csr.m = m; A_csr.n = n;
    A_csr.fmt = matrix_csr;
    A_csr.zero_tolerance = 0.0;
    A_csr.data.csr.nnz = 5;
    A_csr.data.csr.row_ptr = csr_row_ptr;
    A_csr.data.csr.col_ind = csr_col_ind;
    A_csr.data.csr.vals = csr_vals;

    // A as a CSC matrix
    static int csc_col_ptr[3] = {0, 2, 5};
    static int csc_row_ind[5] = {0, 2, 0, 1, 2};
    static double csc_vals[5] = {1, 3, 2, 1, 2};

    // describe A using matrix_desc_t
    matrix_desc_t A_csc;
    A_csc.m = m; A_csc.n = n;
    A_csc.fmt = matrix_csc;
    A_csc.zero_tolerance = 0.0;
    A_csc.data.csc.nnz = 5;
    A_csc.data.csc.col_ptr = csc_col_ptr;
    A_csc.data.csc.row_ind = csc_row_ind;
    A_csc.data.csc.vals = csc_vals;

    // A as a COO matrix
    static int coo_row_ind[5] = {0, 0, 1, 2, 2};
    static int coo_col_ind[5] = {0, 1, 1, 0, 1};
    static double coo_vals[5] = {1, 2, 1, 3, 2};

    // describe A using matrix_desc_t
    matrix_desc_t A_coo;
    A_coo.m = m; A_coo.n = n;
    A_coo.fmt = matrix_coo;
    A_coo.zero_tolerance = 0.0;
    A_coo.data.coo.nnz = 5;
    A_coo.data.coo.row_ind = coo_row_ind;
    A_coo.data.coo.col_ind = coo_col_ind;
    A_coo.data.coo.vals = coo_vals;

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

     lp_problem_t* prob = create_lp_problem(
        c,              // c
        &A_dense,       // A
        l,              // con_lb
        u,              // con_ub
        NULL,           // var_lb
        NULL,           // var_ub
        NULL            // objective_constant
    );
    if (!prob) {
        fprintf(stderr, "[test] create_lp_problem failed.\n");
        return 1;
    }

    run_once("Test 1: Dense Matrix", &A_dense, c, l, u);
    run_once("Test 2: CSR Matrix",   &A_csr,   c, l, u);
    run_once("Test 3: CSC Matrix",   &A_csc,   c, l, u);
    run_once("Test 4: COO Matrix",   &A_coo,   c, l, u);

    test_warm_start("Test 5: Dense Matrix", &A_dense, c, l, u);
    test_warm_start("Test 6: CSR Matrix", &A_csr, c, l, u);
    test_warm_start("Test 7: CSC Matrix", &A_csc, c, l, u);
    test_warm_start("Test 8: COO Matrix", &A_coo, c, l, u);

    return 0;
}