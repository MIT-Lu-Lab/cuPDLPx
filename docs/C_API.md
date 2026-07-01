
### C Interface

cuPDLPx provides a C API for directly solving LPs in memory, defined in header file [`include/cupdlpx.h`](../include/cupdlpx.h). This is useful when integrating cuPDLPx into other C/C++ projects or when generating problems programmatically.

#### Functions and Parameters

The C API involves two main functions:

```c
lp_problem_t *create_lp_problem(
    const double *objective_c,           // objective vector (length n)
    const matrix_desc_t *A_desc,         // constraint matrix (m×n)
    const double *con_lb,                // constraint lower bounds (length m)
    const double *con_ub,                // constraint upper bounds (length m)
    const double *var_lb,                // variable lower bounds (length n)
    const double *var_ub,                // variable upper bounds (length n)
    const double *objective_constant,    // scalar objective offset
    const objective_sense_t *objective_sense // objective sense (NULL → minimize)
);

cupdlpx_result_t* solve_lp_problem(
    lp_problem_t* prob,
    const pdhg_parameters_t* params    // NULL → use default parameters
);
```

`create_lp_problem` parameters:
- `objective_c`: Objective vector. If `NULL`, defaults to all zeros.
- `A_desc`: Matrix descriptor. Supports `matrix_dense`, `matrix_csr`, `matrix_csc`, `matrix_coo`.
- `con_lb`: Constraint lower bounds. If `NULL`, defaults to all `-INFINITY`.
- `con_ub`: Constraint upper bounds. If `NULL`, defaults to all `+INFINITY`.
- `var_lb`: Variable lower bounds. If `NULL`, defaults to all `-INFINITY`.
- `var_ub`: Variable upper bounds. If `NULL`, defaults to all `+INFINITY`.
- `objective_constant`: Scalar constant term added to the objective value. If `NULL`, defaults to `0.0`.
- `objective_sense`: Objective sense, `OBJECTIVE_SENSE_MINIMIZE` or `OBJECTIVE_SENSE_MAXIMIZE`. If `NULL`, defaults to minimize.


`solve_lp_problem` parameters:
- `prob`: An LP problem built with `create_LP_problem`. The solver may clean up the matrix (e.g., drop near-zero entries), so the struct must be mutable.
- `params`: Solver parameters. If `NULL`, the solver will use default parameters.

#### Example: Solving a Small LP
```c
#include "cupdlpx.h"
#include <math.h>
#include <stdio.h>

int main() {
    int m = 3; // number of constraints
    int n = 2; // number of variables

    // Dense matrix A
    double A[3][2] = {
        {1.0, 2.0},
        {0.0, 1.0},
        {3.0, 2.0}
    };

    // Describe A
    matrix_desc_t A_desc;
    A_desc.m = m; A_desc.n = n;
    A_desc.fmt = matrix_dense;
    A_desc.data.dense.A = &A[0][0];

    // Objective coefficients
    double c[2] = {1.0, 1.0};

    // Constraint bounds: l <= A x <= u
    double l[3] = {5.0, -INFINITY, -INFINITY};
    double u[3] = {5.0, 2.0, 8.0};

    // Build the problem
    lp_problem_t* prob = create_lp_problem(
        c, &A_desc, l, u, NULL, NULL, NULL, NULL);

    // Solve (NULL → use default parameters)
    cupdlpx_result_t* res = solve_lp_problem(prob, NULL);

    printf("Termination reason: %d\n", res->termination_reason);
    printf("Primal objective: %.6f\n", res->primal_objective_value);
    printf("Dual objective:   %.6f\n", res->dual_objective_value);
    for (int j = 0; j < res->num_variables; ++j) {
        printf("x[%d] = %.6f\n", j, res->primal_solution[j]);
    }

    lp_problem_free(prob);
    cupdlpx_result_free(res);

    return 0;
}
```