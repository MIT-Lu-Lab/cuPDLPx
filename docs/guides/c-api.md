---
description: Native C data structures, solver lifecycle, parameters, and result handling for cuPDLPx.
---

# C interface

The C API solves LPs from arrays in host memory. Functions are declared in
[`include/cupdlpx.h`](https://github.com/MIT-Lu-Lab/cuPDLPx/blob/main/include/cupdlpx.h),
with public structures and enums in `include/cupdlpx_types.h`.

## Installation

The C library and command-line executable are built from the same source tree.
See [hardware requirements](../getting-started/index.md#hardware-requirements)
for supported GPUs and required CUDA or ROCm versions.

Follow the [native installation instructions](command-line.md#installation)
to build the library.

## Create a problem

```c
lp_problem_t *create_lp_problem(
    const double *objective_c,
    const matrix_desc_t *A_desc,
    const double *con_lb,
    const double *con_ub,
    const double *var_lb,
    const double *var_ub,
    const double *objective_constant,
    const objective_sense_t *objective_sense
);
```

`create_lp_problem` copies the input arrays into a new `lp_problem_t`.
The caller retains ownership of the inputs. Free the problem with
`lp_problem_free` and the result of `solve_lp_problem` with
`cupdlpx_result_free`.

Only `A_desc` is required. Passing `NULL` for another argument selects its
default:

| Argument | Length | `NULL` default |
| --- | ---: | --- |
| `objective_c` | $n$ | all zeros |
| `con_lb` | $m$ | all $-\infty$ |
| `con_ub` | $m$ | all $+\infty$ |
| `var_lb` | $n$ | all $-\infty$ |
| `var_ub` | $n$ | all $+\infty$ |
| `objective_constant` | 1 | `0.0` |
| `objective_sense` | 1 | `OBJECTIVE_SENSE_MINIMIZE` |

## Matrix descriptors

`matrix_desc_t` accepts four host-memory layouts:

| Format | Enum | Required arrays |
| --- | --- | --- |
| Row-major dense | `matrix_dense` | `A` with $m n$ values |
| CSR | `matrix_csr` | `row_ptr`, `col_ind`, `vals`, `nnz` |
| CSC | `matrix_csc` | `col_ptr`, `row_ind`, `vals`, `nnz` |
| COO | `matrix_coo` | `row_ind`, `col_ind`, `vals`, `nnz` |

Indices are zero-based `int` values and numeric data uses `double`.

See the
[cuSPARSE matrix formats documentation](https://docs.nvidia.com/cuda/cusparse/index.html#matrix-formats).

## Solve a small LP

```c
#include "cupdlpx.h"
#include <math.h>
#include <stdio.h>

int main(void) {
    double A[3][2] = {
        {1.0, 2.0},
        {0.0, 1.0},
        {3.0, 2.0}
    };
    double c[2] = {1.0, 1.0};
    double l[3] = {5.0, -INFINITY, -INFINITY};
    double u[3] = {5.0, 2.0, 8.0};

    matrix_desc_t matrix = {
        .m = 3,
        .n = 2,
        .fmt = matrix_dense,
        .data.dense = {.A = &A[0][0]},
    };

    lp_problem_t *problem = create_lp_problem(
        c, &matrix, l, u, NULL, NULL, NULL, NULL);
    if (problem == NULL) {
        return 1;
    }

    pdhg_parameters_t params;
    set_default_parameters(&params);
    params.verbose = false;
    params.termination_criteria.eps_optimal_relative = 1e-6;
    params.termination_criteria.eps_feasible_relative = 1e-6;

    cupdlpx_result_t *result = solve_lp_problem(problem, &params);
    if (result == NULL) {
        lp_problem_free(problem);
        return 1;
    }

    printf("termination reason: %d\n", result->termination_reason);
    printf("objective: %.6f\n", result->primal_objective_value);
    for (int j = 0; j < result->num_variables; ++j) {
        printf("x[%d] = %.6f\n", j, result->primal_solution[j]);
    }

    cupdlpx_result_free(result);
    lp_problem_free(problem);
    return 0;
}
```

## Warm starts

To warm-start the solver, disable presolve and provide a primal vector,
a dual vector, or both:

```c
double x0[2] = {1.0, 2.0};
double y0[3] = {1.0, -1.0, 0.0};

params.presolve = false;
set_start_values(problem, x0, y0);
```

The function copies the supplied arrays. Passing `NULL` clears the
corresponding starting vector.

## Default parameters

Call `set_default_parameters` before changing fields:

```c
pdhg_parameters_t params;
set_default_parameters(&params);

params.termination_criteria.time_sec_limit = 300.0;
params.feasibility_polishing = true;
```

Passing `NULL` as the second argument of `solve_lp_problem` also selects all
defaults:

```c
cupdlpx_result_t *result = solve_lp_problem(problem, NULL);
```

See the [parameter reference](../reference/parameters.md#usage) for
the main fields and defaults.

## Result fields

`solve_lp_problem` returns a `cupdlpx_result_t *`. Its main fields are:

| Result | `cupdlpx_result_t` field |
| --- | --- |
| Termination status | `termination_reason` |
| Original dimensions | `num_variables`, `num_constraints`, `num_nonzeros` |
| Reduced dimensions | `num_reduced_variables`, `num_reduced_constraints`, `num_reduced_nonzeros` |
| Primal solution, dual solution, dual slacks | `primal_solution`, `dual_solution`, `reduced_cost` |
| Objectives | `primal_objective_value`, `dual_objective_value` |
| Gaps | `objective_gap`, `relative_objective_gap` |
| Primal residuals | `absolute_primal_residual`, `relative_primal_residual` |
| Dual residuals | `absolute_dual_residual`, `relative_dual_residual` |
| Iterations | `total_count`, `feasibility_iteration` |
| Timings | `cumulative_time_sec`, `rescaling_time_sec`, `presolve_time`, `feasibility_polishing_time` |
| Ray measures | `max_primal_ray_infeasibility`, `max_dual_ray_infeasibility`, `primal_ray_linear_objective`, `dual_ray_objective` |

The arrays remain valid until `cupdlpx_result_free(result)` is called.
Compare `termination_reason` with symbolic members of `termination_reason_t`
rather than their integer values. See [results and
status](../getting-started/results.md) for interpretation.

## Error handling

Creation and solve functions return `NULL` on failure. Check each returned
pointer before dereferencing it and release any objects already created on the
error path.
