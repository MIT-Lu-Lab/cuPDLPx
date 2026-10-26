---
description: Interpret cuPDLPx progress tables, residuals, timings, and feasibility-polishing logs.
---

# Log interpretation

All interfaces use the same solver log. Logging is controlled by
`OutputFlag` in Python, `verbose` in Julia and C, and `--verbose` or
`--quiet` on the command line.

The example uses CUDA with feasibility polishing disabled. Timings depend
on the problem, GPU, and system configuration.

??? example "Complete solver log"

    ```text
    ---------------------------------------------------------------------------------------
                                        cuPDLPx v0.3.0
                            A GPU-Accelerated First-Order LP Solver
                   (c) Haihao Lu, Massachusetts Institute of Technology, 2025
    ---------------------------------------------------------------------------------------
    Problem: 17013 rows, 200 columns, 104811 nonzeros
    Settings:
      iter_limit         : 2147483647
      time_limit         : 3600.00 sec
      eps_opt            : 1.0e-04
      eps_feas           : 1.0e-04
      spmv_backend       : cusparseSpMVOp (auto)

    Running presolver (PSLP v0.0.8)...
      status          : REDUCED
      presolve time   : 0.00853 sec
      reduced problem : 16997 rows, 200 columns, 104747 nonzeros

    Preconditioning
      Ruiz scaling (10 iterations)
      Pock-Chambolle scaling (alpha=1.0000)
      Bound-objective scaling

    ---------------------------------------------------------------------------------------
       runtime     |     objective      |   absolute residuals    |   relative residuals
      iter   time  |  pr obj    du obj  |  pr res  du res   gap   |  pr res  du res   gap
    ---------------------------------------------------------------------------------------
         0 0.0e+00 |  0.0e+00   0.0e+00 | 0.0e+00 0.0e+00 0.0e+00 | 0.0e+00 0.0e+00 0.0e+00
       200 0.0e+00 | -1.3e+02  -1.2e+02 | 1.1e+00 2.5e-01 9.7e+00 | 8.5e-03 1.7e-02 3.9e-02
       400 8.7e-03 | -1.2e+02  -1.2e+02 | 2.3e-01 7.6e-02 9.2e-01 | 1.7e-03 5.0e-03 3.8e-03
       600 1.2e-02 | -1.2e+02  -1.2e+02 | 4.6e-02 8.5e-02 5.2e-03 | 3.5e-04 5.6e-03 2.1e-05
       800 1.5e-02 | -1.2e+02  -1.2e+02 | 2.5e-02 4.7e-02 3.3e-02 | 1.9e-04 3.1e-03 1.4e-04
      1000 1.8e-02 | -1.2e+02  -1.2e+02 | 1.2e-02 3.4e-02 7.3e-03 | 8.8e-05 2.2e-03 3.0e-05
      1200 2.1e-02 | -1.2e+02  -1.2e+02 | 4.3e-03 2.5e-02 7.6e-04 | 3.2e-05 1.6e-03 3.1e-06
      1400 2.4e-02 | -1.2e+02  -1.2e+02 | 3.1e-03 1.9e-02 1.4e-03 | 2.4e-05 1.3e-03 5.9e-06
      1600 2.7e-02 | -1.2e+02  -1.2e+02 | 3.2e-04 1.8e-02 2.6e-03 | 2.4e-06 1.2e-03 1.1e-05
      1800 3.0e-02 | -1.2e+02  -1.2e+02 | 2.8e-04 1.2e-02 6.5e-03 | 2.1e-06 8.0e-04 2.7e-05
      2000 3.3e-02 | -1.2e+02  -1.2e+02 | 2.1e-04 1.3e-02 3.5e-03 | 1.6e-06 8.3e-04 1.4e-05
      2200 3.6e-02 | -1.2e+02  -1.2e+02 | 3.0e-03 1.1e-02 5.8e-03 | 2.3e-05 7.2e-04 2.4e-05
      2400 4.0e-02 | -1.2e+02  -1.2e+02 | 3.1e-03 5.9e-03 1.9e-03 | 2.4e-05 3.9e-04 7.8e-06
      2600 4.3e-02 | -1.2e+02  -1.2e+02 | 2.2e-03 3.1e-03 1.4e-03 | 1.7e-05 2.1e-04 5.7e-06
      2800 4.6e-02 | -1.2e+02  -1.2e+02 | 1.0e-03 2.1e-03 1.7e-03 | 7.6e-06 1.4e-04 6.8e-06
      3000 4.8e-02 | -1.2e+02  -1.2e+02 | 6.4e-04 3.6e-04 4.6e-04 | 4.9e-06 2.4e-05 1.9e-06
    ---------------------------------------------------------------------------------------
    Solution Summary
      Status                 : OPTIMAL
      Presolve time          : 0.00853 sec
      Precondition time      : 0.003735 sec
      Solve time             : 0.0517 sec
      Iterations             : 3000
      Primal objective       : -121.2216698
      Dual objective         : -121.2221271
      Objective gap          : 1.879e-06
      Primal infeas          : 4.889e-06
      Dual infeas            : 2.399e-05
    ```

## Problem and settings

<div class="cupdlpx-log" markdown>

```text
---------------------------------------------------------------------------------------
                                    cuPDLPx v0.3.0
                        A GPU-Accelerated First-Order LP Solver
               (c) Haihao Lu, Massachusetts Institute of Technology, 2025
---------------------------------------------------------------------------------------
Problem: 17013 rows, 200 columns, 104811 nonzeros
Settings:
  iter_limit         : 2147483647
  time_limit         : 3600.00 sec
  eps_opt            : 1.0e-04
  eps_feas           : 1.0e-04
  spmv_backend       : cusparseSpMVOp (auto)
```

</div>

The problem dimensions, four core settings, and selected sparse matrix–vector
backend are always shown. Additional settings appear when they differ from
their defaults. See [Parameters](../reference/parameters.md) for the
corresponding interface names.

## Presolve and preconditioning

<div class="cupdlpx-log" markdown>

```text
Running presolver (PSLP v0.0.8)...
  status          : REDUCED
  presolve time   : 0.00853 sec
  reduced problem : 16997 rows, 200 columns, 104747 nonzeros

Preconditioning
  Ruiz scaling (10 iterations)
  Pock-Chambolle scaling (alpha=1.0000)
  Bound-objective scaling
```

</div>

`REDUCED` means that presolve produced an equivalent smaller LP. Postsolve
recovers the solution vectors for the original LP. The reported residuals
and gap are computed on the presolved LP.

Preconditioning rescales the LP before the main iteration. Its work is not
included in the PDHG iteration count.

## Progress table

For this example, the iteration log is:

<div class="cupdlpx-log" markdown>

```text
---------------------------------------------------------------------------------------
   runtime     |     objective      |   absolute residuals    |   relative residuals
  iter   time  |  pr obj    du obj  |  pr res  du res   gap   |  pr res  du res   gap
---------------------------------------------------------------------------------------
     0 0.0e+00 |  0.0e+00   0.0e+00 | 0.0e+00 0.0e+00 0.0e+00 | 0.0e+00 0.0e+00 0.0e+00
   200 0.0e+00 | -1.3e+02  -1.2e+02 | 1.1e+00 2.5e-01 9.7e+00 | 8.5e-03 1.7e-02 3.9e-02
   400 8.7e-03 | -1.2e+02  -1.2e+02 | 2.3e-01 7.6e-02 9.2e-01 | 1.7e-03 5.0e-03 3.8e-03
   600 1.2e-02 | -1.2e+02  -1.2e+02 | 4.6e-02 8.5e-02 5.2e-03 | 3.5e-04 5.6e-03 2.1e-05
   800 1.5e-02 | -1.2e+02  -1.2e+02 | 2.5e-02 4.7e-02 3.3e-02 | 1.9e-04 3.1e-03 1.4e-04
  1000 1.8e-02 | -1.2e+02  -1.2e+02 | 1.2e-02 3.4e-02 7.3e-03 | 8.8e-05 2.2e-03 3.0e-05
  1200 2.1e-02 | -1.2e+02  -1.2e+02 | 4.3e-03 2.5e-02 7.6e-04 | 3.2e-05 1.6e-03 3.1e-06
  1400 2.4e-02 | -1.2e+02  -1.2e+02 | 3.1e-03 1.9e-02 1.4e-03 | 2.4e-05 1.3e-03 5.9e-06
  1600 2.7e-02 | -1.2e+02  -1.2e+02 | 3.2e-04 1.8e-02 2.6e-03 | 2.4e-06 1.2e-03 1.1e-05
  1800 3.0e-02 | -1.2e+02  -1.2e+02 | 2.8e-04 1.2e-02 6.5e-03 | 2.1e-06 8.0e-04 2.7e-05
  2000 3.3e-02 | -1.2e+02  -1.2e+02 | 2.1e-04 1.3e-02 3.5e-03 | 1.6e-06 8.3e-04 1.4e-05
  2200 3.6e-02 | -1.2e+02  -1.2e+02 | 3.0e-03 1.1e-02 5.8e-03 | 2.3e-05 7.2e-04 2.4e-05
  2400 4.0e-02 | -1.2e+02  -1.2e+02 | 3.1e-03 5.9e-03 1.9e-03 | 2.4e-05 3.9e-04 7.8e-06
  2600 4.3e-02 | -1.2e+02  -1.2e+02 | 2.2e-03 3.1e-03 1.4e-03 | 1.7e-05 2.1e-04 5.7e-06
  2800 4.6e-02 | -1.2e+02  -1.2e+02 | 1.0e-03 2.1e-03 1.7e-03 | 7.6e-06 1.4e-04 6.8e-06
  3000 4.8e-02 | -1.2e+02  -1.2e+02 | 6.4e-04 3.6e-04 4.6e-04 | 4.9e-06 2.4e-05 1.9e-06
---------------------------------------------------------------------------------------
```

</div>

| Column | Meaning |
| --- | --- |
| `iter` | Total PDHG iterations completed. |
| `time` | Elapsed main-solve time in seconds. |
| `pr obj` | Primal objective value. |
| `du obj` | Dual objective value. |
| `pr res` | Primal feasibility residual. |
| `du res` | Dual feasibility residual. |
| `gap` | Primal–dual objective gap. |

The log reports absolute and relative primal residuals, dual residuals, and
gaps. The optimality test requires the relative residuals to be below
`FeasibilityTol` and the relative gap to be below `OptimalityTol`.
See [Termination criteria](../algorithm/termination.md#optimality-conditions)
for the formulas.

`TermCheckFreq` sets the termination-check interval, `200` in this example.
Progress is logged at these checks, with fewer rows printed as the iteration
count grows. Restarts are checked at the same interval and have no separate
log marker. Residuals and objective values need not decrease monotonically.

## Feasibility polishing

When feasibility polishing runs, the log shows separate primal and dual
tables with objective values and absolute and relative residuals. The
polishing summary reports each phase's status, iteration count, and time.

Polishing can reduce feasibility residuals while increasing the primal–dual
gap. The main solve's termination status is unchanged. See
[Feasibility polishing](../algorithm/feasibility-polishing.md).

## Solution summary

Read the final summary in this order:

1. **Status:** check why the solver stopped. `OPTIMAL` means it reached the
   requested accuracy; a time or iteration limit does not establish that.
2. **Primal and dual residuals:** compare `Primal infeas` and `Dual infeas`
   with `FeasibilityTol`.
3. **Objective gap:** compare `Objective gap` with `OptimalityTol`.

For example:

<div class="cupdlpx-log" markdown>

```text
Solution Summary
  Status                 : OPTIMAL
  Presolve time          : 0.00853 sec
  Precondition time      : 0.003735 sec
  Solve time             : 0.0517 sec
  Iterations             : 3000
  Primal objective       : -121.2216698
  Dual objective         : -121.2221271
  Objective gap          : 1.879e-06
  Primal infeas          : 4.889e-06
  Dual infeas            : 2.399e-05
```

</div>

Both residuals and the relative gap are below the requested `1e-4` tolerances.

The fields have the following meanings:

| Field | Meaning |
| --- | --- |
| `Status` | Termination reason, such as `OPTIMAL`, `TIME_LIMIT`, or `ITERATION_LIMIT`. |
| `Presolve time` | Time spent in PSLP, shown when presolve is enabled. |
| `Precondition time` | Time spent scaling the LP and preparing the scaled problem. |
| `Solve time` | Time spent in the main iteration. |
| `Iterations` | Total main-solve iteration count. |
| `Primal objective`, `Dual objective` | Final objective values, including the objective constant and original objective sense. |
| `Objective gap` | Final relative primal–dual gap on the presolved LP when presolve is enabled. |
| `Primal infeas`, `Dual infeas` | Final relative residuals on the presolved LP when presolve is enabled. |

`Primal infeas` and `Dual infeas` are residual magnitudes. Use `Status` to
determine whether the solver detected infeasibility. See
[Results and status](results.md) for termination reasons and result fields.
