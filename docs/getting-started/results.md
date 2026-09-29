---
description: Solution fields, termination statuses, and infeasibility information returned by cuPDLPx.
---

# Results and status

cuPDLPx reports a termination status, primal and dual vectors, residuals,
objective values, and solve statistics. Check the status before using the
returned vectors. Field names differ across interfaces.

## Termination status

<div class="status-guide" markdown>

| Status | Meaning | Next step |
| --- | --- | --- |
| Optimal | Presolve solved the LP, or the main iteration met the feasibility and optimality tolerances. | Check the final residuals and gap, especially if polishing is enabled. |
| Primal infeasible | The solver detected primal infeasibility. | Review the constraints and bounds, along with any returned infeasibility information. |
| Dual infeasible | The solver detected dual infeasibility, which can indicate an unbounded primal problem. | Review any returned infeasibility information and check for unbounded directions. |
| Infeasible or unbounded | Presolve or the solver could not distinguish infeasibility from unboundedness. | Inspect the model and log for more detail. |
| Time limit | The wall-clock limit was reached. | Check residuals and the gap; increase the time limit if needed. |
| Iteration limit | The iteration limit was reached. | Check residuals and the gap; increase the iteration limit if needed. |
| Feasibility polishing succeeded | A feasibility-polishing phase reached the polishing tolerance. | Check the final residuals and gap; polishing alone does not establish optimality. |
| Unspecified | No more specific termination reason is available. | Inspect the log and input data. |

</div>

[Feasibility polishing](../algorithm/feasibility-polishing.md) preserves the
main solve's termination status. The final gap can exceed the optimality
tolerance even when the status is `OPTIMAL`.

!!! warning "Always inspect the status"

    Reaching a time or iteration limit does not establish feasibility or
    optimality. Check the residuals and primal–dual gap before using the
    returned iterate.

## Solution and quality measures

The result contains:

| Result | Description |
| --- | --- |
| Primal solution $x$ | One value for each variable. |
| Dual solution $y$ | One multiplier for each constraint. |
| Dual slacks $r$ | One dual slack for each variable. |
| Primal objective | $c^\top x+c_0$ in the original objective sense. |
| Dual objective | Reported dual objective value; a valid bound requires dual feasibility. |
| Objective gap | Absolute and relative primal–dual gaps. |
| Primal residual | Absolute and relative violation of the primal constraints. |
| Dual residual | Absolute and relative violation of $c-A^\top y-r=0$. |
| Infeasibility information | Primal- and dual-ray quality measures when applicable. |
| Work statistics | Iteration counts and phase timings. |

With presolve enabled, the reported residuals and objective gap refer to
the [presolved model](../algorithm/presolve.md#presolve), while the solution
vectors are recovered for the original model.

See [Python results](../guides/python.md#solve-and-inspect-results), the [Julia
interface](../guides/julia.md), [C result fields](../guides/c-api.md#result-fields),
and [command-line output files](../guides/command-line.md#output-files)
for field names and access methods.

## Status constants

Use symbolic constants to compare statuses; Python and C use different
integer values. Julia maps the native termination reason to a
MathOptInterface status; see the [Julia interface](../guides/julia.md).

<div class="status-reference" markdown>

=== "Python"

    | Status | Constant |
    | --- | --- |
    | Optimal | `PDLP.OPTIMAL` |
    | Primal infeasible | `PDLP.PRIMAL_INFEASIBLE` |
    | Dual infeasible | `PDLP.DUAL_INFEASIBLE` |
    | Infeasible or unbounded | `PDLP.INFEASIBLE_OR_UNBOUNDED` |
    | Time limit | `PDLP.TIME_LIMIT` |
    | Iteration limit | `PDLP.ITERATION_LIMIT` |
    | Feasibility polishing succeeded | `PDLP.FEAS_POLISH_SUCCESS` |
    | Unspecified | `PDLP.UNSPECIFIED` |

=== "C"

    | Status | Constant |
    | --- | --- |
    | Optimal | `TERMINATION_REASON_OPTIMAL` |
    | Primal infeasible | `TERMINATION_REASON_PRIMAL_INFEASIBLE` |
    | Dual infeasible | `TERMINATION_REASON_DUAL_INFEASIBLE` |
    | Infeasible or unbounded | `TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED` |
    | Time limit | `TERMINATION_REASON_TIME_LIMIT` |
    | Iteration limit | `TERMINATION_REASON_ITERATION_LIMIT` |
    | Feasibility polishing succeeded | `TERMINATION_REASON_FEAS_POLISH_SUCCESS` |
    | Unspecified | `TERMINATION_REASON_UNSPECIFIED` |

</div>
