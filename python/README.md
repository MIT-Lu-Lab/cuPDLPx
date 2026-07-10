# **Python Interface for cuPDLPx**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![GitHub release](https://img.shields.io/github/release/MIT-Lu-Lab/cuPDLPx.svg)](https://github.com/MIT-Lu-Lab/cuPDLPx/releases)
[![PyPI version](https://badge.fury.io/py/cupdlpx.svg)](https://pypi.org/project/cupdlpx/)
[![arXiv](https://img.shields.io/badge/arXiv-2407.16144-B31B1B.svg)](https://arxiv.org/abs/2407.16144)
[![arXiv](https://img.shields.io/badge/arXiv-2507.14051-B31B1B.svg)](https://arxiv.org/abs/2507.14051)

There is the Python interface to **[`cuPDLPx`](../README.md)**, a GPU-accelerated first-order solver for large-scale linear programming (LP).  
It provides a high-level, Pythonic API for constructing, modifying, and solving LPs using NumPy and SciPy data structures.

## Installation

### Requirements
- Python ≥ 3.9  
- NumPy ≥ 1.21  
- SciPy ≥ 1.8  
- An NVIDIA GPU with CUDA support (≥12.4 required)  
- A C/C++ toolchain with GCC and NVCC  

> **SpMV backend** is selected automatically at compile time based on cuSPARSE version:
> - `cusparseSpMV` — CUDA 12.4 – 13.1 (cuSPARSE < 12.7.3)
> - `cusparseSpMVOp` — CUDA 13.1 Update 1+ (cuSPARSE ≥ 12.7.3)


### Install
Install from PyPI:

```bash
pip install cupdlpx
```

Or build from source:

```bash
git clone https://github.com/MIT-Lu-Lab/cuPDLPx.git
cd cuPDLPx
pip install .
```

## Quick Start

```python
import numpy as np
from cupdlpx import Model, PDLP

# Example: minimize c^T x
# subject to l <= A x <= u,  lb <= x <= ub
c  = np.array([1.0, 1.0])
A  = np.array([[1.0, 2.0],
               [0.0, 1.0],
               [3.0, 2.0]])
l  = np.array([5.0, -np.inf, -np.inf])
u  = np.array([5.0, 2.0, 8.0])
lb = np.zeros(2)   # x >= 0
ub = None          # no upper bound

# Create LP model
m = Model(objective_vector=c,
          constraint_matrix=A,
          constraint_lower_bound=l,
          constraint_upper_bound=u,
          variable_lower_bound=lb,
          variable_upper_bound=ub)

# Set model sense
m.ModelSense = PDLP.MAXIMIZE

# Parameters can be set in multiple ways
m.Params.TimeLimit = 60        # attribute style
m.setParam("FeasibilityTol", 1e-6)
m.setParams(OutputFlag=True, OptimalityTol=1e-8)

# Solve
m.optimize()

# Retrieve results
print("Status:", m.StatusName)
print("Objective:", m.ObjVal)
print("Primal solution:", m.X)
print("Dual solution:", m.Pi)
print("Reduced cost:", m.RC)
```

## Modeling

The `Model` class represents a linear programming problem of the form:

$$
\min c^\top x + c_0 \quad
\text{s.t.} \; \ell \le A x \le u, \quad
\text{lb} \le x \le \text{ub}.
$$

### Arguments

- **objective_vector** (`c`): Coefficients of the objective function.  
- **constraint_matrix** (`A`): Coefficient matrix for the constraints. Both dense (`numpy.ndarray`) and sparse (`scipy.sparse.csr_matrix`) inputs are supported. Internally stored in double precision (`float64`).
- **constraint_lower_bound** (`l`): Lower bounds for each constraint. Use `-np.inf` or `None` for no lower bound.
- **constraint_upper_bound** (`u`): Upper bounds for each constraint. Use `+np.inf` or `None` for no upper bound.
- **variable_lower_bound** (`lb`, optional): Lower bounds for the decision variables. Defaults to `-np.inf` for all variables if not provided.
- **variable_upper_bound** (`ub`, optional): Upper bounds for the decision variables. Defaults to `+np.inf` for all variables if not provided.
- **objective_constant** (`c0`, optional): Constant offset in the objective function. Defaults to `0.0`.

To initialize a linear programming problem, you need to provide the objective vector, constraint matrix, and bounds on both constraints and variables.  

```python
c  = np.array([1.0, 1.0])
A  = np.array([[1.0, 2.0],
               [0.0, 1.0],
               [3.0, 2.0]])
l  = np.array([5.0, -np.inf, -np.inf])
u  = np.array([5.0, 2.0, 8.0])
lb = np.zeros(2)   # x >= 0
ub = None          # no upper bound

# Create LP model
m = Model(objective_vector=c,
          constraint_matrix=A,
          constraint_lower_bound=l,
          constraint_upper_bound=u,
          variable_lower_bound=lb,
          variable_upper_bound=ub)
```

### Reading from MPS Files

A `Model` can also be created directly from an MPS file (plain or gzip-compressed) with `cupdlpx.read`:

```python
import cupdlpx

m = cupdlpx.read("problem.mps")      # or "problem.mps.gz"
m.optimize()
```

The objective sense declared in the file (`OBJSENSE` section) is applied to `ModelSense` automatically.


## Model Sense

By default, `cupdlpx` solves **minimization problems**.  

To switch between minimization and maximization, set the attribute `ModelSense`:

```python
# Set model sense
m.ModelSense = PDLP.MAXIMIZE
```

## Parameters

Solver parameters control termination criteria, logging, scaling, and restart behavior.

Below is a list of commonly used parameters, their internal keys, and descriptions.

| Alias | Internal Key | Type | Default | Description |
|---|---|---|---|---|
| `TimeLimit` | `time_sec_limit` | float | `3600.0` | Maximum wall-clock time in seconds. The solver terminates if the limit is reached. |
| `IterationLimit` | `iteration_limit` | int | `2147483647` | Maximum number of iterations. |
| `OutputFlag`, `LogToConsole` | `verbose` | bool | `False` | Enable (`True`) or disable (`False`) console logging output. |
| `TermCheckFreq` | `termination_evaluation_frequency` | int | `200` | Frequency (in iterations) at which termination conditions are evaluated. |
| `OptimalityNorm` | `optimality_norm` | string | `"l2"` | Norm for optimality criteria. Use `"l2"` for L2 norm or `"linf"` for infinity norm. |
| `OptimalityTol` | `eps_optimal_relative` | float | `1e-4` | Relative tolerance for optimality gap. Solver stops if the relative primal-dual gap ≤ this value. |
| `FeasibilityTol` | `eps_feasible_relative` | float | `1e-4` | Relative feasibility tolerance for primal/dual residuals. |
| `RuizIters` | `l_inf_ruiz_iterations` | int | `10` | Number of iterations for L∞ Ruiz scaling. Improves numerical conditioning. |
| `UsePCAlpha` | `has_pock_chambolle_alpha` | bool | `True` | Whether to use the Pock–Chambolle α step size adjustment. |
| `PCAlpha` | `pock_chambolle_alpha` | float | `1.0` | Value of the Pock–Chambolle α parameter. |
| `BoundObjRescaling` | `bound_objective_rescaling` | bool | `True` | Whether to rescale the objective vector during preprocessing. |
| `RestartArtificialThresh` | `artificial_restart_threshold` | float | `0.36` | Threshold for artificial restart. |
| `RestartSufficientReduction` | `sufficient_reduction_for_restart` | float | `0.2` | Sufficient reduction factor to justify a restart. |
| `RestartNecessaryReduction` | `necessary_reduction_for_restart` | float | `0.5` | Necessary reduction factor required for a restart. |
| `RestartKp` | `k_p` | float | `0.99` | Proportional coefficient for PID-controlled primal weight updates. |
| `ReflectionCoeff` | `reflection_coefficient` | float | `1.0` | Reflection coefficient. |
| `SVMaxIter` | `sv_max_iter` | int | 5000 | Maximum number of iterations for the power method |
| `SVTol`| `sv_tol` | float | `1e-4` | Termination tolerance for the power method |
| `Presolve`| `presolve` | bool | `True` | Whether to use presolve. |
| `FeasibilityPolishing` | `feasibility_polishing` | bool | `False` | Run feasibility polishing process.|
| `FeasibilityPolishingTol` | `eps_feas_polish_relative` | float | `1e-6` | Relative tolerance for primal/dual residual.  |

They can be set in multiple ways:

```python
# Method 1: single parameter
m.setParam("TimeLimit", 300)
m.setParam("FeasibilityTol", 1e-6)

# Method 2: multiple parameters
m.setParams(TimeLimit=300, FeasibilityTol=1e-6)

# Method 3: attribute-style access
m.Params.TimeLimit = 300
m.Params.FeasibilityTol = 1e-6

# Reset all parameters to backend defaults
m.resetParams()
```

## Solution Attributes

After calling `m.optimize()`, the solver stores results in a set of read-only attributes. `optimize()` returns the model itself, so chained access like `m.optimize().Status` is also supported. These attributes provide access to primal/dual solutions, objective values, residuals, and runtime statistics.

### Attribute Reference

| Attribute | Type | Description |
|---|---|---|
| `Status` | int | Integer termination status code; compare against `cupdlpx.PDLP` constants: `OPTIMAL=0`, `PRIMAL_INFEASIBLE=1`, `DUAL_INFEASIBLE=2`, `TIME_LIMIT=3`, `ITERATION_LIMIT=4`, `INFEASIBLE_OR_UNBOUNDED=5`, `FEAS_POLISH_SUCCESS=6`, `UNSPECIFIED=-1`. |
| `StatusName` | str | Human-readable status name, e.g. `"OPTIMAL"`, `"PRIMAL_INFEASIBLE"`. |
| `ObjVal` | float | Primal objective value at termination (sign-adjusted according to `ModelSense`). |
| `DualObj` | float | Dual objective value at termination. |
| `Gap` | float | Absolute primal-dual gap. |
| `RelGap` | float | Relative primal-dual gap. |
| `X` | numpy.ndarray | Primal solution vector \(x\). May be `None` if no feasible solution was found. |
| `Pi` | numpy.ndarray | Dual solution vector (Lagrange multipliers). |
| `RC` | numpy.ndarray | Reduced Cost vector. |
| `IterCount` | int | Number of iterations performed. |
| `Runtime` | float | Total wall-clock runtime in seconds. |
| `RescalingTime` | float | Time spent on preprocessing and rescaling (seconds). |
| `RelPrimalResidual` | float | Relative primal residual. |
| `RelDualResidual` | float | Relative dual residual. |
| `MaxPrimalRayInfeas` | float | Maximum primal ray infeasibility (indicator for infeasibility). |
| `MaxDualRayInfeas` | float | Maximum dual ray infeasibility. |
| `PrimalRayLinObj` | float | Linear objective value along a primal ray (used in infeasibility detection). |
| `DualRayObj` | float | Objective value along a dual ray (used in unboundedness detection). |

All solution-related information can then be queried directly from the `Model` object:

```python
m.optimize()

print("Status:", m.StatusName, "(code:", m.Status, ")")
print("Primal objective:", m.ObjVal)
print("Dual objective:", m.DualObj)
print("Relative gap:", m.RelGap)
print("Iterations:", m.IterCount, " Runtime (s):", m.Runtime)

# Access solutions
print("Primal solution:", m.X)
print("Dual solution:", m.Pi)
print("Reduced cost:", m.RC)

# Check residuals
print("Primal residual:", m.RelPrimalResidual)
print("Dual residual:", m.RelDualResidual)
```

## Warm Start

`cupdlpx` supports warm starting from user-provided primal and/or dual solutions.
This allows resuming from a previous iterate or reusing solutions from a related instance, often reducing iterations needed to reach optimality.

```python
# Warm starting solution
x_init = [1.0, 2.0]
pi_init = [1.0, -1.0, 0.0]

# Set warm start
m.setWarmStart(primal=x_init, dual=pi_init)

# Solve
m.optimize()
```

Both primal and dual arguments are optional. You may specify only one of them if desired:

```python
# Only provide primal start
m.setWarmStart(primal=x_init)

# Only provide dual start
m.setWarmStart(dual=pi_init)
```

If the warm-start vectors have incorrect dimensions, `setWarmStart` raises a `ValueError`. Omitting an argument leaves that side unchanged; passing `None` clears it.

To clear existing warm-start values:

```python
m.clearWarmStart()
```

or

```python
m.setWarmStart(primal=None, dual=None)
```
