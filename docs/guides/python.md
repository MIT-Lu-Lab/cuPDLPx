---
description: Install and use the cuPDLPx Python interface, including its model API, parameters, warm starts, and results.
---

# Python interface

The `cupdlpx` package solves LPs from [NumPy](https://numpy.org/doc/stable/)
arrays or [SciPy sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html)
matrices. The `Model` class stores problem data, solver settings, and results.
Solution vectors are returned as read-only NumPy arrays.

To solve an MPS file, see [Read MPS data](#read-mps-data).
To build an LP from arrays, see the [Quickstart](#quickstart).

## Software requirements

| Component | Requirement |
| --- | --- |
| Python | 3.8+ |
| NumPy | 1.21+ |
| SciPy | 1.8+ |

NumPy and SciPy are installed automatically with the package.

See [hardware requirements](../getting-started/index.md#hardware-requirements)
for supported GPUs and required CUDA or ROCm versions.

## Installation

Install from PyPI:

```bash
python -m pip install cupdlpx
```

Confirm that the package imports:

```bash
python -c "import cupdlpx; print(cupdlpx.__version__)"
```

To install the current source tree instead:

```bash
git clone https://github.com/MIT-Lu-Lab/cuPDLPx.git
cd cuPDLPx
python -m pip install .
```

Building from source requires CMake 3.21+ and a
[CUDA or ROCm toolchain](command-line.md#build-requirements). Pip may build
the package locally if no compatible wheel is available.

## Read MPS data

To solve an existing LP, replace `instance.mps` with the path to your file:

```python
import cupdlpx

model = cupdlpx.read("instance.mps")
model.optimize()

if model.Status == cupdlpx.PDLP.OPTIMAL:
    print("objective:", model.ObjVal)
    print("x:", model.X)
else:
    print("solver stopped with", model.StatusName)
```

`cupdlpx.read(filename)` accepts any path-like object and returns a new `Model`;
both `.mps` and `.mps.gz` files are supported. A missing path raises
`FileNotFoundError`; a file rejected by the native parser raises `RuntimeError`.

See [Solve and inspect results](#solve-and-inspect-results) for residuals,
timings, and other solution fields.

## Quickstart

This example solves

$$
\begin{aligned}
\operatorname*{minimize}_{x_1,x_2}\quad & x_1+x_2 \\
\text{subject to}\quad
& x_1 + 2x_2 = 5, \\
& x_2 \le 2, \\
& 3x_1 + 2x_2 \le 8, \\
& x_1,x_2 \ge 0.
\end{aligned}
$$

Every constraint is represented by a lower and an upper bound:

```python
import numpy as np
from cupdlpx import Model, PDLP

c = np.array([1.0, 1.0])
A = np.array([
    [1.0, 2.0],
    [0.0, 1.0],
    [3.0, 2.0],
])

model = Model(
    objective_vector=c,
    constraint_matrix=A,
    constraint_lower_bound=np.array([5.0, -np.inf, -np.inf]),
    constraint_upper_bound=np.array([5.0, 2.0, 8.0]),
    variable_lower_bound=np.zeros(2),
    variable_upper_bound=None,
)

model.Params.TimeLimit = 60
model.setParams(
    OptimalityTol=1e-6,
    FeasibilityTol=1e-6,
    OutputFlag=False,
)
model.optimize()

if model.Status == PDLP.OPTIMAL:
    print(f"objective: {model.ObjVal:.4f}")
    print("x:", np.round(model.X, 4))
else:
    print("solver stopped with", model.StatusName)
```

Output:

```text
objective: 3.0000
x: [1. 2.]
```

Passing `None` omits the corresponding lower or upper bounds.
Set `OutputFlag=True` to print the solver progress table; its fields are explained under [Log
interpretation](../getting-started/log-interpretation.md).

## Create a model

```python
from cupdlpx import Model

model = Model(
    objective_vector=c,
    constraint_matrix=A,
    constraint_lower_bound=l,
    constraint_upper_bound=u,
    variable_lower_bound=lb,
    variable_upper_bound=ub,
    objective_constant=0.0,
)
```

`constraint_matrix` accepts a two-dimensional `numpy.ndarray` or a SciPy
sparse matrix.

<div class="model-arguments" markdown>

| Argument | Type | Shape | Default | Description |
| --- | --- | --- | --- | --- |
| `objective_vector` | array-like | `(n,)` | required | Finite objective coefficients $c$. |
| `constraint_matrix` | NumPy array or SciPy sparse matrix | `(m,n)` | required | Finite constraint matrix $A$. |
| `constraint_lower_bound` | array-like or `None` | `(m,)` | required | Lower bounds $\ell_c$; `None` means all $-\infty$. |
| `constraint_upper_bound` | array-like or `None` | `(m,)` | required | Upper bounds $u_c$; `None` means all $+\infty$. |
| `variable_lower_bound` | array-like or `None` | `(n,)` | `None` | Lower bounds $\ell_v$; `None` means all $-\infty$. |
| `variable_upper_bound` | array-like or `None` | `(n,)` | `None` | Upper bounds $u_v$; `None` means all $+\infty$. |
| `objective_constant` | real number | scalar | `0.0` | Finite objective offset $c_0$. |

</div>

Dense vectors and matrices are copied into C-contiguous `float64` arrays.
Sparse matrices are copied to CSR with `float64` values and 32-bit indices.
The stored arrays are read-only.

Construction raises `TypeError` for unsupported input types, `ValueError` for
invalid dimensions, bounds, or nonfinite data, and `OverflowError` when a
sparse matrix cannot use 32-bit CSR indices.

Models minimize by default. To maximize:

```python
from cupdlpx import PDLP

model.ModelSense = PDLP.MAXIMIZE
```

## Solve and inspect results

Call `optimize()` to solve the model, then read its status and solution attributes:

```python
model.optimize()
status = model.StatusName
```

| Property | Type after a solve | Description |
| --- | --- | --- |
| `Status` | `int` | Termination code; compare with a constant in `cupdlpx.PDLP`. |
| `StatusName` | `str` | Symbolic termination name. |
| `X` | `numpy.ndarray`<br>shape `(n,)` | Primal solution. |
| `Pi` | `numpy.ndarray`<br>shape `(m,)` | Dual solution. |
| `RC` | `numpy.ndarray`<br>shape `(n,)` | Dual slacks. |
| `ObjVal` | `float` | Primal objective value. |
| `DualObj` | `float` | Dual objective value. |
| `Gap` | `float` | Absolute primal–dual objective gap. |
| `RelGap` | `float` | Relative primal–dual objective gap. |
| `RelPrimalResidual` | `float` | Relative primal feasibility residual. |
| `RelDualResidual` | `float` | Relative dual feasibility residual. |
| `IterCount` | `int` | Total iteration count. |
| `Runtime` | `float` | Solve time in seconds. |
| `RescalingTime` | `float` | Preconditioning time in seconds. |
| `MaxPrimalRayInfeas` | `float` | Maximum primal-ray infeasibility. |
| `MaxDualRayInfeas` | `float` | Maximum dual-ray infeasibility. |
| `PrimalRayLinObj` | `float` | Primal-ray linear objective. |
| `DualRayObj` | `float` | Dual-ray objective. |

Solution attributes are `None` before a solve and are cleared whenever model
data changes. `PrimalInfeas` and `DualInfeas` are aliases for
`RelPrimalResidual` and `RelDualResidual`. Invalid model bounds or objective
sense raise `ValueError`; native solver failures raise `RuntimeError`. See
[results and status](../getting-started/results.md) for status interpretation.

## Set parameters

```python
model.setParam(name, value)
model.getParam(name)
model.setParams(**kwargs)
model.resetParams()
```

Each of these forms sets the same time limit:

```python
model.Params.TimeLimit = 120
model.setParam("TimeLimit", 120)
model.setParams(TimeLimit=120)
```

To set several parameters at once, use
`model.setParams(TimeLimit=120, OptimalityTol=1e-6)`.

C parameter names are also accepted:

```python
model.setParam("time_sec_limit", 120)
```

Read or reset parameters with:

```python
print(model.getParam("TimeLimit"))
print(dict(model.Params.items()))
model.resetParams()
```

`setParams` validates every update before applying any of them. `model.Params`
is a live parameter view that supports attribute and item access, iteration,
and `keys()`, `values()`, and `items()`.

An unknown name raises `KeyError`, a value of the wrong type raises
`TypeError`, and a value outside its allowed range raises `ValueError`. See the
[parameter reference](../reference/parameters.md).

## Update model data

Update model data by assigning properties or calling setter methods.
Arrays returned by model properties are read-only; assign a new array
instead of modifying individual entries in place:

```python
model.c = new_objective
model.A = new_constraint_matrix
model.lb = new_variable_lower_bounds
model.ub = new_variable_upper_bounds

model.setObjectiveConstant(2.0)
model.setConstraintLowerBound(new_l)
model.setConstraintUpperBound(new_u)
```

| Property | Type | Validating setter |
| --- | --- | --- |
| `c` | `numpy.ndarray`<br>shape `(n,)` | `setObjectiveVector(c)` |
| `c0` | `float` | `setObjectiveConstant(c0)` |
| `A` | NumPy array or CSR matrix<br>shape `(m,n)` | `setConstraintMatrix(A)` |
| `constr_lb` | `numpy.ndarray` or `None`<br>shape `(m,)` | `setConstraintLowerBound(lower)` |
| `constr_ub` | `numpy.ndarray` or `None`<br>shape `(m,)` | `setConstraintUpperBound(upper)` |
| `lb` | `numpy.ndarray` or `None`<br>shape `(n,)` | `setVariableLowerBound(lower)` |
| `ub` | `numpy.ndarray` or `None`<br>shape `(n,)` | `setVariableUpperBound(upper)` |
| `ModelSense` | `int` | Assign `PDLP.MINIMIZE` or `PDLP.MAXIMIZE`. |
| `num_vars` | `int` | — |
| `num_constrs` | `int` | — |

Changing model data clears the cached solution. The number of variables is
fixed at construction. To change the number of constraints, clear the
constraint bounds, replace `A`, and set new bounds with matching lengths.

## Warm start

```python
model.setWarmStart(primal=..., dual=...)
model.clearWarmStart()
```

To warm-start the solver, disable presolve and provide a primal vector,
a dual vector, or both:

```python
model.setParam("Presolve", False)
model.setWarmStart(primal=x0, dual=y0)
model.optimize()
```

Omitting an argument preserves its previous starting vector. Passing `None`
clears it:

```python
model.setWarmStart(primal=None)
model.clearWarmStart()
```

The primal vector must have length `model.num_vars`; the dual vector must have
length `model.num_constrs`. A length mismatch or nonfinite value raises
`ValueError`.
