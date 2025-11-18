# **cuPDLPx: A GPU-Accelerated First-Order LP Solver**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![GitHub release](https://img.shields.io/github/release/MIT-Lu-Lab/cuPDLPx.svg)](https://github.com/MIT-Lu-Lab/cuPDLPx/releases)
[![PyPI version](https://badge.fury.io/py/cupdlpx.svg)](https://pypi.org/project/cupdlpx/)
[![arXiv](https://img.shields.io/badge/arXiv-2407.16144-B31B1B.svg)](https://arxiv.org/abs/2407.16144)
[![arXiv](https://img.shields.io/badge/arXiv-2507.14051-B31B1B.svg)](https://arxiv.org/abs/2507.14051)

**cuPDLPx** is a GPU-accelerated linear programming solver based on a restarted Halpern PDHG method specifically tailored for GPU architectures. It incorporates  a Halpern update scheme, an adaptive restart scheme, and a PID-controlled primal weight, resulting in substantial empirical improvements over its predecessor, **[cuPDLP](https://github.com/jinwen-yang/cuPDLP.jl)**, on standard LP benchmark suites.

Our work is presented in two papers:

* **Computational Paper:** [cuPDLPx: A Further Enhanced GPU-Based First-Order Solver for Linear Programming](https://arxiv.org/abs/2507.14051) details the practical innovations that give **cuPDLPx** its performance edge.

* **Theoretical Paper:** [Restarted Halpern PDHG for Linear Programming](https://arxiv.org/pdf/2407.16144) provides the mathematical foundation for our method.

## Getting started
Follow these steps to build the solver and verify its installation.

### Requirements
- An NVIDIA GPU with CUDA support (≥12.4 required).
- A C toolchain (gcc) and the NVIDIA CUDA Compiler (nvcc).

### Build from Source
Clone the repository and compile the project using `CMake`.
```bash
git clone git@github.com:MIT-Lu-Lab/cuPDLPx.git
cd cuPDLPx
cmake -B build
cmake --build build --clean-first
```
This will create the solver binary at `./build/cupdlpx`.

### Verifying the Installation
Run a small test problem to confirm that the solver was built correctly.
```bash
# 1. Download a test instance from the MIPLIB library
wget -P test/ https://miplib.zib.de/WebData/instances/2club200v15p5scn.mps.gz

# 2. Solve the problem and write output to the current directory (.)
./build/cupdlpx test/2club200v15p5scn.mps.gz test/
```
If the solver runs and creates output files, your installation is successful.

## Usage Guide

### Command-Line Interface

The solver is invoked with the following syntax, specifying an input file and an output directory.

```bash
./build/cupdlpx [OPTIONS] <mps_file> <output_directory>
```

### Arguments
- `<mps_file>`: The path to the input linear programming problem. Both plain (`.mps`) and gzipped (`.mps.gz`) files are supported.
- `<output_directory>`: The directory where the output files will be saved.

### Solver Options

| Option | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `-h`, `--help` | `flag` | Display the help message. | N/A |
| `-v`, `--verbose` | `flag` | Enable verbose logging. | `false` |
| `--time_limit` | `double` | Time limit in seconds. | `3600.0` |
| `--iter_limit` | `int` | Iteration limit. | `2147483647` |
| `--eps_opt` | `double` | Relative optimality tolerance. | `1e-4` |
| `--eps_feas` | `double` | Relative feasibility tolerance. | `1e-4` |
| `--eps_infeas_detect` | `double` | Infeasibility detection tolerance. | `1e-10` |
| `-f`,`--feasibility_polishing` |`flag`   | Run the polishing loop           | `false` |
| `--eps_feas_polish` | `double` | Relative tolerance for polishing | `1e-6`  |

### Output Files
The solver generates three text files in the specified <output_directory>. The filenames are derived from the input file's basename. For an input `INSTANCE.mps.gz`, the output will be:
- `INSTANCE_summary.txt`: Contains solver statistics, timings, and termination information.
- `INSTANCE_primal_solution.txt`: The primal solution vector.
- `INSTANCE_dual_solution.txt`: The dual solution vector.

### Python Interface

In addition to the command-line, cuPDLPx provides a Python interface (`cupdlpx`)  
for building and solving LPs directly with NumPy and SciPy.  

- High-level, Pythonic API similar to commercial solvers.  
- Supports dense and sparse matrices.  
- Provides easy parameter management and solution attributes.  

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

See the [cupdlpx guide](https://github.com/MIT-Lu-Lab/cuPDLPx/tree/main/python/README.md) for full usage instructions, examples, and API details.

### C Interface

Besides the command-line and Python interfaces, cuPDLPx also provides a C API for directly solving LPs in memory. This is useful when integrating cuPDLPx into other C/C++ projects or when generating problems programmatically.

#### Functions and Parameters

The C API involves two main functions:

```c
lp_problem_t* create_lp_problem(
    const double* c,                   // objective vector (length n)
    const matrix_desc_t* A,            // constraint matrix (m×n)
    const double* con_lb,              // constraint lower bounds (length m)
    const double* con_ub,              // constraint upper bounds (length m)
    const double* var_lb,              // variable lower bounds (length n)
    const double* var_ub,              // variable upper bounds (length n)
    const double* c0                   // scalar objective offset
);

cupdlpx_result_t* solve_lp_problem(
    const lp_problem_t* prob,
    const pdhg_parameters_t* params    // NULL → use default parameters
);
```

`create_lp_problem` parameters:
- `objective_c`: Objective vector. If NULL, defaults to all zeros.
- `A_desc`: Matrix descriptor. Supports `matrix_dense`, `matrix_csr`, `matrix_csc`, `matrix_coo`.
- `con_lb`: Constraint lower bounds. If NULL, defaults to all -INFINITY.
- `con_ub`: Constraint upper bounds. If NULL, defaults to all +INFINITY.
- `var_lb`: Variable lower bounds. If NULL, defaults to all 0.0.
- `var_ub`: Variable upper bounds. If NULL, defaults to all +INFINITY.
- `objective_constant`: Scalar constant term added to the objective value. If NULL, defaults to 0.0.


`solve_lp_problem` parameters:
- `prob`: An LP problem built with `create_LP_problem`.
- `params`: Solver parameters. If `NULL`, the solver will use default parameters.

#### Feasibility Polishing (optional post-solve refinement)

After the normal PDHG solve you can ask cuPDLPx to run an extra feasibility-polishing phase that tries to improve primal/dual feasibility.

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
    A_desc.zero_tolerance = 0.0;
    A_desc.data.dense.A = &A[0][0];

    // Objective coefficients
    double c[2] = {1.0, 1.0};

    // Constraint bounds: l <= A x <= u
    double l[3] = {5.0, -INFINITY, -INFINITY};
    double u[3] = {5.0, 2.0, 8.0};

    // Build the problem
    lp_problem_t* prob = create_lp_problem(
        c, &A_desc, l, u, NULL, NULL, NULL);

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

## Reference
If you use cuPDLPx or the ideas in your work, please cite the source below.

```bibtex
@article{lu2025cupdlpx,
  title={cuPDLPx: A Further Enhanced GPU-Based First-Order Solver for Linear Programming},
  author={Lu, Haihao and Peng, Zedong and Yang, Jinwen},
  journal={arXiv preprint arXiv:2507.14051},
  year={2025}
}

@article{lu2024restarted,
  title={Restarted Halpern PDHG for linear programming},
  author={Lu, Haihao and Yang, Jinwen},
  journal={arXiv preprint arXiv:2407.16144},
  year={2024}
}
```

## License
cuPDLPx is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
