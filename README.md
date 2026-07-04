# **cuPDLPx: A GPU-Accelerated First-Order LP Solver**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![GitHub release](https://img.shields.io/github/release/MIT-Lu-Lab/cuPDLPx.svg)](https://github.com/MIT-Lu-Lab/cuPDLPx/releases)
[![PyPI version](https://badge.fury.io/py/cupdlpx.svg)](https://pypi.org/project/cupdlpx/)
[![arXiv](https://img.shields.io/badge/arXiv-2407.16144-B31B1B.svg)](https://arxiv.org/abs/2407.16144)
[![arXiv](https://img.shields.io/badge/arXiv-2507.14051-B31B1B.svg)](https://arxiv.org/abs/2507.14051)

**cuPDLPx** is a GPU-accelerated linear programming solver based on a restarted Halpern PDHG method specifically tailored for GPU architectures. It incorporates  a Halpern update scheme, an adaptive restart scheme, and a PID-controlled primal weight, resulting in substantial empirical improvements over its predecessor, **[cuPDLP](https://github.com/jinwen-yang/cuPDLP.jl)**, on standard LP benchmark suites.

cuPDLPx solves linear programs of the form
```math
\begin{aligned}
\min_{x} \quad & c^\top x \\
\text{s.t.} \quad & \ell_c \le Ax \le u_c, \\
                  & \ell_v \le x \le u_v.
\end{aligned}
```

Our work is presented in two papers:

* **Computational Paper:** [cuPDLPx: A Further Enhanced GPU-Based First-Order Solver for Linear Programming](https://arxiv.org/abs/2507.14051) details the practical innovations that give **cuPDLPx** its performance edge.

* **Theoretical Paper:** [Restarted Halpern PDHG for Linear Programming](https://arxiv.org/pdf/2407.16144) provides the mathematical foundation for our method.

## Installation

### Requirements
* **GPU:** NVIDIA GPU with CUDA 12.4+, or AMD GPU with ROCm 7.2+.
* **Build Tools:** CMake (≥ 3.20), GCC, and NVCC (CUDA) or hipcc (ROCm).

> **SpMV backend** is selected automatically at compile time based on cuSPARSE version:
> - `cusparseSpMV` — CUDA 12.4 – 13.2 (cuSPARSE < 12.8.1)
> - `cusparseSpMVOp` — CUDA 13.3+ (cuSPARSE ≥ 12.8.1)
>
> On AMD GPUs the solver uses the `hipsparseSpMV` backend via hipSPARSE.

### Build from Source
Clone the repository and compile the project using CMake.
```bash
git clone git@github.com:MIT-Lu-Lab/cuPDLPx.git
cd cuPDLPx
cmake -B build
cmake --build build --clean-first
```
This will create the solver binary at `./build/cupdlpx`.

#### Building for AMD GPUs (ROCm/HIP)
To target AMD GPUs, configure with `-DUSE_HIP=ON` and select the GPU
architecture with `-DCMAKE_HIP_ARCHITECTURES`. The CUDA sources are compiled
as HIP and the cuBLAS/cuSPARSE/CUB calls are mapped to hipBLAS/hipSPARSE/hipCUB.
```bash
cmake -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build --clean-first
```
Set `CMAKE_HIP_ARCHITECTURES` to match your GPU (for example `gfx90a` for
MI200, `gfx1100` for RDNA3 desktop, or `gfx1201` for RDNA4). If the ROCm
install is not on CMake's default search path, point `-DCMAKE_PREFIX_PATH` at
it (e.g. `/opt/rocm`) so `find_package` can locate hip, hipBLAS, hipSPARSE,
and hipCUB. The resulting `./build/cupdlpx` binary is used exactly as in the
CUDA build.

#### Verifying the Installation
Run a small test problem to confirm that the solver was built correctly.
```bash
# 1. Download a test instance from the MIPLIB library
wget -P test/ https://miplib.zib.de/WebData/instances/2club200v15p5scn.mps.gz

# 2. Solve the problem and write output to the current directory (.)
./build/cupdlpx test/2club200v15p5scn.mps.gz test/
```
If the solver runs and creates output files, your installation is successful.

### Python Package Installation
To use cuPDLPx in Python, you can install the pre-built package `cupdlpx` directly from PyPI:
```bash
pip install cupdlpx
```
Or build from source:
```
git clone https://github.com/MIT-Lu-Lab/cuPDLPx.git
cd cuPDLPx
pip install .
```

## Usage & Interfaces
### Command-line Interface

After building the project, the `./build/cupdlpx` binary can be invoked from the command line as follows:

```bash
./build/cupdlpx [OPTIONS] <mps_file> <output_directory>
```

#### Arguments
- `<mps_file>`: The path to the input linear programming problem. Both plain (`.mps`) and gzipped (`.mps.gz`) files are supported.
- `<output_directory>`: The directory where the output files will be saved.

#### Solver Options

| Option | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `-h`, `--help` | `flag` | Display the help message. | N/A |
| `-v`, `--verbose` | `flag` | Verbose logging (enabled by default). | `true` |
| `-q`, `--quiet` | `flag` | Disable verbose logging. | `false` |
| `--time_limit` | `double` | Time limit in seconds. | `3600.0` |
| `--iter_limit` | `int` | Iteration limit. | `2147483647` |
| `--opt_norm` | `string` | Norm for optimality criteria: `l2` or `linf` | `l2` |
| `--eps_opt` | `double` | Relative optimality tolerance. | `1e-4` |
| `--eps_feas` | `double` | Relative feasibility tolerance. | `1e-4` |
| `--l_inf_ruiz_iter` | `int` | Iterations for L-inf Ruiz rescaling| `10` |
| `--no_pock_chambolle` | `flag` | Disable Pock-Chambolle rescaling | `enabled` |
| `--pock_chambolle_alpha` | `float` | Value for Pock-Chambolle alpha | `1.0` |
| `--no_bound_obj_rescaling` | `flag` | Disable bound objective rescaling | `enabled` |
| `--eval_freq` | `int` | Termination evaluation frequency | `200` |
| `--sv_max_iter` | `int` | Max iterations for singular value estimation | `5000` |
| `--sv_tol` | `float` | Tolerance for singular value estimation | `1e-4` |
| `--no_presolve` | `flag` | Disable presolve | `enabled` |
| `-f`,`--feasibility_polishing` |`flag` | Run the polishing loop | `false` |
| `--eps_feas_polish` | `double` | Relative tolerance for polishing | `1e-6`  |

#### Output Files
The solver generates three text files in the specified <output_directory>. The filenames are derived from the input file's basename. For an input `INSTANCE.mps.gz`, the output will be:
```
<output_directory>/
├── INSTANCE_summary.txt          # Statistics, timings, and termination status
├── INSTANCE_primal_solution.txt  # Primal solution vector
└── INSTANCE_dual_solution.txt    # Dual solution vector
```

### Python Interface
The `cupdlpx` Python package supports building and solving LPs directly with `NumPy` and `SciPy`.
Documentation and examples are available in the [Python API Guide](python/README.md).

### Julia Interface
`CuPDLPx.jl` provides a `JuMP`/`MathOptInterface` wrapper for cuPDLPx, enabling cuPDLPx to be used as a solver backend for `JuMP` models. Documentation and examples are available at: [CuPDLPx.jl](https://github.com/MIT-Lu-Lab/CuPDLPx.jl).

### C Interface
The public C API is defined in header file [`include/cupdlpx.h`](include/cupdlpx.h). A detailed description with usage examples can be found in the [C API Guide](docs/C_API.md).

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
