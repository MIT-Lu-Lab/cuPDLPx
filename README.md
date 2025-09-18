# **cuPDLPx: A GPU-Accelerated First-Order LP Solver**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![GitHub release](https://img.shields.io/github/release/MIT-Lu-Lab/cuPDLPx.svg)](https://github.com/MIT-Lu-Lab/cuPDLPx/releases)
[![arXiv](https://img.shields.io/badge/arXiv-2407.16144-B31B1B.svg)](https://arxiv.org/abs/2407.16144)
[![arXiv](https://img.shields.io/badge/arXiv-2507.14051-B31B1B.svg)](https://arxiv.org/abs/2507.14051)

**cuPDLPx** is a GPU-accelerated linear programming solver based on a restarted Halpern PDHG method specifically tailored for GPU architectures. It incorporates  a Halpern update scheme, an adaptive restart scheme, and a PID-controlled primal weight, resulting in substantial empirical improvements over its predecessor, **[cuPDLP](https://github.com/jinwen-yang/cuPDLP.jl)**, on standard LP benchmark suites.

Our work is presented in two papers:

* **Computational Paper:** [cuPDLPx: A Further Enhanced GPU-Based First-Order Solver for Linear Programming](https://arxiv.org/abs/2507.14051) details the practical innovations that give **cuPDLPx** its performance edge.

* **Theoretical Paper:** [Restarted Halpern PDHG for Linear Programming](https://arxiv.org/pdf/2407.16144) provides the mathematical foundation for our method.

## Getting started
Follow these steps to build the solver and verify its installation.

### Requirements
- An NVIDIA GPU with CUDA support (CUDA 12.x is recommended).
- A C toolchain (gcc) and the NVIDIA CUDA Compiler (nvcc).

### Build from Source
Clone the repository and compile the project using `make`.
```bash
make clean
make build
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

### Output Files
The solver generates three text files in the specified <output_directory>. The filenames are derived from the input file's basename. For an input `INSTANCE.mps.gz`, the output will be:
- `INSTANCE_summary.txt`: Contains solver statistics, timings, and termination information.
- `INSTANCE_primal_solution.txt`: The primal solution vector.
- `INSTANCE_dual_solution.txt`: The dual solution vector.

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
