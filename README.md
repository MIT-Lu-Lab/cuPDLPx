# **cuPDLPx: A GPU-Accelerated First-Order LP Solver**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![GitHub release](https://img.shields.io/github/release/MIT-Lu-Lab/cuPDLPx.svg)](https://github.com/MIT-Lu-Lab/cuPDLPx/releases)
[![PyPI version](https://badge.fury.io/py/cupdlpx.svg)](https://pypi.org/project/cupdlpx/)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://mit-lu-lab.github.io/cuPDLPx/)
[![arXiv](https://img.shields.io/badge/arXiv-2407.16144-B31B1B.svg)](https://arxiv.org/abs/2407.16144)
[![arXiv](https://img.shields.io/badge/arXiv-2507.14051-B31B1B.svg)](https://arxiv.org/abs/2507.14051)

**cuPDLPx** is a GPU-accelerated linear programming solver based on a restarted Halpern PDHG method specifically tailored for GPU architectures. It incorporates a Halpern update scheme, an adaptive restart scheme, and a PID-controlled primal weight, resulting in substantial empirical improvements over its predecessor, **[cuPDLP](https://github.com/jinwen-yang/cuPDLP.jl)**, on standard LP benchmark suites.

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

For installation instructions, examples, solver parameters, and algorithm details, see the [cuPDLPx documentation](https://mit-lu-lab.github.io/cuPDLPx/).

## Interfaces

| Interface | Description |
| --- | --- |
| [Command line](https://mit-lu-lab.github.io/cuPDLPx/guides/command-line/) | Solve MPS files from a shell. |
| [Python](https://mit-lu-lab.github.io/cuPDLPx/guides/python/) | Build and solve LPs with [NumPy](https://numpy.org/doc/stable/) and [SciPy](https://docs.scipy.org/doc/scipy/). |
| [Julia](https://mit-lu-lab.github.io/cuPDLPx/guides/julia/) | Use cuPDLPx through [JuMP](https://jump.dev/JuMP.jl/stable/) and [MathOptInterface](https://jump.dev/MathOptInterface.jl/stable/). |
| [C](https://mit-lu-lab.github.io/cuPDLPx/guides/c-api/) | Embed cuPDLPx in native applications. |

## References
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
