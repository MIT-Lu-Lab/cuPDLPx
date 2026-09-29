---
description: Recommended papers, BibTeX entries, and related projects for cuPDLPx.
---

# Citation

If you use cuPDLPx in published work, please cite the computational paper. If
your work also relies on the restarted Halpern PDHG method or its convergence
theory, please cite the theoretical paper as well.

## Computational paper

Algorithmic enhancements, GPU implementation, and numerical results are
described in this paper:

```bibtex
@article{lu2025cupdlpx,
  title   = {{cuPDLPx}: A Further Enhanced GPU-Based First-Order Solver
             for Linear Programming},
  author  = {Lu, Haihao and Peng, Zedong and Yang, Jinwen},
  journal = {arXiv preprint arXiv:2507.14051},
  year    = {2025},
  url     = {https://arxiv.org/abs/2507.14051}
}
```

## Theoretical paper

Restarted Halpern PDHG and its reflected variant are developed in this paper:

```bibtex
@article{lu2024restarted,
  title   = {Restarted Halpern PDHG for Linear Programming},
  author  = {Lu, Haihao and Yang, Jinwen},
  journal = {arXiv preprint arXiv:2407.16144},
  year    = {2024},
  url     = {https://arxiv.org/abs/2407.16144}
}
```

## Background

### Papers

- David Applegate et al., [*Practical Large-Scale Linear Programming Using
  Primal-Dual Hybrid
  Gradient*](https://proceedings.neurips.cc/paper/2021/hash/a8fbbd3b11424ce032ba813493d95ad7-Abstract.html),
  NeurIPS 2021.
- David Applegate et al., [*PDLP: A Practical First-Order Method for
  Large-Scale Linear Programming*](https://arxiv.org/abs/2501.07018), 2025.
- Haihao Lu and Jinwen Yang, [*cuPDLP.jl: A GPU Implementation of Restarted
  Primal-Dual Hybrid Gradient for Linear Programming in
  Julia*](https://arxiv.org/abs/2311.12180), 2023.
- Haihao Lu et al., [*cuPDLP-C: A Strengthened Implementation of cuPDLP for
  Linear Programming by C language*](https://arxiv.org/abs/2312.14832), 2023.
- Haihao Lu and Jinwen Yang, [*An Overview of GPU-based First-Order Methods
  for Linear Programming and Extensions*](https://arxiv.org/abs/2506.02174),
  2025.
- Daniel Cederberg and Stephen Boyd, [*Presolving for GPU-Accelerated
  First-Order LP Solvers*](https://arxiv.org/abs/2604.23951), 2026.

### Blogs

- Cara Touretzky, Robert Luce, and David Torres Sanchez, [*Using GPUs to Solve
  LPs: What's in It for
  Me?*](https://www.gurobi.com/resources/blog/using-gpus-to-solve-lps-what-s-in-it-for-me),
  Gurobi blog, 2025.
- Cara Touretzky and Robert Luce, [*Introducing Gurobi's First GPU-Accelerated
  Solver: Test a Beta Version of Gurobi's PDHG Implementation on NVIDIA's GPU
  Hardware*](https://www.gurobi.com/resources/blog/introducing-gurobi-s-first-gpu-accelerated-solver-test-a-beta-version-of-gurobi-s-pdhg-implementation-on-nvidia-s-gpu-hardware),
  Gurobi blog, 2025.
- Imre Pólik, [*GPU Acceleration of the Hybrid Gradient Algorithm in FICO
  Xpress*](https://www.fico.com/blogs/gpu-acceleration-hybrid-gradient-algorithm-fico-xpress),
  FICO blog, 2025.
- Nicolas Blin, [*Accelerate Large Linear Programming Problems with NVIDIA
  cuOpt*](https://developer.nvidia.com/blog/accelerate-large-linear-programming-problems-with-nvidia-cuopt/),
  NVIDIA Technical Blog, 2024.
- Artelys, [*Artelys Knitro 15.0: New Tools for Your Large-Scale
  Models*](https://www.artelys.com/news/knitro-15-0-new-tools-for-your-large-scale-models/),
  Artelys news, 2025.
- [Scaling up linear programming with PDLP](https://research.google/blog/scaling-up-linear-programming-with-pdlp/),
  a Google Research article by Haihao Lu and David Applegate.
- [Mathematical background for PDLP](https://developers.google.com/optimization/lp/pdlp_math),
  the OR-Tools reference for PDLP formulations, residuals, rescaling, and
  infeasibility certificates.

## Related projects

### Open-source solvers

- [cuPDLP.jl](https://github.com/jinwen-yang/cuPDLP.jl), the earlier Julia GPU solver.
- [cuPDLP-C](https://github.com/COPT-Public/cuPDLP-C), the C implementation of
  cuPDLP.
- [PDQP.jl](https://github.com/jinwen-yang/PDQP.jl), a Julia first-order solver
  for convex quadratic programming on CPUs and NVIDIA GPUs.
- [PDHCG](https://github.com/Lhongpei/PDHCG), a GPU-accelerated first-order
  solver for convex quadratic and conic quadratic programming.
- [HPR-LP-C](https://github.com/PolyU-IOR/HPR-LP-C), a C GPU solver for LP
  based on the Halpern Peaceman–Rachford method.
- [cuOpt](https://github.com/NVIDIA/cuopt), NVIDIA's open-source GPU-accelerated
  optimization library for LP, MIP, and vehicle routing.
- [HiGHS](https://github.com/ERGO-Code/HiGHS), an open-source solver for LP,
  MIP, and QP that includes a PDLP-based first-order LP solver.
- [D-PDLP](https://github.com/Lhongpei/D-PDLP), a distributed LP solver built
  on cuPDLPx for execution across multiple GPUs.
- [CoolPDLP.jl](https://github.com/JuliaDecisionFocusedLearning/CoolPDLP.jl),
  a Julia implementation of PDLP and its variants with support for CPUs,
  multiple GPU architectures, and batched solves.

### Commercial solvers (implementing PDLP)

- [Gurobi](https://www.gurobi.com/)
- [FICO Xpress](https://www.fico.com/en/products/fico-xpress-optimization)
- [Knitro](https://www.artelys.com/solvers/knitro/)
- [COPT](https://www.shanshu.ai/copt)

### Benchmarks

- [MIPLIB 2017](https://miplib.zib.de/), a source of LP relaxation benchmark instances.
- [Mittelmann benchmarks](https://plato.asu.edu/bench.html), independent
  benchmarks and test sets for optimization software.

## Acknowledgements

The development of cuPDLPx is partially supported by AFOSR Grant No.
FA9550-24-1-0051, ONR Grant No. N000142412735, and the NVIDIA Academic Grant
Program.
