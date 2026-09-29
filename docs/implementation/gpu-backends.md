---
description: CUDA and ROCm build requirements, architecture selection, and backend capabilities for cuPDLPx.
---

# GPU backends

cuPDLPx supports NVIDIA GPUs through CUDA and AMD GPUs through ROCm/HIP.
Both backends use the same algorithm and Python, command-line, and C APIs.
The compatibility layer selects the GPU libraries.

## Backend matrix

| Target | Compiler | Dense algebra | Sparse algebra | Parallel primitives |
| --- | --- | --- | --- | --- |
| NVIDIA CUDA | NVCC | cuBLAS | cuSPARSE | CUB |
| AMD ROCm | `hipcc` | hipBLAS | hipSPARSE | hipCUB / rocPRIM |

With `USE_HIP=ON`, compatibility headers map CUDA calls to HIP so that the
same `.cu` sources compile for AMD GPUs.

See [SpMV](sparse-matrix-vector-products.md) for sparse matrix storage,
backend selection, and workspace reuse.

## Select an architecture

### CUDA

CMake selects default architectures based on the CUDA version. To reduce
binary size, specify the target architecture:

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build --parallel
```

Use an architecture supported by the installed CUDA toolkit and the deployment
GPU.

### ROCm

HIP builds require a target architecture. The default is `gfx90a`; override it
for other devices:

```bash
cmake -B build \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_PREFIX_PATH=/opt/rocm
```

## Platform notes

- CI builds both the CUDA and ROCm configurations on Linux.
- CUDA builds are also exercised on Windows.
- The native CLI is disabled automatically for MSVC builds because its current
  argument parser depends on POSIX headers; the libraries and Python binding are
  separate build targets.
- ROCm packages installed outside CMake's search path may require an explicit
  `CMAKE_PREFIX_PATH`.

## Vendor documentation

- NVIDIA: [CUDA Toolkit](https://docs.nvidia.com/cuda/),
  [cuBLAS](https://docs.nvidia.com/cuda/cublas/),
  [cuSPARSE](https://docs.nvidia.com/cuda/cusparse/), and
  [CUB](https://nvidia.github.io/cccl/unstable/cub/index.html).
- AMD: [ROCm](https://rocm.docs.amd.com/),
  [hipBLAS](https://rocm.docs.amd.com/projects/hipBLAS/en/latest/),
  [hipSPARSE](https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/), and
  [rocPRIM](https://rocm.docs.amd.com/projects/rocPRIM/en/latest/).
- The backend dispatch described here is implemented in
  [`internal/cusparse_compat.h`](https://github.com/MIT-Lu-Lab/cuPDLPx/blob/main/internal/cusparse_compat.h)
  and
  [`src/spmv_backend.cu`](https://github.com/MIT-Lu-Lab/cuPDLPx/blob/main/src/spmv_backend.cu).
