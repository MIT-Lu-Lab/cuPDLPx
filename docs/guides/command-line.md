---
description: Build and use the cuPDLPx command-line interface with CUDA or ROCm.
---

# Command-line interface

The `cupdlpx` executable solves LPs in `.mps` or `.mps.gz` format and writes
solution vectors and a solve summary to the output directory.
Build it from source with the C libraries.

## Build requirements

| Component | CUDA build | ROCm build |
| --- | --- | --- |
| Compiler | GCC and NVCC | GCC and `hipcc` |
| Build system | CMake 3.20+ | CMake 3.20+ |

See [hardware requirements](../getting-started/index.md#hardware-requirements)
for supported GPUs and required CUDA or ROCm versions.

## Installation

Clone the repository:

```bash
git clone https://github.com/MIT-Lu-Lab/cuPDLPx.git
cd cuPDLPx
```

### CUDA

Configure CMake and build:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --clean-first --parallel
```

The build selects the sparse matrix–vector backend from the cuSPARSE
version:

| CUDA toolkit | Backend |
| --- | --- |
| CUDA 12.4–13.2 | `cusparseSpMV` |
| CUDA 13.3+ | `cusparseSpMVOp` |

To target a specific CUDA architecture, pass
`-DCMAKE_CUDA_ARCHITECTURES=<architecture>` when configuring.

### ROCm

Enable HIP and provide the architecture of the target AMD GPU:

```bash
cmake -B build \
  -DUSE_HIP=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build --clean-first --parallel
```

Common architecture values include `gfx90a` for MI200, `gfx1100` for RDNA3,
and `gfx1201` for RDNA4. Replace the value with the architecture supported by
the installed ROCm toolchain.

On HIP builds, cuBLAS, cuSPARSE, and CUB calls are mapped to hipBLAS,
hipSPARSE, and hipCUB.

### Build outputs

Both configurations create:

- `build/cupdlpx`, the command-line solver;
- a static core library; and
- a shared `cupdlpx` library.

## Usage

```text
cupdlpx [OPTIONS] <mps_file> <output_directory>
```

For a source build, the executable is normally `./build/cupdlpx`:

```bash
mkdir -p results
./build/cupdlpx problem.mps.gz results
```

The output directory must already exist.

## Solver options

Pass solver options before the input and output paths. See
[Parameters](../reference/parameters.md) for defaults and allowed values,
or run `./build/cupdlpx --help`.

```bash
./build/cupdlpx \
  --time_limit 600 \
  --eps_opt 1e-6 \
  --eps_feas 1e-6 \
  --opt_norm linf \
  problem.mps.gz results
```

## Output files

For `problem.mps.gz`, cuPDLPx creates:

```text
results/
├── problem_summary.txt
├── problem_primal_solution.txt
└── problem_dual_solution.txt
```

The primal and dual files contain one value per line. The summary records the
termination reason, model dimensions, objective values, residuals, iteration
count, and phase timings. Dual slacks are available through the in-memory
Python, Julia, and C interfaces rather than as a separate CLI output file.

!!! tip "In-memory results"

    Use the [Python interface](python.md) or [C interface](c-api.md) when a
    program needs results in memory instead of text files.

## Advanced build options

The native build exposes the following CMake options:

| Option | Default | Purpose |
| --- | --- | --- |
| `CUPDLPX_BUILD_STATIC_LIB` | `ON` | Build the static core library. |
| `CUPDLPX_BUILD_SHARED_LIB` | `ON` | Build the shared library. |
| `CUPDLPX_BUILD_CLI` | `ON` | Build the `cupdlpx` executable. |
| `CUPDLPX_BUILD_PYTHON` | `OFF` | Build the Python extension. |
| `CUPDLPX_BUILD_TESTS` | `OFF` | Build the native test suite. |

For example, build only the shared library:

```bash
cmake -B build \
  -DCUPDLPX_BUILD_STATIC_LIB=OFF \
  -DCUPDLPX_BUILD_CLI=OFF
cmake --build build --parallel
```

See [GPU backends](../implementation/gpu-backends.md) for architecture
defaults and platform requirements.
