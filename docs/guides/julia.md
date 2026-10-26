---
description: Requirements and installation for using cuPDLPx from Julia through CuPDLPx.jl.
---

# Julia interface

CuPDLPx.jl provides a Julia wrapper for cuPDLPx and an interface to JuMP.

## Software requirements

The Julia interface requires Julia 1.10+.

See [hardware requirements](../getting-started/index.md#hardware-requirements)
for supported GPUs and required CUDA or ROCm versions.

## Installation

Install CuPDLPx.jl from the Julia package manager:

```julia
import Pkg
Pkg.add("CuPDLPx")
```

The package installs the cuPDLPx binaries automatically; a separate source
build is not normally required.

CuPDLPx.jl uses the standard JuMP and MathOptInterface APIs for model
construction, parameters, and results. See the [CuPDLPx.jl
documentation](https://jump.dev/JuMP.jl/stable/packages/CuPDLPx/){ target="_blank" rel="noopener noreferrer" }.
