#pragma once

// On HIP builds, cuda_to_hip.h handles the cusparse -> hipsparse mapping.
// Include it first so the defines are active when we check for SpMVOp.
#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
// hipSPARSE does not have SpMVOp; force the standard SpMV path.
#define CUPDLPX_HAS_SPMVOP 0

// Provide fallback typedefs for compilation (never used at runtime on HIP).
typedef void *cusparseSpMVOpDescr_t;
typedef void *cusparseSpMVOpPlan_t;

#else // CUDA build

#include <cusparse.h>

// cusparseSpMVOp_bufferSize was introduced in cuSPARSE 12.7.3 (CUDA 13.1 Update 1).
// CUSPARSE_VERSION encoding: major*1000 + minor*100 + patch.
#if defined(CUSPARSE_VERSION) && CUSPARSE_VERSION >= 12703
#define CUPDLPX_HAS_SPMVOP 1
#else
#define CUPDLPX_HAS_SPMVOP 0
#endif

#if !CUPDLPX_HAS_SPMVOP
// The SpMVOp types were added to cusparse.h before the functions
// (e.g. CUDA 13.1 base has the types but not the functions).
// Only provide fallback typedefs for cuSPARSE versions that lack them entirely.
#if !defined(CUSPARSE_VERSION) || CUSPARSE_VERSION < 12700
typedef void *cusparseSpMVOpDescr_t;
typedef void *cusparseSpMVOpPlan_t;
#endif
#endif

#endif // USE_HIP
