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
// cuSPARSE 12.8.1 (CUDA 13.3.0) introduced the SpMVOp ALG1/ALG2 algorithms and
// added a cusparseSpMVOpAlg_t parameter to cusparseSpMVOp_bufferSize/createDescr.
// CUSPARSE_VERSION encoding: major*1000 + minor*100 + patch.
#if defined(CUSPARSE_VERSION) && CUSPARSE_VERSION >= 12801
#define CUPDLPX_HAS_SPMVOP 1
#else
#define CUPDLPX_HAS_SPMVOP 0
#endif

#endif // USE_HIP
