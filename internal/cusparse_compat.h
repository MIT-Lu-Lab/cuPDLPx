#pragma once

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

#if !CUPDLPX_HAS_SPMVOP
// The SpMVOp types were added to cusparse.h before the functions
// (e.g. CUDA 13.1 base has the types but not the functions).
// Only provide fallback typedefs for cuSPARSE versions that lack them entirely.
#if !defined(CUSPARSE_VERSION) || CUSPARSE_VERSION < 12700
typedef void *cusparseSpMVOpDescr_t;
typedef void *cusparseSpMVOpPlan_t;
#endif
#endif
