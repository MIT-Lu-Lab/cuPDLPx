/*
Copyright 2025 Haihao Lu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/*
 * CUDA to HIP compatibility header for cuPDLPx.
 *
 * This header maps CUDA API symbols to their HIP equivalents when building
 * with USE_HIP. On CUDA builds, it simply includes the standard CUDA headers.
 * Source files keep their CUDA spelling; this header handles the translation.
 */

#pragma once

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

// HIP runtime
#include <hip/hip_runtime.h>

// hipBLAS
#include <hipblas/hipblas.h>

// hipSPARSE
#include <hipsparse/hipsparse.h>

// hipCUB (for cub::DeviceReduce) - C++ only
#ifdef __cplusplus
#include <hipcub/hipcub.hpp>
#endif

// ----------------------------------------------------------------------------
// CUDA Runtime API -> HIP Runtime API
// ----------------------------------------------------------------------------

// Memory management
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemset hipMemset

// Memory copy kinds
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice

// Error handling
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetLastError hipGetLastError
#define cudaGetErrorName hipGetErrorName

// Streams
#define cudaStream_t hipStream_t
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy

// Device synchronization
#define cudaDeviceSynchronize hipDeviceSynchronize

// ----------------------------------------------------------------------------
// CUDA Graph API -> HIP Graph API
// ----------------------------------------------------------------------------

#define cudaGraph_t hipGraph_t
#define cudaGraphExec_t hipGraphExec_t
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaStreamCaptureModeGlobal hipStreamCaptureModeGlobal
#define cudaGraphInstantiate hipGraphInstantiate
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraphExecDestroy hipGraphExecDestroy

// ----------------------------------------------------------------------------
// cuBLAS -> hipBLAS
// ----------------------------------------------------------------------------

#define cublasHandle_t hipblasHandle_t
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasSetStream hipblasSetStream
#define cublasSetPointerMode hipblasSetPointerMode
#define cublasStatus_t hipblasStatus_t
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_POINTER_MODE_HOST HIPBLAS_POINTER_MODE_HOST

// cuBLAS functions
#define cublasDnrm2 hipblasDnrm2
#define cublasDdot hipblasDdot
#define cublasDscal hipblasDscal
#define cublasDaxpy hipblasDaxpy
#define cublasIdamax hipblasIdamax

// cublasDnrm2_v2_64 maps to hipblasDnrm2 (hipBLAS uses 32-bit by default,
// but LP problem sizes should fit; use hipblasDnrm2_64 if needed)
#define cublasDnrm2_v2_64 hipblasDnrm2

// Error name helper
static inline const char *cublasGetStatusName(hipblasStatus_t status)
{
    return hipblasStatusToString(status);
}

// ----------------------------------------------------------------------------
// cuSPARSE -> hipSPARSE
// ----------------------------------------------------------------------------

#define cusparseHandle_t hipsparseHandle_t
#define cusparseCreate hipsparseCreate
#define cusparseDestroy hipsparseDestroy
#define cusparseSetStream hipsparseSetStream
#define cusparseStatus_t hipsparseStatus_t
#define CUSPARSE_STATUS_SUCCESS HIPSPARSE_STATUS_SUCCESS
#define cusparseGetErrorName hipsparseGetErrorName

// Sparse matrix and dense vector descriptors
#define cusparseSpMatDescr_t hipsparseSpMatDescr_t
#define cusparseDnVecDescr_t hipsparseDnVecDescr_t

// Index types and base
#define CUSPARSE_INDEX_32I HIPSPARSE_INDEX_32I
#define CUSPARSE_INDEX_BASE_ZERO HIPSPARSE_INDEX_BASE_ZERO

// Operations
#define CUSPARSE_OPERATION_NON_TRANSPOSE HIPSPARSE_OPERATION_NON_TRANSPOSE
#define CUSPARSE_ACTION_NUMERIC HIPSPARSE_ACTION_NUMERIC
#define CUSPARSE_CSR2CSC_ALG_DEFAULT HIPSPARSE_CSR2CSC_ALG_DEFAULT
#define CUSPARSE_SPMV_CSR_ALG2 HIPSPARSE_SPMV_CSR_ALG2

// Functions
#define cusparseCreateCsr hipsparseCreateCsr
#define cusparseDestroySpMat hipsparseDestroySpMat
#define cusparseCreateDnVec hipsparseCreateDnVec
#define cusparseDestroyDnVec hipsparseDestroyDnVec
#define cusparseDnVecSetValues hipsparseDnVecSetValues
#define cusparseSpMV hipsparseSpMV
#define cusparseSpMV_bufferSize hipsparseSpMV_bufferSize
#define cusparseSpMV_preprocess hipsparseSpMV_preprocess
#define cusparseCsr2cscEx2 hipsparseCsr2cscEx2
#define cusparseCsr2cscEx2_bufferSize hipsparseCsr2cscEx2_bufferSize

// ----------------------------------------------------------------------------
// Data types
// ----------------------------------------------------------------------------

#define CUDA_R_64F HIP_R_64F

// ----------------------------------------------------------------------------
// CUB -> hipCUB (C++ only)
// ----------------------------------------------------------------------------

#ifdef __cplusplus
namespace cub = hipcub;
#endif

#else // CUDA build

// Standard CUDA headers
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

// cub is C++ only; C translation units (cli.c, cupdlpx.c, ...) reach this
// header transitively and must not pull it in.
#ifdef __cplusplus
#include <cub/device/device_reduce.cuh>
#endif

#endif // USE_HIP
