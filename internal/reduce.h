// Copyright 2025 Haihao Lu
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Reductions of a termination check. C++ only.
//
// Every scalar of a check is one cub::DeviceReduce call over a
// thrust::transform_iterator: a device lambda computes the quantity from the
// element index on the fly, and the result is written to a member of the
// device-side device_scalars_t (internal_types.h), never to the host. The
// calls only queue kernels and never wait; the whole struct is copied to the
// host once per check (termination.h).
// DeviceReduce uses no atomics ("run-to-run" determinism), so a given problem
// size always produces the same result.
//
// The lambdas capture device pointers by value from local variables (never
// the solver state itself) and are marked __host__ __device__ so that the
// iterator can deduce their result type on the host; nvcc needs
// --extended-lambda for that (set in CMakeLists.txt).

#ifndef CUPDLPX_REDUCE_H
#define CUPDLPX_REDUCE_H

#include "utils.h"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

// Temporary storage of cub::DeviceReduce, sized once for the longest vector of
// the problem and shared by every call (reduce.cu).
void reduce_temp_storage_create(pdhg_solver_state_t *state);

// Exits with a message when a call would need more temporary storage than
// reduce_temp_storage_create allocated.
void reduce_check_temp_storage(const pdhg_solver_state_t *state, size_t bytes);

// out = sum over i < n of f(i), on the stream, into device memory.
template <typename F> void reduce_sum(pdhg_solver_state_t *state, int n, F f, double *out)
{
    if (n <= 0)
    {
        CUDA_CHECK(cudaMemsetAsync(out, 0, sizeof(double), state->stream));
        return;
    }
    auto it = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), f);
    size_t bytes = 0;
    CUDA_CHECK(cub::DeviceReduce::Sum(NULL, bytes, it, out, n, state->stream));
    reduce_check_temp_storage(state, bytes);
    bytes = state->reduce_temp_bytes;
    CUDA_CHECK(cub::DeviceReduce::Sum(state->reduce_temp_d, bytes, it, out, n, state->stream));
}

// out = max over i < n of f(i), on the stream, into device memory. Every
// maximum the solver takes is a norm or a violation, hence nonnegative, so an
// empty vector yields 0.
template <typename F> void reduce_max(pdhg_solver_state_t *state, int n, F f, double *out)
{
    if (n <= 0)
    {
        CUDA_CHECK(cudaMemsetAsync(out, 0, sizeof(double), state->stream));
        return;
    }
    auto it = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), f);
    size_t bytes = 0;
    CUDA_CHECK(cub::DeviceReduce::Max(NULL, bytes, it, out, n, state->stream));
    reduce_check_temp_storage(state, bytes);
    bytes = state->reduce_temp_bytes;
    CUDA_CHECK(cub::DeviceReduce::Max(state->reduce_temp_d, bytes, it, out, n, state->stream));
}

#endif // CUPDLPX_REDUCE_H
