/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/03/11.
*/

#pragma once

#include "DeviceDefs.cuh"
#include <cuda.h>

namespace faiss_v { namespace gpu {

template <typename T>
inline __device__ T shfl_xor(const T val,
                             int laneMask, int width = kWarpSize) {
#if CUDA_VERSION >= 9000
    return __shfl_xor_sync(0xffffffff, val, laneMask, width);
#else
    return __shfl_xor(val, laneMask, width);
#endif
}

// CUDA SDK does not provide specializations for T*
template <typename T>
inline __device__ T* shfl_xor(T* const val,
int laneMask, int width = kWarpSize) {
static_assert(sizeof(T*) == sizeof(long long), "pointer size");
long long v = (long long) val;
return (T*) shfl_xor(v, laneMask, width);
}

}}
