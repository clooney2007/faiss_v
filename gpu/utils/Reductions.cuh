/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/03/11.
*/

#pragma once
#include "ReductionOperators.cuh"
#include "WarpShuffles.cuh"
#include "DeviceDefs.cuh"
#include <cuda.h>

namespace faiss_v { namespace gpu {

template <typename T, typename Op, int ReduceWidth = kWarpSize>
__device__ inline T warpReduceAll(T val, Op op) {
#pragma unroll
for (int mask = ReduceWidth / 2; mask > 0; mask >>= 1) {
val = op(val, shfl_xor(val, mask));
}

return val;
}

/// Sums a register value across all warp threads
template <typename T, int ReduceWidth = kWarpSize>
__device__ inline T warpReduceAllSum(T val) {
return warpReduceAll<T, Sum<T>, ReduceWidth>(val, Sum<T>());
}

}}
