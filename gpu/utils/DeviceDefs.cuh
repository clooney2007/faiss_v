/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/03/11.
*/

#pragma once

namespace faiss_v { namespace gpu {

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ <= 700
constexpr int kWarpSize = 32;
#else
#error Unknown __CUDA_ARCH__; please define parameters for compute capability
#endif // __CUDA_ARCH__ types
#endif // __CUDA_ARCH__

#ifndef __CUDA_ARCH__
// dummy value for host compiler
constexpr int kWarpSize = 32;
#endif // !__CUDA_ARCH__

// This is a memory barrier for intra-warp writes to shared memory.
__forceinline__ __device__ void warpFence() {

#if __CUDA_ARCH__ >= 700
    __syncwarp();
#else
    // For the time being, assume synchronicity.
    //  __threadfence_block();
#endif
}

} } // namespace
