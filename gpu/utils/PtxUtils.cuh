/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/03/11.
*/

#pragma once

#include <cuda.h>

namespace faiss_v { namespace gpu {

__device__ __forceinline__ int getLaneId() {
    int laneId;
    asm("mov.s32 %0, %laneid;" : "=r"(laneId) );
    return laneId;
}

}}
