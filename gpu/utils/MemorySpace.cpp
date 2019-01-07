/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/07.
*/


#include "MemorySpace.h"
#include <cuda_runtime.h>

namespace faiss_v { namespace gpu {

/// Allocates CUDA memory for a given memory space
void allocMemorySpace(MemorySpace space, void **p, size_t size) {
    if(space == MemorySpace::Device) {
        FAISSV_ASSERT_FMT(cudaMalloc(p, size) == cudaSuccess,
                         "Failed to cudaMalloc %zu bytes", size);
    }
#ifdef FAISSV_UNIFIED_MEM
    else if(space == MemorySpace::Unified) {
    FAISSV_ASSERT_FMT(cudaMallocManaged(p, size) == cudaSuccess,
                     "Failed to cudaMallocManaged %zu bytes", size);
    }
#endif
    else {
        FAISSV_ASSERT_FMT(false, "Unknown MemorySpace %d", (int) space);
    }
}

}}
