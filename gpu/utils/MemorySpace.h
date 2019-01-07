/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/07.
*/


#pragma once

#include "../../FaissVAssert.h"
#include <cuda.h>

#if CUDA_VERSION >= 8000
#define FAISSV_UNIFIED_MEM 1
#endif

namespace faiss_v { namespace gpu {
enum MemorySpace {
    ///  Managed using cudaMalloc/cudaFree
    Device = 1,
    ///  Mananged using cudaMallocManaged/cudaFree
    Unified = 2,
};

/// Allocates CUDA memory for a given memory space
void allocMemorySpace(MemorySpace space, void **p, size_t size);

}}