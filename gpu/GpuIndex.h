/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/06.
*/


#pragma once
#include "../Index.h"
#include <utils/MemorySpace.h>

namespace faiss_v { namespace gpu {

class GpuResources;

struct GpuIndexConfig {
    inline GpuIndexConfig()
        : device(0), memorySpace(MemorySpace::Device){

    }

    /// GPU device
    int device;

    /// Memory space to use for primary storage
    MemorySpace memorySpace;
};

} }
