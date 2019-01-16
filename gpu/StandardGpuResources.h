/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/16.
*/

#pragma once
#include "GpuResources.h"

namespace faiss_v { namespace gpu {

class StandardGpuResources : public GpuResources {
public:
    StandardGpuResources();

    ~StandardGpuResources() override;

    /// Disable allocation of temporary memory; all temporary memory
    /// requests will call cudaMalloc / cudaFree at the point of use
    void noTempMemory();
};

}}