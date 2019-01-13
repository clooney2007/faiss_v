/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/07.
*/

#pragma once

#include "utils/DeviceMemory.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <utility>
#include <vector>

namespace faiss_v { namespace gpu {

/// Base class of GPU-side resource provider;
/// hides provision of cuBLAS handles, CUDA streams and a temporary memory mananger
class GpuResources {
public:
    virtual ~GpuResources();

    /// Call to pre-allocate resources for a particular device. If theis is not called,
    /// then resources will be allocated at the first time of demand
    virtual void initializeForDevice(int device) = 0;

    /// Returns the cuBLAS handle that we use for the given device
    virtual cublasHandle_t getBlasHandle(int device) = 0;

    /// Returns the stream that we order all computation on for the given device
    virtual cudaStream_t getDefaultStream(int device) = 0;

    /// Returns the temporary memory manager for the given device
    virtual DeviceMemory& getMemoryManager(int device) = 0;

    /// Calls getMemoryManager for the current device
    virtual DeviceMemory& getMemoryManagerCurrentDevice() = 0;
};

} }

