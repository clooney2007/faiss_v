/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/16.
*/

#pragma once

#include "GpuResources.h"
#include "utils/DeviceUtils.h"
#include "utils/StackDeviceMemory.h"
#include <unordered_map>
#include <vector>

namespace faiss_v { namespace gpu {

class StandardGpuResources : public GpuResources {
public:
    StandardGpuResources();

    ~StandardGpuResources() override;

    /// Disable allocation of temporary memory; all temporary memory
    /// requests will call cudaMalloc / cudaFree at the point of use
    void noTempMemory();

public:
    /// Internal system calls
    void initializeForDevice(int device) override;

private:
    /// Our default stream that work is ordered on, one per each device
    std::unordered_map<int, cudaStream_t> defaultStreams_;

    /// This contains particular streams as set by the user for
    /// ordering, if any
    std::unordered_map<int, cudaStream_t> userDefaultStreams_;

    /// Other streams we can use, per each device
    std::unordered_map<int, std::vector<cudaStream_t> > alternateStreams_;

    /// Async copy stream to use for GPU <-> CPU pinned memory copies
    std::unordered_map<int, cudaStream_t> asyncCopyStreams_;

    /// cuBLAS handle for each device
    std::unordered_map<int, cublasHandle_t> blasHandles_;

    /// Temporary memory provider, per each device
    std::unordered_map<int, std::unique_ptr<StackDeviceMemory> > memory_;

    /// Pinned memory allocation for use with this GPU
    void* pinnedMemAlloc_;
    size_t pinnedMemAllocSize_;

    /// By default, we reserve this fraction of memory on all devices
    float tempMemFraction_;

    /// Another option is to use a specified amount of memory on all
    /// devices
    size_t tempMemSize_;

    /// Whether we look at tempMemFraction_ or tempMemSize_
    bool useFraction_;

    /// Amount of pinned memory we should allocate
    size_t pinnedMemSize_;
};

}}