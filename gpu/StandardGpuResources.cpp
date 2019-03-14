/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/16.
*/
#include <cstdio>
#include "StandardGpuResources.h"

namespace faiss_v { namespace gpu {

constexpr int kNumStreams = 2;

/// Use 18% of GPU memory for temporary space by default
constexpr float kDefaultTempMemFraction = 0.18f;

/// Default pinned memory allocation size
constexpr size_t kDefaultPinnedMemoryAllocation = (size_t) 256 * 1024 * 1024;


StandardGpuResources::StandardGpuResources()
    : pinnedMemAlloc_(nullptr),
      pinnedMemAllocSize_(0),
      tempMemFraction_(kDefaultTempMemFraction),
      tempMemSize_(0),
      useFraction_(true),
      pinnedMemSize_(kDefaultPinnedMemoryAllocation) {
}

StandardGpuResources::~StandardGpuResources() {
    for (auto& entry : defaultStreams_) {
        DeviceScope scope(entry.first);

        auto it = userDefaultStreams_.find(entry.first);
        if (it == userDefaultStreams_.end()) {
            CUDA_VERIFY(cudaStreamDestroy(entry.second));
        }
    }

    for (auto& entry : alternateStreams_) {
        DeviceScope scope(entry.first);

        for (auto stream : entry.second) {
            CUDA_VERIFY(cudaStreamDestroy(stream));
        }
    }

    for (auto& entry : asyncCopyStreams_) {
        DeviceScope scope(entry.first);

        CUDA_VERIFY(cudaStreamDestroy(entry.second));
    }

    for (auto& entry : blasHandles_) {
        DeviceScope scope(entry.first);

        auto blasStatus = cublasDestroy(entry.second);
        FAISSV_ASSERT(blasStatus == CUBLAS_STATUS_SUCCESS);
    }

    if (pinnedMemAlloc_) {
        CUDA_VERIFY(cudaFreeHost(pinnedMemAlloc_));
    }
}

void
StandardGpuResources::noTempMemory() {
    setTempMemory(0);
}

void
StandardGpuResources::setTempMemory(size_t size) {
    useFraction_ = false;
    tempMemSize_ = size;
}

void
StandardGpuResources::initializeForDevice(int device) {
    if (defaultStreams_.count(device) != 0) {
        return;
    }

    if (defaultStreams_.empty() && pinnedMemSize_ > 0) {
        CUDA_VERIFY(cudaHostAlloc(&pinnedMemAlloc_,
                                  pinnedMemSize_,
                                  cudaHostAllocDefault));
        pinnedMemAllocSize_ = pinnedMemSize_;
    }

    FAISSV_ASSERT(device < getNumDevices());
    DeviceScope scope(device);

    auto& prop = getDeviceProperties(device);

    FAISSV_ASSERT_FMT(prop.major >= 3,
                     "Device id %d with CC %d.%d not supported, "
                         "need 3.0+ compute capability",
                     device, prop.major, prop.minor);

    // Create streams
    cudaStream_t defaultStream = 0;
    auto it = userDefaultStreams_.find(device);
    if (it != userDefaultStreams_.end()) {
        defaultStream = it->second;
    } else {
        CUDA_VERIFY(cudaStreamCreateWithFlags(&defaultStream,
                                              cudaStreamNonBlocking));
    }

    defaultStreams_[device] = defaultStream;

    cudaStream_t asyncCopyStream = 0;
    CUDA_VERIFY(cudaStreamCreateWithFlags(&asyncCopyStream,
                                          cudaStreamNonBlocking));
    asyncCopyStreams_[device] = asyncCopyStream;

    std::vector<cudaStream_t > deviceStreams;
    for (int j = 0; j < kNumStreams; ++j) {
        cudaStream_t stream = 0;
        CUDA_VERIFY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        deviceStreams.push_back(stream);
    }

    alternateStreams_[device] = std::move(deviceStreams);

    // Create cuBLAS handle
    cublasHandle_t  blasHndle = 0;
    auto blasStatus = cublasCreate(&blasHndle);
    FAISSV_ASSERT(blasStatus == CUBLAS_STATUS_SUCCESS);
    blasHandles_[device] = blasHndle;

    size_t toAlloc = 0;
    if (useFraction_) {
        size_t devFree = 0;
        size_t devTotal = 0;

        CUDA_VERIFY(cudaMemGetInfo(&devFree, &devTotal));
        toAlloc = (size_t)(tempMemFraction_ * devTotal);
    } else {
        toAlloc = tempMemSize_;
    }

    FAISSV_ASSERT(memory_.count(device) == 0);
    memory_.emplace(device,
                    std::unique_ptr<StackDeviceMemory>(
                        new StackDeviceMemory(device, toAlloc)));
}

cudaStream_t
StandardGpuResources::getDefaultStream(int device) {
    initializeForDevice(device);

    return defaultStreams_[device];
}

DeviceMemory& StandardGpuResources::getMemoryManager(int device) {
    initializeForDevice(device);

    return *memory_[device];
}

}}

