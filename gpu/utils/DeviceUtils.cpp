/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/07.
*/


#include "DeviceUtils.h"
#include "../../FaissVAssert.h"
#include <mutex>
#include <unordered_map>

namespace faiss_v { namespace gpu {

int getCurrentDevice() {
    int dev = -1;
    CUDA_VERIFY(cudaGetDevice(&dev));
    FAISSV_ASSERT(dev != -1);

    return dev;
}

void setCurrentDevice(int device) {
    CUDA_VERIFY(cudaSetDevice(device));
}

int getNumDevices() {
    int numDev = -1;
    CUDA_VERIFY(cudaGetDeviceCount(&numDev));
    FAISSV_ASSERT(numDev != -1);

    return numDev;
}

const cudaDeviceProp& getDeviceProperties(int device) {
    static std::mutex mutex;
    static std::unordered_map<int, cudaDeviceProp> properties;

    std::lock_guard<std::mutex> guard(mutex);

    auto it = properties.find(device);
    if (it == properties.end()) {
        cudaDeviceProp prop;
        CUDA_VERIFY(cudaGetDeviceProperties(&prop, device));

        properties[device] = prop;
        it = properties.find(device);
    }

    return it->second;
}

const cudaDeviceProp& getCurrentDeviceProperties() {
    return getDeviceProperties(getCurrentDevice());
}

int getMaxThreads(int device) {
    return getDeviceProperties(device).maxThreadsPerBlock;
}

int getMaxThreadsCurrentDevice() {
    return getMaxThreads(getCurrentDevice());
}

int getDeviceForAddress(const void* p) {
    if (!p) {
        return -1;
    }

    cudaPointerAttributes att;
    cudaError_t err = cudaPointerGetAttributes(&att, p);
    FAISSV_ASSERT(err == cudaSuccess || err == cudaErrorInvalidValue);

    if (err == cudaErrorInvalidValue) {
        err = cudaGetLastError();
        FAISSV_ASSERT(err == cudaErrorInvalidValue);

        return -1;
    } else if (att.memoryType == cudaMemoryTypeHost) {
        return -1;
    } else {
        return att.device;
    }
}

bool getFullUnifiedMemSupport(int device) {
    const auto& prop = getDeviceProperties(device);

    return (prop.major >= 6);
}

bool getFullUnifiedMemSupportCurrentDevice() {
    return getFullUnifiedMemSupport(getCurrentDevice());
}

DeviceScope::DeviceScope(int device) {
    prevDevice_ = getCurrentDevice();

    if(prevDevice_ != device) {
        setCurrentDevice(device);
    } else {
        prevDevice_ = -1;
    }
}

DeviceScope::~DeviceScope() {
    if(prevDevice_ != -1) {
        setCurrentDevice(prevDevice_);
    }
}

}}