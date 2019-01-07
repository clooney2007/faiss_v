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