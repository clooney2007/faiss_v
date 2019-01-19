/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/10.
*/

#pragma once

#include "DeviceTensor.cuh"
#include "DeviceUtils.h"
#include "../GpuResources.h"

namespace faiss_v { namespace gpu {

/// Ensure the memory at 'p' is either on the given device,
/// otherwise copy it to the device in a new allocation
template <typename T, int Dim>
DeviceTensor<T, Dim, true> toDevice(GpuResources* resources,
                                    int dstDevice,
                                    T* src,
                                    cudaStream_t stream,
                                    std::initializer_list<int> sizes) {
    int dev = getDeviceForAddress(src);

    if (dev == dstDevice) {
        // On device

        return DeviceTensor<T, Dim, true>(src, sizes);
    } else {
        // On different device or on host
        DeviceScope scope(dstDevice);

        Tensor<T, Dim, true> oldT(src, sizes);
        if (resources) {
            DeviceTensor<T, Dim, true> newT(resources->getMemoryManager(dstDevice), sizes, stream);
            newT.copyFrom(oldT, stream);

            return newT;
        } else {
            DeviceTensor<T, Dim, true> newT(sizes);
            newT.copyFrom(oldT, stream);

            return newT;
        }
    }
};

}}
