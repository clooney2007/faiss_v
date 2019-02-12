/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/07.
*/

#pragma once

#include <cuda_runtime.h>
#include <string>

namespace faiss_v { namespace gpu {

class DeviceMemory;

class DeviceMemoryReservation {
public:
    DeviceMemoryReservation();
    DeviceMemoryReservation(DeviceMemory* state,
                            int device,
                            void* p,
                            size_t size,
                            cudaStream_t stream);

    void* get() { return data_; }

private:
    DeviceMemory* state_;

    int device_;
    void* data_;
    size_t size_;
    cudaStream_t stream_;
};

class DeviceMemory {
public:
    virtual ~DeviceMemory();

    /// Obtains a temporary memory allocation for our device,
    /// whose usage is ordered with respect to the given stream.
    virtual DeviceMemoryReservation getMemory(cudaStream_t stream,
                                              size_t size) = 0;

protected:
    friend class DeviceMemoryReservation;
};

}}
