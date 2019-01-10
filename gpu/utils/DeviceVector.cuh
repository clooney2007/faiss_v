/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/10.
*/


#pragma once
#include "../../FaissVAssert.h"
#include "MemorySpace.h"
#include "StaticUtils.h"
#include "DeviceUtils.h"
#include <algorithm>
#include <cuda.h>
#include <vector>

namespace faiss_v { namespace gpu {

/// A simple version of thrust::device_vector<T>
template <typename T>
class DeviceVector {
public:
    DeviceVector(MemorySpace space = MemorySpace::Device)
        : data_(nullptr),
          num_(0),
          capacity_(0),
          space_(space) {

    }

    ~DeviceVector() {
        clear();
    }

    // Clear all allocated memory
    void clear() {
        CUDA_VERIFY(cudaFree(data_));
        data_ = nullptr;
        num_ = 0;
        capacity_ = 0;
    }

    size_t size() const { return num_; }
    size_t capacity() const { return capacity_; }
    T* data() { return data_; }
    const T* data() const { return data_; }

    // Returns true if we actually reallocated memory
    bool append(const T* d, size_t n, cudaStream_t stream, bool reserveExact = false) {
        bool mem = false;

        if (n > 0) {
            size_t  reserveSize = num_ + n;
            if (!reserveExact) {
                reserveSize = getNewCapacity_(reserveSize);
            }

            mem = reserve(reserveSize, stream);

            int dev = getDeviceForAddress(d);
            if (dev == -1) {
                CUDA_VERIFY(cudaMemcpyAsync(data_ + num_,
                                            d,
                                            n * sizeof(T),
                                            cudaMemcpyHostToDevice,
                                            stream));
            } else {
                CUDA_VERIFY(cudaMemcpyAsync(data_ + num_,
                                            d,
                                            n * sizeof(T),
                                            cudaMemcpyDeviceToDevice,
                                            stream));
            }

            num_ += n;
        }

        return mem;
    }

    // Reutrns true if we actually reallocated memory
    bool reserve(size_t newCapacity, cudaStream_t stream) {
        if (newCapacity <= capacity_) {
            return false;
        }

        // Otherwise, allocate new space
        realloc_(newCapacity, stream);

        return true;
    }

private:
    void realloc_(size_t newCapacity, cudaStream_t stream) {
        FAISSV_ASSERT(num_ <= newCapacity);

        T* newData = nullptr;
        allocMemorySpace(space_, (void**) &newData, newCapacity* sizeof(T));
        CUDA_VERIFY(cudaMemcpyAsync(newData,
                                    data_,
                                    num_ * sizeof(T),
                                    cudaMemcpyDeviceToDevice,
                                    stream));

        // FIXME: keep on reclamation queue to avoid harmmering cudaFree
        CUDA_VERIFY(cudaFree(data_));

        data_ = newData;
        capacity_ = newCapacity;
    }

    size_t getNewCapacity_(size_t preferredSize) {
        return utils::nextHighestPowerOf2(preferredSize);
    }

    T* data_;
    size_t num_;
    size_t capacity_;
    MemorySpace space_;
};

}}
