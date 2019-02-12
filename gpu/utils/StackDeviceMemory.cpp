/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/19.
*/


#include "StackDeviceMemory.h"
#include "DeviceUtils.h"
#include "StaticUtils.h"
#include "../../FaissVAssert.h"

namespace faiss_v {namespace gpu {

StackDeviceMemory::Stack::Stack(int device, size_t size)
    : device_(device),
      isOwner_(false),
      start_(nullptr),
      end_(nullptr),
      size_(size),
      head_(nullptr),
      mallocCurrent_(0),
      highWaterMemoryUsed_(0),
      highWaterMalloc_(0) {
    DeviceScope s(device_);

    CUDA_VERIFY(cudaMalloc(&start_, size_));

    head_ = start_;
    end_ = start_ + size_;
}

StackDeviceMemory::Stack::~Stack() {
    if (isOwner_) {
        DeviceScope s(device_);

        CUDA_VERIFY(cudaFree(start_));
    }
}

char*
StackDeviceMemory::Stack::getAlloc(size_t size, cudaStream_t stream) {
    if (size > (end_ - head_)) {
        // Too large for our stack
        DeviceScope s(device_);

        // Print our requested size before we attempt the allocation
        fprintf(stderr, "WARN: increase temp memory to avoid cudaMalloc, "
                    "or decrease query/add size (alloc %zu B, highwater %zu B)\n",
                size, highWaterMalloc_);

        char* p = nullptr;
        auto err = cudaMalloc(&p, size);
        FAISSV_ASSERT_FMT(err == cudaSuccess,
                         "cudaMalloc error %d on alloc size %zu",
                         (int) err, size);

        mallocCurrent_ += size;
        highWaterMalloc_ = std::max(highWaterMalloc_, mallocCurrent_);

        return p;
    } else {
        // We can make the allocation out of our stack
        // Find all the ranges that we overlap that may have been
        // previously allocated; our allocation will be [head, endAlloc)
        char* startAlloc = head_;
        char* endAlloc = head_ + size;

        while (lastUsers_.size() > 0) {
            auto & prevUser = lastUsers_.back();

            FAISSV_ASSERT(prevUser.start_ <= endAlloc && prevUser.end_ >= startAlloc);

            if (stream != prevUser.stream_) {
                // TODO
                FAISSV_ASSERT(false);
            }

            if (endAlloc < prevUser.end_) {
                // Update the previous user info
                prevUser.start_ = endAlloc;
                break;
            }

            bool done = (prevUser.end_ == endAlloc);
            lastUsers_.pop_back();
            if (done) {
                break;
            }
        }

        head_ = endAlloc;
        FAISSV_ASSERT(head_ <= end_);

        highWaterMemoryUsed_ = std::max(highWaterMemoryUsed_, (size_t)(head_ - start_));

        return startAlloc;
    }
}

void
StackDeviceMemory::Stack::returnAlloc(char *p, size_t size, cudaStream_t stream) {
    if (p < start_ || p > end_) {
        // not this stack; one-off allocation
        DeviceScope s(device_);

        auto err = cudaFree(p);
        FAISSV_ASSERT_FMT(err == cudaSuccess,
                          "cudaFree error %d (addr %p size %zu)",
                          (int) err, p, size);

        FAISSV_ASSERT(mallocCurrent_ >= size);
        mallocCurrent_ -= size;
    } else {
        FAISSV_ASSERT(p + size == head_);

        head_ = p;
        lastUsers_.push_back(Range(p, p + size, stream));
    }
}

StackDeviceMemory::StackDeviceMemory(int device, size_t allocPerDevice)
    : device_(device),
      stack_(device, allocPerDevice) {
}

DeviceMemoryReservation
StackDeviceMemory::getMemory(cudaStream_t stream, size_t size) {
    // alignment of 16
    size = utils::roundUp(size, (size_t) 16);

    return DeviceMemoryReservation(this, device_,
                                   stack_.getAlloc(size, stream),
                                   size,
                                   stream);
}
}}