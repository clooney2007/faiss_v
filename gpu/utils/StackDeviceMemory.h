/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/19.
*/


#pragma once
#include "DeviceMemory.h"
#include <list>
#include <memory>

namespace faiss_v { namespace gpu {
class StackDeviceMemory : public DeviceMemory {
public:
    /// Allocate a new region of memory that we manage
    explicit StackDeviceMemory(int device, size_t allocPerDevice);

    DeviceMemoryReservation getMemory(cudaStream_t stream,
                                      size_t size) override;

protected:
    /// Previous allocation ranges and the streams for which
    /// synchronization is required
    struct Range {
        inline Range(char* s, char* e, cudaStream_t str) :
            start_(s), end_(e), stream_(str) {
        }

        // References a memory range [start, end)
        char* start_;
        char* end_;
        cudaStream_t stream_;
    };

    struct Stack {
        Stack(int device, size_t size);

        ~Stack();

        /// Obtains an allocation; all allocations are guaranteed to be 16
        /// byte aligned
        char* getAlloc(size_t size, cudaStream_t stream);

        /// Returns an allocation
        void returnAlloc(char* p, size_t size, cudaStream_t stream);

        /// Device this allocation is on
        int device_;

        /// Do we own our region of memory?
        bool isOwner_;

        /// Where our allocation begins and ends
        /// [start_, end_) is valid
        char* start_;
        char* end_;

        /// Total size end_ - start_
        size_t size_;

        /// Stack head within [start, end)
        char* head_;

        /// List of previous last users of allocations on our stack, for
        /// possible synchronization purposes
        std::list<Range> lastUsers_;

        /// How much cudaMalloc memory is currently outstanding?
        size_t mallocCurrent_;

        /// What's the high water mark in terms of memory used from the
        /// temporary buffer?
        size_t highWaterMemoryUsed_;

        /// What's the high water mark in terms of memory allocated via
        /// cudaMalloc?
        size_t highWaterMalloc_;
    };

    /// Our device
    int device_;

    /// Memory stack
    Stack stack_;
};

}}

