/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/19.
*/


#pragma once

namespace faiss_v { namespace gpu {
class StackDeviceMemory {
public:
    /// Allocate a new region of memory that we manage
    explicit StackDeviceMemory(int device, size_t allocPerDevice);
};

}}

