/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/07.
*/


#include "DeviceMemory.h"

namespace faiss_v { namespace gpu {

DeviceMemoryReservation::DeviceMemoryReservation()
    : state_(NULL),
      device_(0),
      data_(NULL),
      size_(0),
      stream_(0) {
}

DeviceMemoryReservation::DeviceMemoryReservation(DeviceMemory* state,
                                                 int device,
                                                 void* p,
                                                 size_t size,
                                                 cudaStream_t stream)
    : state_(state),
      device_(device),
      data_(p),
      size_(size),
      stream_(stream) {
}

DeviceMemory::~DeviceMemory() {
}

}}