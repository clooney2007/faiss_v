/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/07.
*/

#pragma once

#include "../../FaissVAssert.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>

namespace faiss_v { namespace gpu {

/// Returns the current thread-local GPU device
int getCurrentDevice();

/// Sets the current thread-local GPU device
void setCurrentDevice(int device);

/// Returns the number of available GPU devices
int getNumDevices();

/// Returns a cached cudaDeviceProp for the given device
const cudaDeviceProp& getDeviceProperties(int device);

/// Returns the cached cudaDeviceProp for the current device
const cudaDeviceProp& getCurrentDeviceProperties();

/// Does the given device support full unified memory sharing host memory?
bool getFullUnifiedMemSupport(int device);

/// Equivalent to getFullUnifiedMemSupport(getCurrentDevice())
bool getFullUnifiedMemSupportCurrentDevice();

/// For a given pointer, returns whether or not it is located on
/// a device (deviceId >= 0) or the host (-1).
int getDeviceForAddress(const void* p);

/// Resource Acquisition Is Initialization object to set the current device,
/// and restore the previous device upon destruction
class DeviceScope {
public:
    explicit DeviceScope(int device);

    ~DeviceScope();

private:
    int prevDevice_;
};

/// Wrapper to test return status of CUDA functions
#define CUDA_VERIFY(X)                                                      \
  do {                                                                      \
    auto err__ = (X);                                                       \
    FAISSV_ASSERT_FMT(err__ == cudaSuccess, "CUDA error %d", (int) err__);   \
  } while (0)

} }
