/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/10.
*/


#pragma once

#include <cuda.h>
#include "../GpuResources.h"
#include "DeviceTensor.cuh"

#if CUDA_VERSION >= 7500
#define FAISSV_USE_FLOAT16 1

if __CUDA_ARCH__ >= 530
#define FAISSV_USE_FULL_FLOAT16 1
#endif

#endif

#ifdef FAISSV_USE_FLOAT16
#include <cuda_fp16.h>
#endif

namespace faiss_v { namespace gpu {

/// Returns true if the given device supports native float16 math
bool getDeviceSupportsFloat16Math(int device);

}}