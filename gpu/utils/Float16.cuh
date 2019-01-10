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

/// Copies `in` to `out` while performing a float32 -> float16 conversion
void runConvertToFloat16(half* out,
                         const float* in,
                         size_t num,
                         cudaStream_t stream);

/// Copies `in` to `out` while performing a float16 -> float32
/// conversion
void runConvertToFloat32(float* out,
                         const half* in,
                         size_t num,
                         cudaStream_t stream);

template <int Dim>
void toHalf(cudaStream_t stream,
            Tensor<float, Dim, true>& in,
            Tensor<half, Dim, true>& out) {
    FAISSV_ASSERT(in.numElements() == out.numElements());

    // The memory is contiguous, so apply a pointwise kernel to convert
    runConvertToFloat16(out.data(), in.data(), in.numElements(), stream);
}

template <int Dim>
DeviceTensor<half, Dim, true> toHalf(GpuResources* resources,
                                     cudaStream_t stream,
                                     Tensor<float, Dim, true>& in) {
    DeviceTensor<half, Dim, true> out;
    if (resources) {
        out = std::move(DeviceTensor<half, Dim, true>(
            resources->getMemoryManagerCurrentDevice(),
            in.sizes(),
            stream));
    } else {
        out = std::move(DeviceTensor<float, Dim, true>(in.sizes()));
    }

    toHalf<Dim>(stream, in, out);

    return out;
};

template <int Dim>
void fromHalf(cudaStream_t stream,
              Tensor<half, Dim, true>& in,
              Tensor<float, Dim, true>& out) {
    FAISSV_ASSERT(in.numElements() == out.numElements());

    // The memory is contiguous, so apply a pointwise kernel to convert
    runConvertToFloat32(out.data(), in.data(), in.numElements(), stream);
}

template <int Dim>
DeviceTensor<float, Dim, true> fromHalf(GpuResources* resources,
                                        cudaStream_t stream,
                                        Tensor<half, Dim, true>& in) {
    DeviceTensor<float, Dim, true> out;
    if (resources) {
        out = std::move(DeviceTensor<float, Dim, true>(
            resources->getMemoryManagerCurrentDevice(),
            in.sizes(),
            stream));
    } else {
        out = std::move(DeviceTensor<float, Dim, true>(in.sizes()));
    }

    fromHalf<Dim>(stream, in, out);

    return out;
};


}}