/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/08.
*/

#pragma once

#include "../utils/DeviceTensor.cuh"
#include "../utils/DeviceVector.cuh"
#include "../utils/Float16.cuh"
#include "../utils/MemorySpace.h"

namespace faiss_v { namespace gpu {

class GpuResources;

/// Holder of GPU resources for a particular flat index
class FlatIndex {
public:
    FlatIndex(GpuResources* res,
              int dim,
              bool l2Distance,
              bool useFloat16,
              bool useFloat16Accumulator,
              bool storeTransposed, MemorySpace space);

    /// Add vectors to ourselves from host or the device
    void add(const float* data, int numVecs, cudaStream_t stream);

private:
    /// Collection of GPU resources used
    GpuResources* resources_;

    /// Dimensionality of our vectors
    const int dim_;

    /// Float16 data format
    const bool useFloat16_;

    /// For supporting hardware, specially Hgemm
    const bool useFloat16Accumulator_;

    /// Store vectors in transposed layout
    const bool storeTransposed_;

    /// L2 or inner product distance
    bool L2Distance_;

    /// Memory space for allocations
    MemorySpace space_;

    /// Vectors num;
    int num_;

    /// The underlying expandable storage
    DeviceVector<char> rawData_;

    /// Vectors in rawData_
    DeviceTensor<float, 2, true> vectors_;
    DeviceTensor<float, 2, true> vectorsTransposed_;

#ifdef FAISSV_USE_FLOAT16
    DeviceTensor<half, 2, true> vectorsHalf_;
    DeviceTensor<half, 2, true> vectorsHalfTransposed_;
#endif

    /// Precomputed L2 norms
    DeviceTensor<float, 1, true> norms_;

#ifdef FAISSV_USE_FLOAT16
    /// Precomputed L2 norms, float16 form
    DeviceTensor<half, 1, true> normsHalf_;
#endif
};

}}
