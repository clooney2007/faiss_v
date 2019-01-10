/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/08.
*/

#include "FlatIndex.cuh"
#include "../utils/Float16.cuh"
#include "../utils/CopyUtils.cuh"
#include "../utils/DeviceUtils.h"

namespace faiss_v { namespace gpu {

void FlatIndex::add(const float *data, int numVecs, cudaStream_t stream) {
    if (numVecs == 0) { return; }

    if (useFloat16_) {
#ifdef FAISSV_USE_FLOAT16
        // Make sure that 'data' is on device
        auto devData = toDevice<float, 2>(resources_,
                                          getCurrentDevice(),
                                          (float*) data,
                                          stream,
                                          {numVecs, dim_};)
        auto devDataHalf = toHalf<2>(resources_, stream, devData);
        rawData_.append((char*) devDataHalf.data(), devDataHalf.getSizeInBytes(), stream, true);
#endif
    } else {
        rawData_.append((char*) data, (size_t) dim_ * numVecs * sizeof(float), stream, true);
    }

    num_ += numVecs;

    if (useFloat16_) {
#ifdef FAISSV_USE_FLOAT16
        DeviceTensor<half, 2, true> vectorsHalf((half*)rawData_.data(), {(int) num_, dim_}, space_);
        vectorsHalf_ = std::move(vectorsHalf);
#endif
    } else {

    }
}

}}