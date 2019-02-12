/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/08.
*/

#include "FlatIndex.cuh"
#include "../utils/CopyUtils.cuh"

namespace faiss_v { namespace gpu {

FlatIndex::FlatIndex(GpuResources *res,
                     int dim,
                     bool l2Distance,
                     bool useFloat16,
                     bool useFloat16Accumulator,
                     bool storeTransposed,
                     MemorySpace space) :
    resources_(res),
    dim_(dim),
    useFloat16_(useFloat16),
    useFloat16Accumulator_(useFloat16Accumulator),
    storeTransposed_(storeTransposed),
    l2Distance_(l2Distance),
    space_(space),
    num_(0),
    rawData_(space) {
#ifndef FAISSV_USE_FLOAT16
    FAISSV_ASSERT(!useFloat16_);
#endif

}

void FlatIndex::reserve(size_t numVecs, cudaStream_t stream) {
    if (useFloat16_) {
        FAISSV_THROW_IF_NOT_MSG(false, "useFloat16_ not implemented");
    } else {
        rawData_.reserve(numVecs * dim_ * sizeof(float), stream);
    }
}

void FlatIndex::add(const float *data, int numVecs, cudaStream_t stream) {
    if (numVecs == 0) { return; }

    if (useFloat16_) {
        FAISSV_ASSERT_MSG(false, "useFloat16_ not implemented");
    } else {
        rawData_.append((char*) data, (size_t) dim_ * numVecs * sizeof(float), stream, true);
    }

    num_ += numVecs;

    if (useFloat16_) {
        FAISSV_ASSERT_MSG(false, "useFloat16_ not implemented");
    } else {
        DeviceTensor<float, 2, true> vectors((float*)rawData_.data(), {(int)num_, dim_}, space_);
        vectors_ = std::move(vectors);
    }

    if (storeTransposed_) {
        FAISSV_ASSERT_MSG(false, "storeTransposed_ not implemented");
    }

    if (l2Distance_) {
        if (useFloat16_) {
            FAISSV_ASSERT_MSG(false, "useFloat16_ not implemented");
        } else {
            DeviceTensor<float, 1, true> norms({(int) num_}, space_);
            runL2Norm(vectors_, norms, true, stream);
            norms_ = std::move(norms);
        }
    }
}

void FlatIndex::reset() {
    rawData_.clear();
    vectors_ = std::move(DeviceTensor<float, 2, true>());
    norms_ = std::move(DeviceTensor<float, 1, true>());
    num_ = 0;
}

}}