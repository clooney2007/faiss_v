/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/08.
*/

#include "GpuIndexFlat.h"
#include "../IndexFlat.h"
#include "utils/DeviceUtils.h"
#include "impl/FlatIndex.cuh"

namespace faiss_v { namespace gpu {

/// Default CPU search size for which we use paged copies
constexpr size_t kMinPageSize = (size_t) 256 * 1024 * 1024;

/// Size above which we page copies from the CPU to GPU (non-paged
/// memory usage)
constexpr size_t kNonPinnedPageSize = (size_t) 256 * 1024 * 1024;

GpuIndexFlat::GpuIndexFlat(GpuResources* resources,
                           const faiss_v::IndexFlat *index,
                           GpuIndexFlatConfig config) :
        GpuIndex(resources, index->d, index->metric_type, config),
        minPagedSize_(kMinPageSize),
        config_(config),
        data_(nullptr) {
    verifySettings_();

    // Flat index doesn't need training
    this->is_trained = true;

    copyFrom(index);
}

GpuIndexFlat::GpuIndexFlat(GpuResources *resources,
                           int dims,
                           MetricType metric,
                           GpuIndexFlatConfig config) :
        GpuIndex(resources, dims, metric, config),
        minPagedSize_(kMinPageSize),
        config_(config),
        data_(nullptr) {
    verifySettings_();

    // Flat index doesn't need training
    this->is_trained = true;

    // Construct index
    DeviceScope scope(device_);
    data_ = new FlatIndex(resources,
                          dims,
                          metric == faiss_v::METRIC_L2,
                          config_.useFlat16,
                          config_.useFloat16Accumulator,
                          config_.storeTransposed,
                          memorySpace_);
}

GpuIndexFlat::~GpuIndexFlat() {
    delete data_;
}


}}