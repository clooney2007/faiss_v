/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/08.
*/

#include "GpuIndexFlat.h"
#include "../IndexFlat.h"
#include "utils/DeviceUtils.h"
#include "impl/FlatIndex.cuh"
#include "GpuResources.h"

#include <limits>

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

void
GpuIndexFlat::copyFrom(const IndexFlat *index) {
    DeviceScope scope(device_);

    this->d = index->d;
    this->metric_type = index->metric_type;

    // GPU code has 32 bit indices
    FAISSV_THROW_IF_NOT_FMT(index->ntotal <= (Index::idx_t) std::numeric_limits<int>::max(),
                            "GPU index only supports up to %zu indices; "
                                "attempting to copy CPU index with %zu parameters",
                            (size_t) std::numeric_limits<int>::max(),
                            (size_t) index->ntotal);
    this->ntotal = index->ntotal;

    delete data_;
    data_ = new FlatIndex(resources_,
                          this->d,
                          index->metric_type == faiss_v::METRIC_L2,
                          config_.useFlat16,
                          config_.useFloat16Accumulator,
                          config_.storeTransposed,
                          memorySpace_);

    // The index could be empty
    if(index->ntotal > 0) {
        data_->add(index->xb.data(), index->ntotal, resources_->getDefaultStream(device_));
    }
}

void
GpuIndexFlat::add(Index::idx_t n, const float *x) {
    DeviceScope scope(device_);

    // To avoid multiple re-allocations, ensure we have enough storage avaliable
    data_->reserve(n, resources_->getDefaultStream(device_));

    if (!config_.useFlat16) {
        addImpl_(n, x, nullptr);
    } else {
        FAISSV_THROW_IF_NOT_MSG(false, "useFlat16 not implemented.");
    }
}

void
GpuIndexFlat::addImpl_(Index::idx_t n, const float *x, const Index::idx_t* ids) {
    FAISSV_THROW_IF_NOT_MSG(!ids, "add_with_ids not supported");
    FAISSV_THROW_IF_NOT(n > 0);

    // Due to GPU indexing in int32, limit the number of bectors
    FAISSV_THROW_IF_NOT_FMT(this->ntotal + n <= (Index::idx_t) std::numeric_limits<int>::max(),
                            "GPU index only supports up to %zu indices; "
                                "attempting to copy CPU index with %zu parameters",
                            (size_t) std::numeric_limits<int>::max(),
                            (size_t) this->ntotal + n);
    data_->add(x, n, resources_->getDefaultStream(device_));
    this->ntotal += n;
}


void
GpuIndexFlat::reset() {
    DeviceScope scope(device_);

    // Free the underlying memory
    data_->reset();
    this->ntotal = 0;
}

void
GpuIndexFlat::verifySettings_() const {
    // Ensure Hgemm is supported on this device
    if(config_.useFloat16Accumulator) {
#ifdef FAISS_USE_FLOAT16
        FAISSV_THROW_IF_NOT_MSG(config_.useFlat16,
                                "useFloat16Accumulator can only be enabled "
                                    "with useFloat16");
        FAISSV_THROW_IF_NOT_FMT(getDeviceSupportsFloat16Math(config_.device),
                                "Device %d does not support Hgemm "
                                    "(useFloat16Accumulator)",
                                config_.device);
#else
        FAISSV_THROW_IF_NOT_MSG(false, "not compiled with float16 support");
#endif
    }
}

}}