/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/06.
*/


#pragma once
#include "../Index.h"
#include "utils/MemorySpace.h"

namespace faiss_v { namespace gpu {

class GpuResources;

struct GpuIndexConfig {
    inline GpuIndexConfig()
        : device(0), memorySpace(MemorySpace::Device) {
    }

    /// GPU device
    int device;

    /// Memory space to use for primary storage
    MemorySpace memorySpace;
};

class GpuIndex : public Index {
public:
    GpuIndex(GpuResources* resources, int dims, MetricType metric, GpuIndexConfig config);

    int getDevice() const {
        return device_;
    }

    GpuResources* getResources() {
        return resources_;
    }

    /// x could be storage on CPU or GPU
    /// Handles paged adds if the add set is too large; calls addInternal_
    void add(Index::idx_t n, const float* x) override;

    /// Handles paged adds if the add set is too large; calls addInternal_
    void add_with_ids(Index::idx_t n, const float* x, const Index::idx_t* ids) override;

    /// Handles paged search; calls searchImpl_
    void search(Index::idx_t n, const float* x, Index::idx_t k, float* distance, Index::idx_t* labels) const override;

protected:
    /// Handles paged adds if the add set is too large, passes to addImpl
    /// to actually perform the add for the current page
    void addInternal_(Index::idx_t n, const float* x, const Index::idx_t* ids);

    /// Overridden to actually perform the add
    virtual void addImpl_(Index::idx_t n, const float* x, const Index::idx_t* ids) = 0;

    /// Overridden to actually perform the search
    virtual void searchImpl_(Index::idx_t n, const float* x, Index::idx_t k,
                             float* distance, Index::idx_t* labels) const = 0;

protected:
    /// Manages streams, cuBLAS handles and scratch memory for devices
    GpuResources* resources_;

    /// GPU device we use
    const int device_;

    /// The memory space use for primary storage
    const MemorySpace memorySpace_;
};

} }
