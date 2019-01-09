/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/08.
*/

#pragma once

#include "GpuIndex.h"

namespace faiss_v {

struct IndexFlat;
struct IndexFlatL2;
struct IndexFlatIP;

}

namespace faiss_v { namespace gpu {

struct FlatIndex;

struct GpuIndexFlatConfig : public GpuIndexConfig {
    inline GpuIndexFlatConfig()
        : useFlat16(false),
          useFloat16Accumulator(false),
          storeTransposed(false) {
    }

    /// Whether data is stored as float16
    bool useFlat16;

    /// Whether math is performed in float16
    bool useFloat16Accumulator;

    /// Whether data is stored in a transposed layout, enabling use of the NN GEMM call
    bool storeTransposed;
};

/// Wrapper around the GPU implementation like faiss_v::IndexFlat;
/// copies data from faiss::IndexFlat
class GpuIndexFlat : public GpuIndex {
public:
    /// Construct from a pre-existing faiss::IndexFlat instance, copying data
    GpuIndexFlat(GpuResources* resources,
                 const IndexFlat* index,
                 GpuIndexFlatConfig config = GpuIndexFlatConfig());

    /// Construct an empty instance
    GpuIndexFlat(GpuResources* resources,
                 int dims,
                 MetricType metricType,
                 GpuIndexFlatConfig config = GpuIndexFlatConfig());

    ~GpuIndexFlat() override;

    /// Set the minimum data size for searches(in Mib), CPU -> GPU paging
    void setMinPagingSize(size_t size);

    /// Returns the current minimum data size for paged searches
    size_t getMinPagingSize() const;

    /// Copy from CPU index
    void copyFrom(const IndexFlat* index);

    /// Copy to a CPU index
    void copyTo(IndexFlat* index) const;

    /// Returns the number of vectors
    size_t getNumVecs() const;

    /// Clears all vectors
    void reset() override ;

    /// Overrides to avoid excessive copies
    void add(Index::idx_t n, const float* x) override ;

    /// Own implementation which handles CPU async copies; not call searchImpl_
    void search(Index::idx_t n,
                const float* x,
                Index::idx_t k,
                float* distances,
                Index::idx_t* labels) const override ;

    /// For internal access
    inline FlatIndex* getGpuData() { return data_; }

protected:
    /// Called from GpuIndex for add
    void addImpl_(Index::idx_t n, const float* x, const Index::idx_t* ids) override ;

    /// Should not be called
    void searchImpl_(Index::idx_t n,
                     const float* x,
                     Index::idx_t k,
                     float* distances,
                     Index::idx_t* labels) const override ;

    /// Called from search when the input data is on the CPU
    void searchFromCpuPaged_(int n,
                             const float* x,
                             int k,
                             float* outDIstancesData,
                             int* outIndicesData) const;

    void searchNonPaged_(int n,
                         const float* x,
                         int k,
                         float* outDistancesData,
                         int* outIndicesData) const;

private:
    /// Checks user settings for consitency
    void verifySettings_() const;

protected:
    /// Our config object
    const GpuIndexFlatConfig config_;

    /// Size above which we page copies from CPU to GPU
    size_t minPagedSize_;

    /// Holds GPU data in list of vectors
    FlatIndex* data_;
};

}}