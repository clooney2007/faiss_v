/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/08.
*/

#pragma once

#include "../GpuResources.h"
#include "../utils/MemorySpace.h"

namespace faiss_v { namespace gpu {

class FlatIndex {
public:
    FlatIndex(GpuResources* res,
              int dim,
              bool l2Distance,
              bool useFloat16,
              bool useFloat16Accumulator,
              bool storeTransposed, MemorySpace space);

    bool getUseFloat16() const;

    /// Returns the number of vectors we contain
    int getSize() const;

    int getDim() const;

    /// Reserve storage that can contain at least this many vectors
    void reserve(size_t numVecs, cudaStream_t stream);
};

}}
