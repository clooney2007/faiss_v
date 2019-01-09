/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/09.
*/

#pragma once

#include "Tensor.cuh"
#include "DeviceMemory.h"
#include "MemorySpace.h"

namespace faiss_v { namespace gpu {

template <typename T,
    int Dim,
    bool InnerContig = false,
    typename IndexT = int,
    template <typename U> class PtrTraits = traits::DefaultPtrTraits>
class DeviceTensor : public Tensor<T, Dim, InnerContig, IndexT, PtrTraits> {
public:
    typedef IndexT IndexType;
    typedef typename PtrTraits<T>::PtrType DataPtrType;

    /// Default constructor
    __host__ DeviceTensor();

    /// Destructor
    __host__ ~DeviceTensor();

private:
    enum AllocState {
        /// Tensor itself owns the memory, which must be freed via cudaFree
        Owner,
        /// Tensor itself is not an owner of the memory; there is nothing to free
        NotOwner,
        /// Tensor has the memory via a temporary memory reservation
        Reservation
    };

    AllocState state_;
    MemorySpace space_;
    DeviceMemoryReservation reservation_;

};

}}