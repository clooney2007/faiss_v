/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/09.
*/

#pragma once

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <initializer_list>

/// Multi-dimensional array class for CUDA device and host usage.
/// Originally from Facebook's fbcunn, since added to the Torch GPU
/// library cutorch as well.

namespace faiss_v { namespace gpu {

template <typename T,
    int Dim,
    bool InnerContig,
    typename IndexT,
    template <typename U> class PtrTraits>
class Tensor;

/// Type of a subspace of a tensor
namespace detail {
template <typename TensorType,
    int SubDim,
    template <typename U> class PtrTraits>
class SubTensor;
}

namespace traits {
template <typename T>
struct RestrictPtrTraits {
    typedef T* __restrict__ PtrType;
};

template <typename T>
struct DefaultPtrTraits {
    typedef T* PtrType;
};
}

/**
   Templated multi-dimensional array that supports strided access of
   elements. Main access is through `operator[]`; e.g.,
   `tensor[x][y][z]`.

   - `T` is the contained type (e.g., `float`)
   - `Dim` is the tensor rank
   - If `InnerContig` is true, then the tensor is assumed to be innermost
   - contiguous, and only operations that make sense on contiguous
   - arrays are allowed (e.g., no transpose). Strides are still
   - calculated, but innermost stride is assumed to be 1.
   - `IndexT` is the integer type used for size/stride arrays, and for
   - all indexing math. Default is `int`, but for large tensors, `long`
   - can be used instead.
   - `PtrTraits` are traits applied to our data pointer (T*). By default,
   - this is just T*, but RestrictPtrTraits can be used to apply T*
   - __restrict__ for alias-free analysis.
*/
template <typename T,
    int Dim,
    bool InnerContig = false,
    typename IndexT = int,
    template <typename U> class PtrTraits = traits::DefaultPtrTraits>
class Tensor {
public:
    enum { NumDim = Dim };
    typedef T DataType;
    typedef IndexT IndexType;
    enum { IsInnerContig = InnerContig};
    typedef typename PtrTraits<T>::PtrType DataPtrType;
    typedef Tensor<T, Dim, InnerContig, IndexT, PtrTraits> TensorType;

    /// Default constructor
    __host__ __device__ Tensor();

    /// Move Assignment
    __host__ __device__ Tensor<T, Dim, InnerContig, IndexT, PtrTraits>&
    operator=(Tensor<T, Dim, InnerContig, IndexT, PtrTraits>&& t);

    /// Constructor that calculates strides with no padding
    __host__ __device__ Tensor(DataPtrType data, const IndexT sizes[Dim]);
    __host__ __device__ Tensor(DataPtrType data, std::initializer_list<IndexT> sizes);

    /// Copies a tensor into ourselves; sizes must match
    __host__ void copyFrom(Tensor<T, Dim, InnerContig, IndexT, PtrTraits>& t,
                           cudaStream_t stream);

    /// Returns a raw pointer to the start of our data.
    __host__ __device__ inline DataPtrType data() {
        return data_;
    }

    /// Returns the size of a given dimension, `[0, Dim - 1]`. No bounds
    /// checking.
    __host__ __device__ inline IndexT getSize(int i) const {
        return size_[i];
    }

    /// Returns the stride of a given dimension, `[0, Dim - 1]`. No bounds
    /// checking.
    __host__ __device__ inline IndexT getStride(int i) const {
        return stride_[i];
    }

    /// Returns the total number of elements contained within our data (product of `getSize(i)`)
    __host__ __device__ size_t numElements() const;

    /// If we are contiguous, returns the total size in bytes of our
    /// data
    __host__ __device__ size_t getSizeInBytes() const {
        return numElements() * sizeof(T);
    }

    /// Returns the size array.
    __host__ __device__ inline const IndexT* sizes() const {
        return size_;
    }

    /// Returns true if there is no padding within the tensor and no
    /// re-ordering of the dimensions.
    /// ~~~
    /// (stride(i) == size(i + 1) * stride(i + 1)) && stride(dim - 1) == 0
    /// ~~~
    __host__ __device__ bool isContiguous() const;

protected:
    /// Raw pointer to where the tensor data begins
    DataPtrType data_;

    /// Array of strides (in sizeof(T) terms) per each dimension
    IndexT stride_[Dim];

    /// Size per each dimension
    IndexT size_[Dim];
};

}}

#include "Tensor-inl.cuh"