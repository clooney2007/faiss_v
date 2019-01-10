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
    typedef Tensor<T, DIm, InnerContig, IndexT, PtrTraits> TensorType;

    /// Default constructor
    __host__ __device__ Tensor();

    /// Copy constructor
    __host__ __device__ Tensor(Tensor<T, Dim, InnerContig, IndexT, PtrTraits> &t);

    /// Move constructor
    __host__ __device__ Tensor(Tensor<T, Dim, InnerContig, IndexT, PtrTraits> &&t);

    /// Assignment
    __host__ __device__ Tensor<T, Dim, InnerContig, IndexT, PtrTraits>&
    operator=(Tensor<T, Dim, InnerContig, IndexT, PtrTraits>& t);

    /// Move Assignment
    __host__ __device__ Tensor<T, Dim, InnerContig, IndexT, PtrTraits>&
    operator=(Tensor<T, Dim, InnerContig, IndexT, PtrTraits>&& t);

    /// Constructor that calculates strides with no padding
    __host__ __device__ Tensor(DataPtrType data, const IndexT sizes[Dim]);
    __host__ __device__ Tensor(DataPtrType data, std::initializer_list<IndexT> sizes);

    /// Constructor that takes arbitrary size/stride arrays.
    __host__ __device__ Tensor(DataPtrType data, const IndexT sizes[Dim], const IndexT strides[Dim]);

    /// Copies a tensor into ourselves;
    __host__ void copyFrom(Tensor<T, Dim, InnerContig, IndexT, PtrTraits>& t, cudaStream_t stream);

    /// Copies ourselves into a tensor;
    __host__ void copyTo(Tensor<T, Dim, InnerContig, IndexT, PtrTraits>& t, cudaStream_t stream);

    /// Returns true if the two tensors are of the same dimensionality
    template <typename OtherT, int OtherDim>
    __host__ __device__ bool isSame(const Tensor<OtherT, OtherDim, InnerContig, IndexT, PtrTraits>& rhs) const;

    /// Returns true if the two tensors are of the same dimensionality
    template <typename OtherT, int OtherDim>
    __host__ __device__ bool isSameSize(const Tensor<OtherT, OtherDim, InnerContig, IndexT, PtrTraits>& rhs) const;

    /// Cast to a tensor of a different type of the same size
    template <typename U>
    __host__ __device__ Tensor<U, Dim, InnerContig, IndexT, PtrTraits> cast();

    /// Const version of 'cast'
    template <typename U>
    __host__ __device__ const Tensor<U, Dim, InnerContig, IndexT, PtrTraits> cast() const;

    /// Returns a raw pointer to the start of our data.
    __host__ __device__ inline DataPtrType data() {
        return data_;
    }

    /// Returns the total number of elements contained within our data (product of `getSize(i)`)
    __host__ __device__ size_t numElements() const;

    /// If we are contiguous, returns the total size in bytes of our data
    __host__ __device__ size_t getSizeInBytes() const {
        return numElements() * sizeof(T);
    }

    /// Returns the size array.
    __host__ __device__ inline const IndexT* sizes() const {
        return size_;
    }

protected:
    /// Raw pointer to where the tensor data begins
    DataPtrType data_;

    /// Array of strides (in sizeof(T) terms) per each dimension
    IndexT stride_[Dim];

    /// Size per each dimension
    IndexT size_[Dim];
};

}}