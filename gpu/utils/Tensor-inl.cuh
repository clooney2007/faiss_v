/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/26.
*/

#include "../GpuFaissVAssert.h"
#include <limits>

namespace faiss_v {namespace gpu {

template <typename T, int Dim, bool InnerContig,
    typename IndexT, template <typename U> class PtrTraits>
__host__ __device__
Tensor<T, Dim, InnerContig, IndexT, PtrTraits>::Tensor()
    : data_(nullptr) {
    static_assert(Dim > 0, "must have > 0 dimensions");

    for (int i = 0; i < Dim; ++i) {
        size_[i] = 0;
        stride_[i] = (IndexT) 1;
    }
}

template <typename T, int Dim, bool InnerContig,
    typename IndexT, template <typename U> class PtrTraits>
__host__ __device__
Tensor<T, Dim, InnerContig, IndexT, PtrTraits>&
Tensor<T, Dim, InnerContig, IndexT, PtrTraits>::operator=(
    Tensor<T, Dim, InnerContig, IndexT, PtrTraits> &&t) {
    data_ = t.data_;
    this->data_ = nullptr;
    for (int i = 0; i < Dim; ++i) {
        stride_[i] = t.stride_[i];
        t.stride_[i] = 0;
        size_[i] = t.size_[i];
        t.size_[i] = 0;
    }

    return *this;
};

template <typename T, int Dim, bool InnerContig,
    typename IndexT, template <typename U> class PtrTraits>
__host__ __device__
Tensor<T, Dim, InnerContig, IndexT, PtrTraits>::Tensor(
    DataPtrType data, const IndexT sizes[Dim])
    : data_(data) {
    static_assert(Dim > 0, "must have > 0 dimensions");

    for (int i = 0; i < Dim; ++i) {
        size_[i] = sizes[i];
    }

    stride_[Dim - 1] = (IndexT) 1;
    for (int i = Dim - 2; i >= 0; --i) {
        stride_[i] = stride_[i + 1] * sizes[i + 1];
    }
};

template <typename T, int Dim, bool InnerContig,
    typename IndexT, template <typename U> class PtrTraits>
__host__ __device__
Tensor<T, Dim, InnerContig, IndexT, PtrTraits>::Tensor(
    DataPtrType data, std::initializer_list<IndexT> sizes)
    : data_(data) {
    GPU_FAISSV_ASSERT(sizes.size() == Dim);
    static_assert(Dim > 0, "must have > 0 dimensions");

    int i = 0;
    for (auto s : sizes) {
        size_[i++] = s;
    }

    stride_[Dim - 1] = (IndexT) 1;
    for (int i = Dim - 2; i >= 0; --i) {
        stride_[i] = stride_[i + 1] * size_[i + 1];
    }
};

template <typename T, int Dim, bool InnerContig, typename IndexT, template <typename U> class PtrTraits>
__host__ void
Tensor<T, Dim, InnerContig, IndexT, PtrTraits>::copyFrom(
    Tensor<T, Dim, InnerContig, IndexT, PtrTraits>& t, cudaStream_t stream) {
    GPU_FAISSV_ASSERT(this->isContiguous());

    GPU_FAISSV_ASSERT(this->numElements() == t.numElements());
    if (t.numElements() > 0) {
        GPU_FAISSV_ASSERT(this->data_);
        GPU_FAISSV_ASSERT(t.data());

        int ourDev = getDeviceForAddress(this->data_);
        int tDev = getDeviceForAddress(t.data());

        if (tDev == -1) {
            CUDA_VERIFY(cudaMemcpyAsync(t.data(),
                                        this->data_,
                                        this->getSizeInBytes(), ourDev == -1 ? cudaMemcpyHostToHost :
                                                               cudaMemcpyDeviceToHost,
                                        stream));
        } else {
            CUDA_VERIFY(cudaMemcpyAsync(t.data(),
                                        this->data_,
                                        this->getSizeInBytes(), ourDev == -1 ? cudaMemcpyHostToDevice :
                                                               cudaMemcpyDeviceToDevice,
                                        stream));
        }
    }
};

template <typename T, int Dim, bool InnerContig,
    typename IndexT, template <typename U> class PtrTraits>
__host__ __device__ size_t
Tensor<T, Dim, InnerContig, IndexT, PtrTraits>::numElements() const {
    size_t size = (size_t) getSize(0);

    for (int i = 1; i < Dim; ++i) {
        size *= (size_t) getSize(i);
    }

    return size;
};

template <typename T, int Dim, bool InnerContig,
    typename IndexT, template <typename U> class PtrTraits>
__host__ __device__ bool
Tensor<T, Dim, InnerContig, IndexT, PtrTraits>::isContiguous() const {
    long prevSize = 1;

    for (int i = Dim - 1; i >= 0; --i) {
        if (getSize(i) != (IndexT) 1) {
            if (getStride(i) == prevSize) {
                prevSize *= getSize(i);
            } else {
                return false;
            }
        }
    }

    return true;
};

}}

