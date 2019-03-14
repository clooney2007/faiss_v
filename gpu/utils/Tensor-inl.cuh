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
Tensor<T, Dim, InnerContig, IndexT, PtrTraits>::Tensor(
    Tensor<T, Dim, InnerContig, IndexT, PtrTraits>& t) {
    this->operator=(t);
}

template <typename T, int Dim, bool InnerContig,
    typename IndexT, template <typename U> class PtrTraits>
__host__ __device__
Tensor<T, Dim, InnerContig, IndexT, PtrTraits>::Tensor(
    Tensor<T, Dim, InnerContig, IndexT, PtrTraits>&& t) {
    this->operator=(std::move(t));
}

template <typename T, int Dim, bool InnerContig,
    typename IndexT, template <typename U> class PtrTraits>
__host__ __device__
Tensor<T, Dim, InnerContig, IndexT, PtrTraits>&
Tensor<T, Dim, InnerContig, IndexT, PtrTraits>::operator=(
    Tensor<T, Dim, InnerContig, IndexT, PtrTraits>& t) {
    data_ = t.data_;
    for (int i = 0; i < Dim; ++i) {
        size_[i] = t.size_[i];
        stride_[i] = t.stride_[i];
    }

    return *this;
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

template <typename T, int Dim, bool InnerContig,
    typename IndexT, template <typename U> class PtrTraits>
__host__ __device__
Tensor<T, Dim, InnerContig, IndexT, PtrTraits>::Tensor(
    DataPtrType data, const IndexT sizes[Dim], const IndexT strides[Dim])
    : data_(data) {
    static_assert(Dim > 0, "must have > 0 dimensions");

    for (int i = 0; i < Dim; ++i) {
        size_[i] = sizes[i];
        stride_[i] = strides[i];
    }
}

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
template <typename U>
__host__ __device__ Tensor<U, Dim, InnerContig, IndexT, PtrTraits>
Tensor<T, Dim, InnerContig, IndexT, PtrTraits>::castResize() {
    static_assert(sizeof(U) >= sizeof(T), "only handles greater sizes");
    constexpr int kMultiple = sizeof(U) / sizeof(T);

    GPU_FAISSV_ASSERT(canCastResize<U>());

    IndexT newSize[Dim];
    IndexT newStride[Dim];
    for (int i = 0; i < Dim - 1; ++i) {
        newSize[i] = size_[i];
        newStride[i] = stride_[i] / kMultiple;
    }

    newStride[Dim - 1] = 1; // this is the same as the old stride
    newSize[Dim - 1] = size_[Dim - 1] / kMultiple;

    return Tensor<U, Dim, InnerContig, IndexT, PtrTraits>(
        reinterpret_cast<U*>(data_), newSize, newStride);
};

template <typename T, int Dim, bool InnerContig,
    typename IndexT, template <typename U> class PtrTraits>
template <typename U>
__host__ __device__ const Tensor<U, Dim, InnerContig, IndexT, PtrTraits>
Tensor<T, Dim, InnerContig, IndexT, PtrTraits>::castResize() const {
    return const_cast<Tensor<T, Dim, InnerContig, IndexT, PtrTraits>*>(this)->
        castResize<U>();
}

template <typename T, int Dim, bool InnerContig,
    typename IndexT, template <typename U> class PtrTraits>
template <typename U>
__host__ __device__ bool
Tensor<T, Dim, InnerContig, IndexT, PtrTraits>::canCastResize() const {
    static_assert(sizeof(U) >= sizeof(T), "only handles greater sizes");
    constexpr int kMultiple = sizeof(U) / sizeof(T);

    // Ensure that the base pointer is sizeof(U) aligned
    if (((uintptr_t) data_) % sizeof(U) != 0) {
        return false;
    }

    // Check all outer strides
    for (int i = 0; i < Dim - 1; ++i) {
        if (stride_[i] % kMultiple != 0) {
            return false;
        }
    }

    // Check inner size
    if (size_[Dim - 1] % kMultiple != 0) {
        return false;
    }

    if (stride_[Dim - 1] != 1) {
        return false;
    }

    return true;
};

template <typename T, int Dim, bool InnerContig,
    typename IndexT, template <typename U> class PtrTraits>
template <typename NewIndexT>
__host__ Tensor<T, Dim, InnerContig, NewIndexT, PtrTraits>
Tensor<T, Dim, InnerContig, IndexT, PtrTraits>::castIndexType() const {
    if (sizeof(NewIndexT) < sizeof(IndexT)) {
        GPU_FAISSV_ASSERT(this->canUseIndexType<NewIndexT>());
    }

    NewIndexT newSize[Dim];
    NewIndexT newStride[Dim];
    for (int i = 0; i < Dim; ++i) {
        newSize[i] = (NewIndexT) size_[i];
        newStride[i] = (NewIndexT) stride_[i];
    }

    return Tensor<T, Dim, InnerContig, NewIndexT, PtrTraits>(
        data_, newSize, newStride);
};

template <typename T, int Dim, bool InnerContig,
    typename IndexT, template <typename U> class PtrTraits>
template <typename NewIndexT>
__host__ bool
Tensor<T, Dim, InnerContig, IndexT, PtrTraits>::canUseIndexType() const {
    static_assert(sizeof(size_t) >= sizeof(IndexT), "index size too large");
    static_assert(sizeof(size_t) >= sizeof(NewIndexT), "new index size too large");

    size_t  maxOffset = 0;

    for (int i = 0; i < Dim; ++i) {
        size_t curMaxOffset = (size_t)size_[i] * (size_t)stride_[i];
        if (curMaxOffset > maxOffset) {
            maxOffset = curMaxOffset;
        }
    }

    if (maxOffset > (size_t) std::numeric_limits<NewIndexT>::max()) {
        return false;
    }

    return true;
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

