/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/24.
*/

#include <utility>

namespace faiss_v {namespace gpu {

template <typename T, int Dim, bool InnerContig, typename IndexT, template <typename U> class PtrTraits>
__host__
DeviceTensor<T, Dim, InnerContig, IndexT, PtrTraits>::DeviceTensor() :
    Tensor<T, Dim, InnerContig, IndexT, PtrTraits>(), state_(AllocState::NotOwner), space_(MemorySpace::Device) {

}

template <typename T, int Dim, bool InnerContig, typename IndexT, template <typename U> class PtrTraits>
__host__
DeviceTensor<T, Dim, InnerContig, IndexT, PtrTraits>::~DeviceTensor() {
    if (state_ == AllocState::Owner) {
        FAISSV_ASSERT(this->data_ || (this->getSizeInBytes() == 0));
        CUDA_VERIFY(cudaFree(this->data_));
        this->data_ = nullptr;
    }
}

template <typename T, int Dim, bool InnerContig,
    typename IndexT, template <typename U> class PtrTraits>
__host__
DeviceTensor<T, Dim, InnerContig, IndexT, PtrTraits>::DeviceTensor(
    const IndexT sizes[Dim],
    MemorySpace space) :
    Tensor<T, Dim, InnerContig, IndexT, PtrTraits>(nullptr, sizes),
    state_(AllocState::Owner),
    space_(space) {

        allocMemorySpace(space, (void**)&this->data_, this->getSizeInBytes());
        FAISSV_ASSERT(this->data_ || (this->getSizeInBytes() == 0));
}

template <typename T, int Dim, bool InnerContig,
    typename IndexT, template <typename U> class PtrTraits>
__host__
DeviceTensor<T, Dim, InnerContig, IndexT, PtrTraits>::DeviceTensor(
    std::initializer_list<IndexT> sizes,
    MemorySpace space) :
    Tensor<T, Dim, InnerContig, IndexT, PtrTraits>(nullptr, sizes),
    state_(AllocState::Owner),
    space_(space) {

        allocMemorySpace(space, (void**)&this->data_, this->getSizeInBytes());
        FAISSV_ASSERT(this->data_ || (this->getSizeInBytes() == 0));
}

template <typename T, int Dim, bool InnerContig, typename IndexT, template <typename U> class PtrTraits>
__host__
DeviceTensor<T, Dim, InnerContig, IndexT, PtrTraits>::DeviceTensor(
    DeviceTensor<T, Dim, InnerContig, IndexT,
                 PtrTraits>&& t) :
    Tensor<T, Dim, InnerContig, IndexT, PtrTraits>(),
        state_(AllocState::NotOwner),
        space_(MemorySpace::Device) {
    this->operator=(std::move(t));
}

template <typename T, int Dim, bool InnerContig, typename IndexT, template <typename U> class PtrTraits>
__host__
DeviceTensor<T, Dim, InnerContig, IndexT, PtrTraits>&
DeviceTensor<T, Dim, InnerContig, IndexT, PtrTraits>::operator=(
    DeviceTensor<T,Dim,InnerContig,IndexT,PtrTraits> &&t) {
    if (this->state_ == AllocState::Owner) {
        CUDA_VERIFY(cudaFree(this->data_));
    }

    this->Tensor<T, Dim, InnerContig, IndexT, PtrTraits>::operator=(
        std::move(t));
    this->state_ = t.state_;
    t.state_ = AllocState ::NotOwner;
    this->space_ = t.space_;
    this->reservation_ = std::move(t.reservation_);

    return *this;
}

// memory reservation constructor
template <typename T, int Dim, bool InnerContig,
    typename IndexT, template <typename U> class PtrTraits>
__host__
DeviceTensor<T, Dim, InnerContig, IndexT, PtrTraits>::DeviceTensor(
    DeviceMemory& m,
    const IndexT sizes[Dim],
    cudaStream_t stream,
    MemorySpace space) :
Tensor<T, Dim, InnerContig, IndexT, PtrTraits>(nullptr, sizes),
state_(AllocState::Reservation),
space_(space) {

    // FIXME: add MemorySpace to DeviceMemory
    auto memory = m.getMemory(stream, this->getSizeInBytes());

    this->data_ = (T*) memory.get();
    FAISSV_ASSERT(this->data_ || (this->getSizeInBytes() == 0));
    reservation_ = std::move(memory);
}

// memory reservation constructor
template <typename T, int Dim, bool InnerContig,
    typename IndexT, template <typename U> class PtrTraits>
__host__
DeviceTensor<T, Dim, InnerContig, IndexT, PtrTraits>::DeviceTensor(
    DeviceMemory& m,
    std::initializer_list<IndexT> sizes,
    cudaStream_t stream,
    MemorySpace space) :
Tensor<T, Dim, InnerContig, IndexT, PtrTraits>(nullptr, sizes),
state_(AllocState::Reservation),
space_(space) {

    // FIXME: add MemorySpace to DeviceMemory
    auto memory = m.getMemory(stream, this->getSizeInBytes());

    this->data_ = (T*) memory.get();
    FAISSV_ASSERT(this->data_ || (this->getSizeInBytes() == 0));
    reservation_ = std::move(memory);
}

template <typename T, int Dim, bool InnerContig, typename IndexT, template <typename U> class PtrTraits>
__host__
DeviceTensor<T, Dim, InnerContig, IndexT, PtrTraits>::DeviceTensor(
    DataPtrType data,
    const IndexT sizes[Dim],
    MemorySpace space) :
        Tensor<T, Dim, InnerContig, IndexT, PtrTraits>(data, sizes),
        state_(AllocState::NotOwner),
        space_(space){

}

template <typename T, int Dim, bool InnerContig, typename IndexT, template <typename U> class PtrTraits>
__host__
DeviceTensor<T, Dim, InnerContig, IndexT, PtrTraits>::DeviceTensor(
    DataPtrType data,
    std::initializer_list<IndexT> sizes,
    MemorySpace space) :
        Tensor<T, Dim, InnerContig, IndexT, PtrTraits>(data, sizes),
        state_(AllocState::NotOwner),
        space_(space) {

}

}}

