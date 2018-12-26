/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2018/12/17.
*/

#include "Heap.h"

namespace faiss_v {

    template<typename C>
    void HeapArray<C>::heapify() {
#pragma omp parallel for
        for (size_t j = 0; j < nh; j++) {
            heap_heapify<C>(k, val + j * k, ids + j * k);
        }
    }

    template<typename C>
    void HeapArray<C>::reorder() {
#pragma omp parallel for
        for (size_t j = 0; j < nh; j++) {
            heap_reorder<C>(k, val + j * k, ids + j * k);
        }
    }

    template<typename C>
    void HeapArray<C>::addn(size_t nj, const T *vin, TI j0,
                            size_t i0, long ni) {
        if (ni == -1) ni = nh;
        assert (i0 >= 0 && i0 + ni <= nh);
#pragma omp parallel for
        for (size_t i = i0; i < i0 + ni; i++) {
            T *__restrict simi = get_val(i);
            TI *__restrict idxi = get_id(i);
            const T *ip_line = vin + (i - i0) * nj;

            for (size_t j = 0; j < nj; j++) {
                T ip = ip_line[j];
                if (C::cmp(simi[0], ip)) {
                    heap_pop<C>(k, simi, idxi);
                    heap_push<C>(k, simi, idxi, ip, j + j0);
                }
            }
        }
    }

    template<typename C>
    void HeapArray<C>::addn_with_ids(size_t nj, const T *vin, const TI *id_in,
                                     long id_stride, size_t i0, long ni) {
        if (id_in == nullptr) {
            addn(nj, vin, 0, i0, ni);
            return;
        }
        if (ni == -1) ni = nh;
        assert (i0 >= 0 && i0 + ni <= nh);
#pragma omp parallel for
        for (size_t i = i0; i < i0 + ni; i++) {
            T *__restrict simi = get_val(i);
            TI *__restrict idxi = get_id(i);
            const T *ip_line = vin + (i - i0) * nj;
            const TI *id_line = id_in + (i - i0) * id_stride;

            for (size_t j = 0; j < nj; j++) {
                T ip = ip_line[j];
                if (C::cmp(simi[0], ip)) {
                    heap_pop<C>(k, simi, idxi);
                    heap_push<C>(k, simi, idxi, ip, id_line[j]);
                }
            }
        }
    }

    // explicit instanciations
    template class HeapArray<CMin <float, long> >;
    template class HeapArray<CMax <float, long> >;
    template class HeapArray<CMin <int, long> >;
    template class HeapArray<CMax <int, long> >;
}