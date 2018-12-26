/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2018/12/17.
*/

#ifndef FAISS_V_HEAP_H
#define FAISS_V_HEAP_H

#include <climits>
#include <cstring>
#include <cmath>

#include <cassert>
#include <cstdio>

#include <limits>

namespace faiss_v {
    /// C object: uniform handling of min and max heap
    template<typename T_, typename TI_>
    struct CMax;

    template<typename T_, typename TI_>
    struct CMin {
        typedef T_ T;
        typedef TI_ TI;
        typedef CMax<T_, TI_> Crev;

        inline static bool cmp(T a, T b) {
            return a < b;
        }

        inline static T neutral() {
            return -std::numeric_limits<T>::max();
        }
    };

    template<typename T_, typename TI_>
    struct CMax {
        typedef T_ T;
        typedef TI_ TI;
        typedef CMin<T_, TI_> Crev;

        inline static bool cmp(T a, T b) {
            return a > b;
        }

        inline static T neutral() {
            return std::numeric_limits<T>::min();
        }
    };

    /// Basic heap operations: push and pop
    /***
     * Pops the top element from the heap defined by bh_val[0..k-1] and
     * bh_ids[0..k-1].
     */
    template<class C>
    inline
    void heap_pop(size_t k, typename C::T *bh_val, typename C::TI *bh_ids) {
        bh_ids--;
        bh_val--;
        typename C::T val = bh_val[k];
        size_t i = 1, i1, i2;
        while (1) {
            i1 = i << 1;
            i2 = i1 + 1;
            if (i1 > k)
                break;
            // left child
            if (i2 == k + 1 || C::cmp(bh_val[i1], bh_val[i2])) {
                if (C::cmp(val, bh_val[i1]))
                    break;
                bh_val[i] = bh_val[i1];
                bh_ids[i] = bh_ids[i1];
                i = i1;
            }
                // right child
            else {
                if (C::cmp(val, bh_val[i2]))
                    break;
                bh_val[i] = bh_val[i2];
                bh_ids[i] = bh_ids[i2];
                i = i2;
            }
        }
        bh_val[i] = bh_val[k];
        bh_ids[i] = bh_ids[k];
    }

    /**
     * Pushes the element(val, ids) into the heap bh_val[0..k-2] and bh_ids[0..k-2].
     * on output the element at k-1 is defined.
     */
    template<class C>
    inline
    void heap_push(size_t k, typename C::T *bh_val, typename C::TI *bh_ids,
                   typename C::T val, typename C::TI ids) {
        bh_val--;
        bh_ids--;
        size_t i = k, i_father;
        while (i > 1) {
            i_father = i >> 1;
            if (!C::cmp(val, bh_val[i_father]))
                break;
            bh_val[i] = bh_val[i_father];
            bh_ids[i] = bh_ids[i_father];
        }
        bh_val[i] = val;
        bh_ids[i] = ids;
    }

    template <typename T> inline
    void minheap_pop (size_t k, T* bh_val, long* bh_ids) {
        heap_pop<CMin<T, long>> (k, bh_val, bh_ids);
    }

    template <typename T> inline
    void maxheap_pop (size_t k, T* bh_val, long* bh_ids) {
        heap_pop<CMax<T, long>> (k, bh_val, bh_ids);
    }

    template <typename T> inline
    void minheap_push (size_t k, T* bh_val, long* bh_ids) {
        heap_pop<CMax<T, long> > (k, bh_val, bh_ids);
    }

    template <typename T> inline
    void maxheap_push (size_t k, T * bh_val, long * bh_ids, T val, long ids) {
        heap_push<CMax<T, long> > (k, bh_val, bh_ids, val, ids);
    }

    /// Heap initialization
    /**
     * Initialization phase for the heap (with unconditional pushes).
     * Store k0 elements in a heap containing up to k values.
     */
    template<class C>
    inline
    void heap_heapify(size_t k, typename C::T *bh_val, typename C::TI *bh_ids,
                      const typename C::T *x = nullptr, const typename C::TI *ids = nullptr,
                      size_t k0 = 0) {
        if (k0 > 0) assert(x);

        if (ids) {
            for (size_t i = 0; i < k0; i++)
                heap_push<C>(i + 1, bh_val, bh_ids, x[i], ids[i]);
        } else {
            for (size_t i = 0; i < k0; i++)
                heap_push<C>(i + 1, bh_val, bh_ids, x[i], i);
        }

        for (size_t i = k0; i < k; i++) {
            bh_val[i] = C::neutral();
            bh_ids[i] = -1;
        }
    }

    template <typename T> inline
    void minheap_heapify (size_t k, T *  bh_val, long * bh_ids,
            const T * x = nullptr, const long * ids = nullptr, size_t k0 = 0)
    {
        heap_heapify< CMin<T, long> > (k, bh_val, bh_ids, x, ids, k0);
    }


    template <typename T> inline
    void maxheap_heapify (size_t k, T * bh_val, long * bh_ids,
            const T * x = nullptr, const long * ids = nullptr, size_t k0 = 0)
    {
        heap_heapify< CMax<T, long> > (k, bh_val, bh_ids, x, ids, k0);
    }

    /// Add some elements to the heap
    template<class C>
    inline
    void heap_addn(size_t k, typename C::T *bh_val, typename C::TI *bh_ids,
                   const typename C::T *x, const typename C::TI *ids, size_t n) {
        size_t i;
        if (ids) {
            for (i = 0; i < n; i++) {
                if (C::cmp(bh_val[0], x[i])) {
                    heap_pop<C>(k, bh_val, bh_ids);
                    heap_push<C>(k, bh_val, bh_ids, x[i], ids[i]);
                }
            }
        } else {
            for (i = 0; i < n; i++) {
                if (C::cmp(bh_val[0], x[i])) {
                    heap_pop<C>(k, bh_val, bh_ids);
                    heap_push<C>(k, bh_val, bh_ids, x[i], i);
                }
            }
        }
    }

    template <typename T> inline
    void minheap_addn (size_t k, T * bh_val, long * bh_ids,
                       const T * x, const long * ids, size_t n)
    {
        heap_addn<CMin<T, long> > (k, bh_val, bh_ids, x, ids, n);
    }

    template <typename T> inline
    void maxheap_addn (size_t k, T * bh_val, long * bh_ids,
                       const T * x, const long * ids, size_t n)
    {
        heap_addn<CMax<T, long> > (k, bh_val, bh_ids, x, ids, n);
    }

    /// Heap finialization (reorder elements)
    template <typename C> inline
    size_t heap_reorder (size_t k, typename C::T* bh_val, typename C::TI* bh_ids) {
        size_t i, ii;
        for (i = 0, ii = 0; i < k; i++) {
            // top element should be put at the end of the list
            typename C::T val = bh_val[0];
            typename C::TI id = bh_ids[0];

            // boundary case: we will over-ride this value if not a true element
            heap_pop<C> (k-i, bh_val, bh_ids);
            bh_val[k-ii-1] = val;
            bh_ids[k-ii-1] = id;
            if (id != -1) ii++;
        }

        // count the number of elements which are effectively returned
        size_t nel = ii;

        memmove(bh_val, bh_val+k-ii, ii * sizeof(*bh_val));
        memmove(bh_ids, bh_ids+k-ii, ii * sizeof(*bh_ids));

        for (; ii < k; ii++) {
            bh_val[ii] = C::neutral();
            bh_ids[ii] = -1;
        }

        return nel;
    }

    template <typename T> inline
    size_t minheap_reorder (size_t k, T * bh_val, long * bh_ids)
    {
        return heap_reorder< CMin<T, long> > (k, bh_val, bh_ids);
    }

    template <typename T> inline
    size_t maxheap_reorder (size_t k, T * bh_val, long * bh_ids)
    {
        return heap_reorder< CMax<T, long> > (k, bh_val, bh_ids);
    }

    /// Operations on heap arrays
    template<typename C>
    struct HeapArray {
        typedef typename C::TI TI;
        typedef typename C::T T;

        size_t nh;  /// number of heap
        size_t k;   /// size per heap
        TI *ids;    /// id (nh * k)
        T *val;     /// distance,size nh * k

        /// return value
        T *get_val(size_t key) {
            return val + key * k;
        }

        /// return ids
        TI *get_id(size_t key) {
            return ids + key * k;
        }

        /// prepare all the heaps before adding
        void heapify();

        /**
         * add nj elements to heap i0:i0+ni
         *
         * @param nj        nb of elements to add
         * @param vin       elements to add
         * @param j0        add this to the ids that are added
         * @param i0        first heap to update
         * @param ni        nb of elements to update (-1 = use nh)
         */
        void addn(size_t nj, const T *vin, TI j0 = 0, size_t i0 = 0, long ni = -1);

        /**
         * same as addn, but add id_in and id_stride
         *
         * @param nj        nb of elements to add
         * @param vin       elements to add
         * @param id_in     add this to the ids that are added
         * @param id_stride stride for id_in
         * @param i0        first heap to update
         * @param ni        nv of elements to update (-1 = use nh)
         */
        void addn_with_ids(size_t nj, const T *vin, const TI *id_in = nullptr, long id_stride = 0, size_t i0 = 0,
                           long ni = -1);

        /// reorder all the heaps
        void reorder();

        /**
         * find the per-line extrema of each line of array D
         *
         * @param vals_out      extrema value of each line(size nh, or NULL)
         * @param idx_out       index of extrema value (size nh or NULL)
         */
        void per_line_extrema(T *vals_out, TI *idx_out) const;
    };

    /* Define useful heaps */
    typedef HeapArray<CMin<float, long> > float_minheap_array_t;
    typedef HeapArray<CMin<int, long> > int_minheap_array_t;

    typedef HeapArray<CMax<float, long> > float_maxheap_array_t;
    typedef HeapArray<CMax<int, long> > int_maxheap_array_t;
}

#endif //FAISS_V_HEAP_H
