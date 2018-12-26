/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2018/12/25.
*/

#ifndef FAISS_V_UTILS_H
#define FAISS_V_UTILS_H


#include "Heap.h"

namespace faiss_v {

    ///Optimized distance/norm/inner prod computations
    /* Squared L2 distance between two vectors */
    float fvec_L2sqr (
            const float * x,
            const float * y,
            size_t d);

    /* SSE-implementation of inner product and L2 distance */
    float  fvec_inner_product (
            const float * x,
            const float * y,
            size_t d);

    /** squared norm of a vector */
    float fvec_norm_L2sqr (const float * x,
                           size_t d);

    /** compute the L2 norms for a set of vectors
     *
     * @param  ip       output norms, size nx
     * @param  x        set of vectors, size nx * d
     */
    void fvec_norms_L2 (float * ip, const float * x, size_t d, size_t nx);

    /// same as fvec_norms_L2, but computes square norms
    void fvec_norms_L2sqr (float * ip, const float * x, size_t d, size_t nx);


    /// KNN functions
    /** Return the k nearest neighors of each of the nx vectors x among the ny
     *  vector y, w.r.t to max inner product
     *
     * @param x    query vectors, size nx * d
     * @param y    database vectors, size ny * d
     * @param res  result array, which also provides k. Sorted on output
     */
    void knn_inner_product(
            const float *x,
            const float *y,
            size_t d, size_t nx, size_t ny,
            float_minheap_array_t *res);

    /** Same as knn_inner_product, for the L2 distance */
    void knn_L2sqr(
            const float *x,
            const float *y,
            size_t d, size_t nx, size_t ny,
            float_maxheap_array_t *res);
}


#endif //FAISS_V_UTILS_H
