/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2018/12/25.
*/

#include "utils.h"

#include <cstdio>
#include <cassert>
#include <cstring>
#include <cmath>

#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <omp.h>


#include <algorithm>
#include <vector>

#include "FaissVAssert.h"

#ifndef FINTEGER
#define FINTEGER long
#endif

extern "C" {
    /* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

    int sgemm_ (const char *transa, const char *transb, FINTEGER *m, FINTEGER *
                n, FINTEGER *k, const float *alpha, const float *a,
                            FINTEGER *lda, const float *b, FINTEGER *
                ldb, float *beta, float *c, FINTEGER *ldc);

    /* Lapack functions, see http://www.netlib.org/clapack/old/single/sgeqrf.c */

    int sgeqrf_ (FINTEGER *m, FINTEGER *n, float *a, FINTEGER *lda,
                 float *tau, float *work, FINTEGER *lwork, FINTEGER *info);

    int sorgqr_(FINTEGER *m, FINTEGER *n, FINTEGER *k, float *a,
                FINTEGER *lda, float *tau, float *work,
                FINTEGER *lwork, FINTEGER *info);
};

namespace faiss_v {
    /// Matrix/vector ops
    void fvec_norms_L2sqr (float* __restrict nr, const float* __restrict x,
            size_t d, size_t nx) {
#pragma omp parallel for
        for (size_t i = 0; i < nx; i++) {
            nr[i] = fvec_norm_L2sqr(x + i * d, d);
        }
    }

    /// KNN functions
    static void knn_L2sqr_sse (const float* x, const float* y,
                               size_t d, size_t nx, size_t ny,
                               float_maxheap_array_t* res) {
        size_t k = res->k;
#pragma omp parallel for
        for (size_t i = 0; i < nx; i++) {
            const float* x_i = x + i * d;
            const float* y_j = y;
            size_t  j;
            float* __restrict simi = res->get_val(i);
            long* __restrict idxi = res->get_id(i);

            maxheap_heapify(k, simi, idxi);
            for (j = 0; j < ny; j++) {
                float disij = fvec_L2sqr(x_i, y_j, d);

                if (disij < simi[0]) {
                    maxheap_pop(k, simi, idxi);
                    maxheap_push(k, simi, idxi, disij, j);
                }
                y_j += d;
            }
            maxheap_reorder(k, simi, idxi);
        }
    }

    template <class DistanceCorrection>
    static void knn_L2sqr_blas (const float* x, const float* y,
            size_t d, size_t nx, size_t ny, float_maxheap_array_t* res,
            const DistanceCorrection &corr) {
        res->heapify();

        // Omit empty matrices
        if (nx == 0 || ny == 0) return;

        size_t k = res->k;

        // block size
        const size_t  bs_x = 4096, bs_y = 1024;
        // const size_t bs_x = 16, bs_y = 16;
        float* ip_block = new float[bs_x * bs_y];

        float* x_norms = new float[nx];
        fvec_norms_L2sqr(x_norms, x, d, nx);

        float* y_norms = new float[ny];
        fvec_norms_L2sqr(y_norms, y, d, ny);

        for (size_t i0 = 0; i0 < nx; i0 += bs_x) {
            size_t i1 = i0 + bs_x;
            if (i1 > nx) i1 = nx;

            for (size_t j0 = 0; j0 < ny; j0 += bs_y) {
                size_t j1 = j0 + bs_y;
                if (j1 > ny) j1 = ny;

                // compute the dot products
                float one = 1, zero = 0;
                FINTEGER nyi = j1 - j0, nxi = i1 - i0, di = d;
                sgemm_ ("Transpose", "Not transpose", &nyi, &nxi, &di, &one,
                        y + j0 * d, &di,
                        x + i0 * d, &di, &zero,
                        ip_block, &nyi);

                // collect minima
#pragma omp parallel for
                for (size_t i = i0; i < i1; ++i) {
                    float* __restrict simi = res->get_val(i);
                    long* __restrict idxi = res->get_id(i);
                    const float* ip_line = ip_block + (i - i0) * (j1 - j0);

                    for (size_t j = j0; j < j1; j++) {
                        float ip = *ip_line++;
                        float dis2 = x_norms[i] + y_norms[j] - 2 * ip;

                        float dis = corr(dis2, i, j);
                        if (dis < simi[0]) {
                            maxheap_pop(k, simi, idxi);
                            maxheap_push(k, simi, idxi, dis, j);
                        }
                    }
                }
            }
        }

        res->reorder();

        delete [] ip_block;
        delete [] x_norms;
        delete [] y_norms;
    }

    /// KNN driver functions
    int distance_compute_blas_threshold = 20;

    struct NopDistanceCorrection {
        float operator()(float dis, size_t qno, size_t bno) const {
            return dis;
        }
    };

    void knn_L2sqr (const float* x, const float* y,
                    size_t d, size_t nx, size_t ny,
                    float_maxheap_array_t* res) {
        if (d % 4 == 0 && nx < distance_compute_blas_threshold) {
            knn_L2sqr_sse(x, y, d, nx, ny, res);
        } else {
            NopDistanceCorrection nop;
            knn_L2sqr_blas(x, y, d, nx, ny, res, nop);
        }
    }
}