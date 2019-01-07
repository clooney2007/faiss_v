/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2018/12/12.
*/

#include "IndexFlat.h"
#include "Heap.h"
#include "utils.h"

namespace faiss_v {
    IndexFlat::IndexFlat(idx_t d, MetricType metric) : Index(d, metric) {

    }

    void IndexFlat::add(idx_t n, const float *x) {
        xb.insert(xb.end(), x, x + n * d);
        ntotal += n;
    }

    void IndexFlat::reset() {
        xb.clear();
        ntotal = 0;
    }

    void IndexFlat::search(idx_t n, const float *x, idx_t k,
                           float *distances, idx_t *labels) const {
        if (metric_type == METRIC_INNER_PRODUCT) {
//            float_minheap_array_t res = {size_t(n), size_t(k), labels, distances};
//            knn_inner_product (x, xb.data(), d, n, ntotal, &res);
        } else if (metric_type == METRIC_L2) {
            float_maxheap_array_t res = {size_t(n), size_t(k), labels, distances};
            knn_L2sqr(x, xb.data(), d, n, ntotal, &res);
        }
    }
}