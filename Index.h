/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2018/12/11.
*/

#ifndef FAISS_V_INDEX_H
#define FAISS_V_INDEX_H

#include "FaissVAssert.h"

namespace faiss_v {

enum MetricType {
    METRIC_INNER_PRODUCT = 0,
    METRIC_L2 = 1,
};

struct Index {
    typedef long idx_t;
    int d;
    idx_t ntotal;

    /// set if the Index does not require training, or if training is done already
    bool is_trained;

    /// type of metric this index uses for search
    MetricType metric_type;

    explicit Index(idx_t d = 0, MetricType metric = METRIC_L2) :
            d(d),
            ntotal(0),
            is_trained(true),
            metric_type(metric) {}

    virtual ~Index();

    /** Add n vectors of dimension d to the index.
     *
     * Vectors are implicitly assigned labels ntotal .. ntotal + n - 1
     * This function slices the input vectors in chuncks smaller than
     * blocksize_add and calls add_core.
     * @param x      input matrix, size n * d
     */
    virtual void add(idx_t n, const float *x) = 0;

    /** Same as add, but stores xids instead of sequential ids.
     *
     * The default implementation fails with an assertion, as it is
     * not supported by all indexes.
     *
     * @param xids if non-null, ids to store for the vectors (size n)
     */
    virtual void add_with_ids(idx_t n, const float *x, const long *xids);

    /** query n vectors of dimension d to the index.
     *
     * return at most k vectors. If there are not enough results for a
     * query, the result array is padded with -1s.
     *
     * @param x           input vectors to search, size n * d
     * @param labels      output labels of the NNs, size n*k
     * @param distances   output pairwise distances, size n*k
     */
    virtual void search(idx_t n, const float *x, idx_t k, float *distance, idx_t *labels) const = 0;

    /// removes all elements from the database.
    virtual void reset() = 0;
};
}


#endif //FAISS_V_INDEX_H
