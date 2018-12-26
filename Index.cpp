/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2018/12/11.
*/

#include "Index.h"

namespace faiss_v {
    Index::~Index() {}

    void Index::add(idx_t n, const float *x) {
        FAISSV_THROW_MSG("add not implement.");
    }

    void Index::add_with_ids(
            idx_t n,
            const float* x,
            const long* xids) {
        FAISSV_THROW_MSG ("add_with_ids not implemented for this type of index");
    }

    void Index::search(idx_t n, const float *x, idx_t k, float *distance, idx_t *labels) const {
        FAISSV_THROW_MSG("search not implement.");
    }

    void Index::reset() {
        FAISSV_THROW_MSG("reset not implement.");
    }
}