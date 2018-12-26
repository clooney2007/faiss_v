/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2018/12/12.
*/

#ifndef FAISS_V_INDEXFLAT_H
#define FAISS_V_INDEXFLAT_H

#include <vector>
#include "Index.h"

namespace faiss_v {
    struct IndexFlat: Index {
        std::vector<float> xb;
        explicit IndexFlat(idx_t d, MetricType metric = METRIC_L2);

        void add(idx_t n, const float* x) override;
        void search(idx_t n, const float *x, idx_t k, float *distance, idx_t *labels) const override;
        void reset() override;

        IndexFlat() {}
    };
}


#endif //FAISS_V_INDEXFLAT_H
