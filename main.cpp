#include <iostream>
#include "Index.h"
#include "IndexFlat.h"

int main() {
    faiss_v::IndexFlat d(1024, faiss_v::MetricType::METRIC_L2);
    d.add_with_ids(0, nullptr, nullptr);
    std::cout << "Hello, World!" << std::endl;
    return 0;
}