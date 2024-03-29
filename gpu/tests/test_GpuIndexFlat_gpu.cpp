/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/16.
*/

#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#include <gtest/gtest.h>
#include "../../Index.h"
#include "../../IndexFlat.h"
#include "../GpuIndexFlat.h"
#include "../StandardGpuResources.h"

namespace {
// FIXME: figure out a better way to test fp16
constexpr float kF16MaxRelErr = 0.07f;
constexpr float kF32MaxRelErr = 6e-3f;

struct TestFlatOptions {
    TestFlatOptions()
        : useL2(false),
          useFloat16(false),
          useTransposed(false),
          numVecsOverride(-1),
          numQueriesOverride(-1),
          kOverride(-1) {
    }

    bool useL2;
    bool useFloat16;
    bool useTransposed;
    int numVecsOverride;
    int numQueriesOverride;
    int kOverride;
};

#define randVal(a,b) a + (lrand48() % (b + 1 - a))

// parameters to use for the test
int d = 1024;
size_t nb = 100000;
size_t nq = 100;
int k = 10;

typedef faiss_v::Index::idx_t idx_t;

struct CommonData {
    std::vector<float> database;
    std::vector <float> queries;
    faiss_v::gpu::GpuIndexFlatL2* gpu_index;
    int numVecs;
    int dim;
    int device = 1;
    int numQuery;

    CommonData(const TestFlatOptions& opt) {
        srand48(time(0));
        numVecs = opt.numVecsOverride > 0 ? opt.numVecsOverride : randVal(50000, 100000);
        dim = randVal(1000, 1500);
        numQuery = opt.numQueriesOverride > 0 ? opt.numQueriesOverride : randVal(1, 512);

        faiss_v::gpu::GpuIndexFlatConfig config;
        config.device = device;
        config.storeTransposed = opt.useTransposed;
        config.useFlat16 = opt.useFloat16;

        int device = 0;
        faiss_v::gpu::StandardGpuResources res;
        res.noTempMemory();
        gpu_index = new faiss_v::gpu::GpuIndexFlatL2(&res, dim, config);

        std::cout << "here" << std::endl;
        database.resize(numVecs * dim);
        for (size_t i = 0; i < numVecs * dim; i++) {
            database[i] = drand48();
        }
        queries.resize(numVecs * dim);
        for (size_t i = 0; i < numVecs * dim; i++) {
            queries[i] = drand48();
        }
        std::cout << "here" << std::endl;
        gpu_index->add(numVecs, database.data());
        while (true) {}
    }
};

TEST(INDEXFLATTEST, INDEXFLAT) {
    TestFlatOptions opt;
    opt.useL2 = true;
    opt.useFloat16 = false;
    opt.useTransposed = false;

    CommonData cd(opt);
    std::vector<idx_t> refI(k * nq);
    std::vector<float> refD(k * nq);

    faiss_v::gpu::GpuIndexFlatL2* index = cd.gpu_index;
    auto  startTime = std::chrono::system_clock::now();
//  index.search(nq, cd.queries.data(), k, refD.data(), refI.data());
    auto endTime = std::chrono::system_clock::now();

    double duration =  double(std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count())
                   * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;

    printf("Search time cost: %fs.\n", (duration));
//  for (int j = 0; j < nq; j++) {
//    for (int i = 0; i < k; i++) {
//      printf("Q: %d, Top: %d, idx: %d, dis: %f\n", j, i, refI[i + j * k], refD[i + j * k]);
//    }
//  }
    printf("Index num: %ld\n", cd.gpu_index->ntotal);
}

}
