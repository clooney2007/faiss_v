/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/05.
*/

#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#include <gtest/gtest.h>
#include <faiss_v/Index.h>
#include <faiss_v/IndexFlat.h>

namespace {
typedef faiss_v::Index::idx_t idx_t;

// parameters to use for the test
int d = 64;
size_t nb = 1000000;
size_t nq = 100;
int k = 10;

typedef faiss_v::Index::idx_t idx_t;

struct CommonData {

  std::vector <float> database;
  std::vector <float> queries;
  faiss_v::IndexFlatL2 index;

  CommonData(): database (nb * d), queries (nq * d), index(d) {
    srand48(time(0));
    for (size_t i = 0; i < nb * d; i++) {
      database[i] = drand48();
    }
    for (size_t i = 0; i < nq * d; i++) {
      queries[i] = drand48();
    }
    index.add(nb, database.data());
  }
};

CommonData cd;

TEST(INDEXFLATTEST, INDEXFLAT) {
  std::vector<idx_t> refI(k * nq);
  std::vector<float> refD(k * nq);

  faiss_v::IndexFlatL2 index = cd.index;
  auto  startTime = std::chrono::system_clock::now();
  index.search(nq, cd.queries.data(), k, refD.data(), refI.data());
  auto endTime = std::chrono::system_clock::now();

  double duration =  double(std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count())
                   * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;

  printf("Search time cost: %fs.\n", (duration));
//  for (int j = 0; j < nq; j++) {
//    for (int i = 0; i < k; i++) {
//      printf("Q: %d, Top: %d, idx: %d, dis: %f\n", j, i, refI[i + j * k], refD[i + j * k]);
//    }
//  }
  printf("Index num: %d\n", cd.index.xb.size());
}

}
