/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/03/13.
*/

#pragma once

#include "MathOperators.cuh"
#include <cuda.h>

namespace faiss_v { namespace gpu {

template<typename T>
struct Sum {
    __device__ inline T operator()(T a, T b) const {
        return Math<T>::add(a, b);
    }

    inline __device__ T identity() const {
        return Math<T>::zero();
    }
};

}}