/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/10.
*/


#pragma once

#include <cuda.h>

namespace faiss_v { namespace gpu { namespace utils {

template <typename U, typename V>
constexpr __host__ __device__ auto divUp(U a, V b) -> decltype(a + b) {
    return (a + b - 1) / b;
}

template <typename U, typename V>
constexpr __host__ __device__ auto roundUp(U a, V b) -> decltype(a + b) {
    return divUp(a, b) * b;
}

template <typename T>
constexpr __host__ __device__ int log2(T n, int p = 0) {
    return (n <= 1) ? p : log2(n / 2, p + 1);
}

template <typename T>
constexpr __host__ __device__ bool isPowerOf2(T v) {
    return (v && !(v & (v - 1)));
}

template <typename T>
constexpr __host__ __device__ T nextHighestPowerOf2(T v) {
    return (isPowerOf2(v) ? (T) 2 * v : ((T) 1 << (log2(v) + 1)));
}

}}}
