/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/03/11.
*/

#pragma once

#include <cuda.h>

namespace faiss_v { namespace gpu {

template <typename T>
struct ConvertTo {
};

template <>
struct ConvertTo<float> {
    static inline __device__ float to(float v) { return v; }
};

template <>
struct ConvertTo<float2> {
    static inline __device__ float2 to(float2 v) { return v; }
};

template <>
struct ConvertTo<float4> {
    static inline __device__ float4 to(float4 v) { return v; }
};

}}
