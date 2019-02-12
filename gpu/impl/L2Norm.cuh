/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/02/12.
*/


#pragma once

#include "../utils/Tensor.cuh"

namespace faiss_v { namespace gpu

void runL2Norm(Tensor<float, 2, true>& input,
               Tensor<float, 1, true>& output,
               bool normSquared,
               cudaStream_t stream);

}}


