/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/01/26.
*/


#ifndef GPU_FAISSV_ASSERT_INCLUDED
#define GPU_FAISSV_ASSERT_INCLUDED

#include "../FaissVAssert.h"
#include <cuda.h>

#ifdef __CUDA_ARCH__
#define GPU_FAISSV_ASSERT(X) assert(X)
#define GPU_FAISSV_ASSERT_MSG(X, MSG) assert(X)
#define GPU_FAISSV_ASSERT_FMT(X, FMT, ...) assert(X)
#else
#define GPU_FAISSV_ASSERT(X) FAISSV_ASSERT(X)
#define GPU_FAISSV_ASSERT_MSG(X, MSG) FAISSV_ASSERT_MSG(X, MSG)
#define GPU_FAISSV_ASSERT_FMT(X, FMT, ...) FAISSV_ASSERT_FMT(X, FMT, __VA_ARGS)
#endif // __CUDA_ARCH__

#endif //GPU_FAISSV_ASSERT_INCLUDED
