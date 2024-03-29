/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2019/02/12.
*/


#include "L2Norm.cuh"
#include "../utils/DeviceDefs.cuh"
#include "../utils/Reductions.cuh"
#include "../utils/MathOperators.cuh"
#include "../utils/PtxUtils.cuh"
#include "../utils/DeviceUtils.h"
#include "../utils/StaticUtils.h"
#include "../utils/ConversionOperators.cuh"

namespace faiss_v { namespace gpu {

// Input: (batch x dim), # repeats
// Output: (# repeats, norm of batch vector)
// Done under the presumption that the dimension size is not too large
// (<10k or so), since there wouldn't be enough parallelism applying a
// single block to the problem. Also that each vector is large enough
// (>64), since a single block works on multiple rows' norms at the
// same time.
// T: the type we are doing the math in (e.g., float, half)
// TVec: the potentially vectorized type we are loading in (e.g.,
// float4, half2)
template <typename T, typename TVec, typename TIndex,
    int RowTileSize, bool NormLoop, bool NormSquared>
__global__ void l2Norm(Tensor<TVec, 2, true, TIndex> input, Tensor<T, 1, true, TIndex> output) {
    extern __shared__ char smemByte[];
    T* smem = (T*)smemByte;

    TIndex numWarps = utils::divUp(blockDim.x, kWarpSize);
    TIndex laneId = getLaneId();
    TIndex warpId = threadIdx.x / kWarpSize;

    bool lastRowTile = (blockIdx.x == (gridDim.x - 1));
    TIndex rowStart = RowTileSize * blockIdx.x;
    T rowNorm[RowTileSize];

    if (lastRowTile) {
        for (TIndex row = 0; row < input.getSize(0) - rowStart; ++row) {
            if (NormLoop) {
                rowNorm[0] = Math<T>::zero();

                for (TIndex col = threadIdx.x; col < input.getSize(1); col += blockDim.x) {
                    TVec val = input[rowStart + row][col];
                    val = Math<TVec>::mul(val, val);
                    rowNorm[0] = Math<T>::add(rowNorm[0], Math<TVec>::reduceAdd(val));
                }
            } else {
                TVec val = input[rowStart + row][threadIdx.x];
                val = Math<TVec>::mul(val, val);
                rowNorm[0] = Math<TVec>::reduceAdd(val);
            }

            rowNorm[0] = warpReduceAllSum(rowNorm[0]);
            if (laneId == 0) {
                smem[row * numWarps + warpId] = rowNorm[0];
            }
        }
    } else {
        // We are guaranteed that all RowTileSize rows are available in
        // [rowStart, rowStart + RowTileSize)
        if (NormLoop) {
            // A single block of threads is not big enough to span each vector
            TVec tmp[RowTileSize];

#pragma unroll
            for (int row = 0; row < RowTileSize; ++row) {
                rowNorm[row] = Math<T>::zero();
            }

            for (TIndex col = threadIdx.x; col < input.getSize(1); col += blockDim.x) {
#pragma unroll
                for (int row = 0; row < RowTileSize; ++row) {
                    tmp[row] = input[rowStart + row][col];
                }

#pragma unroll
                for (int row = 0; row < RowTileSize; ++row) {
                    tmp[row] = Math<TVec>::mul(tmp[row], tmp[row]);
                }

#pragma unroll
                for (int row = 0; row < RowTileSize; ++row) {
                    rowNorm[row] = Math<T>::add(rowNorm[row], Math<TVec>::reduceAdd(tmp[row]));
                }
            }
        } else {
            TVec tmp[RowTileSize];

            // A block of threads is the exact size of the vector
#pragma unroll
            for (int row = 0; row < RowTileSize; ++row) {
                tmp[row] = input[rowStart + row][threadIdx.x];
            }

#pragma unroll
            for (int row = 0; row < RowTileSize; ++row) {
                tmp[row] = Math<TVec>::mul(tmp[row], tmp[row]);
            }

#pragma unroll
            for (int row = 0; row < RowTileSize; ++row) {
                rowNorm[row] = Math<TVec>::reduceAdd(tmp[row]);
            }
        }

        // Sum up all parts in each warp
#pragma unroll
        for (int row = 0; row < RowTileSize; ++row) {
            rowNorm[row] = warpReduceAllSum(rowNorm[row]);
        }

        if (laneId == 0) {
#pragma unroll
            for (int row = 0; row < RowTileSize; ++row) {
                smem[row * numWarps + warpId] = rowNorm[row];
            }
        }
    }

    __syncthreads();

    // Sum across warps
    if (warpId == 0) {
#pragma unroll
        for (int row = 0; row < RowTileSize; ++row) {
            rowNorm[row] = laneId < numWarps ? smem[row * numWarps + laneId] : Math<T>::zero();
        }

#pragma unroll
        for (int row = 0; row < RowTileSize; ++row) {
            rowNorm[row] = warpReduceAllSum(rowNorm[row]);
        }

        // Write out answer
        if (laneId == 0) {
#pragma unroll
            for (int row = 0; row < RowTileSize; ++row) {
                int outCol = rowStart + row;

                if (lastRowTile) {
                    if (outCol < output.getSize(0)) {
                        output[outCol] = NormSquared ? rowNorm[row] :
                                         ConvertTo<T>::to(sqrtf(ConvertTo<float>::to(rowNorm[row])));
                    }
                } else {
                    output[outCol] = NormSquared ? rowNorm[row] :
                                     ConvertTo<T>::to(sqrtf(ConvertTo<float>::to(rowNorm[row])));
                }
            }
        }
    }
};


template <typename T, typename TVec, typename TIndex>
void runL2Norm(Tensor<T, 2, true, TIndex>& input,
               Tensor<T, 1, true, TIndex>& output,
               bool normSquared,
               cudaStream_t stream) {
    FAISSV_ASSERT(input.getSize(0) == output.getSize(0));

    TIndex maxTHreads = (TIndex) getMaxThreadsCurrentDevice();
    constexpr int rowTileSize = 8;

#define RUN_L2(TYPE_T, TYPE_TVEC, INPUT)                                        \
    do {                                                                        \
        if (normLoop) {                                                         \
            if (normSquared) {                                                  \
                l2Norm<TYPE_T, TYPE_TVEC, TIndex, rowTileSize, true, true>      \
                    <<<grid, block, smem, stream>>>(INPUT, output);             \
            } else {                                                            \
                l2Norm<TYPE_T, TYPE_TVEC, TIndex, rowTileSize, true, false>     \
                <<<grid, block, smem, stream>>>(INPUT, output);                 \
            }                                                                   \
        } else {                                                                \
            if (normSquared) {                                                  \
                l2Norm<TYPE_T, TYPE_TVEC, TIndex, rowTileSize, false, true>     \
                <<<grid, block, smem, stream>>>(INPUT, output);                 \
            } else {                                                            \
                l2Norm<TYPE_T, TYPE_TVEC, TIndex, rowTileSize, false, false>    \
                <<<grid, block, smem, stream>>>(INPUT, output);                 \
            }                                                                   \
        }                                                                       \
    } while (0)

    if (input.template canCastResize<TVec> ()) {
        auto inputV = input.template castResize<TVec>();

        auto dim = inputV.getSize(1);
        bool normLoop = dim > maxTHreads;
        auto numThreads = min(dim, maxTHreads);

        auto grid = dim3(utils::divUp(inputV.getSize(0), rowTileSize));
        auto block = dim3(numThreads);

        auto smem = sizeof(T) * rowTileSize * utils::divUp(numThreads, kWarpSize);

        RUN_L2(T, TVec, inputV);
    } else {
        // Can't load using the vectorized type

        auto dim = input.getSize(1);
        bool normLoop = dim > maxTHreads;
        auto numThreads = min(dim, maxTHreads);

        auto grid = dim3(utils::divUp(input.getSize(0), rowTileSize));
        auto block = dim3(numThreads);

        auto smem = sizeof(T) * rowTileSize * utils::divUp(numThreads, kWarpSize);

        RUN_L2(T, T, input);
    }

#undef RUN_L2

    CUDA_TEST_ERROR();
};

void runL2Norm(Tensor<float, 2, true>& input,
               Tensor<float, 1, true>& output,
               bool normSquared,
               cudaStream_t stream) {
    if (input.canUseIndexType<int>()) {
        runL2Norm<float, float4, int>(input, output, normSquared, stream);
    } else {
        auto inputCast = input.castIndexType<long>();
        auto outputCast = output.castIndexType<long>();
        runL2Norm<float, float4, long>(inputCast, outputCast, normSquared, stream);
    }
}

}}