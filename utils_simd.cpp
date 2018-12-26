/*
 * Author: VincentLee
 * Email:  lichlee@yeah.net
 * Created on 2018/12/26.
*/

#include "utils.h"

#ifdef __SSE__

#include <immintrin.h>

#endif

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#include <omp.h>

namespace faiss_v {
#ifdef __AVX__
#define USE_AVX
#endif

    /// Reference implementations
    float fvec_L2sqr_ref(const float *x, const float *y,
                         size_t d) {
        size_t i;
        float res = 0;
        for (i = 0; i < d; i++) {
            const float tmp = x[i] - y[i];
            res += tmp * tmp;
        }

        return res;
    }

    float fvec_inner_product_ref(const float *x, size_t d) {
        size_t i;
        double res = 0;
        for (i = 0; i < d; i++) {
            res += x[i] * x[i];
        }

        return res;
    }

    float fvec_norm_L2sqr_ref(const float *x, size_t d) {
        size_t i;
        double res = 0;
        for (i = 0; i < d; i++) {
            res += x[i] * x[i];
        }

        return res;
    }

    /// SSE and AVX implementations
#ifdef __SSE__

    // reads 0 <= d < 4 floats as __m128
    static inline __m128 masked_read (int d, const float* x) {
        assert (0 <= d && d < 4);
        __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};
        switch (d) {
            case 3:
                buf[2] = x[2];
            case 2:
                buf[1] = x[1];
            case 1:
                buf[0] = x[0];
        }

        return _mm_load_ps(buf);
    }

    float fvec_norm_L2sqr (const float* x, size_t d) {
        __m128 mx;
        __m128 msum1 = _mm_setzero_ps();

        while (d >= 4) {
            mx = _mm_loadu_ps(x); x += 4;
            msum1 = _mm_add_ps(msum1, _mm_mul_ps(mx, mx));
            d -= 4;
        }

        mx = masked_read(d, x);
        msum1 = _mm_add_ps(msum1, _mm_mul_ps(mx, mx));

        msum1 = _mm_hadd_ps(msum1, msum1);
        msum1 = _mm_hadd_ps(msum1, msum1);

        return _mm_cvtss_f32(msum1);
    }
#endif

#ifdef USE_AVX

    // reads 0 <= d < 8 floats as __m256
    static inline __m256 msked_red_8 (int d, const float* x) {
        assert (0 <= d && d < 8);
        if (d < 4) {
            __m256 res = _mm256_setzero_ps();
            res = _mm256_insertf128_ps(res, masked_read(d, x), 0);

            return res;
        } else {
            __m256 res = _mm256_setzero_ps();
            res = _mm256_insertf128_ps(res, _mm_loadu_ps(x), 0);
            res = _mm256_insertf128_ps(res, masked_read(d-4, x+4), 1);

            return res;
        }
    }

    // VAX-implementation of L2 distance
    float fevc_L2sqr (const float* x, const float* y, size_t d) {
        __m256 msum1 = _mm256_setzero_ps();

        while (d >= 8) {
            __m256 mx = _mm256_loadu_ps(x); x += 8;
            __m256 my = _mm256_loadu_ps(y); y += 8;
            const __m256 a_m_b1 = mx - my;
            msum1 += a_m_b1 * a_m_b1;
            d -= 8;
        }

        __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
        msum2 +=       _mm256_extractf128_ps(msum1, 0);

        if (d >= 4) {
            __m128 mx = _mm_loadu_ps(x); x += 4;
            __m128 my = _mm_loadu_ps(y); y += 4;
            const _m128 a_m_b1 = mx - my;
            msum2 += a_m_b1 * a_m_b1;
            d -= 4;
        }

        if (d > 0) {
            __m128 mx = masked_read(d, x);
            __m128 my = masked_read(d, y);
            __m128 a_m_b1 = mx - my;
            msum2 += a_m_b1 * a_m_b1;
        }

        msum2 = _mm_hadd_ps(msum2, msum2);
        msum2 = _mm_hadd_ps(msum2, msum2);

        return _mm_cvtss_f32(msum2);
    }
#elif defined(__SSE__)

    // SSE-implementation of L2 distance
    float fevc_L2sqr (const float* x, const float* y, size_t d) {
        __m128 msum1 = _mm_setzero_ps();

        while (d >= 4) {
            __m128 mx = _mm_loadu_ps(x); x += 4;
            __m128 my = _mm_loadu_ps(y); y += 4;
            const __m128 a_m_b1 = mx - my;
            msum1 += a_m_b1 * a_m_b1;
            d -= 4;
        }

        if (d > 0) {
            // add the last values
            __m128 mx = masked_read(d, x);
            __m128 my = masked_read(d, y);
            __m128 a_m_b1 = mx -my;
            msum1 += a_m_b1 * a_m_b1;
        }

        msum1 = _mm_hadd_ps(msum1, msum1);
        msum1 = _mm_hadd_ps(msum1, msum1);

        return _mm_cvtss_f32(msum1);
    }

#elif defined(__aarch64__)

    float fevc_L2sqr(const float* x, const float* y, size_t d) {
        if (d & 3) return fevc_L2sqr_ref(x, y, d);
        float32x4_t accu = vdupq_n_f32(0);
        for (size_t i = 0; i < d; i += 4) {
            float32x4_t xi = vld1q_f32(x + i);
            float32x4_t yi = vld1q_f32(y + i);
            float32x4_t sq = vsubq_f32(xi, yi);
            accu = vfmaq_f32(accu, sq, sq);
        }
        float32x4_t a2 = vpaddq_f32(accu, accu);

        return vdups_laneq_f32(a2, 0) + vdups_laneq_f32(a2, 1)
    }

    float fvec_norm_L2sqr (const float* x, size_t d) {
        if (d & 3) return fvec_norm_L2sqr_ref(x, d);
        float32x4_t accu = vdupq_n_f32(0);
        for (size_t i = 0; i < d; i += 4) {
            float32x4_t xi = vld1q_f32(x+i);
            accu = vfmaq_f32(accu, xi, xi);
        }
        float32x4_t a2 = vpaddq_f32(accu, accu);

        return vdups_laneq_f32(a2, 0) + vdups_laneq_f32(a2, 1);
    }

#else
    // scalar implementation
    float fvec_L2sqr (const float * x, const float * y, size_t d) {
        return fvec_L2sqr_ref (x, y, d);
    }

    float fvec_norm_L2sqr (const float *x, size_t d) {
        return fvec_norm_L2sqr_ref (x, d);
    }

#endif
} // namespace faiss

