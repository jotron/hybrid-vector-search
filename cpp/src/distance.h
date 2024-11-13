#pragma once
#include <emmintrin.h>
#include <immintrin.h>
#include <xmmintrin.h>

inline float avx2_l2_distance(const float *a, const float *b)
{
    unsigned dim = 104;
    __m256 sum = _mm256_setzero_ps(); // Initialize sum to 0
    unsigned i;
    for (i = 0; i + 7 < dim; i += 8)
    {
        __m256 a_vec = _mm256_load_ps(&a[i]);      // Load 8 floats from a
        __m256 b_vec = _mm256_load_ps(&b[i]);      // Load 8 floats from b
        __m256 diff = _mm256_sub_ps(a_vec, b_vec); // Calculate difference
        sum = _mm256_fmadd_ps(diff, diff, sum);    // Calculate sum of squares
    }
    float result = 0;
    for (unsigned j = 0; j < 8; ++j)
    {
        result += ((float *)&sum)[j];
    }
    return result; // Return distance squared
}

inline float normal_l2(float const *a, float const *b, unsigned dim)
{
    float r = 0;
    for (unsigned i = 0; i < dim; ++i)
    {
        float v = float(a[i]) - float(b[i]);
        v *= v;
        r += v;
    }
    return r;
}