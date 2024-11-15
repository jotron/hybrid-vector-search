#pragma once

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