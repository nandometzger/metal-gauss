#include <metal_stdlib>
using namespace metal;

// Fused Adam. torch's Adam makes ~5 full passes over every tensor
// (lerp_, mul_, addcmul_, sqrt/add, addcdiv_), each one reading and writing
// whole buffers. At 35.4M parameters that is ~1 GB of traffic per step and
// measured 2.6x above this machine's elementwise bandwidth floor. One pass
// does the same arithmetic.
//
// The formula is torch's EXACTLY, including where eps enters -- it is added
// AFTER dividing by sqrt(bias_correction2), not before. Getting that wrong
// changes the update by a factor that grows as v shrinks, so it is not a
// detail that shows up as a small drift.

struct AdamP {
    uint  n;
    float lr;
    float b1;
    float b2;
    float eps;
    float bc1;      // 1 - b1^t
    float sqrt_bc2; // sqrt(1 - b2^t)
    float omb1;     // 1 - b1, rounded from double as torch does: in float,
    float omb2;     // 1.0f - 0.999f is 1.3e-5 off, and v inherits it
};

kernel void adam_step(device float*       p   [[buffer(0)]],
                      device const float* g   [[buffer(1)]],
                      device float*       m   [[buffer(2)]],
                      device float*       v   [[buffer(3)]],
                      constant AdamP&     P   [[buffer(4)]],
                      uint i [[thread_position_in_grid]])
{
    if (i >= P.n) return;

    const float gi = g[i];
    const float mi = fma(P.b1, m[i], P.omb1 * gi);
    const float vi = fma(P.b2, v[i], P.omb2 * gi * gi);
    m[i] = mi;
    v[i] = vi;

    const float denom = sqrt(vi) / P.sqrt_bc2 + P.eps;
    p[i] = p[i] - (P.lr / P.bc1) * mi / denom;
}
