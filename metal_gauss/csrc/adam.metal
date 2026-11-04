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

// Selective Adam: SelectiveAdam's update for the rows the view saw, nothing at
// all for the rest. Dense Adam spends ~29% of a training step streaming every
// parameter and both moments through memory, and on room1 a training view
// sees only ~23% of the gaussians. A hidden row costs one mask byte here.
//
// The arithmetic is selective_adam.py's, not adam_step's: eps enters after the
// sqrt(bc2) scaling. The bias correction uses either the optimiser's global
// step count (k_global, gsplat's choice) or the ROW's own count, already
// incremented by the caller, as if the steps it was hidden for had not
// happened.
struct AdamRowsP {
    uint  n;        // elements in the parameter
    uint  cols;     // elements per row
    uint  rows;     // mask length; rows past it are hidden
    float lr;
    float b1;
    float b2;
    float eps;
    float omb1;     // as in AdamP
    float omb2;
    float k_global; // > 0: bias-correct every row with this step count
};

kernel void adam_step_rows(device float*       p       [[buffer(0)]],
                           device const float* g       [[buffer(1)]],
                           device float*       m       [[buffer(2)]],
                           device float*       v       [[buffer(3)]],
                           device const float* steps   [[buffer(4)]],
                           device const uchar* visible [[buffer(5)]],
                           constant AdamRowsP& P       [[buffer(6)]],
                           uint i [[thread_position_in_grid]])
{
    if (i >= P.n) return;
    const uint row = i / P.cols;
    if (row >= P.rows || visible[row] == 0) return;

    const float gi = g[i];
    const float mi = P.b1 * m[i] + P.omb1 * gi;
    const float vi = P.b2 * v[i] + gi * gi * P.omb2;
    m[i] = mi;
    v[i] = vi;

    const float k = P.k_global > 0.0f ? P.k_global : steps[row];
    const float step_size = P.lr * sqrt(1.0f - pow(P.b2, k)) / (1.0f - pow(P.b1, k));
    p[i] = p[i] - step_size * mi / (sqrt(vi) + P.eps);
}
