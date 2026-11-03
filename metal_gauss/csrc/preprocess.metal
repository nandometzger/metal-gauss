#include <metal_stdlib>
using namespace metal;

// Fused projection + SH + activation, forward and analytic backward.
// One thread per gaussian, no atomics (each gaussian owns its outputs).
//
// This kernel exists because profiling showed the torch host side eating 90%
// of a training step: projection forward was 2.47s of MPS dispatch soup and
// its autograd backward 5.3s, while the actual rasterization kernels took
// 157ms combined. The LichtFeld/Faster-GS lesson applies verbatim: nothing
// belongs outside the fused kernels. Math matches metal_gauss/torch_ref.py exactly
// -- torch_ref stays the gradcheck-verified oracle.

constant float C0 = 0.28209479177387814f;
constant float C1 = 0.4886025119029199f;
constant float C2_0 = 1.0925484305920792f, C2_1 = -1.0925484305920792f,
               C2_2 = 0.31539156525252005f, C2_3 = -1.0925484305920792f,
               C2_4 = 0.5462742152960396f;
constant float C3_0 = -0.5900435899266435f, C3_1 = 2.890611442640554f,
               C3_2 = -0.4570457994644658f, C3_3 = 0.3731763325901154f,
               C3_4 = -0.4570457994644658f, C3_5 = 1.445305721320277f,
               C3_6 = -0.5900435899266435f;

struct Params {
    float4 R0, R1, R2;        // world2cam rows (xyz = row, w = t component)
    float4 intr;              // fx fy cx cy
    float4 lims;              // near far blur max_radius
    float4 cam_center;        // xyz
    uint4  misc;              // N, W, H, sh_degree
    // SH storage layout, in packed_float3 rows. The trainer keeps the DC band
    // and bands 1+ as SEPARATE tensors so Adam can give them different learning
    // rates; concatenating them every step cost 11.2 ms fwd+bwd at 600k. With
    // this the kernel reads both layouts directly and the cat disappears.
    //   fused  (N,16,3) single tensor : (16, 16, 1)
    //   split  (N,1,3) + (N,15,3)     : ( 1, 15, 0)
    uint4  shl;               // stride_dc, stride_rest, offset_rest, unused
};

inline float3 sh_read(device const packed_float3* dc,
                      device const packed_float3* rest,
                      uint g, int i, uint4 L) {
    return (i == 0) ? float3(dc[g * L.x])
                    : float3(rest[g * L.y + L.z + uint(i - 1)]);
}

inline void sh_write(device packed_float3* d_dc,
                     device packed_float3* d_rest,
                     uint g, int i, uint4 L, float3 v) {
    if (i == 0) d_dc[g * L.x] = packed_float3(v);
    else        d_rest[g * L.y + L.z + uint(i - 1)] = packed_float3(v);
}

inline float3x3 quat_to_R(float4 q_wxyz, thread float* qnorm_out) {
    float n = length(q_wxyz);
    float4 q = q_wxyz / max(n, 1e-12f);
    *qnorm_out = n;
    float w = q.x, x = q.y, y = q.z, z = q.w;
    // float3x3(c0,c1,c2) is column-major; each float3 below is a COLUMN of the
    // standard rotation matrix, so this constructs R itself. (An earlier
    // version added a spurious transpose() at the call sites -- identity
    // quaternions masked it, random ones broke every conic. Validated against
    // torch_ref, which is why the identity-vs-random A/B test exists.)
    return float3x3(
        float3(1 - 2*(y*y + z*z), 2*(x*y + w*z),     2*(x*z - w*y)),
        float3(2*(x*y - w*z),     1 - 2*(x*x + z*z), 2*(y*z + w*x)),
        float3(2*(x*z + w*y),     2*(y*z - w*x),     1 - 2*(x*x + y*y)));
}

// SH basis and its gradient wrt the (unnormalised-then-normalised) direction.
inline void sh_basis(int deg, float3 d, thread float* B) {
    float x = d.x, y = d.y, z = d.z;
    B[0] = C0;
    if (deg < 1) return;
    B[1] = -C1 * y; B[2] = C1 * z; B[3] = -C1 * x;
    if (deg < 2) return;
    float xx = x*x, yy = y*y, zz = z*z;
    B[4] = C2_0 * x * y; B[5] = C2_1 * y * z;
    B[6] = C2_2 * (2*zz - xx - yy);
    B[7] = C2_3 * x * z; B[8] = C2_4 * (xx - yy);
    if (deg < 3) return;
    B[9]  = C3_0 * y * (3*xx - yy);
    B[10] = C3_1 * x * y * z;
    B[11] = C3_2 * y * (4*zz - xx - yy);
    B[12] = C3_3 * z * (2*zz - 3*xx - 3*yy);
    B[13] = C3_4 * x * (4*zz - xx - yy);
    B[14] = C3_5 * z * (xx - yy);
    B[15] = C3_6 * x * (xx - 3*yy);
}

inline void sh_basis_grad(int deg, float3 d, int b, thread float3* g) {
    float x = d.x, y = d.y, z = d.z;
    float xx = x*x, yy = y*y, zz = z*z;
    switch (b) {
        case 0:  *g = float3(0); break;
        case 1:  *g = float3(0, -C1, 0); break;
        case 2:  *g = float3(0, 0, C1); break;
        case 3:  *g = float3(-C1, 0, 0); break;
        case 4:  *g = C2_0 * float3(y, x, 0); break;
        case 5:  *g = C2_1 * float3(0, z, y); break;
        case 6:  *g = C2_2 * float3(-2*x, -2*y, 4*z); break;
        case 7:  *g = C2_3 * float3(z, 0, x); break;
        case 8:  *g = C2_4 * float3(2*x, -2*y, 0); break;
        case 9:  *g = C3_0 * float3(6*x*y, 3*xx - 3*yy, 0); break;
        case 10: *g = C3_1 * float3(y*z, x*z, x*y); break;
        case 11: *g = C3_2 * float3(-2*x*y, 4*zz - xx - 3*yy, 8*y*z); break;
        case 12: *g = C3_3 * float3(-6*x*z, -6*y*z, 6*zz - 3*xx - 3*yy); break;
        case 13: *g = C3_4 * float3(4*zz - 3*xx - yy, -2*x*y, 8*x*z); break;
        case 14: *g = C3_5 * float3(2*x*z, -2*y*z, xx - yy); break;
        case 15: *g = C3_6 * float3(3*xx - 3*yy, -6*x*y, 0); break;
        default: *g = float3(0);
    }
}

kernel void preprocess_forward(
    device const packed_float3* means      [[buffer(0)]],
    device const float4*        quats      [[buffer(1)]],   // wxyz
    device const packed_float3* scales     [[buffer(2)]],
    device const packed_float3* sh         [[buffer(3)]],   // DC band (or fused)
    device const packed_float3* sh_rest    [[buffer(12)]],  // bands 1+ (or fused)
    device const float*         opacities  [[buffer(11)]],
    device float2*              out_uv     [[buffer(4)]],
    device packed_float3*       out_conic  [[buffer(5)]],
    device float*               out_depth  [[buffer(6)]],
    device float2*              out_rxy    [[buffer(7)]],   // rect half-extents
    device int*                 out_valid  [[buffer(8)]],
    device packed_float3*       out_color  [[buffer(9)]],
    constant Params&            P          [[buffer(10)]],
    uint g [[thread_position_in_grid]])
{
    if (g >= P.misc.x) return;
    const float fx = P.intr.x, fy = P.intr.y, cx = P.intr.z, cy = P.intr.w;
    const float near = P.lims.x, far = P.lims.y, blur = P.lims.z, max_r = P.lims.w;
    const uint W = P.misc.y, H = P.misc.z;
    const int deg = int(P.misc.w);

    const float3 mean = float3(means[g]);
    const float3 pc = float3(
        dot(P.R0.xyz, mean) + P.R0.w,
        dot(P.R1.xyz, mean) + P.R1.w,
        dot(P.R2.xyz, mean) + P.R2.w);
    const float z = pc.z;
    const float zc = max(z, near);

    const float u = fx * pc.x / zc + cx;
    const float v = fy * pc.y / zc + cy;

    // Jacobian evaluation point clamped to just outside the frustum
    const float lim_x = 1.3f * (0.5f * float(W) / fx);
    const float lim_y = 1.3f * (0.5f * float(H) / fy);
    const float tx = clamp(pc.x / zc, -lim_x, lim_x) * zc;
    const float ty = clamp(pc.y / zc, -lim_y, lim_y) * zc;

    float qn;
    const float3x3 Rg = quat_to_R(quats[g], &qn);
    const float3 s = float3(scales[g]);
    // Sigma = M M^T, M = Rg * diag(s)
    const float3x3 M = float3x3(Rg[0] * s.x, Rg[1] * s.y, Rg[2] * s.z); // columns scaled
    const float3x3 Sigma = M * transpose(M);

    // T = J * R_w2c   (2x3, stored as rows)
    const float izc = 1.0f / zc, izc2 = izc * izc;
    const float3 J0 = float3(fx * izc, 0.0f, -fx * tx * izc2);
    const float3 J1 = float3(0.0f, fy * izc, -fy * ty * izc2);
    const float3 Rw0 = P.R0.xyz, Rw1 = P.R1.xyz, Rw2 = P.R2.xyz;
    const float3 T0 = J0.x * Rw0 + J0.y * Rw1 + J0.z * Rw2;
    const float3 T1 = J1.x * Rw0 + J1.y * Rw1 + J1.z * Rw2;

    const float a = dot(T0, Sigma * T0) + blur;
    const float b = dot(T0, Sigma * T1);
    const float c = dot(T1, Sigma * T1) + blur;
    const float det = a * c - b * b;
    const float dets = max(det, 1e-12f);
    const float3 conic = float3(c / dets, -b / dets, a / dets);

    const float mid = 0.5f * (a + c);
    const float disc = sqrt(max(mid * mid - dets, 0.0f));
    const float lam = max(mid + disc, 1e-12f);
    const float radius = 3.0f * sqrt(lam);

    // SnugBox (Speedy-Splat): the alpha = 1/255 level set of
    // op*exp(-0.5 d^T C d) has exact per-axis extent sqrt(2 ln(255 op)) *
    // sqrt(cov_axis). Pairs outside this rectangle would fail the kernel's
    // own alpha >= 1/255 test, so tightening the binning to it is EXACT --
    // identical image, strictly fewer (gaussian, tile) pairs to sort and walk.
    // Faint splats shrink dramatically; op <= 1/255 has no level set at all.
    const float op = opacities[g];
    const float twoL = 2.0f * log(max(255.0f * op, 1.0f));
    const float rs = sqrt(twoL);                    // 0 when op <= 1/255
    const float rx = min(rs * sqrt(max(a, 0.0f)), radius);
    const float ry = min(rs * sqrt(max(c, 0.0f)), radius);

    bool ok = (z > near) && (z < far) && (det > 1e-12f)
              && (radius < max_r) && (rx > 0.0f) && (ry > 0.0f)
              && (u + rx > 0) && (u - rx < float(W))
              && (v + ry > 0) && (v - ry < float(H));

    out_uv[g] = float2(u, v);
    out_conic[g] = packed_float3(conic);
    out_depth[g] = z;
    out_rxy[g] = float2(rx, ry);
    out_valid[g] = ok ? 1 : 0;

    // Only binned gaussians are ever read by the rasteriser, and binning skips
    // !ok, so an invalid gaussian's colour is never seen. Skipping SH for it
    // drops 16 SH rows of reads -- and on room1 only 23% of 600k gaussians are
    // in a given training view (3DGS CUDA blog, "visibility-gated SH").
    if (!ok) { out_color[g] = packed_float3(0.0f); return; }

    // SH -> colour
    float3 dirv = mean - P.cam_center.xyz;
    float invn = 1.0f / max(length(dirv), 1e-12f);
    float3 dir = dirv * invn;
    float B[16];
    sh_basis(deg, dir, B);
    int nb = (deg + 1) * (deg + 1);
    float3 col = float3(0);
    for (int i = 0; i < nb; ++i)
        col += B[i] * sh_read(sh, sh_rest, g, i, P.shl);
    col += 0.5f;
    out_color[g] = packed_float3(max(col, float3(0)));
}

kernel void preprocess_backward(
    device const packed_float3* means      [[buffer(0)]],
    device const float4*        quats      [[buffer(1)]],
    device const packed_float3* scales     [[buffer(2)]],
    device const packed_float3* sh         [[buffer(3)]],   // DC band (or fused)
    device const packed_float3* sh_rest    [[buffer(14)]],  // bands 1+ (or fused)
    device const float2*        d_uv       [[buffer(4)]],
    device const packed_float3* d_conic    [[buffer(5)]],
    device const packed_float3* d_color    [[buffer(6)]],
    device const int*           valid      [[buffer(7)]],
    device packed_float3*       d_means    [[buffer(8)]],
    device float4*              d_quats    [[buffer(9)]],
    device packed_float3*       d_scales   [[buffer(10)]],
    device packed_float3*       d_sh       [[buffer(11)]],  // DC band (or fused)
    device packed_float3*       d_sh_rest  [[buffer(13)]],  // bands 1+ (or fused)
    constant Params&            P          [[buffer(12)]],
    uint g [[thread_position_in_grid]])
{
    if (g >= P.misc.x) return;
    const int deg = int(P.misc.w);
    const int nb = (deg + 1) * (deg + 1);
    const float3 mean = float3(means[g]);

    float3 dmean = float3(0);

    // ---------------- SH branch ------------------------------------------
    // d_color is the SH branch's only input gradient. When it is zero -- every
    // gaussian the rasteriser never blended, i.e. most of them in any one view
    // -- the branch writes zeros and adds nothing to dmean, so skip its two
    // passes over the 16 SH rows and write the zeros directly.
    const float3 dcol = float3(d_color[g]);
    if (all(dcol == 0.0f)) {
        for (int i = 0; i < 16; ++i) sh_write(d_sh, d_sh_rest, g, i, P.shl, float3(0));
    } else {
        float3 dirv = mean - P.cam_center.xyz;
        float n = max(length(dirv), 1e-12f);
        float3 dir = dirv / n;
        float B[16];
        sh_basis(deg, dir, B);

        // recompute pre-clamp colour for the ReLU gate
        float3 pre = float3(0);
        for (int i = 0; i < nb; ++i) pre += B[i] * sh_read(sh, sh_rest, g, i, P.shl);
        pre += 0.5f;
        const float3 dpre = select(float3(0), dcol, pre > 0.0f);

        float3 ddir = float3(0);
        for (int i = 0; i < nb; ++i) {
            const float3 shi = sh_read(sh, sh_rest, g, i, P.shl);
            sh_write(d_sh, d_sh_rest, g, i, P.shl, B[i] * dpre);
            float3 gB;
            sh_basis_grad(deg, dir, i, &gB);
            ddir += gB * dot(shi, dpre);
        }
        for (int i = nb; i < 16; ++i) sh_write(d_sh, d_sh_rest, g, i, P.shl, float3(0));

        // dir = v/|v| : d_v = (I - dir dir^T)/|v| * d_dir
        dmean += (ddir - dir * dot(dir, ddir)) / n;
    }

    // ---------------- projection branch ----------------------------------
    if (valid[g] != 0) {
        const float fx = P.intr.x, fy = P.intr.y;
        const float near = P.lims.x, blur = P.lims.z;
        const uint W = P.misc.y, H = P.misc.z;

        const float3 pc = float3(
            dot(P.R0.xyz, mean) + P.R0.w,
            dot(P.R1.xyz, mean) + P.R1.w,
            dot(P.R2.xyz, mean) + P.R2.w);
        const float z = pc.z, zc = max(z, near);
        const float izc = 1.0f / zc, izc2 = izc * izc;

        const float lim_x = 1.3f * (0.5f * float(W) / fx);
        const float lim_y = 1.3f * (0.5f * float(H) / fy);
        const float txn = pc.x / zc, tyn = pc.y / zc;
        const bool clx = (txn < -lim_x) || (txn > lim_x);
        const bool cly = (tyn < -lim_y) || (tyn > lim_y);
        const float tx = clamp(txn, -lim_x, lim_x) * zc;
        const float ty = clamp(tyn, -lim_y, lim_y) * zc;

        float qn;
        const float3x3 Rg = quat_to_R(quats[g], &qn);
        const float3 s = float3(scales[g]);
        const float3x3 M = float3x3(Rg[0] * s.x, Rg[1] * s.y, Rg[2] * s.z);
        const float3x3 Sigma = M * transpose(M);

        const float3 J0 = float3(fx * izc, 0.0f, -fx * tx * izc2);
        const float3 J1 = float3(0.0f, fy * izc, -fy * ty * izc2);
        const float3 Rw0 = P.R0.xyz, Rw1 = P.R1.xyz, Rw2 = P.R2.xyz;
        const float3 T0 = J0.x * Rw0 + J0.y * Rw1 + J0.z * Rw2;
        const float3 T1 = J1.x * Rw0 + J1.y * Rw1 + J1.z * Rw2;

        const float3 ST0 = Sigma * T0, ST1 = Sigma * T1;
        const float a = dot(T0, ST0) + blur;
        const float b = dot(T0, ST1);
        const float c = dot(T1, ST1) + blur;
        const float det = a * c - b * b;
        const float dets = max(det, 1e-12f);
        const float id = 1.0f / dets;

        // conic = (c,-b,a)/det ; d w.r.t. (a,b,c) with det chain
        const float3 dcn = float3(d_conic[g]);
        float da, db, dc;
        {
            // conic.x = c/det, conic.y = -b/det, conic.z = a/det
            const float ddet = -(dcn.x * c - dcn.y * b + dcn.z * a) * id * id;
            da = dcn.z * id + ddet * c;
            db = -dcn.y * id - 2.0f * ddet * b;
            dc = dcn.x * id + ddet * a;
            if (det <= 1e-12f) { da = db = dc = 0.0f; }  // through the clamp
        }

        // a = T0^T S T0, b = T0^T S T1, c = T1^T S T1
        // dSigma = da*T0 T0^T + db*T0 T1^T + dc*T1 T1^T (unsym), symmetrised
        // before use. float3x3(c0,c1,c2) is column-major: column j of
        // outer(u,v) = u * v[j].
        #define OUTER(u, v) float3x3((u) * (v).x, (u) * (v).y, (u) * (v).z)
        const float3x3 dSigma = da * OUTER(T0, T0) + db * OUTER(T0, T1)
                              + dc * OUTER(T1, T1);
        #undef OUTER

        const float3 dT0 = 2.0f * (da * ST0) + db * ST1;
        const float3 dT1 = 2.0f * (dc * ST1) + db * ST0;

        // T0 = J0.x Rw0 + J0.y Rw1 + J0.z Rw2  ->  dJ0 = (dT0.Rw0, dT0.Rw1, dT0.Rw2)
        const float3 dJ0 = float3(dot(dT0, Rw0), dot(dT0, Rw1), dot(dT0, Rw2));
        const float3 dJ1 = float3(dot(dT1, Rw0), dot(dT1, Rw1), dot(dT1, Rw2));

        // J0 = (fx/zc, 0, -fx tx/zc^2), J1 = (0, fy/zc, -fy ty/zc^2)
        float dtx = -fx * izc2 * dJ0.z;
        float dty = -fy * izc2 * dJ1.z;
        float dzc = -fx * izc2 * dJ0.x - fy * izc2 * dJ1.y
                    + 2.0f * fx * tx * izc2 * izc * dJ0.z
                    + 2.0f * fy * ty * izc2 * izc * dJ1.z;

        // uv gradients
        const float2 duv = d_uv[g];
        float dpx = fx * izc * duv.x;
        float dpy = fy * izc * duv.y;
        dzc += -fx * pc.x * izc2 * duv.x - fy * pc.y * izc2 * duv.y;

        // tx = clamp(px/zc)*zc : if unclamped tx==px -> dtx flows to px;
        // if clamped, tx = ±lim*zc -> flows to zc.
        if (!clx) dpx += dtx; else dzc += dtx * (tx * izc);
        if (!cly) dpy += dty; else dzc += dty * (ty * izc);

        float dpz = (z > near) ? dzc : 0.0f;
        const float3 dpc = float3(dpx, dpy, dpz);
        dmean += float3(dot(float3(Rw0.x, Rw1.x, Rw2.x), dpc),
                        dot(float3(Rw0.y, Rw1.y, Rw2.y), dpc),
                        dot(float3(Rw0.z, Rw1.z, Rw2.z), dpc));

        // dSigma -> dM = 2 * sym(dSigma) * M ; here dSigma already near-sym
        const float3x3 symd = 0.5f * (dSigma + transpose(dSigma));
        const float3x3 dM = 2.0f * (symd * M);

        // M columns: col_i = Rg_col_i * s_i  (built as float3x3(colums))
        const float3 dRc0 = dM[0] * s.x, dRc1 = dM[1] * s.y, dRc2 = dM[2] * s.z;
        d_scales[g] = packed_float3(float3(dot(dM[0], Rg[0]), dot(dM[1], Rg[1]), dot(dM[2], Rg[2])));

        // quaternion backward. dRc_i is the gradient w.r.t. COLUMN i of the
        // standard R, i.e. the element-gradient matrix G with G[:,i] = dRc_i:
        //   G00=dRc0.x G01=dRc1.x G02=dRc2.x
        //   G10=dRc0.y G11=dRc1.y G12=dRc2.y
        //   G20=dRc0.z G21=dRc1.z G22=dRc2.z
        // Standard VJP of R(q), q=(w,x,y,z) normalised:
        float4 q = quats[g] / max(qn, 1e-12f);
        const float w = q.x, x = q.y, y = q.z, zq = q.w;
        const float G00 = dRc0.x, G10 = dRc0.y, G20 = dRc0.z;
        const float G01 = dRc1.x, G11 = dRc1.y, G21 = dRc1.z;
        const float G02 = dRc2.x, G12 = dRc2.y, G22 = dRc2.z;
        const float dw = 2.0f * (x * (G21 - G12) + y * (G02 - G20) + zq * (G10 - G01));
        const float dx = 2.0f * (w * (G21 - G12) + y * (G10 + G01) + zq * (G20 + G02)
                                 - 2.0f * x * (G11 + G22));
        const float dy = 2.0f * (w * (G02 - G20) + x * (G10 + G01) + zq * (G21 + G12)
                                 - 2.0f * y * (G00 + G22));
        const float dz = 2.0f * (w * (G10 - G01) + x * (G20 + G02) + y * (G21 + G12)
                                 - 2.0f * zq * (G00 + G11));
        float4 dqn = float4(dw, dx, dy, dz);
        // through normalisation: d_q = (dqn - q * dot(q,dqn)) / |q|
        d_quats[g] = (dqn - q * dot(q, dqn)) / max(qn, 1e-12f);
    } else {
        d_scales[g] = packed_float3(float3(0));
        d_quats[g] = float4(0);
    }

    d_means[g] = packed_float3(dmean);
}
