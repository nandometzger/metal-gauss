// PyTorch <-> Metal bindings for the Gaussian rasteriser.
//
// The .metal source is compiled at runtime with newLibraryWithSource because
// this machine has Command Line Tools without full Xcode, so `xcrun metal` is
// unavailable and a precompiled .metallib cannot be produced. Runtime
// compilation needs no Xcode and costs a few hundred ms once per process.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/native/mps/OperationUtils.h>
#include <torch/mps.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

using at::native::mps::getMTLBufferStorage;

static id<MTLLibrary> gLib = nil;
static NSMutableDictionary* gPipelines = nil;

static void initLibrary(const std::string& src) {
    if (gLib) return;
    NSError* err = nil;
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    TORCH_CHECK(dev, "no Metal device");
    MTLCompileOptions* options = [MTLCompileOptions new];
    if (@available(macOS 13.0, *)) {
        options.languageVersion = MTLLanguageVersion3_0;
    }
    gLib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                             options:options error:&err];
    TORCH_CHECK(gLib, "Metal compile failed: ",
                err ? err.localizedDescription.UTF8String : "unknown");
    gPipelines = [NSMutableDictionary new];
}

static id<MTLComputePipelineState> pso(const char* name) {
    TORCH_CHECK(gLib, "call metal_gauss_metal.init(source) first");
    NSString* key = [NSString stringWithUTF8String:name];
    id<MTLComputePipelineState> p = gPipelines[key];
    if (p) return p;
    NSError* err = nil;
    id<MTLFunction> fn = [gLib newFunctionWithName:key];
    TORCH_CHECK(fn, "kernel not found: ", name);
    p = [MTLCreateSystemDefaultDevice() newComputePipelineStateWithFunction:fn error:&err];
    TORCH_CHECK(p, "pipeline failed for ", name);
    gPipelines[key] = p;
    return p;
}

static void checkMPS(const torch::Tensor& t, const char* what) {
    TORCH_CHECK(t.device().is_mps(), what, " must be an MPS tensor");
    TORCH_CHECK(t.is_contiguous(), what, " must be contiguous");
}

#define SETBUF(enc, t, idx) \
    [enc setBuffer:getMTLBufferStorage(t) \
            offset:(t).storage_offset() * (t).element_size() atIndex:idx]

std::vector<torch::Tensor> rasterize_forward(
    torch::Tensor uv, torch::Tensor conic, torch::Tensor opacity, torch::Tensor color,
    torch::Tensor gauss_ids, torch::Tensor tile_offsets,
    int64_t W, int64_t H, int64_t tile, int64_t tiles_x)
{
    for (auto& t : {uv, conic, opacity, color, gauss_ids, tile_offsets}) checkMPS(t, "input");

    auto fopt = torch::TensorOptions().dtype(torch::kFloat).device(uv.device());
    auto iopt = torch::TensorOptions().dtype(torch::kInt32).device(uv.device());
    auto out_rgb   = torch::zeros({H, W, 3}, fopt);
    auto out_alpha = torch::zeros({H, W}, fopt);
    auto out_T     = torch::ones({H, W}, fopt);
    auto out_n     = torch::zeros({H, W}, iopt);

    @autoreleasepool {
        id<MTLCommandBuffer> cb = torch::mps::get_command_buffer();
        dispatch_sync(torch::mps::get_dispatch_queue(), ^{
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso("rasterize_forward")];
            SETBUF(enc, uv, 0);           SETBUF(enc, conic, 1);
            SETBUF(enc, opacity, 2);      SETBUF(enc, color, 3);
            SETBUF(enc, gauss_ids, 4);    SETBUF(enc, tile_offsets, 5);
            SETBUF(enc, out_rgb, 6);      SETBUF(enc, out_alpha, 7);
            SETBUF(enc, out_T, 8);        SETBUF(enc, out_n, 9);
            uint dims[4] = {(uint)W, (uint)H, (uint)tile, (uint)tiles_x};
            [enc setBytes:dims length:sizeof(dims) atIndex:10];
            [enc dispatchThreads:MTLSizeMake(W, H, 1)
              threadsPerThreadgroup:MTLSizeMake(tile, tile, 1)];
            [enc endEncoding];
            torch::mps::commit();
        });
    }
    return {out_rgb, out_alpha, out_T, out_n};
}

std::vector<torch::Tensor> pack_intersections(
    torch::Tensor uv, torch::Tensor conic, torch::Tensor opacity,
    torch::Tensor color, torch::Tensor gauss_ids)
{
    auto fopt = torch::TensorOptions().dtype(torch::kFloat).device(uv.device());
    const int64_t M = gauss_ids.numel();
    auto p_xy_opac = torch::empty({M, 3}, fopt);
    auto p_conic   = torch::empty({M, 3}, fopt);
    auto p_rgb     = torch::empty({M, 3}, fopt);
    if (M == 0) return {p_xy_opac, p_conic, p_rgb};

    @autoreleasepool {
        id<MTLCommandBuffer> cb = torch::mps::get_command_buffer();
        dispatch_sync(torch::mps::get_dispatch_queue(), ^{
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso("pack_intersections")];
            SETBUF(enc, uv, 0);         SETBUF(enc, conic, 1);
            SETBUF(enc, opacity, 2);    SETBUF(enc, color, 3);
            SETBUF(enc, gauss_ids, 4);
            SETBUF(enc, p_xy_opac, 5);  SETBUF(enc, p_conic, 6);
            SETBUF(enc, p_rgb, 7);
            uint n = (uint)M;
            [enc setBytes:&n length:sizeof(n) atIndex:8];
            [enc dispatchThreads:MTLSizeMake(M, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            torch::mps::commit();
        });
    }
    return {p_xy_opac, p_conic, p_rgb};
}

std::vector<torch::Tensor> rasterize_backward(
    torch::Tensor uv, torch::Tensor conic, torch::Tensor opacity, torch::Tensor color,
    torch::Tensor gauss_ids, torch::Tensor tile_offsets,
    torch::Tensor final_T, torch::Tensor n_contrib,
    torch::Tensor grad_rgb, torch::Tensor grad_alpha,
    int64_t W, int64_t H, int64_t tile, int64_t tiles_x,
    bool want_absgrad)
{
    auto fopt = torch::TensorOptions().dtype(torch::kFloat).device(uv.device());
    const int64_t N = uv.size(0);
    auto d_uv      = torch::zeros({N, 2}, fopt);
    auto d_conic   = torch::zeros({N, 3}, fopt);
    auto d_opacity = torch::zeros({N}, fopt);
    auto d_color   = torch::zeros({N, 3}, fopt);
    // absgrad statistic, not an adjoint: per-gaussian sum of per-PIXEL |d_uv|.
    // Allocated at full size only when asked for; the kernel skips the
    // reduction, the sqrt and the tenth atomic otherwise.
    auto d_absuv   = torch::zeros({want_absgrad ? N : 1}, fopt);

    grad_rgb = grad_rgb.contiguous();
    grad_alpha = grad_alpha.contiguous();

    @autoreleasepool {
        id<MTLCommandBuffer> cb = torch::mps::get_command_buffer();
        dispatch_sync(torch::mps::get_dispatch_queue(), ^{
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso("rasterize_backward")];
            SETBUF(enc, uv, 0);          SETBUF(enc, conic, 1);
            SETBUF(enc, opacity, 2);     SETBUF(enc, color, 3);
            SETBUF(enc, gauss_ids, 4);   SETBUF(enc, tile_offsets, 5);
            SETBUF(enc, final_T, 6);     SETBUF(enc, n_contrib, 7);
            SETBUF(enc, grad_rgb, 8);    SETBUF(enc, grad_alpha, 9);
            SETBUF(enc, d_uv, 10);       SETBUF(enc, d_conic, 11);
            SETBUF(enc, d_opacity, 12);  SETBUF(enc, d_color, 13);
            uint dims[4] = {(uint)W, (uint)H, (uint)tile, (uint)tiles_x};
            [enc setBytes:dims length:sizeof(dims) atIndex:14];
            SETBUF(enc, d_absuv, 15);
            uint wa = want_absgrad ? 1u : 0u;
            [enc setBytes:&wa length:sizeof(wa) atIndex:16];
            [enc dispatchThreads:MTLSizeMake(W, H, 1)
              threadsPerThreadgroup:MTLSizeMake(tile, tile, 1)];
            [enc endEncoding];
            torch::mps::commit();
        });
    }
    return {d_uv, d_conic, d_opacity, d_color, d_absuv};
}

void init(const std::string& src) { initLibrary(src); }



// ---- fused preprocess (projection + SH + activation) ----------------------

struct PreParams {
    float R0[4], R1[4], R2[4];
    float intr[4];
    float lims[4];
    float cam_center[4];
    uint32_t misc[4];
    uint32_t shl[4];   // SH layout: stride_dc, stride_rest, offset_rest, unused
};

// The trainer stores the DC band and bands 1+ as separate tensors so Adam can
// give them different learning rates. Concatenating them each step cost 11.2 ms
// fwd+bwd at 600k, so the kernel reads whichever layout it is handed.
static void setShLayout(PreParams& P, const torch::Tensor& sh,
                        const torch::Tensor& sh_rest) {
    const bool fused = (sh.size(1) == 16);
    TORCH_CHECK(fused || (sh.size(1) == 1 && sh_rest.size(1) == 15),
                "SH must be either one (N,16,3) tensor or (N,1,3)+(N,15,3); got ",
                sh.sizes(), " and ", sh_rest.sizes());
    P.shl[0] = fused ? 16u : 1u;
    P.shl[1] = fused ? 16u : 15u;
    P.shl[2] = fused ? 1u : 0u;
    P.shl[3] = 0u;
}

static PreParams makeParams(torch::Tensor viewmat, double fx, double fy, double cx, double cy,
                            double near, double far, double blur, double max_radius,
                            torch::Tensor cam_center, int64_t N, int64_t W, int64_t H,
                            int64_t sh_degree) {
    PreParams P;
    auto vm = viewmat.contiguous().cpu();
    auto a = vm.accessor<float, 2>();
    for (int j = 0; j < 4; ++j) { P.R0[j] = a[0][j]; P.R1[j] = a[1][j]; P.R2[j] = a[2][j]; }
    P.intr[0] = fx; P.intr[1] = fy; P.intr[2] = cx; P.intr[3] = cy;
    P.lims[0] = near; P.lims[1] = far; P.lims[2] = blur; P.lims[3] = max_radius;
    auto cc = cam_center.contiguous().cpu();
    auto c = cc.accessor<float, 1>();
    P.cam_center[0] = c[0]; P.cam_center[1] = c[1]; P.cam_center[2] = c[2]; P.cam_center[3] = 0;
    P.misc[0] = (uint32_t)N; P.misc[1] = (uint32_t)W; P.misc[2] = (uint32_t)H;
    P.misc[3] = (uint32_t)sh_degree;
    return P;
}

std::vector<torch::Tensor> preprocess_forward(
    torch::Tensor means, torch::Tensor quats, torch::Tensor scales, torch::Tensor sh,
    torch::Tensor sh_rest, torch::Tensor opacities,
    torch::Tensor viewmat, double fx, double fy, double cx, double cy,
    int64_t W, int64_t H, double near, double far, double blur, double max_radius,
    torch::Tensor cam_center, int64_t sh_degree)
{
    const int64_t N = means.size(0);
    auto fopt = torch::TensorOptions().dtype(torch::kFloat).device(means.device());
    auto iopt = torch::TensorOptions().dtype(torch::kInt32).device(means.device());
    auto uv = torch::empty({N, 2}, fopt);
    auto conic = torch::empty({N, 3}, fopt);
    auto depth = torch::empty({N}, fopt);
    auto rxy = torch::empty({N, 2}, fopt);
    auto valid = torch::empty({N}, iopt);
    auto color = torch::empty({N, 3}, fopt);

    PreParams P = makeParams(viewmat, fx, fy, cx, cy, near, far, blur, max_radius,
                             cam_center, N, W, H, sh_degree);
    setShLayout(P, sh, sh_rest);
    @autoreleasepool {
        id<MTLCommandBuffer> cb = torch::mps::get_command_buffer();
        dispatch_sync(torch::mps::get_dispatch_queue(), ^{
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso("preprocess_forward")];
            SETBUF(enc, means, 0); SETBUF(enc, quats, 1); SETBUF(enc, scales, 2);
            SETBUF(enc, sh, 3); SETBUF(enc, sh_rest, 12);
            SETBUF(enc, uv, 4); SETBUF(enc, conic, 5); SETBUF(enc, depth, 6);
            SETBUF(enc, rxy, 7); SETBUF(enc, valid, 8); SETBUF(enc, color, 9);
            [enc setBytes:&P length:sizeof(P) atIndex:10];
            torch::Tensor opc = opacities.contiguous();
            SETBUF(enc, opc, 11);
            NSUInteger tg = 256;
            [enc dispatchThreads:MTLSizeMake(N, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];
            torch::mps::commit();
        });
    }
    return {uv, conic, depth, rxy, valid, color};
}

std::vector<torch::Tensor> preprocess_backward(
    torch::Tensor means, torch::Tensor quats, torch::Tensor scales, torch::Tensor sh,
    torch::Tensor sh_rest, torch::Tensor d_uv, torch::Tensor d_conic, torch::Tensor d_color, torch::Tensor valid,
    torch::Tensor viewmat, double fx, double fy, double cx, double cy,
    int64_t W, int64_t H, double near, double far, double blur, double max_radius,
    torch::Tensor cam_center, int64_t sh_degree)
{
    const int64_t N = means.size(0);
    auto fopt = torch::TensorOptions().dtype(torch::kFloat).device(means.device());
    auto d_means = torch::empty({N, 3}, fopt);
    auto d_quats = torch::empty({N, 4}, fopt);
    auto d_scales = torch::empty({N, 3}, fopt);
    // Gradients mirror the input layout: one (N,16,3) when fused, or a
    // (N,1,3)+(N,15,3) pair, so the trainer never splits a concatenated grad.
    const bool fused = (sh.size(1) == 16);
    auto d_sh = torch::empty({N, fused ? 16 : 1, 3}, fopt);
    auto d_sh_rest = fused ? d_sh : torch::empty({N, 15, 3}, fopt);

    PreParams P = makeParams(viewmat, fx, fy, cx, cy, near, far, blur, max_radius,
                             cam_center, N, W, H, sh_degree);
    setShLayout(P, sh, sh_rest);
    d_uv = d_uv.contiguous(); d_conic = d_conic.contiguous(); d_color = d_color.contiguous();
    @autoreleasepool {
        id<MTLCommandBuffer> cb = torch::mps::get_command_buffer();
        dispatch_sync(torch::mps::get_dispatch_queue(), ^{
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso("preprocess_backward")];
            SETBUF(enc, means, 0); SETBUF(enc, quats, 1); SETBUF(enc, scales, 2);
            SETBUF(enc, sh, 3); SETBUF(enc, sh_rest, 14);
            SETBUF(enc, d_uv, 4); SETBUF(enc, d_conic, 5); SETBUF(enc, d_color, 6);
            SETBUF(enc, valid, 7);
            SETBUF(enc, d_means, 8); SETBUF(enc, d_quats, 9); SETBUF(enc, d_scales, 10);
            SETBUF(enc, d_sh, 11); SETBUF(enc, d_sh_rest, 13);
            [enc setBytes:&P length:sizeof(P) atIndex:12];
            [enc dispatchThreads:MTLSizeMake(N, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            torch::mps::commit();
        });
    }
    return {d_means, d_quats, d_scales, d_sh, d_sh_rest};
}

void adam_step(torch::Tensor p, torch::Tensor g, torch::Tensor m, torch::Tensor v,
               double lr, double b1, double b2, double eps, int64_t step)
{
    for (auto& t : {p, g, m, v}) checkMPS(t, "adam tensor");
    TORCH_CHECK(g.numel() == p.numel() && m.numel() == p.numel() && v.numel() == p.numel(),
                "adam_step: parameter/grad/moment numel mismatch");
    const uint n = (uint)p.numel();
    if (n == 0) return;

    struct { uint n; float lr, b1, b2, eps, bc1, sqrt_bc2; } P = {
        n, (float)lr, (float)b1, (float)b2, (float)eps,
        (float)(1.0 - std::pow(b1, (double)step)),
        (float)std::sqrt(1.0 - std::pow(b2, (double)step)),
    };

    @autoreleasepool {
        id<MTLCommandBuffer> cb = torch::mps::get_command_buffer();
        dispatch_sync(torch::mps::get_dispatch_queue(), ^{
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            id<MTLComputePipelineState> ps = pso("adam_step");
            [enc setComputePipelineState:ps];
            SETBUF(enc, p, 0); SETBUF(enc, g, 1);
            SETBUF(enc, m, 2); SETBUF(enc, v, 3);
            [enc setBytes:&P length:sizeof(P) atIndex:4];
            NSUInteger tg = MIN((NSUInteger)256, ps.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake(n, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];
            torch::mps::commit();
        });
    }
}

static void ssim_dispatch(const char* kern, std::vector<torch::Tensor> bufs,
                          int64_t npix, int64_t stride)
{
    @autoreleasepool {
        id<MTLCommandBuffer> cb = torch::mps::get_command_buffer();
        dispatch_sync(torch::mps::get_dispatch_queue(), ^{
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            id<MTLComputePipelineState> ps = pso(kern);
            [enc setComputePipelineState:ps];
            for (size_t k = 0; k < bufs.size(); ++k) SETBUF(enc, bufs[k], k);
            uint dim[2] = {(uint)npix, (uint)stride};
            [enc setBytes:dim length:sizeof(dim) atIndex:bufs.size()];
            NSUInteger tg = MIN((NSUInteger)256, ps.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake((NSUInteger)(3 * npix), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];
            torch::mps::commit();
        });
    }
}

torch::Tensor ssim_tail_forward(torch::Tensor b, int64_t H, int64_t W) {
    checkMPS(b, "blurred stack");
    TORCH_CHECK(b.numel() == 15 * H * W, "ssim_tail_forward: expected 15*H*W");
    auto out = torch::empty({1, 3, H, W}, b.options());
    ssim_dispatch("ssim_tail_forward", {b, out}, H * W, H * W);
    return out;
}

torch::Tensor ssim_tail_backward(torch::Tensor b, torch::Tensor gout,
                                 int64_t H, int64_t W) {
    checkMPS(b, "blurred stack"); checkMPS(gout, "grad");
    auto gb = torch::empty_like(b);
    ssim_dispatch("ssim_tail_backward", {b, gout, gb}, H * W, H * W);
    return gb;
}

torch::Tensor ssim_stack_blur_h(torch::Tensor x, torch::Tensor y, torch::Tensor w,
                                int64_t H, int64_t W) {
    checkMPS(x, "x"); checkMPS(y, "y"); checkMPS(w, "weights");
    TORCH_CHECK(x.numel() == 3 * H * W && y.numel() == x.numel(), "expected (H,W,3)");
    auto out = torch::empty({1, 15, H, W}, x.options());
    @autoreleasepool {
        id<MTLCommandBuffer> cb = torch::mps::get_command_buffer();
        dispatch_sync(torch::mps::get_dispatch_queue(), ^{
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso("ssim_stack_blur_h")];
            SETBUF(enc, x, 0); SETBUF(enc, y, 1); SETBUF(enc, w, 2); SETBUF(enc, out, 3);
            uint hw[2] = {(uint)W, (uint)H};
            [enc setBytes:hw length:sizeof(hw) atIndex:4];
            [enc dispatchThreads:MTLSizeMake(W, H, 3)
              threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc endEncoding];
            torch::mps::commit();
        });
    }
    return out;
}

torch::Tensor ssim_blur15(torch::Tensor src, torch::Tensor w, int64_t dir,
                          int64_t H, int64_t W) {
    checkMPS(src, "src"); checkMPS(w, "weights");
    auto out = torch::empty_like(src);
    @autoreleasepool {
        id<MTLCommandBuffer> cb = torch::mps::get_command_buffer();
        dispatch_sync(torch::mps::get_dispatch_queue(), ^{
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso("ssim_blur15")];
            SETBUF(enc, src, 0); SETBUF(enc, w, 1); SETBUF(enc, out, 2);
            uint hwd[4] = {(uint)W, (uint)H, (uint)dir, 0};
            [enc setBytes:hwd length:sizeof(hwd) atIndex:3];
            [enc dispatchThreads:MTLSizeMake(W, H, 15)
              threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc endEncoding];
            torch::mps::commit();
        });
    }
    return out;
}

torch::Tensor ssim_chain_backward(torch::Tensor x, torch::Tensor y,
                                  torch::Tensor d_stack, int64_t H, int64_t W) {
    checkMPS(x, "x"); checkMPS(y, "y"); checkMPS(d_stack, "d_stack");
    auto d_x = torch::empty_like(x);
    @autoreleasepool {
        id<MTLCommandBuffer> cb = torch::mps::get_command_buffer();
        dispatch_sync(torch::mps::get_dispatch_queue(), ^{
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso("ssim_chain_backward")];
            SETBUF(enc, x, 0); SETBUF(enc, y, 1); SETBUF(enc, d_stack, 2); SETBUF(enc, d_x, 3);
            uint hw[2] = {(uint)W, (uint)H};
            [enc setBytes:hw length:sizeof(hw) atIndex:4];
            [enc dispatchThreads:MTLSizeMake(W, H, 3)
              threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc endEncoding];
            torch::mps::commit();
        });
    }
    return d_x;
}

struct BinParamsHost { uint N, W, H, tile, tiles_x, tiles_y, pad0, pad1; };

static void bin_dispatch(const char* kern, std::vector<torch::Tensor> bufs,
                         BinParamsHost P, int bufIndexForParams)
{
    @autoreleasepool {
        id<MTLCommandBuffer> cb = torch::mps::get_command_buffer();
        dispatch_sync(torch::mps::get_dispatch_queue(), ^{
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            id<MTLComputePipelineState> ps = pso(kern);
            [enc setComputePipelineState:ps];
            for (size_t k = 0; k < bufs.size(); ++k) SETBUF(enc, bufs[k], k);
            [enc setBytes:&P length:sizeof(P) atIndex:bufIndexForParams];
            NSUInteger tg = MIN((NSUInteger)256, ps.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake(P.N, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];
            torch::mps::commit();
        });
    }
}

torch::Tensor bin_count(torch::Tensor uv, torch::Tensor rxy, torch::Tensor conic,
                        torch::Tensor opacity, torch::Tensor valid,
                        int64_t W, int64_t H, int64_t tile,
                        int64_t tiles_x, int64_t tiles_y)
{
    const int64_t N = uv.size(0);
    auto counts = torch::empty({N}, torch::TensorOptions()
                               .dtype(torch::kInt32).device(uv.device()));
    BinParamsHost P{(uint)N, (uint)W, (uint)H, (uint)tile,
                    (uint)tiles_x, (uint)tiles_y, 0, 0};
    bin_dispatch("bin_count", {uv, rxy, conic, opacity, valid, counts}, P, 6);
    return counts;
}

std::vector<torch::Tensor> bin_write(torch::Tensor uv, torch::Tensor rxy,
                                     torch::Tensor conic, torch::Tensor opacity,
                                     torch::Tensor valid, torch::Tensor depth,
                                     torch::Tensor offsets, int64_t total,
                                     int64_t W, int64_t H, int64_t tile,
                                     int64_t tiles_x, int64_t tiles_y)
{
    const int64_t N = uv.size(0);
    auto keys = torch::empty({total}, torch::TensorOptions()
                             .dtype(torch::kInt64).device(uv.device()));
    auto ids  = torch::empty({total}, torch::TensorOptions()
                             .dtype(torch::kInt32).device(uv.device()));
    BinParamsHost P{(uint)N, (uint)W, (uint)H, (uint)tile,
                    (uint)tiles_x, (uint)tiles_y, 0, 0};
    bin_dispatch("bin_write",
                 {uv, rxy, conic, opacity, valid, depth, offsets, keys, ids}, P, 9);
    return {keys, ids};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init", &init, "compile the Metal library from source");
    m.def("rasterize_forward", &rasterize_forward);
    m.def("rasterize_backward", &rasterize_backward);
    m.def("pack_intersections", &pack_intersections);
    m.def("preprocess_forward", &preprocess_forward);
    m.def("preprocess_backward", &preprocess_backward);
    m.def("adam_step", &adam_step);
    m.def("ssim_tail_forward", &ssim_tail_forward);
    m.def("ssim_tail_backward", &ssim_tail_backward);
    m.def("ssim_stack_blur_h", &ssim_stack_blur_h);
    m.def("ssim_blur15", &ssim_blur15);
    m.def("ssim_chain_backward", &ssim_chain_backward);
    m.def("bin_count", &bin_count);
    m.def("bin_write", &bin_write);
}
