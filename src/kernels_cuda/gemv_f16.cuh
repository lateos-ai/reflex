// Shared by gemv.cu and gemv_gather.cu: one warp's dot product of an f16
// weight row with an f32 input vector, accumulated in f32. Same warp-per-row
// layout and shuffle-tree reduction as the f32 `gemv_kernel` (see gemv.cu for
// the coalescing rationale); only the weight load is narrower.
//
// Widest path first: 16-byte loads (8 halves per lane, so one warp reads 512
// contiguous bytes of the row per iteration) when `in_features` is a multiple
// of 8 and both the row and `x` are 16-byte aligned. Then `half2` loads when
// the row is 4-byte and `x` 8-byte aligned with an even `in_features`, then a
// scalar fallback for any size. Alignment is checked at run time because
// callers pass zero-copy views into stacked per-expert/per-head tensors whose
// offsets the kernel can't assume. Every lane of a warp sees the same pointers
// and sizes, so the branch is uniform.
//
// Weights are converted to f32 exactly (`__half22float2`), each product is an
// f32 multiply, and the sum is f32: the only difference from the f32 kernel
// is the weight values themselves (f16-rounded at load).

#pragma once

#include <cuda_fp16.h>

__device__ __forceinline__ float warp_dot_f16(
    const float* __restrict__ x,
    const __half* __restrict__ row,
    unsigned int n,
    unsigned int lane
) {
    const unsigned long long row_addr = reinterpret_cast<unsigned long long>(row);
    const unsigned long long x_addr = reinterpret_cast<unsigned long long>(x);
    float sum = 0.0f;

    if ((n & 7u) == 0u && (row_addr & 15ull) == 0ull && (x_addr & 15ull) == 0ull) {
        const uint4* row8 = reinterpret_cast<const uint4*>(row);
        const float4* x4 = reinterpret_cast<const float4*>(x);
        const unsigned int n8 = n >> 3;
        for (unsigned int i = lane; i < n8; i += 32u) {
            const uint4 raw = row8[i];
            const __half2* h = reinterpret_cast<const __half2*>(&raw);
            const float2 w0 = __half22float2(h[0]);
            const float2 w1 = __half22float2(h[1]);
            const float2 w2 = __half22float2(h[2]);
            const float2 w3 = __half22float2(h[3]);
            const float4 xa = x4[2u * i];
            const float4 xb = x4[2u * i + 1u];
            sum += xa.x * w0.x + xa.y * w0.y + xa.z * w1.x + xa.w * w1.y
                 + xb.x * w2.x + xb.y * w2.y + xb.z * w3.x + xb.w * w3.y;
        }
    } else if ((n & 1u) == 0u && (row_addr & 3ull) == 0ull && (x_addr & 7ull) == 0ull) {
        const __half2* row2 = reinterpret_cast<const __half2*>(row);
        const float2* x2 = reinterpret_cast<const float2*>(x);
        const unsigned int n2 = n >> 1;
        for (unsigned int i = lane; i < n2; i += 32u) {
            const float2 w = __half22float2(row2[i]);
            const float2 xv = x2[i];
            sum += xv.x * w.x + xv.y * w.y;
        }
    } else {
        for (unsigned int i = lane; i < n; i += 32u) {
            sum += x[i] * __half2float(row[i]);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    return sum;
}
