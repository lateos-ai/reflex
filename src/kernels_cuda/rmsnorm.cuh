// The row reduction shared by rmsnorm_kernel (rmsnorm.cu) and the fused decode
// norms (decode_ops.cu): 1 / sqrt(mean(x^2) + eps), with the sum of squares taken by
// one thread in index order. Every RMSNorm in the project sums in exactly this order
// (as fused multiply-adds, which nvcc emits for `sum_sq += v * v`), so outputs stay
// bit-identical across kernels; a parallel tree reduction would change them.
//
// The order is fixed, so the sum is a chain of dependent adds; what can be cut is the
// time between them. With `vec4`, `s` must be 16-byte aligned (shared memory declared
// as float4): the loop reads four floats per load and unrolls, so the loads run ahead
// of the chain instead of waiting on shared-memory latency for every element. The
// additions themselves are the same, in the same order.
#pragma once

__device__ __forceinline__ float rmsnorm_rms_inv(
    const float* __restrict__ s,
    unsigned int hidden_size,
    float eps,
    bool vec4
) {
    float sum_sq = 0.0f;
    unsigned int i = 0;
    if (vec4) {
        const float4* s4 = reinterpret_cast<const float4*>(s);
        const unsigned int n4 = hidden_size / 4;
        #pragma unroll 8
        for (unsigned int j = 0; j < n4; j++) {
            const float4 v = s4[j];
            sum_sq += v.x * v.x;
            sum_sq += v.y * v.y;
            sum_sq += v.z * v.z;
            sum_sq += v.w * v.w;
        }
        i = n4 * 4;
    }
    for (; i < hidden_size; i++) {
        float v = s[i];
        sum_sq += v * v;
    }
    return 1.0f / sqrtf(sum_sq / hidden_size + eps);
}
