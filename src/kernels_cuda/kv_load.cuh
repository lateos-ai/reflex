// K/V cache element loads for the attention kernels, which are templated on the
// cache's element type: f32 (the default) or f16 (REFLEX_KV_CACHE=f16). Every load
// widens to f32; all arithmetic stays f32. `kv_ld(p, i)` is `p[i]` with the caller's
// own index expression, so the f32 instantiations compile to exactly the code the
// untemplated kernels did (their cubins are byte-identical).
#pragma once
#include <cuda_fp16.h>

template <typename I>
__device__ __forceinline__ float kv_ld(const float* p, I i) {
    return p[i];
}

template <typename I>
__device__ __forceinline__ float kv_ld(const __half* p, I i) {
    return __half2float(p[i]);
}

// f32 -> f16 for a cache write: round to nearest even, saturating at +-65504 (f16's
// largest finite value) so an outlier can't turn into inf and then NaN in softmax;
// NaN stays NaN. The same expression f16 cache-file export uses
// (cast_act_f16_kernel), so a cache row and an exported one round alike.
__device__ __forceinline__ __half kv_to_f16(float x) {
    return __float2half_rn(fabsf(x) > 65504.0f ? copysignf(65504.0f, x) : x);
}
