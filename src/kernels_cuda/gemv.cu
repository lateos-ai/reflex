// f32 matrix-vector product: y = x @ W^T (the nn.Linear convention), for a
// single input row (this MVP always forwards one token at a time). `w` is
// row-major (out_features, in_features); each output element is one dot
// product over in_features.
//
// Warp-per-row (not thread-per-row, see git history for the original naive
// version): with one thread computing one full row's dot product, threads
// within a warp read DIFFERENT rows at the same loop iteration --
// `w[j*in_features + i]` for consecutive j is strided by `in_features`
// elements, so no two lanes of a warp ever land in the same 128-byte cache
// line. Measured at ~9% of a T4's peak memory bandwidth on the decode path
// (65ms/token) -- see STATUS.md's "warm-latency perf vs. TypeSafe Jev" entry.
// Assigning one WARP to each output row instead means all 32 lanes read the
// SAME row at the same iteration, just 32 consecutive elements apart --
// fully coalesced. `float4` loads add a second 4x on top when `in_features`
// is a multiple of 4 (true for every hidden/FFN size this project's real
// fixtures use); a scalar fallback keeps this correct for any input size
// instead of relying on that always holding.
//
// Reduction order changes from the original (strict left-to-right
// accumulation) to a per-lane partial-sum + warp-shuffle tree reduction --
// floating-point addition isn't associative, so results can differ at the
// ULP level from before. Needs real-hardware byte-exact re-verification
// against llama.cpp before being trusted for `reflex check`'s methodology
// (see STATUS.md) -- not yet done as of this kernel's rewrite.
extern "C" __global__ void gemv_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    unsigned int in_features,
    unsigned int out_features
) {
    const unsigned int warps_per_block = blockDim.x >> 5;
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int row = blockIdx.x * warps_per_block + warp_id;
    if (row >= out_features) {
        return;
    }

    const float* row_ptr = w + (unsigned long long)row * in_features;
    float sum = 0.0f;

    if ((in_features & 3u) == 0u) {
        const float4* x4 = reinterpret_cast<const float4*>(x);
        const float4* row4 = reinterpret_cast<const float4*>(row_ptr);
        unsigned int vec_count = in_features >> 2;
        for (unsigned int idx = lane; idx < vec_count; idx += 32u) {
            float4 xv = x4[idx];
            float4 wv = row4[idx];
            sum += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
        }
    } else {
        for (unsigned int i = lane; i < in_features; i += 32u) {
            sum += x[i] * row_ptr[i];
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0u) {
        y[row] = sum;
    }
}
