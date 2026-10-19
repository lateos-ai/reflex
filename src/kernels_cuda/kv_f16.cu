// Kernels for an f16 K/V cache (REFLEX_KV_CACHE=f16), in a module of their own that
// only a process with that setting loads, so the default f32 build's modules don't
// grow. Activations, scores and accumulators stay f32; only the cache is f16.

#include "attention_online.cuh"

// attention_online_kernel for an f16 K/V cache: the same code, widening each K/V
// element to f32 as it loads. Its split partials go through the f32 build's
// attention_online_combine_kernel unchanged.
extern "C" __global__ void attention_online_f16_kernel(
    const float* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    float* __restrict__ dst,
    AttnParams p
) {
    attention_online<__half>(q, k, v, dst, p);
}

// Writes n f32 values into an f16 cache (kv_to_f16: round to nearest even,
// saturating at +-65504). Every K/V cache write of the f16 path goes through this.
extern "C" __global__ void kv_store_f16_kernel(
    const float* __restrict__ src,
    __half* __restrict__ dst,
    unsigned long long n
) {
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = kv_to_f16(src[i]);
    }
}

// kv_store_f16_kernel for a K and a V write of n values each, in one launch: the
// shape of every dense/hybrid cache write (decode row or prefill block).
extern "C" __global__ void kv_store_kv_f16_kernel(
    const float* __restrict__ k,
    const float* __restrict__ v,
    __half* __restrict__ k_dst,
    __half* __restrict__ v_dst,
    unsigned long long n
) {
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        k_dst[i] = kv_to_f16(k[i]);
    } else if (i < 2 * n) {
        v_dst[i - n] = kv_to_f16(v[i - n]);
    }
}

// mla_write_kv_cache_batch_kernel (elementwise.cu) for an f16 latent cache.
extern "C" __global__ void mla_write_kv_cache_batch_f16_kernel(
    __half* __restrict__ kv_cache,
    const float* __restrict__ kv_cmpr,
    const float* __restrict__ k_pe,
    unsigned int start_pos,
    unsigned int rows,
    unsigned int kv_lora,
    unsigned int qk_rope
) {
    unsigned int qk_dim = kv_lora + qk_rope;
    unsigned long long total = (unsigned long long)rows * qk_dim;
    unsigned long long global_tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned long long idx = global_tid; idx < total; idx += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned int col = idx % qk_dim;
        unsigned long long row = idx / qk_dim;
        unsigned long long dst_idx = (start_pos + row) * (unsigned long long)qk_dim + col;
        kv_cache[dst_idx] = kv_to_f16(
            (col < kv_lora) ? kv_cmpr[row * kv_lora + col] : k_pe[row * qk_rope + (col - kv_lora)]
        );
    }
}
