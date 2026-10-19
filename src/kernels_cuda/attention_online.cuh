// The online-softmax attention kernel body behind attention_online.cu (f32 K/V) and
// kv_f16.cu (f16 K/V cache, REFLEX_KV_CACHE=f16), templated on the cache's element
// type T. See attention_online.cu for the design, layout and launch.
#pragma once
#include "kv_load.cuh"

#define ATTN_WARPS 8
#define ATTN_MAX_V_PER_LANE 32

// Must match `AttnParams` in src/model/kernels.rs field for field (repr(C)).
struct AttnParams {
    unsigned long long k_pos_stride;
    unsigned long long k_head_stride;
    unsigned long long v_pos_stride;
    unsigned long long v_head_stride;
    unsigned int num_q_heads;
    unsigned int group_size;
    unsigned int qk_dim;
    unsigned int v_dim;
    unsigned int start_pos;
    unsigned int num_splits;
    unsigned int split_len;
    unsigned int window;  // 0 = no sliding window
    float scale;
};

__device__ __forceinline__ float warp_sum(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffffu, x, offset);
    }
    return x;
}

template <typename T>
__device__ __forceinline__ void attention_online(
    const float* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    float* __restrict__ dst,
    const AttnParams& p
) {
    extern __shared__ float smem[];
    float* q_s = smem;                         // [qk_dim]
    float* warp_m = q_s + p.qk_dim;            // [ATTN_WARPS]
    float* warp_l = warp_m + ATTN_WARPS;       // [ATTN_WARPS]
    float* warp_acc = warp_l + ATTN_WARPS;     // [ATTN_WARPS][v_dim]

    const unsigned int row = blockIdx.x;
    const unsigned int qh = blockIdx.y;
    const unsigned int split = blockIdx.z;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int kvh = qh / p.group_size;

    const unsigned long long qrow = (unsigned long long)row * p.num_q_heads + qh;
    for (unsigned int d = tid; d < p.qk_dim; d += blockDim.x) {
        q_s[d] = q[qrow * p.qk_dim + d];
    }
    __syncthreads();

    // This block's slice of the causal window [lo, seq_len).
    const unsigned int seq_len = p.start_pos + row + 1;
    const unsigned int lo = (p.window != 0 && seq_len > p.window) ? seq_len - p.window : 0;
    const unsigned int begin = max(split * p.split_len, lo);
    const unsigned int end = min(seq_len, begin + p.split_len);

    float m = -INFINITY;
    float l = 0.0f;
    float acc[ATTN_MAX_V_PER_LANE];
#pragma unroll
    for (int i = 0; i < ATTN_MAX_V_PER_LANE; i++) {
        acc[i] = 0.0f;
    }

    const T* k_head = k + (unsigned long long)kvh * p.k_head_stride;
    const T* v_head = v + (unsigned long long)kvh * p.v_head_stride;
    for (unsigned int pos = begin + warp; pos < end; pos += ATTN_WARPS) {
        const T* kp = k_head + (unsigned long long)pos * p.k_pos_stride;
        float dot = 0.0f;
        for (unsigned int d = lane; d < p.qk_dim; d += 32) {
            dot += q_s[d] * kv_ld(kp, d);
        }
        const float s = warp_sum(dot) * p.scale;

        const float m_new = fmaxf(m, s);
        const float correction = expf(m - m_new);  // 0 on the first position (m = -inf)
        const float weight = expf(s - m_new);
        l = l * correction + weight;
        const T* vp = v_head + (unsigned long long)pos * p.v_pos_stride;
#pragma unroll
        for (int i = 0; i < ATTN_MAX_V_PER_LANE; i++) {
            const unsigned int d = i * 32 + lane;
            if (d >= p.v_dim) {
                break;
            }
            acc[i] = acc[i] * correction + weight * kv_ld(vp, d);
        }
        m = m_new;
    }

    // Merge this block's warps. A warp that saw no positions has m = -inf, l = 0.
    if (lane == 0) {
        warp_m[warp] = m;
        warp_l[warp] = l;
    }
#pragma unroll
    for (int i = 0; i < ATTN_MAX_V_PER_LANE; i++) {
        const unsigned int d = i * 32 + lane;
        if (d >= p.v_dim) {
            break;
        }
        warp_acc[warp * p.v_dim + d] = acc[i];
    }
    __syncthreads();

    float block_m = -INFINITY;
    for (int w = 0; w < ATTN_WARPS; w++) {
        block_m = fmaxf(block_m, warp_m[w]);
    }
    float block_l = 0.0f;
    if (block_m != -INFINITY) {
        for (int w = 0; w < ATTN_WARPS; w++) {
            block_l += warp_l[w] * expf(warp_m[w] - block_m);
        }
    }

    if (p.num_splits == 1) {
        float* out = dst + qrow * p.v_dim;
        for (unsigned int d = tid; d < p.v_dim; d += blockDim.x) {
            float num = 0.0f;
            for (int w = 0; w < ATTN_WARPS; w++) {
                num += warp_acc[w * p.v_dim + d] * expf(warp_m[w] - block_m);
            }
            out[d] = num / block_l;
        }
    } else {
        float* part = dst + (qrow * p.num_splits + split) * (p.v_dim + 2);
        for (unsigned int d = tid; d < p.v_dim; d += blockDim.x) {
            float num = 0.0f;
            if (block_m != -INFINITY) {
                for (int w = 0; w < ATTN_WARPS; w++) {
                    num += warp_acc[w * p.v_dim + d] * expf(warp_m[w] - block_m);
                }
            }
            part[d] = num;
        }
        if (tid == 0) {
            part[p.v_dim] = block_m;
            part[p.v_dim + 1] = block_l;
        }
    }
}

