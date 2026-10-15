// GQA-aware decode attention: one new query row against the K/V cache, for head dims
// 64, 128 and 256 (qk_dim == v_dim). Same layout, masking and partial format as
// attention_online_kernel (see attention_online.cu), which still serves every other
// shape (MLA, Gemma 4's d512, prefill).
//
// Why: attention_online_kernel gives each query head its own blocks, so a KV head's
// K/V rows are read once per query head of its group, and each warp walks its
// positions one at a time with a full memory latency exposed per position. On a T4 at
// ~1,550 cached positions that ran at ~28% of memory bandwidth, 3.8 ms of an 11.3 ms
// Qwen3-0.6B decode step. Here:
//   - a block takes HB query heads that share one KV head and reads each K and V row
//     once for all of them (scores, online softmax and accumulators per head, in
//     registers);
//   - a warp handles two positions per iteration, issuing both K and V loads before
//     using them;
//   - positions are split across blocks (grid.y) by the launcher to fill the GPU, and
//     each block writes the (acc, m, l) partials attention_online_combine_kernel merges.
//
// Lane l holds dims l, l + 32, ... of a row (DL = D / 32 of them), so every load is
// one coalesced 128-byte segment per 32 dims.
//
// Launch: grid (num_kv_heads * group_size / HB, num_splits), block DEC_WARPS * 32
// threads, static shared memory. dst: num_splits == 1 the output [num_q_heads, D];
// otherwise partials [num_q_heads, num_splits, D + 2] (row index 0 only).

#define DEC_WARPS 4

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

__device__ __forceinline__ float dec_warp_sum(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffffu, x, offset);
    }
    return x;
}

template <int DL, int HB>
__device__ __forceinline__ void attention_decode(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ dst,
    const AttnParams& p
) {
    constexpr int D = DL * 32;
    __shared__ float m_s[DEC_WARPS][HB];
    __shared__ float l_s[DEC_WARPS][HB];
    __shared__ float acc_s[DEC_WARPS][HB][D];

    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int chunks = p.group_size / HB;
    const unsigned int kvh = blockIdx.x / chunks;
    const unsigned int head0 = kvh * p.group_size + (blockIdx.x % chunks) * HB;
    const unsigned int split = blockIdx.y;

    // This block's queries, pre-scaled, in registers.
    float qr[HB][DL];
#pragma unroll
    for (int h = 0; h < HB; h++) {
#pragma unroll
        for (int j = 0; j < DL; j++) {
            qr[h][j] = q[(unsigned long long)(head0 + h) * D + j * 32 + lane] * p.scale;
        }
    }

    const unsigned int seq_len = p.start_pos + 1;
    const unsigned int lo = (p.window != 0 && seq_len > p.window) ? seq_len - p.window : 0;
    const unsigned int begin = max(split * p.split_len, lo);
    const unsigned int end = min(seq_len, split * p.split_len + p.split_len);

    float m[HB], l[HB], acc[HB][DL];
#pragma unroll
    for (int h = 0; h < HB; h++) {
        m[h] = -INFINITY;
        l[h] = 0.0f;
#pragma unroll
        for (int j = 0; j < DL; j++) {
            acc[h][j] = 0.0f;
        }
    }

    const float* k_head = k + (unsigned long long)kvh * p.k_head_stride;
    const float* v_head = v + (unsigned long long)kvh * p.v_head_stride;

    // One position's update for every head, from its K and V rows in registers.
    auto update = [&](const float (&kr)[DL], const float (&vr)[DL]) {
#pragma unroll
        for (int h = 0; h < HB; h++) {
            float dot = 0.0f;
#pragma unroll
            for (int j = 0; j < DL; j++) {
                dot += qr[h][j] * kr[j];
            }
            const float s = dec_warp_sum(dot);
            const float m_new = fmaxf(m[h], s);
            const float correction = expf(m[h] - m_new);  // 0 on the first position
            const float weight = expf(s - m_new);
            l[h] = l[h] * correction + weight;
#pragma unroll
            for (int j = 0; j < DL; j++) {
                acc[h][j] = acc[h][j] * correction + weight * vr[j];
            }
            m[h] = m_new;
        }
    };

    unsigned int pos = begin + warp;
    for (; pos + DEC_WARPS < end; pos += 2 * DEC_WARPS) {
        const float* k0 = k_head + (unsigned long long)pos * p.k_pos_stride;
        const float* k1 = k0 + (unsigned long long)DEC_WARPS * p.k_pos_stride;
        const float* v0 = v_head + (unsigned long long)pos * p.v_pos_stride;
        const float* v1 = v0 + (unsigned long long)DEC_WARPS * p.v_pos_stride;
        float ka[DL], kb[DL], va[DL], vb[DL];
#pragma unroll
        for (int j = 0; j < DL; j++) {
            ka[j] = k0[j * 32 + lane];
            kb[j] = k1[j * 32 + lane];
            va[j] = v0[j * 32 + lane];
            vb[j] = v1[j * 32 + lane];
        }
        update(ka, va);
        update(kb, vb);
    }
    if (pos < end) {
        const float* k0 = k_head + (unsigned long long)pos * p.k_pos_stride;
        const float* v0 = v_head + (unsigned long long)pos * p.v_pos_stride;
        float ka[DL], va[DL];
#pragma unroll
        for (int j = 0; j < DL; j++) {
            ka[j] = k0[j * 32 + lane];
            va[j] = v0[j * 32 + lane];
        }
        update(ka, va);
    }

    // Merge the block's warps. A warp that saw no positions has m = -inf, l = 0.
#pragma unroll
    for (int h = 0; h < HB; h++) {
        if (lane == 0) {
            m_s[warp][h] = m[h];
            l_s[warp][h] = l[h];
        }
#pragma unroll
        for (int j = 0; j < DL; j++) {
            acc_s[warp][h][j * 32 + lane] = acc[h][j];
        }
    }
    __syncthreads();

    for (unsigned int e = threadIdx.x; e < HB * D; e += DEC_WARPS * 32) {
        const unsigned int h = e / D, d = e % D;
        float bm = -INFINITY;
#pragma unroll
        for (int w = 0; w < DEC_WARPS; w++) {
            bm = fmaxf(bm, m_s[w][h]);
        }
        float bl = 0.0f, num = 0.0f;
        if (bm != -INFINITY) {
#pragma unroll
            for (int w = 0; w < DEC_WARPS; w++) {
                const float f = expf(m_s[w][h] - bm);
                bl += l_s[w][h] * f;
                num += acc_s[w][h][d] * f;
            }
        }
        const unsigned int qh = head0 + h;
        if (p.num_splits == 1) {
            dst[(unsigned long long)qh * D + d] = num / bl;
        } else {
            float* part = dst + ((unsigned long long)qh * p.num_splits + split) * (D + 2);
            part[d] = num;
            if (d == 0) {
                part[D] = bm;
                part[D + 1] = bl;
            }
        }
    }
}

#define DEC_KERNEL(D, HB)                                                                   \
    extern "C" __global__ void __launch_bounds__(DEC_WARPS * 32)                            \
        attention_decode_d##D##_h##HB##_kernel(const float* __restrict__ q,                \
                                               const float* __restrict__ k,                \
                                               const float* __restrict__ v,                \
                                               float* __restrict__ dst, AttnParams p) {    \
        attention_decode<D / 32, HB>(q, k, v, dst, p);                                      \
    }

DEC_KERNEL(64, 1)
DEC_KERNEL(64, 2)
DEC_KERNEL(64, 4)
DEC_KERNEL(64, 8)
DEC_KERNEL(128, 1)
DEC_KERNEL(128, 2)
DEC_KERNEL(128, 4)
DEC_KERNEL(128, 8)
DEC_KERNEL(256, 1)
DEC_KERNEL(256, 2)
DEC_KERNEL(256, 4)
