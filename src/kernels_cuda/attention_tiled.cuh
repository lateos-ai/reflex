// Tiled causal attention for batched prefill (flash-attention style, f32 SIMT, no
// tensor cores), templated on the K/V cache's element type T (f32, or f16 with
// REFLEX_KV_CACHE=f16: attention_tiled*_f16.cu), the head dim D and the query vectors per
// block M. Each
// .cu file that includes this header instantiates the head dims one model family needs,
// so a model JIT-compiles (in a PTX build) only its own module:
//   attention_tiled.cu       D 256 and 512 (Gemma 4; Qwen3.5's gated attention, D 256)
//   attention_tiled_d128.cu  D 128 (Qwen3, Llama, Mistral, most Qwen2.5 and MoE files)
//   attention_tiled_d64.cu   D 64 (Qwen2.5-0.5B, TinyLlama)
//
// Why: attention_online_kernel gives every (query row, head) pair its own block, which
// streams that row's whole K/V range from global memory one position per warp. Nothing
// is reused across query rows or across the query heads of a GQA group. On Gemma 4 12B
// at 3,713 tokens that was 86% of prefill, and on Qwen3-0.6B at 2K tokens 91%. Here one
// block takes M query vectors that
// share a KV head -- `rows_per_block` consecutive rows times the `heads_per_block` query
// heads of the group -- and walks the K/V range in tiles of BC positions staged in shared
// memory, so each K/V element loaded serves all M queries.
//
// Per K/V tile:
//   1. S[M][BC] = Q K^T * scale, accumulated over D in chunks of DC (Q and K chunks in
//      shared memory, a 4x4 or 2x4 register tile of S per thread), then masked
//      (causal, sliding window, rows past `rows`) to -inf.
//   2. Online softmax per query: running max m, running sum l, correction
//      alpha = exp(m_old - m_new), P = exp(S - m_new) (0 where S is -inf).
//   3. O = O * alpha + P V, with V streamed through shared memory VC positions at a
//      time. Each thread keeps M*D/256 output values in registers.
// Finally O / l. Numerics: f32 throughout and expf, like attention_online_kernel; only
// the summation order differs (per-tile instead of per-position online softmax).
//
// Layout and masking are attention_online_kernel's (see attention_online.cu): q [rows,
// num_q_heads, D], K/V position p of KV head h at p*pos_stride + h*head_stride, dst
// [rows, num_q_heads, D]; query row r is at position start_pos + r and sees key j iff
// j <= start_pos + r and (window == 0 or start_pos + r - j < window). K/V must hold
// positions 0 .. start_pos + rows. qk_dim == v_dim == D.
//
// Launch: block 256 threads, no dynamic shared memory; grid (ceil(rows / rows_per_block),
// num_kv_heads * group_size / heads_per_block), where heads_per_block = min(group_size,
// M) and rows_per_block = M / heads_per_block. A group larger than M must be a multiple
// of it. A smaller group that doesn't divide M (3, 6, 7) leaves M % group_size query
// slots per block unused: they are masked like rows past `rows`. M is 64, except 32
// for D = 512, which keeps the output registers per thread at 64 or fewer and the
// static shared memory under 48 KB.

#pragma once
#include "kv_load.cuh"

// Must match `AttnParams` in src/model/kernels.rs (and attention_online.cu) field for
// field. num_splits and split_len are unused here.
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

#define TA_THREADS 256
#define TA_BC 64  // K/V positions per tile
#define TA_DC 32  // head-dim chunk for Q K^T

template <typename T, int D, int M>
__device__ __forceinline__ void attention_tiled(
    const float* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    float* __restrict__ dst,
    const AttnParams& p,
    unsigned int rows
) {
    static_assert(M % 16 == 0 && D % 16 == 0 && D % TA_DC == 0, "tile shape");
    constexpr int VC = 4096 / D < TA_BC ? 4096 / D : TA_BC;  // V positions staged at a time
    constexpr int TM = M / 16;         // query vectors per thread (both phases)
    constexpr int TN = TA_BC / 16;     // S columns per thread
    constexpr int TD = D / 16;         // output dims per thread

    // Q/K chunks for step 1 and the V chunk for step 3 never live at the same time.
    // Rows of q_s, k_s and s_tile are padded by one float so that the two query groups
    // (or 16 K rows) a warp reads at once fall in different banks.
    constexpr int QS = TA_DC + 1, SS = TA_BC + 1;
    constexpr int QK_FLOATS = M * QS + TA_BC * QS;
    constexpr int V_FLOATS = VC * D;
    constexpr int STAGE_FLOATS = QK_FLOATS > V_FLOATS ? QK_FLOATS : V_FLOATS;
    __shared__ float stage[STAGE_FLOATS];
    __shared__ float s_tile[M * SS];
    __shared__ float m_run[M];
    __shared__ float l_run[M];
    __shared__ float alpha_s[M];
    float* q_s = stage;                // [M][QS]
    float* k_s = stage + M * QS;       // [BC][QS]
    float* v_s = stage;                // [VC][D]

    const unsigned int tid = threadIdx.x;
    const unsigned int tx = tid & 15u;  // S column / output dim lane
    const unsigned int ty = tid >> 4;   // query-vector group

    const unsigned int g = p.group_size;
    const unsigned int heads_per_block = g < (unsigned int)M ? g : (unsigned int)M;
    const unsigned int rows_per_block = M / heads_per_block;
    // Query slots in use; the rest (when g doesn't divide M) are masked out.
    const unsigned int used = rows_per_block * heads_per_block;
    const unsigned int chunks_per_group = g / heads_per_block;
    const unsigned int kvh = blockIdx.y / chunks_per_group;
    const unsigned int head0 = kvh * g + (blockIdx.y % chunks_per_group) * heads_per_block;
    // Heaviest (latest) row tiles first, so the causal imbalance doesn't leave a tail.
    const unsigned int r0 = (gridDim.x - 1 - blockIdx.x) * rows_per_block;
    const unsigned int kv_len = p.start_pos + rows;

    // Query vector mi is row r0 + mi / heads_per_block, head head0 + mi % heads_per_block.
    const unsigned int last_row = min(r0 + rows_per_block, rows) - 1;
    const unsigned int q_lo = p.start_pos + r0;
    const unsigned int q_hi = p.start_pos + last_row;  // inclusive
    const unsigned int key_lo = (p.window != 0 && q_lo + 1 > p.window) ? q_lo + 1 - p.window : 0;
    const unsigned int tile_lo = key_lo / TA_BC;
    const unsigned int tile_hi = q_hi / TA_BC + 1;  // exclusive

    if (tid < M) {
        m_run[tid] = -INFINITY;
        l_run[tid] = 0.0f;
    }

    float o[TM][TD];
#pragma unroll
    for (int i = 0; i < TM; i++) {
#pragma unroll
        for (int j = 0; j < TD; j++) {
            o[i][j] = 0.0f;
        }
    }

    const T* k_head = k + (unsigned long long)kvh * p.k_head_stride;
    const T* v_head = v + (unsigned long long)kvh * p.v_head_stride;

    for (unsigned int tile = tile_lo; tile < tile_hi; tile++) {
        const unsigned int kbase = tile * TA_BC;

        // 1. S = Q K^T over D, in DC-wide chunks.
        float s[TM][TN];
#pragma unroll
        for (int i = 0; i < TM; i++) {
#pragma unroll
            for (int j = 0; j < TN; j++) {
                s[i][j] = 0.0f;
            }
        }
        for (int dc = 0; dc < D; dc += TA_DC) {
            __syncthreads();  // previous users of `stage` are done
#pragma unroll
            for (int e = tid; e < M * TA_DC; e += TA_THREADS) {
                const unsigned int mi = e / TA_DC, kk = e % TA_DC;
                const unsigned int row = r0 + mi / heads_per_block;
                const unsigned int qh = head0 + mi % heads_per_block;
                q_s[mi * QS + kk] = mi < used && row < rows
                    ? q[((unsigned long long)row * p.num_q_heads + qh) * D + dc + kk]
                    : 0.0f;
            }
#pragma unroll
            for (int e = tid; e < TA_BC * TA_DC; e += TA_THREADS) {
                const unsigned int n = e / TA_DC, kk = e % TA_DC;
                const unsigned int pos = kbase + n;
                k_s[n * QS + kk] = pos < kv_len
                    ? kv_ld(k_head, (unsigned long long)pos * p.k_pos_stride + dc + kk)
                    : 0.0f;
            }
            __syncthreads();
#pragma unroll 8
            for (int kk = 0; kk < TA_DC; kk++) {
                float qv[TM], kv[TN];
#pragma unroll
                for (int i = 0; i < TM; i++) {
                    qv[i] = q_s[(ty * TM + i) * QS + kk];
                }
#pragma unroll
                for (int j = 0; j < TN; j++) {
                    kv[j] = k_s[(tx + 16 * j) * QS + kk];
                }
#pragma unroll
                for (int i = 0; i < TM; i++) {
#pragma unroll
                    for (int j = 0; j < TN; j++) {
                        s[i][j] += qv[i] * kv[j];
                    }
                }
            }
        }
#pragma unroll
        for (int i = 0; i < TM; i++) {
            const unsigned int mi = ty * TM + i;
            const unsigned int row = r0 + mi / heads_per_block;
            const unsigned int qpos = p.start_pos + row;
#pragma unroll
            for (int j = 0; j < TN; j++) {
                const unsigned int pos = kbase + tx + 16 * j;
                const bool visible = mi < used && row < rows && pos <= qpos
                    && (p.window == 0 || qpos - pos < p.window);
                s_tile[mi * SS + tx + 16 * j] = visible ? s[i][j] * p.scale : -INFINITY;
            }
        }
        __syncthreads();

        // 2. Online softmax, one warp per query vector at a time (lanes over BC = 64).
        {
            const unsigned int warp = tid >> 5, lane = tid & 31u;
            for (unsigned int mi = warp; mi < (unsigned int)M; mi += TA_THREADS / 32) {
                float* srow = s_tile + mi * SS;
                const float a = srow[lane], b = srow[lane + 32];
                float mx = fmaxf(a, b);
                for (int off = 16; off > 0; off >>= 1) {
                    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
                }
                const float m_old = m_run[mi];
                const float m_new = fmaxf(m_old, mx);
                const float pa = a == -INFINITY ? 0.0f : expf(a - m_new);
                const float pb = b == -INFINITY ? 0.0f : expf(b - m_new);
                srow[lane] = pa;
                srow[lane + 32] = pb;
                float sum = pa + pb;
                for (int off = 16; off > 0; off >>= 1) {
                    sum += __shfl_xor_sync(0xffffffffu, sum, off);
                }
                if (lane == 0) {
                    const float alpha = m_new == -INFINITY ? 1.0f : expf(m_old - m_new);
                    alpha_s[mi] = alpha;
                    l_run[mi] = l_run[mi] * alpha + sum;
                    m_run[mi] = m_new;
                }
            }
        }
        __syncthreads();

        // 3. O = O * alpha + P V.
#pragma unroll
        for (int i = 0; i < TM; i++) {
            const float alpha = alpha_s[ty * TM + i];
#pragma unroll
            for (int j = 0; j < TD; j++) {
                o[i][j] *= alpha;
            }
        }
        for (int vc = 0; vc < TA_BC; vc += VC) {
            __syncthreads();  // `stage` free (Q/K chunks or the previous V chunk)
#pragma unroll
            for (int e = tid; e < VC * D; e += TA_THREADS) {
                const unsigned int kr = e / D, d = e % D;
                const unsigned int pos = kbase + vc + kr;
                v_s[e] = pos < kv_len ? kv_ld(v_head, (unsigned long long)pos * p.v_pos_stride + d) : 0.0f;
            }
            __syncthreads();
#pragma unroll
            for (int kr = 0; kr < VC; kr++) {
                float pv[TM];
#pragma unroll
                for (int i = 0; i < TM; i++) {
                    pv[i] = s_tile[(ty * TM + i) * SS + vc + kr];
                }
#pragma unroll
                for (int j = 0; j < TD; j++) {
                    const float vv = v_s[kr * D + tx + 16 * j];
#pragma unroll
                    for (int i = 0; i < TM; i++) {
                        o[i][j] += pv[i] * vv;
                    }
                }
            }
        }
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < TM; i++) {
        const unsigned int mi = ty * TM + i;
        const unsigned int row = r0 + mi / heads_per_block;
        if (mi >= used || row >= rows) {
            continue;
        }
        const unsigned int qh = head0 + mi % heads_per_block;
        const float inv_l = 1.0f / l_run[mi];
        float* out = dst + ((unsigned long long)row * p.num_q_heads + qh) * D;
#pragma unroll
        for (int j = 0; j < TD; j++) {
            out[tx + 16 * j] = o[i][j] * inv_l;
        }
    }
}
