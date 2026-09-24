// On-GPU dequantization for the K-quant block formats that dominate weight
// bytes in a typical Q4_K_M/Q5_K_M GGUF (Q4_K/Q5_K attention/FFN projections,
// Q6_K for a handful of higher-precision tensors like attn_output/ffn_down).
// Phase 2 (Fast IO) round 3 added Q4_K/Q6_K; Q5_K followed the same pattern
// as a post-MVP extension. Each closes the remaining cold-start gap vs.
// llama.cpp, which never materializes a full-f32 host copy of quantized
// weights -- unlike this project's model.rs before Phase 2 round 3, which did
// `dequant::dequantize` (CPU, src/dequant.rs) then `htod_sync_copy` for every
// tensor.
//
// Each kernel is a line-for-line port of its host counterpart in
// src/dequant.rs (`dequantize_block_q4_k`/`dequantize_block_q5_k`/
// `dequantize_block_q6_k`, themselves line-for-line ports of upstream
// ggml-quants.c) -- variable names
// (`d`, `dmin`, `sc`, `m`, `ql`, `qh`, `is`, `shift`, ...) are kept identical
// on purpose so the CUDA and Rust/C versions can be diffed by eye. One CUDA
// thread handles one whole 256-element block (correctness-first, matching
// this project's existing `gemv_kernel`/`rmsnorm_kernel` style -- no
// warp-level tricks), parallelized across the (typically tens of thousands
// of) blocks in a weight tensor.

#include <cuda_fp16.h>
#include <cstring>

__device__ __forceinline__ float le_f16(const unsigned char* b) {
    unsigned short bits = (unsigned short)b[0] | ((unsigned short)b[1] << 8);
    __half h;
    memcpy(&h, &bits, sizeof(h));
    return __half2float(h);
}

// Port of `get_scale_min_k4` (ggml-quants.c / src/dequant.rs): unpacks one of
// 8 (scale, min) pairs from a Q4_K/Q5_K block's 12-byte `scales` field, each
// 6 bits wide.
__device__ __forceinline__ void get_scale_min_k4(unsigned int j, const unsigned char* q, unsigned char* d_out, unsigned char* m_out) {
    if (j < 4) {
        *d_out = q[j] & 63;
        *m_out = q[j + 4] & 63;
    } else {
        *d_out = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m_out = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Port of `dequantize_row_q4_K`. Block layout (144 bytes): `d: f16`,
// `dmin: f16`, `scales[12]`, `qs[128]`.
extern "C" __global__ void dequantize_q4k_kernel(
    const unsigned char* __restrict__ blocks,
    float* __restrict__ y,
    unsigned int num_blocks
) {
    unsigned int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= num_blocks) {
        return;
    }
    const unsigned char* block = blocks + (unsigned long long)bi * 144;
    float* yb = y + (unsigned long long)bi * 256;

    float d = le_f16(block);
    float dmin = le_f16(block + 2);
    const unsigned char* scales = block + 4;
    const unsigned char* q = block + 16;

    unsigned int is = 0;
    unsigned int y_off = 0;
    unsigned int q_off = 0;
    unsigned int j = 0;
    while (j < 256) {
        unsigned char sc, m;
        get_scale_min_k4(is, scales, &sc, &m);
        float d1 = d * (float)sc;
        float m1 = dmin * (float)m;
        get_scale_min_k4(is + 1, scales, &sc, &m);
        float d2 = d * (float)sc;
        float m2 = dmin * (float)m;

        for (unsigned int l = 0; l < 32; l++) {
            yb[y_off + l] = d1 * (float)(q[q_off + l] & 0xF) - m1;
        }
        for (unsigned int l = 0; l < 32; l++) {
            yb[y_off + 32 + l] = d2 * (float)(q[q_off + l] >> 4) - m2;
        }
        q_off += 32;
        is += 2;
        y_off += 64;
        j += 64;
    }
}

// Port of `dequantize_row_q5_K`. Block layout (176 bytes): `d: f16`,
// `dmin: f16`, `scales[12]`, `qh[32]`, `qs[128]`.
extern "C" __global__ void dequantize_q5k_kernel(
    const unsigned char* __restrict__ blocks,
    float* __restrict__ y,
    unsigned int num_blocks
) {
    unsigned int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= num_blocks) {
        return;
    }
    const unsigned char* block = blocks + (unsigned long long)bi * 176;
    float* yb = y + (unsigned long long)bi * 256;

    float d = le_f16(block);
    float dmin = le_f16(block + 2);
    const unsigned char* scales = block + 4;
    const unsigned char* qh = block + 16;
    const unsigned char* ql = block + 48;

    unsigned int is = 0;
    unsigned char u1 = 1;
    unsigned char u2 = 2;
    unsigned int y_off = 0;
    unsigned int ql_off = 0;
    unsigned int j = 0;
    while (j < 256) {
        unsigned char sc, m;
        get_scale_min_k4(is, scales, &sc, &m);
        float d1 = d * (float)sc;
        float m1 = dmin * (float)m;
        get_scale_min_k4(is + 1, scales, &sc, &m);
        float d2 = d * (float)sc;
        float m2 = dmin * (float)m;

        for (unsigned int l = 0; l < 32; l++) {
            unsigned int hi = (qh[l] & u1) ? 16 : 0;
            yb[y_off + l] = d1 * (float)((ql[ql_off + l] & 0xF) + hi) - m1;
        }
        for (unsigned int l = 0; l < 32; l++) {
            unsigned int hi = (qh[l] & u2) ? 16 : 0;
            yb[y_off + 32 + l] = d2 * (float)((ql[ql_off + l] >> 4) + hi) - m2;
        }
        ql_off += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
        y_off += 64;
        j += 64;
    }
}

// Port of `dequantize_row_q6_K`. Block layout (210 bytes): `ql[128]`,
// `qh[64]`, `scales[16]` (i8), `d: f16`.
extern "C" __global__ void dequantize_q6k_kernel(
    const unsigned char* __restrict__ blocks,
    float* __restrict__ y,
    unsigned int num_blocks
) {
    unsigned int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= num_blocks) {
        return;
    }
    const unsigned char* block = blocks + (unsigned long long)bi * 210;
    float* yb = y + (unsigned long long)bi * 256;

    const unsigned char* ql_all = block;
    const unsigned char* qh_all = block + 128;
    const signed char* sc_all = (const signed char*)(block + 192);
    float d = le_f16(block + 208);

    unsigned int y_off = 0;
    unsigned int ql_off = 0;
    unsigned int qh_off = 0;
    unsigned int sc_off = 0;
    unsigned int n = 0;
    while (n < 256) {
        for (unsigned int l = 0; l < 32; l++) {
            unsigned int is = l / 16;
            unsigned char qh = qh_all[qh_off + l];
            int q1 = (int)((ql_all[ql_off + l] & 0xF) | (((qh >> 0) & 3) << 4)) - 32;
            int q2 = (int)((ql_all[ql_off + l + 32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32;
            int q3 = (int)((ql_all[ql_off + l] >> 4) | (((qh >> 4) & 3) << 4)) - 32;
            int q4 = (int)((ql_all[ql_off + l + 32] >> 4) | (((qh >> 6) & 3) << 4)) - 32;
            yb[y_off + l] = d * (float)sc_all[sc_off + is] * (float)q1;
            yb[y_off + l + 32] = d * (float)sc_all[sc_off + is + 2] * (float)q2;
            yb[y_off + l + 64] = d * (float)sc_all[sc_off + is + 4] * (float)q3;
            yb[y_off + l + 96] = d * (float)sc_all[sc_off + is + 6] * (float)q4;
        }
        y_off += 128;
        ql_off += 64;
        qh_off += 32;
        sc_off += 8;
        n += 128;
    }
}
