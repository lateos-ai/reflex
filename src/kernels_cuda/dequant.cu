// On-GPU dequantization for GGUF block-quantized weight formats. Phase 2
// (Fast IO) round 3 added Q4_K/Q6_K (the formats that dominate weight bytes
// in a typical Q4_K_M GGUF); Q5_K followed as a post-MVP extension; this pass
// adds the legacy 32-element-block formats (Q4_0/1, Q5_0/1, Q8_0/1) and the
// remaining 256-element K-quant formats (Q2_K, Q3_K, Q8_K). Each closes the
// remaining cold-start gap vs. llama.cpp, which never materializes a full-f32
// host copy of quantized weights -- unlike this project's model.rs before
// Phase 2 round 3, which did `dequant::dequantize` (CPU, src/dequant.rs) then
// `htod_sync_copy` for every tensor. The 8 IQ-family formats (IQ2_XXS/XS/S,
// IQ3_XXS/S, IQ1_S/M, IQ4_XS) are deliberately left on the host path for now
// -- they need constant-memory lookup tables ported from ggml-common.h, not
// just this file's per-block-loop pattern; left for a future round.
//
// Each kernel is a line-for-line port of its host counterpart in
// src/dequant.rs (`dequantize_block_q4_k`/`dequantize_block_q5_k`/
// `dequantize_block_q6_k`/etc., themselves line-for-line ports of upstream
// ggml-quants.c) -- variable names
// (`d`, `dmin`, `sc`, `m`, `ql`, `qh`, `is`, `shift`, ...) are kept identical
// on purpose so the CUDA and Rust/C versions can be diffed by eye. One CUDA
// thread handles one whole block (correctness-first, matching this project's
// existing `gemv_kernel`/`rmsnorm_kernel` style -- no warp-level tricks),
// parallelized across the (typically tens of thousands of) blocks in a
// weight tensor.

#include <cuda_fp16.h>
#include <cstring>

__device__ __forceinline__ float le_f16(const unsigned char* b) {
    unsigned short bits = (unsigned short)b[0] | ((unsigned short)b[1] << 8);
    __half h;
    memcpy(&h, &bits, sizeof(h));
    return __half2float(h);
}

// GGUF stores f32 little-endian; on this (little-endian) GPU a raw byte copy
// already produces the right value, same trick `le_f16` uses above. The
// `memcpy` (not a cast-and-deref) avoids unaligned-access UB, since `b` is
// `block + <offset>` and blocks aren't guaranteed 4-byte aligned in the
// uploaded buffer.
__device__ __forceinline__ float le_f32(const unsigned char* b) {
    float f;
    memcpy(&f, b, sizeof(f));
    return f;
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

// Port of `dequantize_row_q4_0`. Block layout (18 bytes): `d: f16`, `qs[16]`.
extern "C" __global__ void dequantize_q4_0_kernel(
    const unsigned char* __restrict__ blocks,
    float* __restrict__ y,
    unsigned int num_blocks
) {
    unsigned int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= num_blocks) {
        return;
    }
    const unsigned char* block = blocks + (unsigned long long)bi * 18;
    float* yb = y + (unsigned long long)bi * 32;

    float d = le_f16(block);
    const unsigned char* qs = block + 2;
    for (unsigned int j = 0; j < 16; j++) {
        int x0 = (int)(qs[j] & 0x0F) - 8;
        int x1 = (int)(qs[j] >> 4) - 8;
        yb[j] = (float)x0 * d;
        yb[j + 16] = (float)x1 * d;
    }
}

// Port of `dequantize_row_q4_1`. Block layout (20 bytes): `d: f16`, `m: f16`,
// `qs[16]`.
extern "C" __global__ void dequantize_q4_1_kernel(
    const unsigned char* __restrict__ blocks,
    float* __restrict__ y,
    unsigned int num_blocks
) {
    unsigned int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= num_blocks) {
        return;
    }
    const unsigned char* block = blocks + (unsigned long long)bi * 20;
    float* yb = y + (unsigned long long)bi * 32;

    float d = le_f16(block);
    float m = le_f16(block + 2);
    const unsigned char* qs = block + 4;
    for (unsigned int j = 0; j < 16; j++) {
        float x0 = (float)(qs[j] & 0x0F);
        float x1 = (float)(qs[j] >> 4);
        yb[j] = x0 * d + m;
        yb[j + 16] = x1 * d + m;
    }
}

// Port of `dequantize_row_q5_0`. Block layout (22 bytes): `d: f16`,
// `qh: u32`, `qs[16]`.
extern "C" __global__ void dequantize_q5_0_kernel(
    const unsigned char* __restrict__ blocks,
    float* __restrict__ y,
    unsigned int num_blocks
) {
    unsigned int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= num_blocks) {
        return;
    }
    const unsigned char* block = blocks + (unsigned long long)bi * 22;
    float* yb = y + (unsigned long long)bi * 32;

    float d = le_f16(block);
    unsigned int qh = (unsigned int)block[2] | ((unsigned int)block[3] << 8) | ((unsigned int)block[4] << 16) | ((unsigned int)block[5] << 24);
    const unsigned char* qs = block + 6;
    for (unsigned int j = 0; j < 16; j++) {
        unsigned int xh_0 = ((qh >> j) << 4) & 0x10;
        unsigned int xh_1 = (qh >> (j + 12)) & 0x10;
        int x0 = (int)((unsigned int)(qs[j] & 0x0F) | xh_0) - 16;
        int x1 = (int)((unsigned int)(qs[j] >> 4) | xh_1) - 16;
        yb[j] = (float)x0 * d;
        yb[j + 16] = (float)x1 * d;
    }
}

// Port of `dequantize_row_q5_1`. Block layout (24 bytes): `d: f16`,
// `m: f16`, `qh: u32`, `qs[16]`.
extern "C" __global__ void dequantize_q5_1_kernel(
    const unsigned char* __restrict__ blocks,
    float* __restrict__ y,
    unsigned int num_blocks
) {
    unsigned int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= num_blocks) {
        return;
    }
    const unsigned char* block = blocks + (unsigned long long)bi * 24;
    float* yb = y + (unsigned long long)bi * 32;

    float d = le_f16(block);
    float m = le_f16(block + 2);
    unsigned int qh = (unsigned int)block[4] | ((unsigned int)block[5] << 8) | ((unsigned int)block[6] << 16) | ((unsigned int)block[7] << 24);
    const unsigned char* qs = block + 8;
    for (unsigned int j = 0; j < 16; j++) {
        unsigned int xh_0 = ((qh >> j) << 4) & 0x10;
        unsigned int xh_1 = (qh >> (j + 12)) & 0x10;
        unsigned int x0 = (unsigned int)(qs[j] & 0x0F) | xh_0;
        unsigned int x1 = (unsigned int)(qs[j] >> 4) | xh_1;
        yb[j] = (float)x0 * d + m;
        yb[j + 16] = (float)x1 * d + m;
    }
}

// Port of `dequantize_row_q8_0`. Block layout (34 bytes): `d: f16`,
// `qs[32]` (i8).
extern "C" __global__ void dequantize_q8_0_kernel(
    const unsigned char* __restrict__ blocks,
    float* __restrict__ y,
    unsigned int num_blocks
) {
    unsigned int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= num_blocks) {
        return;
    }
    const unsigned char* block = blocks + (unsigned long long)bi * 34;
    float* yb = y + (unsigned long long)bi * 32;

    float d = le_f16(block);
    const signed char* qs = (const signed char*)(block + 2);
    for (unsigned int j = 0; j < 32; j++) {
        yb[j] = (float)qs[j] * d;
    }
}

// Port of `dequantize_row_q8_1`'s value semantics (see `dequant.rs`'s doc
// comment on `dequantize_block_q8_1`: no upstream `dequantize_row_q8_1`
// exists, but `x = d*q` is the same as Q8_0 -- the `s` field at bytes 2..4 is
// a precomputed dot-product helper, not part of value reconstruction).
// Block layout (36 bytes): `d: f16`, `s: f16`, `qs[32]` (i8).
extern "C" __global__ void dequantize_q8_1_kernel(
    const unsigned char* __restrict__ blocks,
    float* __restrict__ y,
    unsigned int num_blocks
) {
    unsigned int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= num_blocks) {
        return;
    }
    const unsigned char* block = blocks + (unsigned long long)bi * 36;
    float* yb = y + (unsigned long long)bi * 32;

    float d = le_f16(block);
    const signed char* qs = (const signed char*)(block + 4);
    for (unsigned int j = 0; j < 32; j++) {
        yb[j] = (float)qs[j] * d;
    }
}

// Port of `dequantize_row_q2_K`. Block layout (84 bytes): `scales[16]`,
// `qs[64]`, `d: f16`, `dmin: f16`.
extern "C" __global__ void dequantize_q2k_kernel(
    const unsigned char* __restrict__ blocks,
    float* __restrict__ y,
    unsigned int num_blocks
) {
    unsigned int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= num_blocks) {
        return;
    }
    const unsigned char* block = blocks + (unsigned long long)bi * 84;
    float* yb = y + (unsigned long long)bi * 256;

    const unsigned char* scales = block;
    const unsigned char* q = block + 16;
    float d = le_f16(block + 80);
    float dmin = le_f16(block + 82);

    unsigned int is = 0;
    unsigned int y_off = 0;
    unsigned int q_off = 0;
    unsigned int n = 0;
    while (n < 256) {
        for (unsigned int si = 0; si < 4; si++) {
            unsigned int shift = si * 2;

            unsigned char sc = scales[is];
            is++;
            float dl = d * (float)(sc & 0xF);
            float ml = dmin * (float)(sc >> 4);
            for (unsigned int l = 0; l < 16; l++) {
                yb[y_off + l] = dl * (float)((q[q_off + l] >> shift) & 3) - ml;
            }

            sc = scales[is];
            is++;
            dl = d * (float)(sc & 0xF);
            ml = dmin * (float)(sc >> 4);
            for (unsigned int l = 0; l < 16; l++) {
                yb[y_off + 16 + l] = dl * (float)((q[q_off + l + 16] >> shift) & 3) - ml;
            }
            y_off += 32;
        }
        q_off += 32;
        n += 128;
    }
}

// Port of `dequantize_row_q3_K`. Block layout (110 bytes): `hmask[32]`,
// `qs[64]`, `scales[12]`, `d: f16`.
extern "C" __global__ void dequantize_q3k_kernel(
    const unsigned char* __restrict__ blocks,
    float* __restrict__ y,
    unsigned int num_blocks
) {
    unsigned int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= num_blocks) {
        return;
    }
    const unsigned char* block = blocks + (unsigned long long)bi * 110;
    float* yb = y + (unsigned long long)bi * 256;

    const unsigned char* hmask = block;
    const unsigned char* q = block + 32;
    const unsigned char* raw_scales = block + 96;
    float d_all = le_f16(block + 108);

    const unsigned int KMASK1 = 0x03030303u;
    const unsigned int KMASK2 = 0x0f0f0f0fu;
    unsigned int aux[4];
    aux[0] = (unsigned int)raw_scales[0] | ((unsigned int)raw_scales[1] << 8) | ((unsigned int)raw_scales[2] << 16) | ((unsigned int)raw_scales[3] << 24);
    aux[1] = (unsigned int)raw_scales[4] | ((unsigned int)raw_scales[5] << 8) | ((unsigned int)raw_scales[6] << 16) | ((unsigned int)raw_scales[7] << 24);
    aux[2] = (unsigned int)raw_scales[8] | ((unsigned int)raw_scales[9] << 8) | ((unsigned int)raw_scales[10] << 16) | ((unsigned int)raw_scales[11] << 24);

    unsigned int tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);

    signed char scales[16];
    for (unsigned int i = 0; i < 4; i++) {
        unsigned int v = aux[i];
        scales[4 * i + 0] = (signed char)(v & 0xFF);
        scales[4 * i + 1] = (signed char)((v >> 8) & 0xFF);
        scales[4 * i + 2] = (signed char)((v >> 16) & 0xFF);
        scales[4 * i + 3] = (signed char)((v >> 24) & 0xFF);
    }

    unsigned char m = 1;
    unsigned int is = 0;
    unsigned int y_off = 0;
    unsigned int q_off = 0;
    unsigned int n = 0;
    while (n < 256) {
        unsigned int shift = 0;
        for (unsigned int rep = 0; rep < 4; rep++) {
            float dl = d_all * (float)((int)scales[is] - 32);
            is++;
            for (unsigned int l = 0; l < 16; l++) {
                int bit = (hmask[l] & m) ? 0 : 4;
                yb[y_off + l] = dl * (float)((int)((q[q_off + l] >> shift) & 3) - bit);
            }
            y_off += 16;

            dl = d_all * (float)((int)scales[is] - 32);
            is++;
            for (unsigned int l = 0; l < 16; l++) {
                int bit = (hmask[l + 16] & m) ? 0 : 4;
                yb[y_off + l] = dl * (float)((int)((q[q_off + l + 16] >> shift) & 3) - bit);
            }
            y_off += 16;

            shift += 2;
            m <<= 1;
        }
        q_off += 32;
        n += 128;
    }
}

// Port of `dequantize_row_q8_K`. Block layout (292 bytes): `d: f32`,
// `qs[256]` (i8), `bsums[16]` (i16, unused for dequant -- see `dequant.rs`'s
// doc comment on `dequantize_block_q8_k`).
extern "C" __global__ void dequantize_q8k_kernel(
    const unsigned char* __restrict__ blocks,
    float* __restrict__ y,
    unsigned int num_blocks
) {
    unsigned int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= num_blocks) {
        return;
    }
    const unsigned char* block = blocks + (unsigned long long)bi * 292;
    float* yb = y + (unsigned long long)bi * 256;

    float d = le_f32(block);
    const signed char* qs = (const signed char*)(block + 4);
    for (unsigned int j = 0; j < 256; j++) {
        yb[j] = (float)qs[j] * d;
    }
}
