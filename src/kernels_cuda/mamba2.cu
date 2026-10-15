// Mamba-2 SSM layer kernels (Granite 4.0-H, `granitehybrid`). The math, the
// notation (D, N, G, H, P, C) and the llama.cpp sources are in
// reference/mamba2_reference.rs, the host oracle each kernel is tested against
// (src/model/granite_hybrid_kernel_tests.rs); design: docs/design/mamba2-hybrid.md.
//
// Prefill is sequential over the prompt on SSM layers: each kernel takes all
// `rows` positions in one launch and walks them in order, carrying the state
// from one position to the next. The in/out projections around them are
// ordinary batched GEMMs. A decode step is the same launch with rows == 1.
//
// `zxbcdt` is ssm_in's output, row-major [rows, in_stride] with
// in_stride = D + C + H: z at column 0, xBC at column D, dt at column D + C.

// Shape arguments shared by the three SSM kernels, passed by value. Must match
// `Mamba2Dims` in src/model/granite_hybrid.rs field for field.
struct Mamba2Dims {
    unsigned int rows;
    unsigned int in_stride;
    unsigned int c_dim;
    unsigned int d_inner;
    unsigned int n_head;
    unsigned int head_dim;
    unsigned int n_group;
    unsigned int d_state;
    unsigned int d_conv;
    float eps;
};

__device__ __forceinline__ float m2_silu(float x) {
    return x / (1.0f + expf(-x));
}

// Causal depthwise conv over xBC, plus bias and SiLU, for `rows` positions in
// order. One thread per channel c < C. `w` is ssm_conv1d.weight
// (channel c's d_conv taps at c * d_conv, oldest first), `state` the
// [(d_conv - 1), C] time-major window of previous raw inputs, advanced in
// place. Output `out` is [rows, C]. Taps are summed oldest first, then the
// bias is added, as llama.cpp's ssm_conv_f32 does.
extern "C" __global__ void mamba2_conv_kernel(
    const float* __restrict__ zxbcdt,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ state,
    float* __restrict__ out,
    Mamba2Dims p
) {
    const unsigned int rows = p.rows, in_stride = p.in_stride, c_dim = p.c_dim;
    const unsigned int d_inner = p.d_inner, d_conv = p.d_conv;
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= c_dim) return;
    const float* wc = w + (unsigned long long)c * d_conv;
    const float bias = b[c];
    for (unsigned int t = 0; t < rows; t++) {
        const float cur = zxbcdt[(unsigned long long)t * in_stride + d_inner + c];
        float acc = 0.0f;
        for (unsigned int k = 0; k + 1 < d_conv; k++) {
            acc += state[k * c_dim + c] * wc[k];
        }
        acc += cur * wc[d_conv - 1];
        acc += bias;
        out[(unsigned long long)t * c_dim + c] = m2_silu(acc);
        for (unsigned int k = 0; k + 2 < d_conv; k++) {
            state[k * c_dim + c] = state[(k + 1) * c_dim + c];
        }
        if (d_conv > 1) {
            state[(d_conv - 2) * c_dim + c] = cur;
        }
    }
}

#define M2_WARP 32
#define M2_WARPS_PER_BLOCK 4
#define M2_MAX_STATE_PER_LANE 8  // d_state <= 256

// The selective-scan state update and readout, for `rows` positions in order.
// One warp per (head h, element p) of the [H, P] x tensor, as llama.cpp's
// ssm_scan_f32_group does: lane l holds state[h, p, 32 * j + l] for
// j < d_state / 32, sums its terms in j order, then the warp reduces with an
// xor butterfly. `conv` is mamba2_conv_kernel's [rows, C] output (x, B, C);
// dt comes from zxbcdt at column D + C. `state` is [H, P, N]
// (n innermost) and advanced in place. Writes y (without the D skip) to
// [rows, D]. Requires d_state % 32 == 0, d_state <= 256 (checked by the
// launcher).
extern "C" __global__ void mamba2_scan_kernel(
    const float* __restrict__ conv,
    const float* __restrict__ zxbcdt,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a,
    float* __restrict__ state,
    float* __restrict__ y,
    Mamba2Dims p
) {
    const unsigned int rows = p.rows, in_stride = p.in_stride, c_dim = p.c_dim;
    const unsigned int n_head = p.n_head, head_dim = p.head_dim;
    const unsigned int n_group = p.n_group, d_state = p.d_state;
    const unsigned int dt_offset = p.d_inner + p.c_dim;
    const unsigned int lane = threadIdx.x % M2_WARP;
    const unsigned int warp_idx = blockIdx.x * M2_WARPS_PER_BLOCK + threadIdx.x / M2_WARP;
    const unsigned int d_inner = n_head * head_dim;
    if (warp_idx >= d_inner) return;  // whole warps exit together
    const unsigned int h = warp_idx / head_dim;
    const unsigned int g = h / (n_head / n_group);
    const unsigned int per_lane = d_state / M2_WARP;
    const unsigned int b_off = d_inner + g * d_state;
    const unsigned int c_off = d_inner + n_group * d_state + g * d_state;
    float* s = state + (unsigned long long)warp_idx * d_state;
    const float a_h = a[h];
    const float bias = dt_bias[h];

    // Fully unrolled with a guard (not `j < per_lane`) so st stays in
    // registers instead of local memory.
    float st[M2_MAX_STATE_PER_LANE];
#pragma unroll
    for (unsigned int j = 0; j < M2_MAX_STATE_PER_LANE; j++) {
        st[j] = j < per_lane ? s[j * M2_WARP + lane] : 0.0f;
    }
    for (unsigned int t = 0; t < rows; t++) {
        const float* conv_t = conv + (unsigned long long)t * c_dim;
        float dt = zxbcdt[(unsigned long long)t * in_stride + dt_offset + h] + bias;
        dt = dt <= 20.0f ? log1pf(expf(dt)) : dt;
        const float da = expf(dt * a_h);
        const float x_dt = conv_t[warp_idx] * dt;
        float sum = 0.0f;
#pragma unroll
        for (unsigned int j = 0; j < M2_MAX_STATE_PER_LANE; j++) {
            if (j >= per_lane) break;
            const unsigned int n = j * M2_WARP + lane;
            st[j] = (st[j] * da) + (conv_t[b_off + n] * x_dt);
            sum += st[j] * conv_t[c_off + n];
        }
        for (int offset = M2_WARP / 2; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xffffffffu, sum, offset, M2_WARP);
        }
        if (lane == 0) {
            y[(unsigned long long)t * d_inner + warp_idx] = sum;
        }
    }
#pragma unroll
    for (unsigned int j = 0; j < M2_MAX_STATE_PER_LANE; j++) {
        if (j < per_lane) s[j * M2_WARP + lane] = st[j];
    }
}

// D skip, z gate, grouped RMSNorm: one block per (row, group). Each element i
// of the group becomes v = (y + x * D[h]) * silu(z); the block's thread 0 then
// sums v^2 over the group in index order (as rmsnorm_kernel does) and every
// element is scaled by 1/sqrt(mean + eps) and norm_w. `x` is read from the
// conv output (columns 0..D of each [C] row), z from zxbcdt (columns 0..D).
// __fmul_rn keeps x * D a separate rounding, as llama.cpp's separate mul and
// add ops do. Output `out` is [rows, D].
extern "C" __global__ void mamba2_gate_norm_kernel(
    const float* __restrict__ y,
    const float* __restrict__ conv,
    const float* __restrict__ zxbcdt,
    const float* __restrict__ d,
    const float* __restrict__ norm_w,
    float* __restrict__ out,
    Mamba2Dims p
) {
    const unsigned int rows = p.rows, in_stride = p.in_stride, c_dim = p.c_dim;
    const unsigned int d_inner = p.d_inner, n_group = p.n_group, head_dim = p.head_dim;
    const float eps = p.eps;
    __shared__ float rms_inv_s;
    const unsigned int group = d_inner / n_group;
    for (unsigned int blk = blockIdx.x; blk < rows * n_group; blk += gridDim.x) {
        const unsigned int t = blk / n_group;
        const unsigned int g = blk % n_group;
        const unsigned int first = g * group;
        const float* y_t = y + (unsigned long long)t * d_inner;
        const float* x_t = conv + (unsigned long long)t * c_dim;
        const float* z_t = zxbcdt + (unsigned long long)t * in_stride;
        float* o_t = out + (unsigned long long)t * d_inner;
        for (unsigned int i = first + threadIdx.x; i < first + group; i += blockDim.x) {
            const float v = y_t[i] + __fmul_rn(x_t[i], d[i / head_dim]);
            o_t[i] = __fmul_rn(v, m2_silu(z_t[i]));
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float sum_sq = 0.0f;
            for (unsigned int i = first; i < first + group; i++) {
                sum_sq += o_t[i] * o_t[i];
            }
            rms_inv_s = 1.0f / sqrtf(sum_sq / group + eps);
        }
        __syncthreads();
        const float rms_inv = rms_inv_s;
        for (unsigned int i = first + threadIdx.x; i < first + group; i += blockDim.x) {
            o_t[i] = o_t[i] * rms_inv * norm_w[i];
        }
        __syncthreads();
    }
}

// In place: h[i] = h[i] + m[i] * s, with the product rounded on its own
// (llama.cpp's ggml_scale then ggml_add, no fused multiply-add). Granite's
// residual_scale applied to a sublayer's output `m` before the residual add.
extern "C" __global__ void scaled_add_kernel(
    float* __restrict__ h,
    const float* __restrict__ m,
    float s,
    unsigned long long n
) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        h[i] = h[i] + __fmul_rn(m[i], s);
    }
}
