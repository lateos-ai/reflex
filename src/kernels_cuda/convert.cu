// Element-type conversions between f32 and f16 device buffers.
//
// All f32 -> f16 conversions round to nearest even (`__float2half_rn`), the
// same rounding the f16-output dequant kernels in dequant.cu use, so a weight
// that is dequantized to f32 first and converted afterwards (a LoRA merge
// target) ends up with exactly the bits a direct f16 dequant would give it.

#include <cuda_fp16.h>

// `REFLEX_F16_ROUNDTRIP=1` numerics probe: rounds an f32 weight buffer to f16
// and back in place, so an `--weights f32` run sees exactly the weight values
// f16 storage would, while every kernel downstream stays the f32 one. Isolates
// "f16 weight rounding" from "f16 kernels" when diagnosing a token mismatch.
extern "C" __global__ void f16_roundtrip_kernel(float* __restrict__ w, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        w[i] = __half2float(__float2half_rn(w[i]));
    }
}

// `out[i] = (f16)in[i]`: narrows a LoRA merge target, merged in f32, to the
// f16 it is stored as.
extern "C" __global__ void f32_to_f16_kernel(
    const float* __restrict__ in,
    __half* __restrict__ out,
    unsigned long long n
) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2half_rn(in[i]);
    }
}

// `out[i] = (f32)in[i]`, exact. Widens an f16 weight back to f32 so a LoRA
// adapter applied after load is still added in f32 (`Model::apply_lora` on a
// weight that was not loaded in f32 for it).
extern "C" __global__ void f16_to_f32_kernel(
    const __half* __restrict__ in,
    float* __restrict__ out,
    unsigned long long n
) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __half2float(in[i]);
    }
}

// Up to F16_RUNS_MAX runs of f16 values, laid out back to back in one landing
// buffer, each widened (exactly) into its own f32 buffer: `runs.dst[r][i] =
// (f32)landing[runs.src[r] + i]` for i < runs.n[r]. An f16 cache file's chunk
// (`WeightLoadPipeline::upload_state`) holds one run per cached buffer, so one
// launch covers what used to be one launch per buffer. Block row `blockIdx.y`
// is the run; each row strides over its run's values. Must match `F16Runs` in
// src/model/loading.rs.
#define F16_RUNS_MAX 64
struct F16Runs {
    unsigned long long dst[F16_RUNS_MAX];
    unsigned int src[F16_RUNS_MAX];
    unsigned int n[F16_RUNS_MAX];
};

extern "C" __global__ void f16_to_f32_runs_kernel(
    const __half* __restrict__ landing,
    F16Runs runs,
    unsigned int count
) {
    const unsigned int r = blockIdx.y;
    if (r >= count) {
        return;
    }
    float* __restrict__ out = reinterpret_cast<float*>(runs.dst[r]);
    const __half* __restrict__ in = landing + runs.src[r];
    const unsigned int n = runs.n[r];
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        out[i] = __half2float(in[i]);
    }
}

// Largest finite f16.
#define F16_MAX 65504.0f

// Prefill activations into an f16 GEMM operand (`--weights f16`,
// `Model::gemm`/`gemm_view`): `out[i] = (f16)in[i]`, round to nearest even,
// except that |in[i]| > 65504 saturates to +-65504 instead of becoming inf
// (NaN stays NaN). Every call also folds into `stats`, a 2-element device
// buffer owned by the model: `stats[0]` is the running max |in[i]| (its f32
// bits; `atomicMax` on the bits orders non-negative floats correctly) and
// `stats[1]` counts saturated elements. `Model::f16_activation_stats` reads
// them back, so an overflow is reported instead of silently clipped. One
// shared-memory reduction per block keeps it to one atomic per 256 elements.
extern "C" __global__ void cast_act_f16_kernel(
    const float* __restrict__ in,
    __half* __restrict__ out,
    unsigned long long n,
    unsigned int* __restrict__ stats
) {
    __shared__ unsigned int block_max[32];
    __shared__ unsigned int block_sat[32];
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;

    float a = 0.0f;
    if (i < n) {
        const float v = in[i];
        a = fabsf(v);
        out[i] = __float2half_rn(a > F16_MAX ? copysignf(F16_MAX, v) : v);
    }

    unsigned int bits = __float_as_uint(a);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        bits = max(bits, __shfl_down_sync(0xffffffffu, bits, offset));
    }
    const unsigned int sat = __popc(__ballot_sync(0xffffffffu, a > F16_MAX));
    if (lane == 0u) {
        block_max[warp] = bits;
        block_sat[warp] = sat;
    }
    __syncthreads();
    if (threadIdx.x == 0u) {
        unsigned int m = 0u;
        unsigned int s = 0u;
        for (unsigned int w = 0; w < (blockDim.x + 31u) / 32u; w++) {
            m = max(m, block_max[w]);
            s += block_sat[w];
        }
        if (m != 0u) {
            atomicMax(&stats[0], m);
        }
        if (s != 0u) {
            atomicAdd(&stats[1], s);
        }
    }
}
