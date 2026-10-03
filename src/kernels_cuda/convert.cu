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
