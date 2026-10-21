// reflex-min-arch: 75
//
// Quantized-resident prefill GEMM on tensor cores: y[rows, out] = x16[rows, in] @ W^T,
// with W the GGUF's raw quantized blocks dequantized to f16 in registers (the same
// values `dequantize_*_coalesced_f16_kernel` writes to the scratch buffer), x16 the
// activations already cast by `cast_act_f16_kernel`, and f32 accumulation. Replaces
// the dequantize-to-scratch + cublasGemmEx path for 2+ rows with `--weights f16`.
// Design: docs/design/quantized-mma-prefill.md.
//
// `mma.sync` and `ldmatrix` need sm_75, hence the directive above (build.rs adds
// `-arch=compute_75` in portable-PTX builds). In cubin/fatbin images for older
// archs the bodies compile to nothing; `Model` checks the compute capability and
// never launches them there.
//
// Step 1 of the design: the module exists and loads lazily, but no GGUF type is
// implemented yet, so `Model::gemm` never dispatches here.

#include <cuda_fp16.h>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750
#define GEMM_Q_MMA_ENABLED 1
#else
#define GEMM_Q_MMA_ENABLED 0
#endif

// Placeholder with the final argument list: raw Q4_K blocks, f16 activations
// `[rows, in_features]`, f32 output `[rows, out_features]`.
extern "C" __global__ void gemm_q4k_mma_kernel(
    const unsigned char* __restrict__ w,
    const __half* __restrict__ x16,
    float* __restrict__ y,
    unsigned int rows,
    unsigned int in_features,
    unsigned int out_features) {
#if GEMM_Q_MMA_ENABLED
#endif
}
