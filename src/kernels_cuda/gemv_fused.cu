// Fused GEMV projections for the decode path.
//
// In the standard decode loop each layer launches 6 separate gemv calls:
//   q = x @ W_q^T, k = x @ W_k^T, v = x @ W_v^T,
//   gate = x @ W_g^T, up = x @ W_u^T, down = act @ W_d^T
//
// For rows==1 (decode) these are bandwidth-bound -- the ~1-2 block launches
// have host overhead comparable to their GPU work. Fusing q/k/v into one
// launch and gate/up into another cuts launches by half while keeping
// outputs bit-identical (same per-row dot product, same warp-shuffle reduction).
//
// Each fused kernel takes separate weight pointers and output pointers; no
// concatenation at load time. One warp handles one output row, so the number
// of warps = sum of out_features across fused projections.

extern "C" __global__ void gemv_qkv_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w_q,
    const float* __restrict__ w_k,
    const float* __restrict__ w_v,
    float* __restrict__ y_q,
    float* __restrict__ y_k,
    float* __restrict__ y_v,
    unsigned int in_features,
    unsigned int q_out,
    unsigned int k_out,
    unsigned int v_out
) {
    const unsigned int warps_per_block = blockDim.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;

    // Total output rows = q_out + k_out + v_out.
    // Block i handles rows [i*warps_per_block .. (i+1)*warps_per_block).
    // Each warp within the block handles consecutive rows.
    const unsigned int total_out = q_out + k_out + v_out;
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int warp_row = blockIdx.x * warps_per_block + warp_id;
    
    if (warp_row >= total_out) {
        return;
    }

    // Determine which projection this row belongs to and the offset.
    const float* w_ptr;
    float* y_ptr;
    unsigned int row_in_proj;
    
    if (warp_row < q_out) {
        w_ptr = w_q;
        y_ptr = y_q;
        row_in_proj = warp_row;
    } else if (warp_row < q_out + k_out) {
        w_ptr = w_k;
        y_ptr = y_k;
        row_in_proj = warp_row - q_out;
    } else {
        w_ptr = w_v;
        y_ptr = y_v;
        row_in_proj = warp_row - q_out - k_out;
    }

    const float* row_ptr = w_ptr + (unsigned long long)row_in_proj * in_features;
    float sum = 0.0f;

    if ((in_features & 3u) == 0u) {
        const float4* x4 = reinterpret_cast<const float4*>(x);
        const float4* row4 = reinterpret_cast<const float4*>(row_ptr);
        unsigned int vec_count = in_features >> 2;
        for (unsigned int idx = lane; idx < vec_count; idx += 32u) {
            float4 xv = x4[idx];
            float4 wv = row4[idx];
            sum += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
        }
    } else {
        for (unsigned int i = lane; i < in_features; i += 32u) {
            sum += x[i] * row_ptr[i];
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0u) {
        y_ptr[row_in_proj] = sum;
    }
}

// f16-weight variant: same structure, loads weights as __half.
#include "gemv_f16.cuh"

extern "C" __global__ void gemv_qkv_f16_kernel(
    const float* __restrict__ x,
    const __half* __restrict__ w_q,
    const __half* __restrict__ w_k,
    const __half* __restrict__ w_v,
    float* __restrict__ y_q,
    float* __restrict__ y_k,
    float* __restrict__ y_v,
    unsigned int in_features,
    unsigned int q_out,
    unsigned int k_out,
    unsigned int v_out
) {
    const unsigned int warps_per_block = blockDim.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;

    const unsigned int total_out = q_out + k_out + v_out;
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int warp_row = blockIdx.x * warps_per_block + warp_id;
    
    if (warp_row >= total_out) {
        return;
    }

    const __half* w_ptr;
    float* y_ptr;
    unsigned int row_in_proj;
    
    if (warp_row < q_out) {
        w_ptr = w_q;
        y_ptr = y_q;
        row_in_proj = warp_row;
    } else if (warp_row < q_out + k_out) {
        w_ptr = w_k;
        y_ptr = y_k;
        row_in_proj = warp_row - q_out;
    } else {
        w_ptr = w_v;
        y_ptr = y_v;
        row_in_proj = warp_row - q_out - k_out;
    }

    // `warp_dot_f16` already performs the warp-shuffle reduction.
    const float sum = warp_dot_f16(x, w_ptr + (unsigned long long)row_in_proj * in_features, in_features, lane);
    if (lane == 0u) {
        y_ptr[row_in_proj] = sum;
    }
}

// Fused gate + up projections (ffn_gate and ffn_up both read from the same input).
// Same pattern as qkv fusion.
extern "C" __global__ void gemv_gate_up_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w_gate,
    const float* __restrict__ w_up,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    unsigned int in_features,
    unsigned int gate_out,
    unsigned int up_out
) {
    const unsigned int warps_per_block = blockDim.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;

    const unsigned int total_out = gate_out + up_out;
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int warp_row = blockIdx.x * warps_per_block + warp_id;
    
    if (warp_row >= total_out) {
        return;
    }

    const float* w_ptr;
    float* y_ptr;
    unsigned int row_in_proj;
    
    if (warp_row < gate_out) {
        w_ptr = w_gate;
        y_ptr = y_gate;
        row_in_proj = warp_row;
    } else {
        w_ptr = w_up;
        y_ptr = y_up;
        row_in_proj = warp_row - gate_out;
    }

    const float* row_ptr = w_ptr + (unsigned long long)row_in_proj * in_features;
    float sum = 0.0f;

    if ((in_features & 3u) == 0u) {
        const float4* x4 = reinterpret_cast<const float4*>(x);
        const float4* row4 = reinterpret_cast<const float4*>(row_ptr);
        unsigned int vec_count = in_features >> 2;
        for (unsigned int idx = lane; idx < vec_count; idx += 32u) {
            float4 xv = x4[idx];
            float4 wv = row4[idx];
            sum += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
        }
    } else {
        for (unsigned int i = lane; i < in_features; i += 32u) {
            sum += x[i] * row_ptr[i];
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0u) {
        y_ptr[row_in_proj] = sum;
    }
}

// f16-weight variant.
extern "C" __global__ void gemv_gate_up_f16_kernel(
    const float* __restrict__ x,
    const __half* __restrict__ w_gate,
    const __half* __restrict__ w_up,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    unsigned int in_features,
    unsigned int gate_out,
    unsigned int up_out
) {
    const unsigned int warps_per_block = blockDim.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;

    const unsigned int total_out = gate_out + up_out;
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int warp_row = blockIdx.x * warps_per_block + warp_id;
    
    if (warp_row >= total_out) {
        return;
    }

    const __half* w_ptr;
    float* y_ptr;
    unsigned int row_in_proj;
    
    if (warp_row < gate_out) {
        w_ptr = w_gate;
        y_ptr = y_gate;
        row_in_proj = warp_row;
    } else {
        w_ptr = w_up;
        y_ptr = y_up;
        row_in_proj = warp_row - gate_out;
    }

    // `warp_dot_f16` already performs the warp-shuffle reduction.
    const float sum = warp_dot_f16(x, w_ptr + (unsigned long long)row_in_proj * in_features, in_features, lane);
    if (lane == 0u) {
        y_ptr[row_in_proj] = sum;
    }
}
