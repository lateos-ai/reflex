use super::*;
use crate::dequant_iq;
use crate::gguf::GgufFile;
use cudarc::driver::CudaDevice;

/// Byte-exact host-vs-device cross-check for the IQ-family on-device
/// dequant kernels (`kernels_cuda/dequant.cu`'s `dequantize_iq*_kernel`
/// functions) against the already-verified host path (`dequant_iq.rs`,
/// unit-tested against hand-computed values since Phase 21.13) -- run
/// against every real IQ-family tensor's actual bytes (not just a
/// synthetic all-zero block), so it also exercises codebook/sign-bit
/// lookup paths the hand-computed unit tests don't reach. `#[ignore]`d
/// like every other real-GGUF test in this file -- run with:
/// `REFLEX_TEST_GGUF=<path to a real GGUF containing IQ-family tensors>
/// cargo test --release -- --ignored iq_dequant_kernel_matches_host_on_real_tensors`
#[test]
#[ignore]
fn iq_dequant_kernel_matches_host_on_real_tensors() {
    let gguf_path = std::env::var("REFLEX_TEST_GGUF").expect(
        "set REFLEX_TEST_GGUF to a real local GGUF path containing IQ-family tensors to run this test",
    );
    let file = GgufFile::open(&gguf_path).expect("failed to open REFLEX_TEST_GGUF");
    let device = CudaDevice::new(0).expect("failed to init CUDA device 0");
    let kernels = load_dequant_kernels(&device).expect("failed to load dequant kernels");
    let mut pipeline =
        WeightLoadPipeline::new(&device).expect("failed to create weight-load pipeline");

    type HostDequantFn = fn(&[u8], &mut [f32]);

    let mut tested_types: Vec<GgmlType> = Vec::new();
    for info in &file.tensors {
        let ggml_type = info.ggml_type;
        let (host_fn, block_bytes): (HostDequantFn, usize) = match ggml_type {
            GgmlType::IQ2XXS => (dequant_iq::dequantize_block_iq2_xxs, IQ2XXS_BLOCK_BYTES),
            GgmlType::IQ2XS => (dequant_iq::dequantize_block_iq2_xs, IQ2XS_BLOCK_BYTES),
            GgmlType::IQ2S => (dequant_iq::dequantize_block_iq2_s, IQ2S_BLOCK_BYTES),
            GgmlType::IQ3XXS => (dequant_iq::dequantize_block_iq3_xxs, IQ3XXS_BLOCK_BYTES),
            GgmlType::IQ3S => (dequant_iq::dequantize_block_iq3_s, IQ3S_BLOCK_BYTES),
            GgmlType::IQ1S => (dequant_iq::dequantize_block_iq1_s, IQ1S_BLOCK_BYTES),
            GgmlType::IQ1M => (dequant_iq::dequantize_block_iq1_m, IQ1M_BLOCK_BYTES),
            GgmlType::IQ4XS => (dequant_iq::dequantize_block_iq4_xs, IQ4XS_BLOCK_BYTES),
            _ => continue,
        };
        if tested_types.contains(&ggml_type) {
            continue; // one real tensor per type is enough
        }
        tested_types.push(ggml_type);

        let bytes = file.tensor_bytes(info).expect("tensor_bytes failed");
        let element_count = info.element_count();
        let num_blocks = bytes.len() / block_bytes;

        let mut host_out = vec![0f32; num_blocks * QK_K];
        for b in 0..num_blocks {
            let block = &bytes[b * block_bytes..(b + 1) * block_bytes];
            host_fn(block, &mut host_out[b * QK_K..(b + 1) * QK_K]);
        }
        host_out.truncate(element_count as usize);

        let device_out =
            dequantize_tensor_to_device(&mut pipeline, &kernels, ggml_type, bytes, element_count)
                .expect("dequantize_tensor_to_device failed");
        let device_host = device.dtoh_sync_copy(&device_out).expect("dtoh failed");

        assert_eq!(
            host_out.len(),
            device_host.len(),
            "{} ({ggml_type:?}): length mismatch",
            info.name
        );
        for (i, (h, d)) in host_out.iter().zip(device_host.iter()).enumerate() {
            assert_eq!(
                h, d,
                "{} ({ggml_type:?}): element {i} host={h} device={d}",
                info.name
            );
        }
        eprintln!(
            "OK {} ({ggml_type:?}): {} elements byte-exact",
            info.name,
            host_out.len()
        );
    }
    assert!(
        !tested_types.is_empty(),
        "REFLEX_TEST_GGUF contained no IQ-family tensor to test"
    );
}
