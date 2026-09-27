#!/usr/bin/env python3
"""Builds test-data/tiny-qwen3moe-lora.gguf: a hand-built, fully synthetic LoRA
adapter targeting test-data/tiny-qwen3moe.gguf's three per-expert-stacked MoE
FFN tensors (ffn_gate_exps/ffn_up_exps/ffn_down_exps, both of that fixture's
two layers), used to verify src/lora.rs's per-expert delta math and
src/model.rs's shape validation without needing a real MoE LoRA adapter (see
HISTORY.md for why no real one was usable: davidanugraha's MoE adapters use a
Megatron/verl fused-expert LoRA representation with no per-expert index in
the tensor name at all, which llama.cpp's own convert_lora_to_gguf.py can't
convert either -- its expert-stacking mechanism requires per-expert-indexed
HF tensor names like "experts.{i}.gate_proj.lora_A.weight" to trigger the
torch.stack it relies on).

Tensor shapes/values are NOT run through the real convert_lora_to_gguf.py
(this fixture has no real base HF checkpoint or safetensors adapter behind
it) -- instead this writes the exact GGUF ne-order shapes and byte layout
that script would have produced for a standard per-expert-Linear PEFT
adapter, derived by tracing its real source (llama.cpp's
Qwen2MoeModel.modify_tensors, which every MoE arch's LoraModel subclass
inherits unchanged): lora_a ends up shape [in_features, rank, expert_count],
lora_b ends up shape [rank, out_features, expert_count], both in the same
expert-major contiguous-per-expert-chunk layout as the base model's own
per-expert-stacked tensors (see model.rs's expert_weight_view doc comment).

Values are a deterministic formula (not random), so the matching Rust test
(lora.rs's moe_expert_lora_fixture_tests) can recompute the expected delta
independently rather than just re-deriving the same random numbers:
    A[e, r, i] = 100000*layer + 10000*kind_id + 1000*e + 10*r + i
    B[e, r, o] = 100000*layer + 10000*kind_id + 1000*e + 100*r + o
where kind_id is 0/1/2 for gate/up/down -- distinct per layer and per tensor
kind so a mixed-up layer or tensor-kind index would produce a visibly wrong
(not just numerically-close) delta.

Run: python scripts/build_tiny_moe_lora_fixture.py
(needs only the `gguf` package, already a dependency of this project's other
fixture-building scripts; matches test-data/tiny-qwen3moe.gguf's real
in_features=out_features=32, expert_count=8, block_count=2.)
"""
import numpy as np
import gguf

OUT_PATH = "test-data/tiny-qwen3moe-lora.gguf"

IN_FEATURES = 32
OUT_FEATURES = 32
EXPERT_COUNT = 8
BLOCK_COUNT = 2
RANK = 2
ALPHA = 8.0

KINDS = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]


def main():
    writer = gguf.GGUFWriter(OUT_PATH, "llama")
    writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
    writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, ALPHA)

    for layer in range(BLOCK_COUNT):
        for kind_id, kind in enumerate(KINDS):
            # numpy shape (expert_count, rank, in_features) -> GGUF ne
            # [in_features, rank, expert_count] (ne0 fastest == last numpy axis).
            a = np.empty((EXPERT_COUNT, RANK, IN_FEATURES), dtype=np.float32)
            for e in range(EXPERT_COUNT):
                for r in range(RANK):
                    for i in range(IN_FEATURES):
                        a[e, r, i] = 100000 * layer + 10000 * kind_id + 1000 * e + 10 * r + i

            # numpy shape (expert_count, out_features, rank) -> GGUF ne
            # [rank, out_features, expert_count].
            b = np.empty((EXPERT_COUNT, OUT_FEATURES, RANK), dtype=np.float32)
            for e in range(EXPERT_COUNT):
                for o in range(OUT_FEATURES):
                    for r in range(RANK):
                        b[e, o, r] = 100000 * layer + 10000 * kind_id + 1000 * e + 100 * r + o

            base_name = f"blk.{layer}.{kind}.weight"
            writer.add_tensor(f"{base_name}.lora_a", a)
            writer.add_tensor(f"{base_name}.lora_b", b)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
