#!/usr/bin/env python3
"""One-shot cold-start driver for vLLM.

Loads a model, greedily generates exactly one token for a fixed prompt,
prints it, and exits. Does no internal timing of its own — this script is
meant to be wrapped externally by `/usr/bin/time -v` (via
scripts/bench_cold_vllm.sh), matching the external-wall-clock methodology
already used for the Reflex-vs-llama.cpp comparison (see
DECISIONS.md's benchmark-methodology entry) so process launch, Python
startup, and vLLM's own engine init (including CUDA graph capture) are all
included on vLLM's side of the comparison, the same way OS exec/dynamic-
linking overhead is included on every other engine measured this way.

Usage: vllm_single_shot.py <model-path-or-hf-id> <prompt> [--tokenizer <hf-repo-id>]

A local .gguf path needs `--tokenizer <hf-repo-id>` alongside it -- vLLM's
GGUF loader reads only the weights from the file and still wants an
HF-format tokenizer/config from a separate source (this is vLLM's own
convention, not a Reflex requirement; Reflex reads the
GGUF's own embedded tokenizer directly, see src/tokenizer.rs).
"""
import argparse

from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="GGUF path or HF model id/path for vLLM to load")
    parser.add_argument("prompt")
    parser.add_argument("--tokenizer", default=None, help="HF repo id to source the tokenizer/config from (required for a local .gguf path)")
    args = parser.parse_args()

    kwargs = {"tokenizer": args.tokenizer} if args.tokenizer else {}
    llm = LLM(model=args.model_path, tokenizer_mode="auto", **kwargs)
    params = SamplingParams(temperature=0, max_tokens=1)
    outputs = llm.generate([args.prompt], params)

    for output in outputs:
        print(f"COLD_VLLM_OK text={output.outputs[0].text!r}")


if __name__ == "__main__":
    main()
