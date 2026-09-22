/* Phase 4 (Embeddability) round 2 verification harness: a real C program
 * that links the compiled coldstart-infer cdylib/staticlib and exercises
 * the full load -> generate -> free surface declared in
 * include/coldstart_infer.h. Not part of the Cargo build -- compile and run
 * by hand (or via a small shell wrapper) on a real GPU instance:
 *
 *   gcc -I include -o ffi-test/smoke_test ffi-test/smoke_test.c \
 *       -L target/release -lcoldstart_infer -ldl -lpthread -lm
 *   LD_LIBRARY_PATH=target/release ./ffi-test/smoke_test <gguf-path> [prompt] [max_new_tokens]
 *
 * The output line is deliberately shaped like qwen3_coldstart's own
 * COLDSTART_QWEN3_OK line (token_ids=[...] token_text="...") so it can be
 * diffed by eye against a real `qwen3_coldstart <gguf> <prompt> --max-tokens
 * <n>` run on the same GGUF+prompt -- that comparison is this round's actual
 * correctness bar (see README.md's Phase 4 round 2 section), not just "it
 * compiles and links".
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "coldstart_infer.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <gguf-path> [prompt] [max_new_tokens]\n", argv[0]);
        return 2;
    }
    const char *gguf_path = argv[1];
    const char *prompt = argc > 2 ? argv[2] : "Once upon a time";
    uintptr_t max_new_tokens = argc > 3 ? (uintptr_t)strtoull(argv[3], NULL, 10) : 1;

    ColdstartModel *model = coldstart_load(gguf_path, NULL);
    if (!model) {
        fprintf(stderr, "coldstart_load failed: %s\n", coldstart_last_error());
        return 1;
    }
    fprintf(stderr, "coldstart_load OK\n");

    ColdstartGenerateResult result;
    int rc = coldstart_generate(model, prompt, max_new_tokens, &result);
    if (rc != 0) {
        fprintf(stderr, "coldstart_generate failed: %s\n", coldstart_last_error());
        coldstart_free(model);
        return 1;
    }

    printf("COLDSTART_FFI_OK num_generated=%zu token_id=%u token_ids=[", result.num_tokens, result.token_ids[0]);
    for (size_t i = 0; i < result.num_tokens; i++) {
        printf(i == 0 ? "%u" : ",%u", result.token_ids[i]);
    }
    printf("] token_text=%s\n", result.text);

    coldstart_free_generate_result(&result);
    coldstart_free(model);

    /* Double-free/use-after-free smoke check on the freed struct/handle:
     * coldstart_free_generate_result on an already-zeroed struct and
     * coldstart_free(NULL) must both be safe no-ops per the header's
     * documented contract. */
    coldstart_free_generate_result(&result);
    coldstart_free(NULL);

    return 0;
}
