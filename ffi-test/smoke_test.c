/* C-FFI verification harness: a real C program that links the compiled
 * reflex-engine cdylib/staticlib and exercises the full
 * load -> generate -> free surface declared in include/reflex_engine.h. Not
 * part of the Cargo build -- compile and run by hand (or via a small shell
 * wrapper) on a real GPU instance:
 *
 *   gcc -I include -o ffi-test/smoke_test ffi-test/smoke_test.c \
 *       -L target/release -lreflex_engine -ldl -lpthread -lm
 *   LD_LIBRARY_PATH=target/release ./ffi-test/smoke_test <gguf-path> [prompt] [max_new_tokens]
 *
 * The output line is deliberately shaped like `reflex generate`'s own
 * REFLEX_GENERATE_OK line (token_ids=[...] token_text="...") so it can be
 * diffed by eye against a real `reflex generate <gguf> <prompt> --max-tokens
 * <n>` run on the same GGUF+prompt -- that comparison is this test's actual
 * correctness bar (see README.md's Embeddability section), not just "it
 * compiles and links".
 *
 * It also checks reflex_last_error_code(): the category each failure reports,
 * and that a successful call resets it to REFLEX_ERROR_CODE_OK. Those checks
 * print REFLEX_FFI_ERROR_CODES_OK and exit 1 on the first mismatch.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "reflex_engine.h"

/* Fails the test unless the last reflex_* call reported `expected`. */
static int expect_code(ReflexErrorCode expected, const char *what) {
    ReflexErrorCode got = reflex_last_error_code();
    if (got != expected) {
        const char *msg = reflex_last_error();
        fprintf(stderr, "error-code check failed for %s: expected %d, got %d (%s)\n", what,
                (int)expected, (int)got, msg ? msg : "no message");
        return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <gguf-path> [prompt] [max_new_tokens]\n", argv[0]);
        return 2;
    }
    const char *gguf_path = argv[1];
    const char *prompt = argc > 2 ? argv[2] : "Once upon a time";
    uintptr_t max_new_tokens = argc > 3 ? (uintptr_t)strtoull(argv[3], NULL, 10) : 1;

    /* Failures that need no GPU work. */
    if (reflex_load(NULL, NULL) != NULL ||
        !expect_code(REFLEX_ERROR_CODE_INVALID_INPUT, "reflex_load(NULL)")) {
        return 1;
    }
    if (reflex_load("/nonexistent/reflex-ffi-test.gguf", NULL) != NULL ||
        !expect_code(REFLEX_ERROR_CODE_IO, "reflex_load(missing file)")) {
        return 1;
    }

    ReflexModel *model = reflex_load(gguf_path, NULL);
    if (!model) {
        fprintf(stderr, "reflex_load failed: %s\n", reflex_last_error());
        return 1;
    }
    fprintf(stderr, "reflex_load OK\n");
    /* A successful call clears the previous failure. */
    if (!expect_code(REFLEX_ERROR_CODE_OK, "successful reflex_load") || reflex_last_error() != NULL) {
        fprintf(stderr, "reflex_last_error() should be NULL after a successful call\n");
        reflex_free(model);
        return 1;
    }

    ReflexGenerateResult scratch;
    if (reflex_generate(model, prompt, 0, &scratch) != -1 ||
        !expect_code(REFLEX_ERROR_CODE_INVALID_INPUT, "max_new_tokens = 0")) {
        reflex_free(model);
        return 1;
    }

    /* A prompt of 2x the engine's position limit in words: over the limit for any
     * tokenizer that gives each " hello" at least one token. */
    size_t words = 2 * (size_t)REFLEX_ATTN_MAX_POSITIONS;
    char *long_prompt = malloc(words * 6 + 1);
    if (!long_prompt) {
        reflex_free(model);
        return 1;
    }
    for (size_t i = 0; i < words; i++) {
        memcpy(long_prompt + i * 6, " hello", 6);
    }
    long_prompt[words * 6] = '\0';
    int overflow_rc = reflex_generate(model, long_prompt, 1, &scratch);
    free(long_prompt);
    if (overflow_rc != -1 ||
        !expect_code(REFLEX_ERROR_CODE_CONTEXT_OVERFLOW, "over-limit prompt")) {
        reflex_free(model);
        return 1;
    }
    printf("REFLEX_FFI_ERROR_CODES_OK\n");

    ReflexGenerateResult result;
    int rc = reflex_generate(model, prompt, max_new_tokens, &result);
    if (rc != 0) {
        fprintf(stderr, "reflex_generate failed: %s\n", reflex_last_error());
        reflex_free(model);
        return 1;
    }
    if (!expect_code(REFLEX_ERROR_CODE_OK, "successful reflex_generate")) {
        reflex_free_generate_result(&result);
        reflex_free(model);
        return 1;
    }

    printf("REFLEX_FFI_OK num_generated=%zu token_id=%u token_ids=[", result.num_tokens, result.token_ids[0]);
    for (size_t i = 0; i < result.num_tokens; i++) {
        printf(i == 0 ? "%u" : ",%u", result.token_ids[i]);
    }
    printf("] token_text=%s\n", result.text);

    reflex_free_generate_result(&result);
    reflex_free(model);

    /* Double-free/use-after-free smoke check on the freed struct/handle:
     * reflex_free_generate_result on an already-zeroed struct and
     * reflex_free(NULL) must both be safe no-ops per the header's
     * documented contract. */
    reflex_free_generate_result(&result);
    reflex_free(NULL);

    return 0;
}
