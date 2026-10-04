#!/usr/bin/env bash
# scripts/verify_f16_weights.sh — every GPU verification step for f16 weight storage
# (`--weights f16`, the default, vs. the `--weights f32` reference mode), in order, on
# one CUDA machine, in one session. Writes every raw log plus a markdown summary.
#
#   0. Phase 0 gate: f16 *weight rounding alone* (REFLEX_F16_ROUNDTRIP=1 on the f32
#      kernels) vs. plain f32, 32 greedy tokens, every model x prompt: matched/total,
#      first divergence, top-2 logit gap there. If a real (non-random-weight) model
#      diverges, the script STOPS here (exit 3) unless CONTINUE_PAST_GATE=1.
#   1. f32 mode is a regression baseline: `--weights f32` vs. `master` must give
#      identical tokens, `reflex check` logit checksums and `system1` scores.
#   2. f16 correctness: f16 vs. f32 tokens (and vs. llama.cpp's `llama-simple` text if
#      LLAMA_SIMPLE is set), first divergence + top-2 gaps, system1 max |score diff|,
#      and the max |activation| fed to the f16 prefill GEMM per model (flagged when
#      within 10x of f16's 65504 limit).
#   3. Cold start, master vs. f32 vs. f16: bench_cold_start_phases_system1.sh and
#      bench_cold_start_phases.sh (generate first token), N_RUNS each.
#   4. Memory and decode: `reflex bench` resident VRAM and decode ms/token, plus the
#      nvidia-smi peak during a 64-token generate, f32 vs. f16, DENSE_GGUF and LARGE_GGUF.
#   5. Tests: host tests (REFLEX_SKIP_CUDA=1), the sidecar's, and every #[ignore]d GPU
#      test via gpu_nightly_tests.sh, once with REFLEX_WEIGHTS=f32 and once with f16.
#
# Usage: DENSE_GGUF=... [other vars] scripts/verify_f16_weights.sh   (repo root, GPU box)
# Environment:
#   DENSE_GGUF        required  Qwen3-0.6B GGUF (e.g. Qwen3-0.6B-Q4_K_M.gguf)
#   LLAMA_GGUF        optional  Llama/Mistral GGUF (README: TinyLlama-1.1B)
#   HYBRID_GGUF       optional  Qwen3.5 hybrid GGUF (README: Qwen3.5-0.8B)
#   MLA_REAL_GGUF     optional  real DeepSeek-V2-Lite GGUF (~17 GB). Its f32 runs need an 80 GB
#                               GPU (~63 GB of f32 weights); f16 should need about half
#   LARGE_GGUF        optional  a bigger model, run in steps 0-2 and 4 (e.g. Qwen3-1.7B). Its f32
#                               weights must fit the GPU (Qwen3-4B needs ~16 GB, too big for a T4)
#   test-data/{tiny-qwen3moe,tiny-qwen35moe,deepseek-tiny-mla}.gguf
#                     optional  random-weight fixtures: run, reported, never gate
#   LLAMA_SIMPLE      optional  path to llama.cpp's llama-simple for step 2's text match
#   N_RUNS            default 10
#   OUT_DIR           default bench-results/verify-f16-<timestamp>
#   CONTINUE_PAST_GATE=1        run steps 1-5 even if step 0 finds a real-model divergence
#   SKIP_MASTER=1               skip everything that needs a master build (steps 1, 3's master row)
# Exit status: 3 = stopped at the Phase 0 gate; 1 = some check failed; 0 = all passed.

set -uo pipefail

: "${DENSE_GGUF:?set DENSE_GGUF to a Qwen3-0.6B GGUF path}"
LLAMA_GGUF="${LLAMA_GGUF:-}"
HYBRID_GGUF="${HYBRID_GGUF:-}"
MLA_REAL_GGUF="${MLA_REAL_GGUF:-}"
LARGE_GGUF="${LARGE_GGUF:-}"
LLAMA_SIMPLE="${LLAMA_SIMPLE:-}"
N_RUNS="${N_RUNS:-10}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root" || exit 1
OUT_DIR="${OUT_DIR:-bench-results/verify-f16-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
SUMMARY="$OUT_DIR/summary.md"
FEATURES="ipc,json-output"
BIN="$repo_root/target/release/reflex"
MASTER_DIR="$OUT_DIR/master-worktree"
MASTER_BIN="$MASTER_DIR/target/release/reflex"
failures=0

log() { echo "[verify-f16] $*" >&2; }
section() { printf '\n## %s\n\n' "$1" >>"$SUMMARY"; log "== $1"; }
fail() { failures=$((failures + 1)); echo "**FAIL:** $*" >>"$SUMMARY"; log "FAIL: $*"; }

# ---- models -------------------------------------------------------------------------
# "name|path|real" -- real=1 models gate Phase 0; real=0 are random-weight fixtures.
models=("qwen3-dense|$DENSE_GGUF|1")
[[ -n "$LARGE_GGUF" ]] && models+=("large|$LARGE_GGUF|1")
[[ -n "$LLAMA_GGUF" ]] && models+=("llama|$LLAMA_GGUF|1")
[[ -n "$HYBRID_GGUF" ]] && models+=("qwen35-hybrid|$HYBRID_GGUF|1")
[[ -n "$MLA_REAL_GGUF" ]] && models+=("deepseek-v2-lite|$MLA_REAL_GGUF|1")
[[ -e test-data/tiny-qwen3moe.gguf ]] && models+=("tiny-qwen3moe (random)|test-data/tiny-qwen3moe.gguf|0")
[[ -e test-data/tiny-qwen35moe.gguf ]] && models+=("tiny-qwen35moe (random)|test-data/tiny-qwen35moe.gguf|0")
[[ -e test-data/deepseek-tiny-mla.gguf ]] && models+=("deepseek-tiny-mla (random)|test-data/deepseek-tiny-mla.gguf|0")

prompts=(
  "The capital of France is"
  "Once upon a time"
  "The following is a detailed technical explanation of how a modern CPU executes instructions. It covers instruction fetch, decode, register renaming, out-of-order issue, the reorder buffer, branch prediction, the cache hierarchy from L1 to L3, and how memory ordering is kept consistent across cores. First, consider instruction fetch:"
)
SYS1_PROMPT="Q: Is the sky blue during the day? A:"

# ---- environment ----------------------------------------------------------------------
{
  echo "# f16 weights verification"
  echo
  echo "- date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- branch: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
  echo "- master: $(git rev-parse --short master 2>/dev/null || echo n/a)"
  echo "- GPU: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi failed')"
  echo "- N_RUNS: $N_RUNS"
  echo "- raw logs: $OUT_DIR"
} >"$SUMMARY"
if ! nvidia-smi >/dev/null 2>&1; then
  log "error: nvidia-smi failed -- this script needs an NVIDIA GPU"
  exit 1
fi

# ---- builds ---------------------------------------------------------------------------
log "building branch (release, $FEATURES)..."
if ! cargo build --release --features "$FEATURES" >"$OUT_DIR/build-branch.log" 2>&1; then
  log "error: branch build failed, see $OUT_DIR/build-branch.log"; tail -20 "$OUT_DIR/build-branch.log" >&2; exit 1
fi
if [[ "${SKIP_MASTER:-0}" != 1 ]]; then
  log "building master in a worktree..."
  if [[ ! -d "$MASTER_DIR" ]]; then
    git worktree add --detach "$MASTER_DIR" master >"$OUT_DIR/worktree.log" 2>&1 || {
      log "error: git worktree add failed, see $OUT_DIR/worktree.log"; exit 1; }
  fi
  if ! (cd "$MASTER_DIR" && cargo build --release --features "$FEATURES") >"$OUT_DIR/build-master.log" 2>&1; then
    log "error: master build failed, see $OUT_DIR/build-master.log"; exit 1
  fi
fi

# ---- helpers ----------------------------------------------------------------------------
slug() { echo "$1" | tr -c 'A-Za-z0-9' '_' | cut -c1-40; }

# run_gen <bin> <out-prefix> <gguf> <prompt> [extra args...]   (32 greedy tokens, traced)
run_gen() {
  local bin="$1" out="$2" gguf="$3" prompt="$4"; shift 4
  REFLEX_TOP2_TRACE=1 REFLEX_F16_ACT_STATS=1 "$bin" generate "$gguf" "$prompt" --max-tokens 32 "$@" \
    >"$out.stdout" 2>"$out.stderr"
  echo $? >"$out.status"
}

# Analyzer: compares two traced generate runs (or prints a table row). Python is used for
# parsing only; no numbers are produced that the binaries didn't print.
analyze() {
  python3 - "$@" <<'PY'
import re, sys

def load(prefix):
    try:
        out = open(prefix + ".stdout").read()
        err = open(prefix + ".stderr").read()
        status = int(open(prefix + ".status").read().strip() or 1)
    except OSError:
        return None
    m = re.search(r"REFLEX_GENERATE_OK .*?token_ids=\[([0-9,]*)\]", out)
    ids = [int(x) for x in m.group(1).split(",") if x] if m else None
    top2 = [tuple(map(float, re.findall(r"=(-?[0-9.eE+-]+|inf|-inf|nan)", l)))
            for l in err.splitlines() if l.startswith("REFLEX_TOP2 ")]
    act = re.search(r"REFLEX_F16_ACT_STATS max_abs=(\S+) saturated=(\d+)", err)
    text = re.search(r'token_text=("(?:[^"\\]|\\.)*")', out)
    return dict(ids=ids, top2=top2, status=status, text=text.group(1) if text else None,
                act=(act.group(1), act.group(2)) if act else None)

mode = sys.argv[1]
if mode == "compare":
    # compare <ref-prefix> <test-prefix> <model> <prompt-label>
    ref, test = load(sys.argv[2]), load(sys.argv[3])
    model, plabel = sys.argv[4], sys.argv[5]
    if not ref or ref["ids"] is None or not test or test["ids"] is None:
        print(f"| {model} | {plabel} | RUN FAILED | | | |")
        print("STATUS=error", file=sys.stderr)
        sys.exit(0)
    a, b = ref["ids"], test["ids"]
    total = min(len(a), len(b))
    first = next((i for i in range(total) if a[i] != b[i]), None)
    if first is None and len(a) != len(b):
        first = total
    matched = total if first is None else first
    def gap(run, i):
        if i is None or i >= len(run["top2"]):
            return ""
        t = run["top2"][i]
        return f"{t[1] - t[3]:.4g} (ids {int(t[0])}/{int(t[2])})"
    div = "none" if first is None else str(first)
    print(f"| {model} | {plabel} | {matched}/{max(len(a), len(b))} | {div} | {gap(ref, first)} | {gap(test, first)} |")
    print("STATUS=" + ("match" if first is None else "diverged"), file=sys.stderr)
elif mode == "act":
    r = load(sys.argv[2])
    print(r["act"][0] + " " + r["act"][1] if r and r["act"] else "n/a n/a")
elif mode == "text":
    r = load(sys.argv[2])
    print(r["text"] if r and r["text"] else "")
PY
}

# ---- step 0: Phase 0 gate -------------------------------------------------------------
section "Step 0 -- Phase 0 gate: f16 weight rounding alone (REFLEX_F16_ROUNDTRIP=1, f32 kernels)"
{
  echo "32 greedy tokens per run. Gap = top1 - top2 logit at the first diverging position."
  echo
  echo "| model | prompt | matched/total | first divergence | gap in f32 run | gap in rounded run |"
  echo "|---|---|---|---|---|---|"
} >>"$SUMMARY"
gate_hit=0
mkdir -p "$OUT_DIR/step0"
for entry in "${models[@]}"; do
  IFS='|' read -r name path real <<<"$entry"
  for pi in "${!prompts[@]}"; do
    p="${prompts[$pi]}"; base="$OUT_DIR/step0/$(slug "$name")_p$pi"
    run_gen "$BIN" "$base.f32" "$path" "$p" --weights f32
    REFLEX_F16_ROUNDTRIP=1 run_gen "$BIN" "$base.rt" "$path" "$p" --weights f32
    st=$(analyze compare "$base.f32" "$base.rt" "$name" "p$pi" 2>&1 >>"$SUMMARY" | sed -n 's/^STATUS=//p')
    if [[ "$st" == error ]]; then fail "step 0 run failed: $name p$pi (logs: $base.*)"; fi
    if [[ "$st" == diverged && "$real" == 1 ]]; then gate_hit=1; fi
  done
done
echo >>"$SUMMARY"
echo "Prompts: $(for i in "${!prompts[@]}"; do printf 'p%s=\"%s\" ' "$i" "${prompts[$i]:0:60}"; done)" >>"$SUMMARY"
if [[ $gate_hit == 1 ]]; then
  echo >>"$SUMMARY"
  echo "**GATE: a real model diverged within 32 tokens from f16 weight rounding alone.**" >>"$SUMMARY"
  if [[ "${CONTINUE_PAST_GATE:-0}" != 1 ]]; then
    log "Phase 0 gate hit -- stopping. Summary: $SUMMARY (set CONTINUE_PAST_GATE=1 to run the rest)"
    cat "$SUMMARY"
    exit 3
  fi
  log "Phase 0 gate hit, continuing because CONTINUE_PAST_GATE=1"
fi

# ---- step 1: f32 regression vs master ------------------------------------------------
if [[ "${SKIP_MASTER:-0}" != 1 ]]; then
  section "Step 1 -- --weights f32 vs. master (must be identical)"
  echo "| model | check | result |" >>"$SUMMARY"; echo "|---|---|---|" >>"$SUMMARY"
  mkdir -p "$OUT_DIR/step1"
  for entry in "${models[@]}"; do
    IFS='|' read -r name path real <<<"$entry"
    for pi in "${!prompts[@]}"; do
      p="${prompts[$pi]}"; base="$OUT_DIR/step1/$(slug "$name")_p$pi"
      "$MASTER_BIN" check "$path" "$p" --max-tokens 32 >"$base.master.check" 2>"$base.master.check.err"
      "$BIN" check "$path" "$p" --max-tokens 32 --weights f32 >"$base.f32.check" 2>"$base.f32.check.err"
      a=$(grep '^REFLEX_CHECK ' "$base.master.check" | sed 's/ weights_dtype=[a-z0-9]*//')
      b=$(grep '^REFLEX_CHECK ' "$base.f32.check" | sed 's/ weights_dtype=[a-z0-9]*//')
      if [[ -n "$a" && "$a" == "$b" ]]; then
        echo "| $name | check p$pi (32 token ids + first-token logit checksum) | identical |" >>"$SUMMARY"
      else
        echo "| $name | check p$pi | **DIFFERS** (see $base.*.check) |" >>"$SUMMARY"
        fail "f32 vs master differs: $name p$pi"
      fi
    done
    base="$OUT_DIR/step1/$(slug "$name")_system1"
    "$MASTER_BIN" system1 "$path" "$SYS1_PROMPT" --candidate " True" --candidate " False" \
      >"$base.master" 2>"$base.master.err"
    "$BIN" system1 "$path" "$SYS1_PROMPT" --candidate " True" --candidate " False" --weights f32 \
      >"$base.f32" 2>"$base.f32.err"
    a=$(grep '^REFLEX_SYSTEM1_CANDIDATE_OK' "$base.master")
    b=$(grep '^REFLEX_SYSTEM1_CANDIDATE_OK' "$base.f32")
    if [[ -z "$a" && -z "$b" ]]; then
      echo "| $name | system1 | not run on either (see $base.*.err) |" >>"$SUMMARY"
    elif [[ "$a" == "$b" ]]; then
      echo "| $name | system1 scores | identical |" >>"$SUMMARY"
    else
      echo "| $name | system1 scores | **DIFFERS** (see $base.*) |" >>"$SUMMARY"
      fail "system1 f32 vs master differs: $name"
    fi
  done
fi

# ---- step 2: f16 correctness -----------------------------------------------------------
section "Step 2 -- f16 correctness"
{
  echo "f16 (default mode) vs. f32, 32 greedy tokens."
  echo
  echo "| model | prompt | matched/total | first divergence | gap in f32 run | gap in f16 run |"
  echo "|---|---|---|---|---|---|"
} >>"$SUMMARY"
mkdir -p "$OUT_DIR/step2"
act_rows=""
for entry in "${models[@]}"; do
  IFS='|' read -r name path real <<<"$entry"
  for pi in "${!prompts[@]}"; do
    p="${prompts[$pi]}"; base="$OUT_DIR/step2/$(slug "$name")_p$pi"
    run_gen "$BIN" "$base.f32" "$path" "$p" --weights f32
    run_gen "$BIN" "$base.f16" "$path" "$p" --weights f16
    st=$(analyze compare "$base.f32" "$base.f16" "$name" "p$pi" 2>&1 >>"$SUMMARY" | sed -n 's/^STATUS=//p')
    [[ "$st" == error ]] && fail "step 2 run failed: $name p$pi (logs: $base.*)"
    read -r amax asat <<<"$(analyze act "$base.f16")"
    act_rows+="| $name | p$pi | $amax | $asat |"$'\n'
    if [[ -n "$LLAMA_SIMPLE" && "$real" == 1 ]]; then
      "$LLAMA_SIMPLE" -m "$path" -n 32 "$p" >"$base.llama.stdout" 2>"$base.llama.stderr"
      # llama-simple echoes the prompt then the continuation; compare continuations as text.
      python3 - "$base" "$p" >>"$OUT_DIR/step2/llama_rows.md" <<'PY'
import ast, sys
base, prompt = sys.argv[1], sys.argv[2]
llama = open(base + ".llama.stdout", errors="replace").read()
cont = llama.split(prompt, 1)[1] if prompt in llama else llama
def reflex_text(suffix):
    import re
    m = re.search(r'token_text=("(?:[^"\\]|\\.)*")', open(base + suffix).read())
    return ast.literal_eval(m.group(1)) if m else None
f32, f16 = reflex_text(".f32.stdout"), reflex_text(".f16.stdout")
def cmp(t):
    if t is None: return "run failed"
    if cont.startswith(t) or t.startswith(cont.strip("\n")): return "match"
    n = next((i for i in range(min(len(t), len(cont))) if t[i] != cont[i]), min(len(t), len(cont)))
    return f"differs at char {n}"
print(f"| {base.rsplit('/', 1)[1]} | {cmp(f32)} | {cmp(f16)} |")
PY
    fi
  done
done
{
  echo
  echo "Max |activation| cast to f16 for the prefill GEMM (REFLEX_F16_ACT_STATS), f16 runs."
  echo "f16's largest finite value is 65504; anything above 6550.4 is within 10x and is flagged."
  echo
  echo "| model | prompt | max abs | saturated |"
  echo "|---|---|---|---|"
  printf '%s' "$act_rows"
} >>"$SUMMARY"
while IFS='|' read -r _ m pl amax asat _; do
  amax="${amax// /}"; asat="${asat// /}"
  [[ -z "$amax" || "$amax" == "n/a" ]] && continue
  if awk -v a="$amax" 'BEGIN { exit !(a > 6550.4 || a != a) }'; then
    fail "activation range: $m $pl max |x| = $amax is within 10x of 65504 (saturated=$asat)"
  fi
done <<<"$act_rows"
if [[ -f "$OUT_DIR/step2/llama_rows.md" ]]; then
  { echo; echo "Text vs. llama.cpp \`llama-simple -n 32\` (greedy)."; echo
    echo "| run | reflex f32 | reflex f16 |"; echo "|---|---|---|"
    cat "$OUT_DIR/step2/llama_rows.md"; } >>"$SUMMARY"
fi

# system1: max |score diff| f16 vs f32
{ echo; echo "system1, f16 vs. f32 on the same prompt/candidates (\" True\"/\" False\" and \" Paris\"/\" London\")."; echo
  echo "| model | max abs score diff | best candidate same |"; echo "|---|---|---|"; } >>"$SUMMARY"
for entry in "${models[@]}"; do
  IFS='|' read -r name path real <<<"$entry"
  base="$OUT_DIR/step2/$(slug "$name")_system1"
  for w in f32 f16; do
    "$BIN" system1 "$path" "$SYS1_PROMPT" --candidate " True" --candidate " False" --weights $w >"$base.a.$w" 2>"$base.a.$w.err"
    "$BIN" system1 "$path" "The capital of France is" --candidate " Paris" --candidate " London" --weights $w >"$base.b.$w" 2>"$base.b.$w.err"
  done
  python3 - "$base" "$name" >>"$SUMMARY" <<'PY'
import re, sys
base, name = sys.argv[1], sys.argv[2]
def scores(f):
    try: t = open(f).read()
    except OSError: return None, None
    s = [float(x) for x in re.findall(r"REFLEX_SYSTEM1_CANDIDATE_OK .*? score=(-?[0-9.]+)", t)]
    b = re.search(r"best_idx=(\d+)", t)
    return (s or None), (b.group(1) if b else None)
diffs, same = [], True
for k in "ab":
    (s32, b32), (s16, b16) = scores(f"{base}.{k}.f32"), scores(f"{base}.{k}.f16")
    if s32 is None or s16 is None:
        continue
    diffs += [abs(x - y) for x, y in zip(s32, s16)]
    same &= b32 == b16
print(f"| {name} | {max(diffs):.6g} | {'yes' if same else 'NO'} |" if diffs else f"| {name} | not run (see {base}.*.err) | |")
PY
done

# ---- step 3: cold start -------------------------------------------------------------------
section "Step 3 -- cold start ($(basename "$DENSE_GGUF"), n=$N_RUNS each, same GPU and session)"
mkdir -p "$OUT_DIR/step3"
cold() {  # cold <label> <dir-with-scripts> <REFLEX_WEIGHTS or empty>
  local label="$1" dir="$2" w="$3"
  for which in system1 generate; do
    local script="$dir/scripts/bench_cold_start_phases_system1.sh"
    [[ $which == generate ]] && script="$dir/scripts/bench_cold_start_phases.sh"
    echo "### $label -- $which" >>"$SUMMARY"
    # Via bash: bench_cold_start_phases.sh is not executable in git (mode 100644, on master too).
    if REFLEX_WEIGHTS="$w" bash "$script" "$DENSE_GGUF" "$N_RUNS" \
         >"$OUT_DIR/step3/${label}_$which.md" 2>"$OUT_DIR/step3/${label}_$which.err"; then
      sed -n '/^| phase/,$p' "$OUT_DIR/step3/${label}_$which.md" >>"$SUMMARY"
    else
      fail "cold-start bench failed: $label $which (see $OUT_DIR/step3/${label}_$which.err)"
    fi
    echo >>"$SUMMARY"
  done
}
[[ "${SKIP_MASTER:-0}" != 1 ]] && cold master "$MASTER_DIR" ""
cold branch-f32 "$repo_root" f32
cold branch-f16 "$repo_root" f16
echo "model_load_ms: master vs. branch-f32 isolates the larger dequant module (both load f32); branch-f32 vs. branch-f16 is the dtype itself." >>"$SUMMARY"

# ---- step 4: memory and decode ------------------------------------------------------------
section "Step 4 -- memory and decode"
{ echo "\`reflex bench\` runs three prompt buckets, one row each. Warm prompt p50: first-token"
  echo "latency of an already-loaded model (prefill + LM head). Decode: the 16 tokens after it."
  echo
  echo "| model | weights | resident MiB (load delta) | nvidia-smi peak MiB over idle | prompt tokens | warm prompt p50 ms | decode ms/token | tokens/s |"
  echo "|---|---|---|---|---|---|---|---|"; } >>"$SUMMARY"
mkdir -p "$OUT_DIR/step4"
for path in "$DENSE_GGUF" ${LARGE_GGUF:+"$LARGE_GGUF"}; do
  for w in f32 f16; do
    base="$OUT_DIR/step4/$(slug "$(basename "$path")")_$w"
    "$BIN" bench "$path" --warmup 3 --iters 20 --weights $w >"$base.bench" 2>"$base.bench.err"
    res=$(grep -o 'model_resident_mib=[0-9]*' "$base.bench" | cut -d= -f2)
    idle=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | head -1)
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 -lms 20 >"$base.smi" 2>/dev/null &
    smi_pid=$!
    "$BIN" generate "$path" "${prompts[2]}" --max-tokens 64 --weights $w >"$base.gen" 2>"$base.gen.err"
    kill "$smi_pid" 2>/dev/null; wait "$smi_pid" 2>/dev/null
    peak=$(sort -n "$base.smi" | tail -1)
    over=$(( ${peak:-0} - ${idle:-0} ))
    # One REFLEX_BENCH_WARM_OK and one REFLEX_BENCH_THROUGHPUT_OK line per prompt bucket.
    rows=$(awk -v m="$(basename "$path")" -v w="$w" -v r="${res:-n/a}" -v o="$over" '
      function f(k,  i) { for (i = 1; i <= NF; i++) if (index($i, k "=") == 1) return substr($i, length(k) + 2) }
      /^REFLEX_BENCH_WARM_OK / { warm[f("prompt_tokens")] = f("p50_ms") }
      /^REFLEX_BENCH_THROUGHPUT_OK / { pt = f("prompt_tokens")
        printf "| %s | %s | %s | %s | %s | %s | %s | %s |\n", m, w, r, o, pt, warm[pt], f("ms_per_token"), f("tokens_per_sec") }
    ' "$base.bench")
    if [[ -z "$res" || -z "$rows" ]]; then
      echo "| $(basename "$path") | $w | ${res:-n/a} | $over | n/a | n/a | n/a | n/a |" >>"$SUMMARY"
      fail "bench failed: $(basename "$path") $w (see $base.bench.err)"
    else
      echo "$rows" >>"$SUMMARY"
    fi
  done
done

# ---- step 5: tests ----------------------------------------------------------------------------
section "Step 5 -- tests"
if REFLEX_SKIP_CUDA=1 cargo test >"$OUT_DIR/host-tests.log" 2>&1; then
  echo "- host tests (REFLEX_SKIP_CUDA=1): $(grep -m1 '^test result' "$OUT_DIR/host-tests.log")" >>"$SUMMARY"
else
  fail "host tests failed (see $OUT_DIR/host-tests.log)"
fi
if (cd sidecar/openai-adapter && cargo test) >"$OUT_DIR/sidecar-tests.log" 2>&1; then
  echo "- sidecar tests: $(grep -m1 '^test result' "$OUT_DIR/sidecar-tests.log")" >>"$SUMMARY"
else
  fail "sidecar tests failed (see $OUT_DIR/sidecar-tests.log)"
fi
for w in f32 f16; do
  echo >>"$SUMMARY"; echo "### #[ignore]d GPU tests, REFLEX_WEIGHTS=$w" >>"$SUMMARY"
  if REFLEX_WEIGHTS=$w LOG_DIR="$OUT_DIR/gpu-tests-$w" CARGO_TEST_FLAGS="--release --features $FEATURES" \
       DENSE_GGUF="$DENSE_GGUF" HYBRID_GGUF="$HYBRID_GGUF" MLA_REAL_GGUF="$MLA_REAL_GGUF" \
       scripts/gpu_nightly_tests.sh >"$OUT_DIR/gpu-tests-$w.md" 2>"$OUT_DIR/gpu-tests-$w.err"; then
    cat "$OUT_DIR/gpu-tests-$w.md" >>"$SUMMARY"
  else
    cat "$OUT_DIR/gpu-tests-$w.md" >>"$SUMMARY"
    fail "GPU tests failed with REFLEX_WEIGHTS=$w"
  fi
  # The prefill tests print their measured hidden-state error; keep it with the result.
  grep -h "rel_l2=" "$OUT_DIR/gpu-tests-$w"/*.log 2>/dev/null | sed 's/^/    /' >>"$SUMMARY"
done

echo >>"$SUMMARY"
echo "**Failures: $failures**" >>"$SUMMARY"
cat "$SUMMARY"
log "summary: $SUMMARY"
[[ $failures -eq 0 ]]
