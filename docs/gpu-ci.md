# GPU nightly CI

`.github/workflows/gpu-nightly.yml` runs the engine's real-GPU checks every night:

1. **All `#[ignore]`d GPU tests**, through `scripts/gpu_nightly_tests.sh`. The list is
   discovered with `cargo test -- --ignored --list`, so new GPU tests are picked up
   automatically. Each test gets the model it needs, and any test whose prerequisite is
   missing is reported as **SKIP** with the reason rather than dropped.
2. **The cold-start phase bench** (`scripts/bench_cold_start_phases_system1.sh`,
   `reflex system1` on Qwen3-0.6B-Q4_K_M, n=10), compared with
   `scripts/bench_compare.py` against `bench/baseline-t4.json`. The job fails if total
   p50 regresses by more than 15%.
3. **The warm agent-relay hops** (`scripts/bench_relay_nightly.sh`): one `reflex stdio`
   on the same model with the shared system prompt pinned, 20 hops of each kind with
   fixed prompts and greedy decoding: a generate hop resumed from the pinned prefix, a
   System-1 `candidates` hop, and a `json_schema` hop. Compared with
   `scripts/bench_compare.py --relay` against `bench/relay-baseline-t4.json` (see
   [The relay baseline](#the-relay-baseline)).
4. **Cold `generate`** (`scripts/bench_cold_start_phases.sh`, first token on the same
   model, n=10): reported, not gated; there is no `generate` baseline.
5. **Whole-process energy** (`scripts/bench_cold_energy.sh`, `reflex system1` under
   `reflex-energy`, which reads the counter on its updates): gross energy, idle
   baseline and net energy, p50/p95. Reported in the run summary only; it does not gate
   the job.
6. **A one-row results table** (`scripts/nightly_summary.py`): date, commit, GPU,
   `system1` and `generate` p50, gross energy p50 and the three relay hops' p50s. See
   [Latest results](#latest-results).

All results go to the run's summary page. The raw logs (per-test logs, per-run
`/usr/bin/time -v` output, bench stdout, `nightly-summary.md`) are uploaded as an
artifact kept for 30 days.

GitHub-hosted runners have no GPU, so the workflow starts an **ephemeral self-hosted
runner** on an EC2 `g4dn.xlarge` (Tesla T4, `sm_75`) with
[`machulav/ec2-github-runner`](https://github.com/machulav/ec2-github-runner), then
terminates the instance in a final job that runs even when earlier jobs fail or are
cancelled.

## Security

The repository is public. A self-hosted runner that ran code from a fork's pull
request would hand that code an AWS machine inside your account. The workflow
therefore has **no `pull_request` trigger**, only `schedule` and `workflow_dispatch`,
and those run the default branch's code. Keep it that way. If you ever want GPU checks
on PRs, gate them behind a maintainer-applied label and `pull_request_target` with an
explicit checkout of the reviewed SHA, and read GitHub's hardening guide before doing
so.

Each runner is single-use. The instance is created for one run and terminated
afterwards, so nothing persists between runs.

## Setup (one-time, needs an AWS account and repo admin)

The workflow skips every job until the repository variable
`GPU_NIGHTLY_ENABLED` is `true`, so merging it changes nothing until you finish this.

### 1. AWS

- **GPU quota.** New accounts start with 0 vCPUs of G/VT instances. Request at least 4
  vCPUs for *Running On-Demand G and VT instances* (quota `L-DB2E81BA`), plus the Spot
  quota `L-3819A6DF` if you plan to use Spot.
- **Network.** A subnet whose instances get a public IP (the default VPC's subnets do)
  and a security group that allows **outbound** HTTPS. No inbound rules are needed: the
  runner connects out to GitHub.
- **An IAM user for the workflow** with a policy scoped to what the action does:

  ```json
  {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "ec2:RunInstances",
          "ec2:TerminateInstances",
          "ec2:DescribeInstances",
          "ec2:DescribeInstanceStatus",
          "ec2:CreateTags"
        ],
        "Resource": "*"
      },
      {
        "Effect": "Allow",
        "Action": "ssm:GetParameter",
        "Resource": "arn:aws:ssm:*::parameter/aws/service/deeplearning/*"
      }
    ]
  }
  ```

  Create an access key for it. (GitHub OIDC with an assumable role is the stronger
  option, since it involves no long-lived key. Switch
  `aws-actions/configure-aws-credentials` to `role-to-assume` if you set that up.)

### 2. GitHub

- **A personal access token** that can register self-hosted runners on this
  repository. That means a classic PAT with `repo` scope, or a fine-grained token with
  *Administration: read and write* on this repository. It is used only to register
  and remove the runner.

**Secrets** (Settings → Secrets and variables → Actions → Secrets):

| Secret | Value |
|---|---|
| `GPU_NIGHTLY_AWS_ACCESS_KEY_ID` | the IAM user's access key id |
| `GPU_NIGHTLY_AWS_SECRET_ACCESS_KEY` | its secret access key |
| `GPU_NIGHTLY_GH_PAT` | the runner-registration token above |
| `GPU_NIGHTLY_FIXTURES_URL` | *optional*: URL of a `.tar.gz` of the synthetic `test-data/*.gguf` fixtures (`deepseek-tiny-mla`, `tiny-qwen3moe`, `tiny-qwen3moe-lora`, `tiny-qwen35moe`; about 100 MB). A presigned S3 URL or a private release asset both work. Without it, the 8 fixture tests SKIP. |

**Variables** (same page, Variables tab):

| Variable | Value |
|---|---|
| `GPU_NIGHTLY_ENABLED` | `true` to turn the workflow on |
| `GPU_NIGHTLY_SUBNET_ID` | e.g. `subnet-0ba8a0f25edfcfe76` |
| `GPU_NIGHTLY_SECURITY_GROUP_ID` | e.g. `sg-0123456789abcdef0` |
| `GPU_NIGHTLY_AWS_REGION` | *optional*, default `us-east-1` |
| `GPU_NIGHTLY_INSTANCE_TYPE` | *optional*, default `g4dn.xlarge`. A different GPU also needs `REFLEX_CUDA_ARCH` changed in the workflow and a new baseline. |
| `GPU_NIGHTLY_MARKET_TYPE` | *optional*: empty for On-Demand (default), `spot` for Spot. Spot costs less but `g4dn` Spot capacity in `us-east-1` has been exhausted in every AZ at once before, which would fail the run at launch. |

### 3. First run

Trigger it manually (Actions → GPU nightly → Run workflow) and check that:

- the summary shows the GPU test table, the bench comparisons and the results table;
- the EC2 console shows the `reflex-gpu-nightly` instance **terminated** afterwards.

Then replace the provisional baseline (next section).

## The baseline

`bench/baseline-t4.json` starts out **provisional**: its numbers come from the historical
manually measured T4 table in [benchmarks.md](benchmarks.md#cold-start-phase-breakdown)
(`f32` weights, before quantized-resident weights became the default), not from this
workflow's own environment. While
`"provisional": true`, a regression is reported as a **warning** and the job still
passes, so small environment differences (a newer DLAMI driver, a different AZ's host)
can't turn the first nightly red. `bench_compare.py --strict` overrides that.

To replace it with a real measurement, or to accept an intentional change such as a
feature that legitimately makes model load slower:

1. Download the `gpu-nightly-<run id>` artifact from a representative run.
2. Write a new baseline from that run's bench output:

   ```
   python3 scripts/bench_compare.py bench-output.txt \
     --write-baseline bench/baseline-t4.json \
     --source "GPU nightly run https://github.com/<org>/<repo>/actions/runs/<id>"
   ```

   This writes `"provisional": false`, so later regressions fail the job. It writes only
   the phases and the run's provenance: copy the descriptive fields
   (`compute_capability`, `kernel_image`, `weights`, `kv_cache`, `published`, `note`)
   over from the old file, since `scripts/prove_it.sh` prints them.
3. Commit it with a message that says why the baseline moved.

Single nightly runs are noisy at the level of individual phases. The gate is on
**total** p50 only. The per-phase table is there to help diagnose a failure, not to
gate on.

The threshold (default 15%) and bench run count (default 10) can be changed per run
with the manual-dispatch inputs.

## The relay baseline

`bench/relay-baseline-t4.json` holds one gated p50 per hop kind: the round trip of the
pinned generate hop, the round trip of the `candidates` hop, and the round trip of the
`json_schema` hop, plus that hop's count of valid JSON outputs. A hop kind regresses
when its p50 grows by more than the threshold (default 15%) **and** by at least 3 ms
(`--min-delta-ms`): a warm hop is short enough for a few ms of jitter to be a large
percentage. Fewer valid JSON outputs than the baseline also count as a regression. The
other p50s (first token, prefix copy, decode per token, masking) are in the report for
diagnosis only. If a hop's token count differs from the baseline's, the report says the
comparison is not like-for-like: the round trips then include different decode work.

The baseline stays **provisional until two scheduled runs exist**, so until then a
regression is a warning. A value of `"TBD"` means nothing was measured yet; the report
says so and the step passes. To write one from a run's `relay-output.txt`:

```
python3 scripts/bench_compare.py relay-output.txt --relay \
  --write-baseline bench/relay-baseline-t4.json [--provisional] \
  --source "GPU nightly run https://github.com/<org>/<repo>/actions/runs/<id>"
```

The relay threshold and hop count can be changed per run with the manual-dispatch
inputs.

## Latest results

Refreshed by hand at release time from a nightly run; the workflow never pushes to the
repository. Every run's summary page ends with this table's row for that run (also in
the artifact as `nightly-summary.md`), so refreshing it means copying that row here and
linking the run. A cell is `n/a` in a run when its bench produced nothing usable, and
the summary says why.

| date | commit | GPU | system1 p50 | generate p50 | gross energy p50 | relay pinned p50 | relay candidates p50 | relay constrained p50 |
|---|---|---|---|---|---|---|---|---|
| 2026-11-03 | 3b904c5 | Tesla T4 | 354.4 ms | 384.7 ms | TBD | 162.5 ms | 15.2 ms | 154.6 ms |

Run: [GPU nightly run 41568038765](https://github.com/lateos-ai/reflex/actions/runs/41568038765)
(manual dispatch on the branch that added this table; constrained hop 20/20 valid JSON).
Gross energy is TBD until a nightly runs with the counter-aligned reads; that run predates
them, and its net figure varied too much from run to run to publish.

What the columns are, all on Qwen3-0.6B-Q4_K_M with the workflow's build (default
weights, `--features ipc,nvml`, `sm_75` cubin):

- **system1 / generate p50**: cold processes, process start to result (`system1`, two
  candidates) and to first token (`generate`), n=10 each.
- **gross energy p50**: whole-process GPU energy of a cold `system1` run, measured by
  `reflex-energy` from outside the process over a window between two counter updates,
  n=10, not net of idle (most of it is the GPU's idle draw). Net of idle is in the
  energy step's own table, but a cold start's ~1 J net is below what this method gets
  right (see [docs/benchmarks.md](benchmarks.md#energy)), so gross is the
  run-to-run signal. The engine itself runs without `REFLEX_ENERGY=1`, so its timings
  are the default path's.
- **relay p50s**: the gated p50 per hop kind on one warm `reflex stdio`, 20 hops each:
  pinned and constrained round trip, `candidates` System-1 round trip (see
  [The relay baseline](#the-relay-baseline)).

## Cost

Everything in this section is an **estimate** from list prices, not a measurement.
Replace it with the real figure after a few runs (EC2 bills per second, and the
instance carries a `GitHubRun` tag you can filter the bill by).

| Item | Estimate |
|---|---|
| `g4dn.xlarge` On-Demand, `us-east-1` | $0.526/hour |
| One run: boot + runner registration, toolchain, cold build, tests, bench | TBD, expected about 15–25 minutes, so roughly $0.13–0.22 |
| 30 nightly runs | TBD, roughly $4–7/month |
| Model cache (`actions/cache`, ~0.9 GB) | free within GitHub's 10 GB per-repo cache |

A run that hangs is capped by the job's 90-minute `timeout-minutes`, and the stop job
still terminates the instance afterwards.

## Alternative: a Runpod pod instead of EC2

The workflow implements the EC2 path only. A Runpod pod can host the runner instead;
the steps after "start runner" stay the same. The outline:

1. **Image.** A CUDA *devel* base (for `nvcc`), such as
   `nvidia/cuda:12.4.1-devel-ubuntu22.04`, with the GitHub Actions runner unpacked
   into it and an entrypoint that runs `./run.sh --jitconfig "$RUNNER_JITCONFIG"`.
2. **Start job.** Call GitHub's
   `POST /repos/{owner}/{repo}/actions/runners/generate-jitconfig` (same PAT) to get a
   single-use runner config with a unique label, then create the pod through Runpod's
   REST API with `RUNNER_JITCONFIG` in its environment and a T4-class GPU.
3. **Stop job.** Delete the pod by id, under `if: always()`, as with EC2.

That path would need these secrets: `GPU_NIGHTLY_RUNPOD_API_KEY` (plus the same
`GPU_NIGHTLY_GH_PAT`) and a published runner image. On any GPU other than the T4, the
baseline and `REFLEX_CUDA_ARCH` also have to be redone for that card.

## Running the same checks by hand

On any Linux machine with an NVIDIA GPU and `nvcc`:

```
export REFLEX_CUDA_ARCH=sm_75            # match your GPU
cargo build --release --features ipc,nvml --bin reflex
DENSE_GGUF=/path/Qwen3-0.6B-Q4_K_M.gguf HYBRID_GGUF=/path/Qwen3.5-0.8B-Q4_K_M.gguf \
  scripts/gpu_nightly_tests.sh
scripts/bench_cold_start_phases_system1.sh /path/Qwen3-0.6B-Q4_K_M.gguf 10 | tee bench.txt
python3 scripts/bench_compare.py bench.txt --baseline bench/baseline-t4.json
scripts/bench_cold_start_phases.sh /path/Qwen3-0.6B-Q4_K_M.gguf 10 | tee generate.txt
python3 scripts/nightly_summary.py --commit "$(git rev-parse HEAD)" \
  --gpu "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" \
  --system1 bench.txt --generate generate.txt
```
