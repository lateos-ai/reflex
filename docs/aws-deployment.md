# AWS deployment: Spot GPU instances with scale-to-zero

This is **one** deployment pattern for running Reflex on AWS, not the only one — see
the root [README](../README.md)'s `## Docker` and `## Kubernetes` sections for the base
container image and a plain Kubernetes `Job` pattern. This guide covers a
cost-optimized pattern for a single, always-cold-starting GPU instance: an EC2 Auto
Scaling Group backed by Spot Instances with **minimum capacity 0**, a `reflex uds`
sidecar container reachable only over a local Unix Domain Socket, and a shared EFS
cache so a freshly-launched Spot instance doesn't re-download model weights.

Nothing here changes the engine itself. Reflex's [Non-goals](../README.md#non-goals)
are unconditional: no in-core HTTP/gRPC server, no request queue, `batch_size` always
1. This guide only wires the engine's existing `uds` subcommand into AWS infrastructure
around it.

## Architecture overview

```
Auto Scaling Group (min=0, max=1, Spot, single instance type)
  └─ EC2 instance (boots from a DLAMI, resolved via SSM at launch)
       ├─ EFS mount: /mnt/reflex-cache  (model weights, survives instance replacement)
       ├─ docker run --entrypoint reflex ... uds /models/<file>.gguf /tmp/reflex-ipc/reflex.sock
       └─ /tmp/reflex-ipc/reflex.sock  (bind-mounted into the container)
            ^
            | local UDS connection only — no ALB, no network socket, no cross-AZ hop
            |
       your application process, same instance
```

The model download and the container start are both driven by
[`scripts/aws_ec2_bootstrap.sh`](../scripts/aws_ec2_bootstrap.sh), run once as EC2
user-data on every instance launch (it's idempotent, so a re-run on restart is safe).

## What this pattern actually buys you (and what it doesn't)

Be precise about this before adopting it: **AWS Warm Pools — the standard mechanism for
hiding EC2 boot latency behind a pool of pre-initialized instances — do not support
Spot Instances in a mixed-instances-policy Auto Scaling Group.** Scaling this ASG from
0 means a real EC2 instance boot (DLAMI init, Docker/NVIDIA Container Toolkit already
baked into a custom AMI is possible, but at minimum the container pull/start and EFS
mount take real time) — on the order of tens of seconds to low minutes, not the
sub-second `process_start_to_first_token_ms` this project's own cold-start benchmarks
report for the `reflex` process itself. Reflex's AOT-compiled-kernel cold start is real
and measured, but it happens *after* the EC2 instance and container are already
running — it is not a substitute for instance boot time.

What this pattern legitimately buys you:
- **No idle GPU-hour cost.** The instance (and its Spot bill) doesn't exist at all when
  the ASG is at 0.
- **No ALB, no cross-AZ egress fee, no request-serialization overhead** for the actual
  inference call — the client talks to the sidecar over a local socket, not a network
  load balancer.
- **Spot pricing** on the GPU-hours you do use, typically 60-70% below on-demand.

What it does *not* buy you: sub-second response to a cold "scale from zero" trigger.
Match the trigger to that reality — a scheduled window (business hours only), a
queue-depth-driven CloudWatch alarm for batch/async workloads, or an interactive
workload where users can tolerate one cold EC2 boot after an idle period are all
realistic fits. A latency-sensitive request-response API expecting sub-second
first-response after long idle periods is not.

## Prerequisites

- An AWS account, a VPC with at least one subnet that has a route to the internet (for
  pulling the base DLAMI's packages, the ECR image, and the Hugging Face download) or a
  NAT gateway if the subnet is private.
- The `reflex` Docker image built with the `ipc` Cargo feature (see below) and pushed to
  a private ECR repository.
- **Pick one GPU instance family up front.** This guide uses `g4dn.xlarge` (NVIDIA T4,
  compute capability `sm_75`) throughout, matched to a `REFLEX_CUDA_ARCH=sm_75` build —
  the zero-driver-JIT cubin path this project's own architecture is built around (see
  the root [README](../README.md)'s "Core technical bet" section). **Do not** put both
  `g4dn.xlarge` and `g5.xlarge` (A10G, `sm_86`) in one mixed-instances-policy ASG with a
  pinned cubin build — a `sm_75` cubin will not run on an A10G or vice versa. If you
  deliberately want a multi-family Spot fleet for capacity diversification, build with
  `REFLEX_CUDA_ARCH` unset instead (portable PTX, JIT'd by the driver at load time on
  whichever GPU the instance actually has) and accept the driver-JIT cold-start cost
  that reintroduces.

### Model sizing for a T4 (`g4dn.xlarge`, 16GB VRAM)

Reflex dequantizes every weight tensor once and holds it GPU-resident as `f32` — VRAM
need is therefore `total_params × 4 bytes`, regardless of the source GGUF's quant
level, and for MoE, regardless of how many experts are actually "active" per token
(every expert is dequantized and resident, since routing happens per-token at
runtime). Real-hardware-verified free VRAM on a `g4dn.xlarge` right after driver
init: **14,775 MiB** (`nvidia-smi` inside a real `docker run --gpus all` container) —
leaving headroom for the KV cache/activation buffers puts the practical ceiling
around **~3-3.5B total parameters**.

| Model class | Fits on one T4? |
|---|---|
| Dense Qwen3-0.6B / Qwen3-1.7B | Yes, comfortably — real-hardware-verified |
| Qwen3.5-0.8B hybrid (Gated DeltaNet) | Yes, comfortably — real-hardware-verified |
| Dense Qwen3-4B | No — 4B × 4B = 16GB, over budget before KV cache/activations are even counted |
| Qwen3-MoE (e.g. `Qwen3-30B-A3B`) | No, not close — every expert is dequantized regardless of top-k routing, so it needs the full ~120GB `f32` footprint, not the "3B active" figure |
| DeepSeek-V2-Lite (MLA) | No — needed ~63GB even at the smallest real checkpoint; this project verified it on a rented 80GB A100, not a T4-class card |

For latency-sensitive single-pass scoring (the `reflex system1` subcommand), smaller is
strictly better, not just VRAM-cheaper — both the cold-load time and the per-token
compute scale with parameter count, so the smallest viable checkpoint (Qwen3-0.6B) is
the right choice for that use case even when a bigger one would technically fit; see
the root [README](../README.md)'s Benchmarks section for the TypeSafe Jev comparison
this reasoning is based on. Going to a bigger GPU instance family (`g5`/A10G-24GB,
`p3`/V100-16-32GB, `p4d`/A100-40-80GB — note every `g4dn` size uses the same 16GB T4,
so a larger `g4dn.*xlarge` does not buy more VRAM) only matters for going up in *model
class* (dense Qwen3-4B+, any MoE, or MLA/DeepSeek-V2-class); it does not meaningfully
help cold-load or single-pass-scoring latency for a model that already fits
comfortably on a T4.

## 1. Build and push the sidecar image

The repo's root `Dockerfile` already builds the `reflex` binary and is used unchanged
by README's Docker/Kubernetes sections. It gained one additive build arg,
`REFLEX_FEATURES`, specifically for this guide — empty by default (identical image to
before), set to `ipc` to compile in the `stdio`/`uds` subcommands this pattern needs:

```bash
docker build \
  --build-arg REFLEX_CUDA_ARCH=sm_75 \
  --build-arg REFLEX_FEATURES=ipc \
  -t reflex:ipc .

aws ecr create-repository --repository-name reflex --region "$AWS_REGION" || true
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
docker tag reflex:ipc "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/reflex:ipc"
docker push "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/reflex:ipc"
```

Note the image is *not* built with the `download` Cargo feature — the sidecar never
downloads models itself; see step 5.

## 2. Shared model cache: EFS, not a single EBS volume

A `gp3` EBS volume is single-attach and locked to one Availability Zone. A plain
(non-warm-pool) Auto Scaling Group gives you no guarantee a replacement instance lands
in the same AZ, let alone that you can cheaply re-attach the exact same volume without
custom attach-by-tag logic in user-data. **EFS (NFS, multi-AZ, mountable by any
instance that can reach it)** is the AWS-documented pattern for a cache that must
survive ASG instance replacement:

```bash
efs_id=$(aws efs create-file-system \
  --creation-token reflex-cache --encrypted \
  --throughput-mode bursting \
  --tags Key=Name,Value=reflex-cache \
  --query 'FileSystemId' --output text)

# One mount target per subnet the ASG can launch into.
aws efs create-mount-target \
  --file-system-id "$efs_id" --subnet-id "$SUBNET_ID" \
  --security-groups "$EFS_SECURITY_GROUP_ID"
```

`$EFS_SECURITY_GROUP_ID` should allow inbound TCP/2049 (NFS) from the ASG's own
security group only — not `0.0.0.0/0`.

If your ASG is deliberately capped at `max-size=1` (this guide's default), a single
`gp3` volume reattached by a fixed, pre-created volume ID via `aws ec2 attach-volume`
in user-data is a simpler alternative that works — but it doesn't generalize past one
instance, and EFS costs cents/month for a model-sized cache, so it's the default here.

## 3. IAM instance profile (least privilege)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PullSidecarImage",
      "Effect": "Allow",
      "Action": ["ecr:GetAuthorizationToken"],
      "Resource": "*"
    },
    {
      "Sid": "PullSidecarImageRepo",
      "Effect": "Allow",
      "Action": ["ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage"],
      "Resource": "arn:aws:ecr:REGION:ACCOUNT_ID:repository/reflex"
    },
    {
      "Sid": "MountModelCache",
      "Effect": "Allow",
      "Action": ["elasticfilesystem:ClientMount", "elasticfilesystem:ClientWrite"],
      "Resource": "arn:aws:elasticfilesystem:REGION:ACCOUNT_ID:file-system/FILE_SYSTEM_ID"
    }
  ]
}
```

If the model repo is gated/private, add a fourth statement scoped to
`ssm:GetParameter` on one specific parameter path holding the HF token as a
`SecureString`, and read it into `HF_TOKEN` from the Launch Template's user-data at
boot — **never** put a token directly in user-data as plaintext; user-data is readable
by anything with access to the instance's metadata service (IMDS).

## 4. AMI resolution via SSM (no hardcoded AMI ID)

Set the Launch Template's `ImageId` to an SSM parameter alias so it always resolves to
the current AMI at launch time, instead of a hardcoded ID that goes stale:

```
resolve:ssm:/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id
```

(Ubuntu 22.04 to match this project's own `Dockerfile` base image.) To inspect what
that currently resolves to:

```bash
aws ssm get-parameter --region "$AWS_REGION" \
  --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
  --query "Parameter.Value" --output text
```

## 5. Bootstrap script (user-data)

[`scripts/aws_ec2_bootstrap.sh`](../scripts/aws_ec2_bootstrap.sh) is the real, runnable
script this Launch Template's user-data invokes. It:

1. Installs Docker from Docker's own `docker-ce` apt repository — not the distro
   `docker.io` package, which is older and doesn't pull in `nvidia-ctk`.
2. Installs NVIDIA Container Toolkit from **its own** apt repository (`nvidia-ctk` ships
   from `nvidia-container-toolkit`, not from `docker.io`), then runs `nvidia-ctk runtime
   configure --runtime=docker`.
3. Mounts the EFS cache at `/mnt/reflex-cache`.
4. Downloads the GGUF with a plain `curl -L -C -` against the Hugging Face resolve URL
   (skipping the download if the file is already cached from a previous instance) — no
   Python, no `download` Cargo feature needed in the sidecar image, since `reflex uds`
   only ever needs a local file path.
5. Pulls the ECR image and starts it with `--entrypoint /usr/local/bin/reflex ... uds
   <path> /tmp/reflex-ipc/reflex.sock`.

Its user-data invocation:

```bash
#!/bin/bash
export EFS_ID=fs-0123456789abcdef0
export ECR_IMAGE=123456789012.dkr.ecr.us-east-1.amazonaws.com/reflex:ipc
export MODEL_REPO=Qwen/Qwen3-0.6B-GGUF
export MODEL_FILE=Qwen3-0.6B-Q8_0.gguf
# export HF_TOKEN populated from SSM here if the repo is gated
curl -fsSL https://raw.githubusercontent.com/YOUR_ORG/reflex/main/scripts/aws_ec2_bootstrap.sh -o /tmp/bootstrap.sh
bash /tmp/bootstrap.sh
```

(Or bake the script into a custom AMI/user-data directly rather than fetching it at
boot, if you'd rather not depend on GitHub availability during instance launch.)

## 6. Launch Template + Auto Scaling Group

```bash
aws ec2 create-launch-template \
  --launch-template-name reflex-g4dn-spot \
  --launch-template-data '{
    "ImageId": "resolve:ssm:/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id",
    "InstanceType": "g4dn.xlarge",
    "IamInstanceProfile": {"Name": "reflex-instance-profile"},
    "SecurityGroupIds": ["'"$INSTANCE_SECURITY_GROUP_ID"'"],
    "InstanceMarketOptions": {
      "MarketType": "spot",
      "SpotOptions": {"SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"}
    },
    "UserData": "'"$(base64 -w0 user-data.sh)"'",
    "TagSpecifications": [{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": "reflex-engine"}]}]
  }'

aws autoscaling create-auto-scaling-group \
  --auto-scaling-group-name reflex-asg \
  --launch-template "LaunchTemplateName=reflex-g4dn-spot,Version=\$Latest" \
  --min-size 0 --max-size 1 --desired-capacity 0 \
  --vpc-zone-identifier "$SUBNET_ID" \
  --tags "Key=Name,Value=reflex-engine,PropagateAtLaunch=true"
```

Deliberately **no mixed-instances-policy** (see Prerequisites above) and **no target
tracking** — target tracking assumes a running fleet whose load it can measure, which
doesn't apply to a min=0 ASG. Drive `desired-capacity` from either a scheduled action
(`aws autoscaling put-scheduled-update-group-action`, e.g. up at 9am/down at 6pm) or a
CloudWatch alarm on a workload-specific metric (queue depth, a custom metric your
application publishes), depending on which trigger from the "What this buys you"
section above matches your traffic pattern.

## 7. Hardened `docker run` flags

The bootstrap script's `docker run` (see step 5) deliberately does **not** use
`--ipc=host`. Docker's `--ipc` flag shares the host's Linux SysV/POSIX IPC namespace —
an unrelated meaning of "IPC" from this project's own JSON-line protocol terminology —
and widens the container's blast radius for no benefit here: the UDS socket only needs
a shared bind-mounted *directory* (`/tmp/reflex-ipc`), not a shared kernel IPC
namespace. It does use:

- `--cap-drop=ALL` — the binary needs no Linux capabilities beyond what the NVIDIA
  runtime injects for GPU access.
- `--security-opt=no-new-privileges` — blocks privilege escalation via setuid binaries
  inside the container.
- `-v "$models_dir:/models:ro"` — the model volume is mounted read-only; the sidecar
  never writes to it.

## 8. Connecting a client: two distinct patterns

Reflex never runs a network server (see Non-goals). "Connecting a client" means one of
two genuinely different things — don't conflate them.

### Pattern A — UDS socket (cross-process, the actual sidecar client)

Any process **on the same instance** that can reach `/tmp/reflex-ipc/reflex.sock` can
speak the one-JSON-request-per-line protocol defined in
[`src/ipc.rs`](../src/ipc.rs). No PyO3, no special client library — plain sockets in
any language. Minimal Python example (stdlib only):

```python
import json
import socket

def call(prompt: str, max_tokens: int = 32) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect("/tmp/reflex-ipc/reflex.sock")
        request = {"prompt": prompt, "max_tokens": max_tokens}
        sock.sendall((json.dumps(request) + "\n").encode())
        response_line = sock.makefile("r").readline()
    return json.loads(response_line)

result = call("Once upon a time")
print(result["text"])
```

The request schema (`IpcRequest`): `{"id": <optional>, "prompt": <string>, "candidates":
[] (default), "max_tokens": 32 (default), "temperature": 1.0 (default)}`. An empty
`candidates` list runs plain generation; a non-empty list runs System1 candidate
scoring instead. The response schema (`IpcResponse`): `{"id", "ok", "error"?,
"token_ids"?, "text"?, "candidates"?, "entropy"?}`.

A Unix Domain Socket is local-to-the-host by construction. A genuinely remote client
(e.g. your laptop) needs either SSH access to run co-located on the same instance, or
an operator-added relay such as `ssh -L /local/socket:remote/socket` or `socat` bridging
TCP to the UDS path — an explicit, external add-on you choose to run, never something
Reflex grows on its own (per the Non-goals' "core engine never grows a network
socket").

### Pattern B — PyO3 / C FFI in-process embedding

This is a different **architecture**, not a client of the sidecar above. The PyO3
bindings (`src/python.rs`, built via `maturin build --features python`) and the C FFI
(`src/ffi.rs`, `include/reflex_engine.h`, with a real working example at
[`ffi-test/smoke_test.c`](../ffi-test/smoke_test.c)) load the model **into the same
process** as your Python or C/C++ application — there is no socket, no sidecar
container, and no network hop at all. Use this pattern when your application can run
directly on the GPU instance in the same process as the engine; use Pattern A when you
want the engine isolated in its own container, reachable by one or more sibling
processes on the same host.

## Cost Savings Breakdown (illustrative)

Pricing below is illustrative only — us-east-1, order-of-magnitude, as of writing.
Spot pricing varies continuously by AZ and demand and is never guaranteed; check the
[AWS Pricing Calculator](https://calculator.aws) for current numbers before budgeting.
The comparison assumes a workload active roughly 2 hours/day (~60 hours/month) — the
scale-to-zero win shrinks as utilization rises toward 24/7, at which point Spot vs.
on-demand pricing (not scale-to-zero) becomes the dominant lever instead.

The "always-on" framing for vLLM isn't a strawman: vLLM's own startup — CUDA graph
capture and model load — takes minutes, which makes true per-request scale-to-zero
impractical for it in practice, so an always-on deployment is the realistic baseline
for that engine, not an unfair comparison point.

| Component | This guide's pattern (Spot `g4dn.xlarge`, scale-to-zero) | Always-on vLLM + ALB (on-demand) |
|---|---|---|
| Compute | ~$0.18/hr Spot × ~60 active hr/mo ≈ **$11/mo** | ~$0.53/hr On-Demand × 730 hr/mo ≈ **$384/mo** |
| Load balancer | none (local UDS) | ALB base + LCU-hours ≈ **$18-25/mo** |
| Cross-AZ / egress | none (same-host socket) | non-zero if targets span AZs |
| Model cache | EFS, few-GB model ≈ **$1-2/mo** | local instance disk (lost on replacement) |
| **Estimated total** | **≈ $12-15/mo** | **≈ $400-410/mo** |

Restated plainly: the saving here is idle-GPU-hour and ALB avoidance for a
low-utilization workload — it is not a claim that this pattern responds to a cold
request faster than an always-on server would (an always-on server, by definition, has
no cold start to pay on the request path at all).

## Known limitations (recap)

- Scaling from 0 pays full EC2 instance boot time; Spot Instances are incompatible with
  AWS Warm Pools in a mixed-instances-policy ASG, so nothing here hides that latency.
- The UDS socket is host-local; there is no remote-network client pattern beyond an
  operator-added SSH/socat relay.
- One GPU instance family (and one pinned `REFLEX_CUDA_ARCH`) per ASG if you want the
  zero-driver-JIT cubin path — no mixing `g4dn`/`g5` in one pinned-build ASG.
- Cost figures above are illustrative, not a quote — verify with the AWS Pricing
  Calculator for your region and Spot market before budgeting.
- **Nothing in this guide has been deployed against a real AWS account or a real EC2
  GPU instance as part of writing it.** It was checked against this repository's actual
  CLI/IPC/Docker behavior and against current AWS documentation, not validated
  end-to-end on real infrastructure — treat it as a verified starting point, and
  confirm each step against your own account before relying on it in production.
