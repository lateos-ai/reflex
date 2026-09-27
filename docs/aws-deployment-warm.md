# AWS deployment: always-warm demo endpoint with autoscaling

This is a **sibling** pattern to [`docs/aws-deployment.md`](aws-deployment.md), not a
replacement for it. That guide covers a Spot, scale-to-zero (min=0), local-UDS-only
instance that's never reachable from outside the host. This guide covers the opposite
end of the spectrum: a permanently-warm On-Demand baseline instance, reachable over
public HTTPS, that scales out under real traffic — built to serve as a public "try
Reflex live" demo endpoint, e.g. for publicizing this project's warm-latency numbers.

Nothing here changes the engine itself. Reflex's [Non-goals](../README.md#non-goals)
are unconditional: no in-core HTTP/gRPC server, no request queue, `batch_size` always
1. This guide wires the existing [`sidecar/openai-adapter`](../sidecar/openai-adapter/README.md)
HTTP sidecar into AWS infrastructure around it — the sidecar's own HTTP front door
already accepts concurrent connections while strictly serializing them into one
`reflex stdio` child underneath (see that crate's README); this guide adds horizontal
scaling on top by running more *instances*, never more concurrency inside one.

## Architecture overview

```
Internet
  -> Route 53 (your subdomain) -> ALB (HTTPS:443, ACM cert, WAF WebACL attached)
       -> Target Group (HTTP:8000, health check "/healthz")
            -> Auto Scaling Group: min=1/max=2, g4dn.xlarge, On-Demand baseline +
               Spot burst (mixed-instances-policy, OnDemandBaseCapacity=1)
                 each instance: sidecar/openai-adapter/Dockerfile's combined image,
                 reflex-openai-adapter bound 0.0.0.0:8000, spawning
                 reflex stdio <gguf> as a child in the same container
```

The model download and container start are driven by
[`scripts/aws_ec2_bootstrap_warm.sh`](../scripts/aws_ec2_bootstrap_warm.sh), run as
EC2 user-data on every instance launch (idempotent, safe to re-run).

## What this pattern buys you (and what it doesn't)

- **No cold "scale from zero" latency for baseline traffic.** The min=1 instance is
  always running — a request during normal load never pays EC2 boot time or even
  Reflex's own cold-start time, since the model is already loaded and warm.
- **Real horizontal scale-out under burst traffic**, via a second, Spot-priced
  instance joining the target group at a meaningful discount over On-Demand (see the
  cost breakdown below) — bounded at max=2 for this rollout, a deliberate cost
  ceiling, not a technical limit.
- **A real public HTTPS endpoint** other people (or OpenRouter, if that application is
  ever approved) can point an OpenAI-SDK-compatible client at directly.

What it does *not* buy you:
- **Scale-out still pays a real EC2 boot** (tens of seconds to ~1-2 minutes) before
  the second instance is healthy and serving — this rollout deliberately skips AWS
  Warm Pools to keep the first rollout simple, and Warm Pools couldn't cover the burst
  instance anyway once it's Spot-priced: **AWS Warm Pools don't support Spot Instances
  in a mixed-instances-policy ASG at all** (same restriction the sibling scale-to-zero
  guide documents), so a Warm Pool here could only ever pre-initialize the On-Demand
  baseline slot, which is already always running and has nothing to hide boot latency
  for. Revisit only if scale-out latency becomes a real problem in practice, and only
  for a redesign that keeps the burst instance On-Demand too (trading back the Spot
  savings below for faster scale-out).
- **Real, continuous On-Demand GPU-hour cost for the baseline instance** (the burst
  instance is Spot, but the baseline instance that makes this pattern "always warm"
  is deliberately not), unlike the scale-to-zero sibling guide's near-zero idle cost.
  This is the explicit tradeoff for "always warm."
- **The burst instance can occasionally fail to launch or get reclaimed** (2-minute
  Spot interruption notice) — acceptable here specifically because it's *extra*
  capacity on top of an always-on On-Demand baseline, not the only capacity.
- **Not a substitute for real per-user auth.** The WAF shared-secret header (Phase 4
  below) keeps casual/automated traffic out; it is not per-user rate-limiting,
  billing, or identity — treat this as a demo endpoint, not a multi-tenant product.

## Prerequisites

- An AWS account, a VPC with public subnets (the ALB needs to be internet-facing) and
  a route to the internet for pulling the ECR image and the Hugging Face model.
- A domain name you control, delegated to Route 53 (or usable via your existing DNS
  provider), for the ACM certificate the ALB's HTTPS listener needs.
- The combined image (see below) built and pushed to a private ECR repository.
- Same GPU-family guidance as the sibling guide: this deployment is pinned to
  `g4dn.xlarge` (T4, `sm_75`) throughout, matching every real-hardware verification in
  this repo for the chosen model (Qwen3-0.6B — see the sibling guide's model-sizing
  table for why this fits a T4 comfortably; do not mix instance families in one ASG if
  using a pinned `REFLEX_CUDA_ARCH` cubin build).

## 1. Build and push the combined image

Unlike the sibling guide (which uses the root `Dockerfile` unchanged), this pattern
needs both `reflex` (with the `ipc` feature, for `reflex stdio`) and
`reflex-openai-adapter` in one image, since the adapter spawns `reflex stdio` as a
literal child process. Build from the repo root using
[`sidecar/openai-adapter/Dockerfile`](../sidecar/openai-adapter/Dockerfile) (build
context = repo root):

```bash
docker build -f sidecar/openai-adapter/Dockerfile \
  --build-arg REFLEX_CUDA_ARCH=sm_75 \
  -t reflex-openai-adapter:latest .

aws ecr create-repository --repository-name reflex-openai-adapter --region "$AWS_REGION" || true
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
docker tag reflex-openai-adapter:latest "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/reflex-openai-adapter:latest"
docker push "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/reflex-openai-adapter:latest"
```

This build was verified locally (Windows, Docker Desktop, no GPU) as part of writing
this guide: the image builds cleanly, and running it without a GPU/valid model path
fails cleanly — the container exits within about a second of the `reflex` child
dying, with a clear log line, instead of hanging or serving a silently-broken
`/healthz` forever (see the sidecar's README "Known limitations" for the fix this
relies on).

## 2. Shared model cache: EFS

Same rationale as the sibling guide — a `gp3` EBS volume is single-attach/single-AZ,
which doesn't survive ASG instance replacement across AZs. EFS (NFS, multi-AZ) does:

```bash
efs_id=$(aws efs create-file-system \
  --creation-token reflex-warm-cache --encrypted \
  --throughput-mode bursting \
  --tags Key=Name,Value=reflex-warm-cache \
  --query 'FileSystemId' --output text)

# One mount target per subnet the ASG can launch into.
aws efs create-mount-target \
  --file-system-id "$efs_id" --subnet-id "$SUBNET_ID" \
  --security-groups "$EFS_SECURITY_GROUP_ID"
```

`$EFS_SECURITY_GROUP_ID` should allow inbound TCP/2049 (NFS) from the ASG's own
security group only.

## 3. IAM instance profile

Same shape as the sibling guide's, scoped to this repository instead:

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
      "Resource": "arn:aws:ecr:REGION:ACCOUNT_ID:repository/reflex-openai-adapter"
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

If the model repo is gated, add a fourth statement scoped to `ssm:GetParameter` on a
specific `SecureString` parameter for `HF_TOKEN`, read at boot — never plaintext in
user-data (readable via the instance metadata service).

## 4. AMI resolution via SSM

Same SSM parameter alias as the sibling guide, so the Launch Template always resolves
to the current AMI rather than a hardcoded, staling ID:

```
resolve:ssm:/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id
```

## 5. Bootstrap script (user-data)

[`scripts/aws_ec2_bootstrap_warm.sh`](../scripts/aws_ec2_bootstrap_warm.sh) is the
real, runnable script the Launch Template's user-data invokes:

```bash
#!/bin/bash
export ECR_IMAGE=123456789012.dkr.ecr.us-east-1.amazonaws.com/reflex-openai-adapter:latest
export EFS_ID=fs-0123456789abcdef0
export MODEL_REPO=Qwen/Qwen3-0.6B-GGUF
export MODEL_FILE=Qwen3-0.6B-Q8_0.gguf
export PORT=8000
# export HF_TOKEN populated from SSM here if the repo is gated
curl -fsSL https://raw.githubusercontent.com/YOUR_ORG/reflex/main/scripts/aws_ec2_bootstrap_warm.sh -o /tmp/bootstrap.sh
bash /tmp/bootstrap.sh
```

It differs from the sibling guide's script in exactly the ways the two patterns
differ: it starts `reflex-openai-adapter` bound to a real TCP port instead of bare
`reflex uds` bound to a local socket, and polls `/healthz` over HTTP for readiness
instead of waiting for a socket file.

## 6. Launch Template + Auto Scaling Group

```bash
aws ec2 create-launch-template \
  --launch-template-name reflex-warm-g4dn \
  --launch-template-data '{
    "ImageId": "resolve:ssm:/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id",
    "InstanceType": "g4dn.xlarge",
    "IamInstanceProfile": {"Name": "reflex-warm-instance-profile"},
    "SecurityGroupIds": ["'"$INSTANCE_SECURITY_GROUP_ID"'"],
    "UserData": "'"$(base64 -w0 user-data.sh)"'",
    "TagSpecifications": [{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": "reflex-warm"}]}]
  }'

aws autoscaling create-auto-scaling-group \
  --auto-scaling-group-name reflex-warm-asg \
  --mixed-instances-policy '{
    "LaunchTemplate": {
      "LaunchTemplateSpecification": {"LaunchTemplateName": "reflex-warm-g4dn", "Version": "$Latest"},
      "Overrides": [{"InstanceType": "g4dn.xlarge"}]
    },
    "InstancesDistribution": {
      "OnDemandBaseCapacity": 1,
      "OnDemandPercentageAboveBaseCapacity": 0,
      "SpotAllocationStrategy": "price-capacity-optimized"
    }
  }' \
  --min-size 1 --max-size 2 --desired-capacity 1 \
  --vpc-zone-identifier "$SUBNET_ID" \
  --target-group-arns "$TARGET_GROUP_ARN" \
  --health-check-type ELB --health-check-grace-period 300 \
  --tags "Key=Name,Value=reflex-warm,PropagateAtLaunch=true"
```

**On-Demand baseline, Spot for burst** — `OnDemandBaseCapacity: 1` pins the
always-warm instance (min=1) to On-Demand, matching the explicit "always warm,
reliability over cost" decision for that instance. `OnDemandPercentageAboveBaseCapacity:
0` means any capacity *above* that baseline — i.e. the one burst instance this ASG can
ever add, since max=2 — is fulfilled from Spot instead, at a meaningful discount (see
the cost breakdown below). This is safe here specifically because there is still only
**one instance type in the `Overrides` list** (`g4dn.xlarge`) — it does not run into
the "don't mix GPU families in one pinned-cubin ASG" problem, which is about mixing
different instance *families/architectures* with a `REFLEX_CUDA_ARCH`-pinned cubin
build, not about mixing On-Demand and Spot *capacity* for the same instance type. If
Spot capacity for `g4dn.xlarge` isn't available in the AZ at scale-out time, the ASG
simply can't add the burst instance until it is — an acceptable failure mode for
*extra* capacity, since the On-Demand baseline instance keeps serving throughout
either way. `SpotAllocationStrategy: price-capacity-optimized` is AWS's current
recommended default, balancing price against interruption risk.

`--health-check-type ELB` means the ASG uses the ALB target group's `/healthz` result
(not just EC2 instance-status checks) to decide an instance is unhealthy and needs
replacing — this is what makes the sidecar's fail-fast-on-dead-child behavior
(Known limitations in its README) actually trigger a real instance replacement, not
just a container restart on the same box.

## 7. Application Load Balancer, target group, ACM, WAF

```bash
# Target group: HTTP to the instance, health check /healthz.
aws elbv2 create-target-group \
  --name reflex-warm-tg --protocol HTTP --port 8000 \
  --vpc-id "$VPC_ID" --target-type instance \
  --health-check-path /healthz \
  --health-check-protocol HTTP

# ACM cert for your subdomain (DNS validation via Route 53).
cert_arn=$(aws acm request-certificate \
  --domain-name "reflex-demo.yourdomain.com" \
  --validation-method DNS \
  --query CertificateArn --output text)
# ... complete DNS validation in Route 53 before proceeding ...

# Internet-facing ALB, HTTPS listener forwarding to the target group above.
alb_arn=$(aws elbv2 create-load-balancer \
  --name reflex-warm-alb --type application --scheme internet-facing \
  --subnets $PUBLIC_SUBNET_IDS --security-groups "$ALB_SECURITY_GROUP_ID" \
  --query 'LoadBalancers[0].LoadBalancerArn' --output text)

aws elbv2 create-listener \
  --load-balancer-arn "$alb_arn" --protocol HTTPS --port 443 \
  --certificates CertificateArn="$cert_arn" \
  --default-actions Type=forward,TargetGroupArn="$TARGET_GROUP_ARN"

# Route 53 record pointing your subdomain at the ALB (alias record).
```

**WAF WebACL**, associated with the ALB, two rules:
- An allow-by-default-deny rule matching a custom header
  (`X-Reflex-Demo-Key: <shared secret>`) via a byte-match statement — this is a shared
  "unlisted door" secret, appropriate for a low-stakes public demo (the sidecar itself
  has zero auth by design; see its README).
- A rate-based rule capping requests per source IP per 5-minute window, as basic abuse
  throttling given the sidecar has none of its own.

Instance security group: allow inbound on port 8000 **from the ALB's security group
only**, never `0.0.0.0/0` — the instance is directly internet-adjacent in this
pattern, unlike the sibling guide's local-UDS-only instance.

## 8. Autoscaling policy

Target tracking on `ALBRequestCountPerTarget`, not CPU/GPU utilization — CPU is a poor
signal for GPU-bound decode, and GPU utilization is bursty/single-flight for a
backend that serializes to ~1 concurrent request per instance underneath (see the
sidecar's `reflex_client.rs`):

```bash
aws autoscaling put-scaling-policy \
  --auto-scaling-group-name reflex-warm-asg \
  --policy-name reflex-warm-request-tracking \
  --policy-type TargetTrackingScaling \
  --target-tracking-configuration '{
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ALBRequestCountPerTarget",
      "ResourceLabel": "'"$ALB_RESOURCE_LABEL"'"
    },
    "TargetValue": 1.0
  }'
```

Start `TargetValue` at `1.0` (roughly "this backend serializes to ~1 concurrent
request") and tune down if scale-out feels late in practice.

**Known caveat, stated plainly**: ALB's `ALBRequestCountPerTarget` metric is
attributed on request *completion*, which is an imperfect signal for a long-lived SSE
streaming chat completion held open for many seconds — a burst of streaming requests
may not register as load as promptly as a true concurrency gauge would. Acceptable as
a starting point for this rollout; a custom CloudWatch metric published by the
sidecar itself would be a more accurate signal, but that's explicitly out of scope
here (it would add a new responsibility to a deliberately simple sidecar — see its
"Why a separate crate" README section).

## 9. Cost containment

- **ASG max-size = 2** is the primary blast-radius limiter on worst-case spend,
  chosen to match the user's explicit "several months of affordable spend" budget,
  not unlimited scale.
- **CloudWatch billing alarm**: enable "Receive Billing Alerts" in the account's
  Billing preferences, then create an alarm on the `EstimatedCharges` metric — this
  metric only exists in **`us-east-1`**, regardless of which region the ASG/ALB
  actually run in (a well-known AWS quirk):

```bash
aws cloudwatch put-metric-alarm --region us-east-1 \
  --alarm-name reflex-warm-billing-alarm \
  --metric-name EstimatedCharges --namespace AWS/Billing \
  --statistic Maximum --period 21600 --threshold "$THRESHOLD_USD" \
  --comparison-operator GreaterThanThreshold --evaluation-periods 1 \
  --dimensions Name=Currency,Value=USD \
  --alarm-actions "$SNS_TOPIC_ARN"
```

Subscribe your own account-owner email to `$SNS_TOPIC_ARN` during setup.

## Cost breakdown

Pricing below is us-east-1 list pricing, checked against AWS's own pricing pages and
current third-party pricing trackers as of writing (not just an order-of-magnitude
guess) — still confirm against the [AWS Pricing Calculator](https://calculator.aws)
before committing real budget, since list prices do change over time. Contrast with
the sibling guide's scale-to-zero numbers (~$12-15/mo for a ~2hr/day workload):

| Component | Rate | Baseline (min=1, no burst) |
|---|---|---|
| Compute, baseline instance | `g4dn.xlarge` On-Demand $0.526/hr × 730 hr/mo | **$383.98/mo** |
| EBS root volume | ~30GB gp3, ~$0.08/GB-mo | **~$2.40/mo** |
| ALB | $0.0225/hr base (730 hr/mo) + $0.008/LCU-hr (light traffic, ~2-3 LCU avg) | **~$21-24/mo** |
| WAF | $5/mo per WebACL + $1/mo per rule × 2 rules + $0.60/million requests | **~$7/mo** (demo-level request volume) |
| Model cache (EFS) | $0.30/GB-mo, model is well under 1GB (Qwen3-0.6B Q8_0 GGUF) | **~$1/mo** (rounds up from a fraction of a GB) |
| Route 53 hosted zone (if not already using one for this domain) | $0.50/mo + negligible query cost | **~$0.50/mo** |
| Data transfer out | first 100GB/mo free, then $0.09/GB | **$0/mo** at demo-level traffic |
| **Baseline total** | | **≈ $416/mo** |

Scale-out adds a second `g4dn.xlarge`, but as **Spot capacity**
(`OnDemandBaseCapacity=1` in step 6 above pins only the baseline instance to
On-Demand), prorated by how much of the month it's actually running (the ASG only
runs it while `ALBRequestCountPerTarget` says it's needed). Spot pricing for
`g4dn.xlarge` currently runs **$0.18-0.30/hr** depending on AZ/demand at the time —
not a fixed or guaranteed rate, unlike the baseline's On-Demand price — so burst
scenarios below are given as a range, not a point estimate:

| Burst scenario | Added Spot compute | **Total** |
|---|---|---|
| Light (≈10% duty cycle — occasional spikes, ~73 hr/mo) | +$13-22/mo | **≈ $429-438/mo** |
| Moderate (≈50% duty cycle — sustained elevated interest, ~365 hr/mo) | +$66-110/mo | **≈ $482-526/mo** |
| Hard ceiling (both instances essentially always on — max=2, ~730 hr/mo, plus data transfer past the free 100GB tier) | +$131-219/mo, +~$20-40/mo transfer | **≈ $567-675/mo** |

Compare the hard-ceiling row to the **~$820-850/mo** it would be with an On-Demand
burst instance — this is the concrete effect of the `OnDemandBaseCapacity`/Spot change
above: roughly **$180-280/mo cheaper at worst case**, at the cost of the burst
instance occasionally being unable to launch or getting reclaimed with a 2-minute
warning (acceptable here since it's *extra* capacity on top of an always-on On-Demand
baseline, not the only capacity — see "What this pattern buys you" above).

The **hard ceiling row is the number that actually matters for "several months of
affordable spend"** — it's the worst case the `max-size=2` ASG cap, the WAF
rate-based rule (Phase 4/step 7 above), and now the Spot burst pricing are bounding,
regardless of how much traffic actually arrives. Realistic month-to-month cost for a
demo that isn't constantly saturated sits in the **$416-526/mo** range.

This is the real cost of "always warm, reliability over cost" — restated plainly, the
scale-to-zero sibling pattern is dramatically cheaper for a low-utilization workload;
this pattern trades that for zero cold-start latency on every request and a real
public endpoint.

## Known limitations (recap)

- Scale-out beyond the baseline still pays a real EC2 boot (Warm Pools deliberately
  skipped for this rollout, and incompatible with the burst instance's Spot pricing
  anyway — see "What this pattern buys you" above).
- The burst instance is Spot-priced and can occasionally fail to launch or be
  reclaimed with a 2-minute warning — acceptable since it's capacity on top of an
  always-on On-Demand baseline, not the only capacity, but not a guarantee.
- The WAF shared-secret header is abuse deterrence, not real per-user
  authentication/billing/rate-limiting.
- `ALBRequestCountPerTarget`'s request-completion attribution is an imperfect signal
  for long-lived SSE streams (see step 8).
- `usage.prompt_tokens` in the sidecar's responses remains a whitespace-word-count
  approximation, not exact (see the sidecar's own README).
- Cost figures above are illustrative, not a quote.
- **This guide's Docker image build and its negative-path failure behavior (dead
  `reflex` child → container exit) were verified locally against real Docker Desktop
  as part of writing it. The full ASG/ALB/WAF/Route 53 topology described above has
  not been deployed against a real AWS account** — treat the account-level steps
  (2 through 9) as a verified-on-paper starting point, and confirm each one against
  your own account before pointing real public traffic at it, same convention as the
  sibling guide.
