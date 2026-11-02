# Contributing to Reflex

## Contributor License Agreement

Before we can accept a pull request, you need to sign our [Contributor License
Agreement](CLA.md). A bot will comment on your first PR with instructions — it
amounts to commenting the agreed sign-off text once; you won't need to do it again
on future PRs from the same GitHub account.

## Scope

Reflex has deliberate, permanent non-goals — see [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md#non-goals)'s "Non-goals" section
before proposing a serving-platform, multi-tenancy, batching, or in-core HTTP/gRPC
server feature. PRs in that direction will likely be declined regardless of
implementation quality, so it's worth raising an issue to discuss scope first for
anything beyond a bug fix or small, self-contained feature.

## Development

See [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) for build/run/test commands and the rules that constrain changes.

## Pull requests

The PR template asks three things; [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md#proposing-a-change)
explains each:

- **What does this do to cold-start latency and energy** on a representative small model
  (Qwen3-0.6B-Q4_K_M)? A number before and after, or "none: host-only change".
- **Does it stay inside the non-goals?**
- **Is it still correct?** CI's checks pass, and for engine changes, `reflex check` and
  the GPU tests on real hardware (say so if you couldn't run them).
