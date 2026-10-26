#!/bin/sh
# Writes the Rust side of the compliance bundle (see docs/compliance/README.md) to
# <out-dir>:
#
#   reflex.cdx.json                    CycloneDX SBOM of the engine crate (features: ipc,
#                                      what the adapter images ship)
#   reflex-openai-adapter.cdx.json     CycloneDX SBOM of sidecar/openai-adapter
#   THIRD_PARTY_LICENSES.html          license text of every crate in both lockfiles
#   THIRD_PARTY_LICENSES-adapter.html  (one file per Cargo project)
#
# Run from the repo root. Needs cargo-cyclonedx and cargo-about (versions pinned in
# .github/workflows/compliance.yml and the Dockerfiles). Reads Cargo.lock; builds
# nothing, so no CUDA toolkit is needed.
#
#   sh scripts/compliance-bundle.sh <out-dir>
set -eu

out=${1:?usage: sh scripts/compliance-bundle.sh <out-dir>}
mkdir -p "$out"
out=$(cd "$out" && pwd)

cargo cyclonedx --format json --features ipc --spec-version 1.5 --override-filename reflex.cdx
mv reflex.cdx.json "$out/reflex.cdx.json"
(
    cd sidecar/openai-adapter
    cargo cyclonedx --format json --spec-version 1.5 --override-filename reflex-openai-adapter.cdx
    mv reflex-openai-adapter.cdx.json "$out/reflex-openai-adapter.cdx.json"
)

cargo about generate --locked --all-features -c about.toml -o "$out/THIRD_PARTY_LICENSES.html" about.hbs
cargo about generate --locked --manifest-path sidecar/openai-adapter/Cargo.toml -c about.toml \
    -o "$out/THIRD_PARTY_LICENSES-adapter.html" about.hbs

echo "compliance bundle written to $out"
