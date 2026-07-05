#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tmp="$(mktemp -d /tmp/concordia-claim-audit.XXXXXX)"
trap 'rm -rf "${tmp}"' EXIT

python3 "${root}/check_claims.py" \
  --claims "${root}/claims.json" \
  --repo-root "$(cd "${root}/../.." && pwd)" \
  --markdown "${tmp}/claims.md"

grep -q "| gpu-delta-checkpoint | implemented |" "${tmp}/claims.md"
grep -q "| cross-architecture-ctx | partial |" "${tmp}/claims.md"
grep -q "Durable AOF disk/file commit is still host-side" "${tmp}/claims.md"

cat >"${tmp}/bad_claims.json" <<'JSON'
[
  {
    "id": "bad-implemented-claim",
    "title": "Bad implemented claim",
    "status": "implemented",
    "paper_claim": "This should be rejected",
    "repo_reality": "No evidence",
    "evidence": [],
    "blockers": []
  }
]
JSON

if python3 "${root}/check_claims.py" \
    --claims "${tmp}/bad_claims.json" \
    --repo-root "$(cd "${root}/../.." && pwd)" >/tmp/bad-claim.out 2>&1; then
  echo "claim audit accepted implemented claim without evidence" >&2
  exit 1
fi

grep -q "implemented claim bad-implemented-claim has no evidence" /tmp/bad-claim.out
echo "claim audit tests passed"
