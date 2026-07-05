# Concordia Claim Audit

This folder keeps the paper-to-repo claim boundary explicit. The matrix in
`claims.json` marks each major Concordia paper claim as `implemented`,
`partial`, `missing`, or `blocked`, with code/benchmark evidence and blockers.

Run:

```bash
bash bench/concordia_claim_audit/test_claim_audit.sh
```

Generate a readable table:

```bash
python3 bench/concordia_claim_audit/check_claims.py \
  --claims bench/concordia_claim_audit/claims.json \
  --repo-root . \
  --markdown /tmp/concordia_claims.md
```
