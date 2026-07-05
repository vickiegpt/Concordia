#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

VALID_STATUSES = {"implemented", "partial", "missing", "blocked"}


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root)
    claims = json.loads(Path(args.claims).read_text(encoding="utf-8"))
    errors = validate_claims(claims, repo_root)
    if errors:
        for error in errors:
            print(error)
        return 1
    if args.markdown:
        write_markdown(Path(args.markdown), claims)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Validate Concordia paper claim matrix")
    parser.add_argument("--claims", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--markdown", default="")
    return parser.parse_args()


def validate_claims(claims, repo_root: Path):
    errors = []
    seen = set()
    if not isinstance(claims, list) or not claims:
        return ["claims file must contain a non-empty list"]

    for claim in claims:
        claim_id = claim.get("id", "")
        status = claim.get("status", "")
        evidence = claim.get("evidence", [])
        blockers = claim.get("blockers", [])

        if not claim_id:
            errors.append("claim is missing id")
            continue
        if claim_id in seen:
            errors.append(f"duplicate claim id {claim_id}")
        seen.add(claim_id)
        if status not in VALID_STATUSES:
            errors.append(f"claim {claim_id} has invalid status {status}")
        if status == "implemented" and not evidence:
            errors.append(f"implemented claim {claim_id} has no evidence")
        if status in {"partial", "missing", "blocked"} and not blockers:
            errors.append(f"{status} claim {claim_id} has no blockers")
        for item in evidence:
            path = item.get("path", "")
            if not path:
                errors.append(f"claim {claim_id} has evidence without path")
                continue
            if not (repo_root / path).exists():
                errors.append(f"claim {claim_id} evidence path does not exist: {path}")
    return errors


def write_markdown(path: Path, claims):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Concordia Claim Audit",
        "",
        "| id | status | title | repo reality | blockers |",
        "| --- | --- | --- | --- | --- |",
    ]
    for claim in claims:
        blockers = "<br>".join(claim.get("blockers", [])) or "-"
        lines.append(
            "| {id} | {status} | {title} | {repo_reality} | {blockers} |".format(
                id=escape_cell(claim.get("id", "")),
                status=escape_cell(claim.get("status", "")),
                title=escape_cell(claim.get("title", "")),
                repo_reality=escape_cell(claim.get("repo_reality", "")),
                blockers=escape_cell(blockers),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def escape_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
