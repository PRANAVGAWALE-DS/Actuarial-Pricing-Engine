#!/usr/bin/env python3
"""
generate_checksums.py
=====================
Run after train.py to regenerate SHA-256 checksum files for model artifacts.
Safe to re-run — always overwrites existing checksum files.

The script discovers all .joblib files in the model directory (excluding
preprocessors and existing checksum files) so it automatically covers
whatever model train.py produced — no hardcoded artifact list.

Usage:
    python scripts/generate_checksums.py
    python scripts/generate_checksums.py --model-dir /custom/models/path
    python scripts/generate_checksums.py --model-dir models/ --dry-run
"""

import argparse
import hashlib
import sys
from pathlib import Path


def compute_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_artifacts(model_dir: Path) -> list[Path]:
    """
    Return all .joblib files that represent model artifacts.
    Excludes:
      - preprocessor_*.joblib  (covered separately by _resolve_preprocessor_path)
      - *_checksum.joblib      (checksum side-files, if any)
      - drift_baseline.joblib  (monitoring artifact, not a prediction model)
    """
    return sorted(
        p
        for p in model_dir.glob("*.joblib")
        if not p.stem.startswith("preprocessor_")
        and not p.stem.endswith("_checksum")
        and p.stem not in {"drift_baseline"}
    )


class ChecksumMismatch(Exception):
    """Raised when a model artifact's live hash differs from its stored checksum."""


def verify_checksums(model_dir: Path) -> list[str]:
    """
    Verify SHA-256 checksums for all model artifacts in *model_dir*.

    For each artifact discovered by find_artifacts(), looks for a
    corresponding ``<stem>_checksum.txt`` file written by generate_checksums.py.
    Compares the stored hash against a freshly computed one.

    Returns:
        List of artifact stem names that passed verification (empty = nothing
        to verify, which is treated as a warning by the caller).

    Raises:
        FileNotFoundError: if a checksum side-file is missing for an artifact.
        ChecksumMismatch: if any live hash does not match the stored hash.
            The exception message names every failing artifact so the caller
            can surface them all in one shot rather than stopping at the first.
    """
    artifacts = find_artifacts(model_dir)
    if not artifacts:
        return []

    failures: list[str] = []
    verified: list[str] = []

    for src in artifacts:
        checksum_file = model_dir / f"{src.stem}_checksum.txt"
        if not checksum_file.exists():
            raise FileNotFoundError(
                f"Checksum file missing for {src.name}: expected {checksum_file}"
            )
        stored = checksum_file.read_text().strip()
        live = compute_sha256(src)
        if live != stored:
            failures.append(f"  {src.name}: stored={stored[:16]}...  live={live[:16]}...")
        else:
            verified.append(src.stem)

    if failures:
        raise ChecksumMismatch(
            "Checksum verification failed for the following artifacts:\n" + "\n".join(failures)
        )

    return verified


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SHA-256 checksum files for model artifacts."
    )
    parser.add_argument(
        "--model-dir",
        default="models/",
        help="Path to models directory (default: models/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without writing anything.",
    )
    args = parser.parse_args()

    base = Path(args.model_dir)
    if not base.exists():
        print(f"[ERROR] Model directory not found: {base}", file=sys.stderr)
        sys.exit(1)

    artifacts = find_artifacts(base)
    if not artifacts:
        print(f"[WARN ] No model artifacts found in {base}", file=sys.stderr)
        sys.exit(0)

    generated = 0
    for src in artifacts:
        sha256 = compute_sha256(src)
        checksum_file = base / f"{src.stem}_checksum.txt"
        if args.dry_run:
            print(f"[DRY  ] {src.name} → {checksum_file.name}  ({sha256[:16]}...)")
        else:
            checksum_file.write_text(sha256 + "\n")
            print(f"[OK   ] {src.name} → {checksum_file.name}  ({sha256[:16]}...)")
        generated += 1

    action = "Would generate" if args.dry_run else "Generated"
    print(f"\n{action} {generated} checksum file(s) in {base}")


if __name__ == "__main__":
    main()
