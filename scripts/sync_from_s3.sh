#!/usr/bin/env bash
# Pull artifacts from the project S3 bucket to the local working tree.
#
# Usage:
#   scripts/sync_from_s3.sh datasets       # only datasets/ -> data/
#   scripts/sync_from_s3.sh scores         # only scores/   -> results/scores/
#   scripts/sync_from_s3.sh checkpoints    # only checkpoints/ -> results/checkpoints/
#   scripts/sync_from_s3.sh exports        # only exports/  -> results/exports/
#   scripts/sync_from_s3.sh all            # all four prefixes
#   scripts/sync_from_s3.sh --dry-run all  # show what would change, transfer nothing
#
# Idempotent: aws s3 sync only transfers files whose size or mtime changed.
#
# Env overrides:
#   MTAD_S3_BUCKET (default: mtad-platform-imanian-2026)
#   AWS_PROFILE    (default: mtad)
set -euo pipefail

BUCKET="${MTAD_S3_BUCKET:-mtad-platform-imanian-2026}"
PROFILE="${AWS_PROFILE:-mtad}"

DRY=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY="--dryrun"
    shift
fi

if [[ -z "${1:-}" ]]; then
    echo "usage: $0 [--dry-run] datasets|scores|checkpoints|exports|all" >&2
    exit 2
fi
TARGET="$1"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

sync_one() {
    local prefix="$1" local_dir="$2"
    mkdir -p "$ROOT/$local_dir"
    aws --profile "$PROFILE" s3 sync $DRY \
        --exclude "*/.keep" \
        --exclude ".keep" \
        "s3://$BUCKET/$prefix/" "$ROOT/$local_dir/"
}

case "$TARGET" in
    datasets)    sync_one datasets    data ;;
    scores)      sync_one scores      results/scores ;;
    checkpoints) sync_one checkpoints results/checkpoints ;;
    exports)     sync_one exports     results/exports ;;
    all)
        sync_one datasets    data
        sync_one scores      results/scores
        sync_one checkpoints results/checkpoints
        sync_one exports     results/exports
        ;;
    *) echo "unknown target: $TARGET" >&2; exit 2 ;;
esac
