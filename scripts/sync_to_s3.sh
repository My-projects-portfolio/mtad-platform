#!/usr/bin/env bash
# Push artifacts from the local working tree to the project S3 bucket.
#
# Usage:
#   scripts/sync_to_s3.sh datasets       # data/             -> datasets/
#   scripts/sync_to_s3.sh scores         # results/scores    -> scores/
#   scripts/sync_to_s3.sh checkpoints    # results/checkpoints -> checkpoints/
#   scripts/sync_to_s3.sh exports        # results/exports   -> exports/
#   scripts/sync_to_s3.sh all            # all four prefixes
#   scripts/sync_to_s3.sh --dry-run all  # show what would change, transfer nothing
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
    local local_dir="$1" prefix="$2"
    if [[ ! -d "$ROOT/$local_dir" ]]; then
        echo "skip $local_dir/ (does not exist locally)" >&2
        return 0
    fi
    aws --profile "$PROFILE" s3 sync $DRY \
        --exclude "*/.keep" \
        --exclude ".keep" \
        "$ROOT/$local_dir/" "s3://$BUCKET/$prefix/"
}

case "$TARGET" in
    datasets)    sync_one data                datasets ;;
    scores)      sync_one results/scores      scores ;;
    checkpoints) sync_one results/checkpoints checkpoints ;;
    exports)     sync_one results/exports     exports ;;
    all)
        sync_one data                datasets
        sync_one results/scores      scores
        sync_one results/checkpoints checkpoints
        sync_one results/exports     exports
        ;;
    *) echo "unknown target: $TARGET" >&2; exit 2 ;;
esac
