#!/usr/bin/env bash
# vendor.sh — Copy MTGFLOW.py and NF.py from a local MTGFLOW clone into _upstream/
# and apply the two mechanical patches documented in ATTRIBUTION.md.
#
# Usage:
#   ./vendor.sh /path/to/MTGFLOW
#
# Where /path/to/MTGFLOW is a clone of https://github.com/zqhang/MTGFLOW with
# the standard layout (a top-level `models/` directory containing MTGFLOW.py
# and NF.py).
#
# Idempotent: re-running overwrites the vendored files. Verify the diff against
# upstream after running (see ATTRIBUTION.md).

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /path/to/MTGFLOW" >&2
    exit 1
fi

SRC="$1"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Sanity check: the source must look like the MTGFLOW repo.
if [[ ! -f "$SRC/models/MTGFLOW.py" ]] || [[ ! -f "$SRC/models/NF.py" ]]; then
    echo "ERROR: $SRC does not look like a MTGFLOW clone." >&2
    echo "       Expected $SRC/models/MTGFLOW.py and $SRC/models/NF.py to exist." >&2
    exit 1
fi

echo "Vendoring MTGFLOW from: $SRC"
echo "Vendoring MTGFLOW into: $DEST"

# Copy NF.py byte-identical.
cp "$SRC/models/NF.py" "$DEST/NF.py"
echo "  - NF.py copied (byte-identical)"

# Copy MTGFLOW.py with two mechanical patches.
#   1.  from models.NF import MAF   ->   from .NF import MAF
#   2.  delete the dead `from turtle import forward, shape` line so the file
#       imports on Pythons without tkinter (Amazon Linux 2023's 3.11 build).
sed 's|^from models\.NF import MAF$|from .NF import MAF|' \
    "$SRC/models/MTGFLOW.py" > "$DEST/MTGFLOW.py"
sed -i '/^from turtle import forward, shape$/d' "$DEST/MTGFLOW.py"
echo "  - MTGFLOW.py copied (MAF import patched, dead turtle import stripped)"

# Verify the MAF patch fired exactly once (otherwise upstream may have moved the import).
if ! grep -qx "from \.NF import MAF" "$DEST/MTGFLOW.py"; then
    echo "WARNING: expected patched import 'from .NF import MAF' not found in MTGFLOW.py." >&2
    echo "         Upstream may have changed the import line. Inspect manually." >&2
    exit 2
fi

# Verify the turtle strip fired (no turtle imports should remain at column 0).
if [[ $(grep -c '^from turtle import' "$DEST/MTGFLOW.py") -ne 0 ]]; then
    echo "WARNING: 'from turtle import' still present in MTGFLOW.py after vendoring." >&2
    echo "         Upstream may have changed the line; inspect manually." >&2
    exit 2
fi

echo ""
echo "Done. Verify with:"
echo "  diff -u <(sed 's|^from models\\.NF import MAF\$|from .NF import MAF|' \\"
echo "             $SRC/models/MTGFLOW.py) \\"
echo "          $DEST/MTGFLOW.py"
echo "  diff -u $SRC/models/NF.py $DEST/NF.py"
echo ""
echo "Both diffs should be empty."