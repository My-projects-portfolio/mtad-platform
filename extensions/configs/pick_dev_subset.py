#!/usr/bin/env python3
"""
pick_dev_subset.py

Select 10 representative files from TSB-AD-M-Eva.csv, one per source family,
preferring smaller files (smallest training-index token) for faster iteration.

Writes a single-column CSV (header: file_name) that the runner accepts.

Usage:
    python pick_dev_subset.py <eva_csv_path> <output_csv_path>

Example:
    python pick_dev_subset.py \
        TSB-AD/Datasets/File_List/TSB-AD-M-Eva.csv \
        extensions/configs/dev_subset.csv
"""

import sys
from pathlib import Path
import pandas as pd


# Source families to include: 5 canonical SOTA-comparison anchors + 5 for
# domain/dimensionality diversity. Edit this list if you want a different mix.
FAMILIES = [
    "SMD", "MSL", "SMAP", "SWaT", "PSM",            # canonical comparison anchors
    "MITDB", "GECCO", "CATSv2", "OPPORTUNITY", "Exathlon",  # diversity
]


def train_size(filename: str) -> int:
    """Extract the train_index from TSB-AD filename schema.

    Filenames look like: 057_SMD_id_1_Facility_tr_4529_1st_4629.csv
    The training-index token is the 3rd-from-last after splitting on '_'.
    Smaller train_index ≈ smaller dataset ≈ faster to run.
    """
    try:
        return int(filename.split(".")[0].split("_")[-3])
    except (ValueError, IndexError):
        return 10**12  # sort un-parseable filenames to the end


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    eva_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    if not eva_path.exists():
        print(f"ERROR: {eva_path} does not exist", file=sys.stderr)
        sys.exit(1)

    eva_files = pd.read_csv(eva_path)["file_name"].tolist()
    print(f"Loaded {len(eva_files)} files from {eva_path}")

    selected = []
    found = []
    missing = []

    for family in FAMILIES:
        # Match files where the token appears between underscores (avoids "MSL"
        # accidentally matching "SLMNOP" or similar).
        matches = [f for f in eva_files if f"_{family}_" in f]
        if not matches:
            missing.append(family)
            continue
        chosen = sorted(matches, key=train_size)[0]
        selected.append(chosen)
        found.append((family, chosen, train_size(chosen)))

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"file_name": selected}).to_csv(out_path, index=False)

    # Report
    print()
    print(f"Selected {len(selected)} files (one per available family):")
    print(f"{'Family':<14} {'Train_idx':>10}  File")
    print("-" * 100)
    for family, fn, ts in found:
        print(f"{family:<14} {ts:>10}  {fn}")

    if missing:
        print()
        print(f"WARNING: families not found in Eva list: {missing}")
        print("Edit FAMILIES in this script to substitute alternatives.")

    print()
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
