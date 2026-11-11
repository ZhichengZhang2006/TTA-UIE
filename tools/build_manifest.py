"""Build the grouped LSUI manifest from existing paired public data."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from datasets.manifest import FIELDS, assign_splits, scan, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v1.yaml", type=Path)
    parser.add_argument("--dataset", default="LSUI")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check-pairs", action="store_true", help="Pair checks always run.")
    parser.add_argument("--hash", action="store_true", help="Hashing always runs.")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = config["data"]
    root = (args.root or Path(data["root"])).resolve()
    output = args.out or Path(data["manifest"])
    rows = scan(root, args.dataset, data["manifest_workers"])
    counts, groups, links = assign_splits(rows, data["split_targets"], config["seed"])
    write_csv(output, rows, FIELDS)
    report = {
        "root": str(root), "entries": len(rows),
        "invalid": dict(Counter(row["error"] for row in rows if not row["valid_pair"])),
        "original": dict(Counter(row["original_split"] for row in rows)),
        "assigned": dict(counts), "groups": len(groups),
        "groups_with_duplicates": sum(len(group) > 1 for group in groups),
        "largest_group": max(map(len, groups), default=0),
        "links": dict(Counter(kind for _, _, kind in links)),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not all(counts[split] for split in data["split_targets"]):
        raise RuntimeError("Insufficient valid independent groups for all four splits")


if __name__ == "__main__":
    main()
