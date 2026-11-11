"""Flag exact and near-duplicate source/target images for future blind scoring."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.manifest import duplicate_links, read_csv, scan, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("manifests/manifest.csv"))
    parser.add_argument("--target-root", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("manifests/cross_dataset_matches.csv"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    source = [row for row in read_csv(args.source) if row["valid_pair"].lower() == "true"]
    rows = source.copy()
    summary = {}
    for target in args.target_root:
        target = target.resolve()
        dataset = target.name
        target_rows = scan(target, dataset, args.workers)
        valid = [row for row in target_rows if row["valid_pair"]]
        rows.extend(valid)
        summary[dataset] = {
            "root": str(target), "valid_pairs": len(valid), "invalid_pairs": len(target_rows) - len(valid),
        }
    matches = []
    for first, second, method in duplicate_links(rows):
        if first < len(source) <= second:
            left, right = rows[first], rows[second]
            matches.append(dict(
                source_id=left["image_id"], source_split=left["split"],
                target_id=right["image_id"], target_dataset=right["dataset"],
                target_original_split=right["original_split"], method=method,
            ))
    fields = ["source_id", "source_split", "target_id", "target_dataset", "target_original_split", "method"]
    write_csv(args.out, matches, fields)
    for dataset, stats in summary.items():
        selected = [row for row in matches if row["target_dataset"] == dataset]
        stats["target_pairs_flagged"] = len({row["target_id"] for row in selected})
        stats["match_links"] = dict(Counter(row["method"] for row in selected))
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
