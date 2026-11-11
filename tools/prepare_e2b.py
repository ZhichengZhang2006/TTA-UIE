"""Reserve whole groups for a once-only E2b acceptance evaluation."""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.manifest import read_csv, write_csv
from prtta.run import load_config, sha_file, stable_seed, write_json


def main():
    config = load_config("configs/e2b_operator_study.yaml")
    remaining = read_csv("manifests/lsui_remaining_utility_fit.csv")
    grouped = defaultdict(list)
    for row in remaining:
        grouped[row["group_id"]].append(row)
    gate, future = [], []
    for group in sorted(grouped, key=lambda key: stable_seed(2026, "e2b_gate_groups", key)):
        (gate if len(gate) < config["gate_count"] else future).extend(grouped[group])
    for name, rows in (("lsui_e2b_gate", gate), ("lsui_e3_remaining", future)):
        contents = [dict(row, split="e2b_gate" if name == "lsui_e2b_gate" else "utility_fit") for row in rows]
        path = Path("manifests") / f"{name}.csv"
        if path.exists():
            assert read_csv(path) == contents
        else:
            write_csv(path, contents, list(contents[0]))
    all_rows = read_csv("manifests/manifest.csv")
    used = read_csv("manifests/lsui_diagnostic_consumed.csv")
    partitions = [
        {r["group_id"] for r in all_rows if r["split"] == split}
        for split in ("train", "tune", "risk_calib")
    ] + [{r["group_id"] for r in rows} for rows in (used, gate, future)]
    for i, left in enumerate(partitions):
        assert all(not left.intersection(right) for right in partitions[i+1:])
    write_json("manifests/e2b_isolation.json", dict(
        gate=len(gate), future=len(future), gate_groups=len(partitions[-2]),
        hashes={name: sha_file("manifests/" + name + ".csv") for name in ("lsui_e2b_gate", "lsui_e3_remaining")},
        risk_calibration_sha256=sha_file("manifests/risk_calibration_sealed.csv"),
    ))
    print(f"E2b gate {len(gate)} images, future Utility Fit {len(future)} images")


if __name__ == "__main__":
    main()
