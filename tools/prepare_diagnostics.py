"""Freeze whole Utility Fit groups for Pilot and Fixed diagnostics."""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.manifest import duplicate_links, read_csv, write_csv
from prtta.run import load_config, sha_file, stable_seed, write_json


def prepare(config_path):
    settings = load_config(config_path)
    base = load_config(settings["base_config"])
    all_rows = read_csv(base["data"]["manifest"])
    groups = defaultdict(list)
    for row in all_rows:
        if row["split"] == "utility_fit":
            groups[row["group_id"]].append(row)
    ordered = sorted(groups, key=lambda key: stable_seed(settings["seed"], "diagnostic_groups", key))
    chosen = {"pilot": [], "fixed": [], "remaining": []}
    for key in ordered:
        phase = "remaining"
        if len(chosen["pilot"]) < settings["pilot_count"]:
            phase = "pilot"
        elif len(chosen["fixed"]) < settings["fixed_count"]:
            phase = "fixed"
        chosen[phase].extend(groups[key])
    fields = list(all_rows[0])
    paths = {
        "pilot": "manifests/lsui_e1b_pilot.csv",
        "fixed": "manifests/lsui_e1b_fixed.csv",
        "remaining": "manifests/lsui_remaining_utility_fit.csv",
    }
    outputs = {paths[k]: v for k, v in chosen.items()}
    outputs["manifests/lsui_diagnostic_consumed.csv"] = chosen["pilot"] + chosen["fixed"]
    outputs["manifests/risk_calibration_sealed.csv"] = [r for r in all_rows if r["split"] == "risk_calib"]
    consumed = {r["image_id"] for r in outputs["manifests/lsui_diagnostic_consumed.csv"]}
    for path, rows in outputs.items():
        contents = [dict(row, split="diagnostic_consumed") if row["image_id"] in consumed else row for row in rows]
        if Path(path).exists():
            if read_csv(path) != contents:
                raise RuntimeError(f"Existing frozen data differs: {path}")
        else:
            write_csv(path, contents, fields)
    valid = [r for r in all_rows if r["valid_pair"].lower() == "true"]
    phase_by_id = {r["image_id"]: r["split"] for r in valid}
    for phase in ("pilot", "fixed"):
        for row in chosen[phase]:
            phase_by_id[row["image_id"]] = phase
    for i, j, _ in duplicate_links(valid):
        assert phase_by_id[valid[i]["image_id"]] == phase_by_id[valid[j]["image_id"]]
    report = dict(
        counts={k: len(v) for k, v in chosen.items()},
        groups={k: len({r["group_id"] for r in v}) for k, v in chosen.items()},
        hashes={path: sha_file(path) for path in outputs},
        original_manifest_sha256=sha_file(base["data"]["manifest"]),
        isolation="No cross-partition exact/corroborated near-duplicate links under the frozen V1 rules",
    )
    write_json("manifests/diagnostic_isolation.json", report)
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e1b_output_matched.yaml")
    prepare(parser.parse_args().config)
