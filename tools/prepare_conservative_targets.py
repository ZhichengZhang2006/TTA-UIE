"""Apply the explicitly conservative candidate exclusion policy without scoring targets."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.manifest import read_csv, scan, write_csv
from prtta.run import write_json


def main():
    links = read_csv("manifests/cross_dataset_matches.csv")
    reviewed = {r["target_id"]: r for r in read_csv("results/duplicates/visual_review.csv")}
    records = []
    flagged = {r["target_id"] for r in links}
    for row in links:
        review = reviewed.get(row["target_id"])
        records.append(dict(
            target_id=row["target_id"], source_id=row["source_id"], matching_rule=row["method"],
            confidence="visual_same_content" if review else "uncalibrated_hash_candidate",
            visual_review_status=review["status"] if review else "pending",
            excluded=True, reason="conservative_candidate_exclusion; candidate status preserved",
        ))
    write_csv("manifests/duplicate_audit_review.csv", records, list(records[0]))
    retained, distribution = [], []
    for dataset in ("UIEB", "EUVP-d", "EUVP-s"):
        rows = scan(Path("/home/user/UnderWaterDataset") / dataset, dataset, 8)
        valid = [r for r in rows if r["valid_pair"]]
        keep = [dict(r, independence_status="pending_scene_audit") for r in valid if r["image_id"] not in flagged]
        retained.extend(keep)
        for split in sorted({r["original_split"] for r in valid}):
            original = [r for r in valid if r["original_split"] == split]
            remaining = [r for r in keep if r["original_split"] == split]
            distribution.append(dict(dataset=dataset, original_split=split, before=len(original),
                                     after=len(remaining), excluded=len(original)-len(remaining),
                                     retained_fraction=len(remaining)/len(original)))
    write_csv("manifests/target_conservative_eligible.csv", retained, list(retained[0]))
    write_csv("manifests/target_exclusion_distribution.csv", distribution, list(distribution[0]))
    write_json("manifests/e2b_target_audit.json", dict(
        flagged_unique_ids=len(flagged), visually_confirmed=len(reviewed),
        pending=len(flagged-reviewed.keys()), conservative_exclusions=len(flagged),
        retained=dict(Counter(r["dataset"] for r in retained)),
        target_quality_scores_computed=False, independence_status="pending_scene_audit",
    ))


if __name__ == "__main__":
    main()
