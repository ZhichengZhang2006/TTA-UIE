"""Verify completed E2b records without opening reference images or changing methods."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    root = Path("results/e2b")
    lock = json.loads((root / "gate/opened.json").read_text())
    assert lock["config_sha256"] == sha("configs/e2b_operator_study.yaml")
    assert lock["manifest_sha256"] == sha("manifests/lsui_e2b_gate.csv")
    assert lock["noise_audit_sha256"] == sha("configs/e2b_noise_audit.yaml")
    for name, value in lock["estimator_hashes"].items():
        assert sha(f"checkpoints/e2b/{name}.pt") == value
        history = rows(root / f"training/{name}.csv")
        assert [int(r["epoch"]) for r in history] == list(range(1, 61))
        assert all(math.isfinite(float(r["train_loss"])) for r in history)
    assert sha("checkpoints/source_seed2026.pt") == "479129a20eb1d8bc1a7b0d83ff7ce736e1b0d4152c17ed9be1b8101572bc2533"
    assert sha("manifests/risk_calibration_sealed.csv") == "7587e7187b45cd9ac36cf8d765d5ca2f78edb03d0e32ac2436c8629a46e7db7d"
    gate = rows("manifests/lsui_e2b_gate.csv")
    ids = {r["image_id"] for r in gate}
    group_ids = {r["group_id"] for r in gate}
    source = rows("manifests/manifest.csv")
    assert not group_ids.intersection(r["group_id"] for r in source if r["split"] in ("train", "tune", "risk_calib"))
    for path in ("manifests/lsui_e1b_pilot.csv", "manifests/lsui_e1b_fixed.csv", "manifests/lsui_e3_remaining.csv"):
        assert not group_ids.intersection(r["group_id"] for r in rows(path))
    records = rows(root / "gate/per_image.csv")
    methods = sorted({r["method"] for r in records})
    counts = Counter((r["seed"], r["method"], float(r["target_rms_8bit"])) for r in records)
    assert len(counts) == 3 * len(methods) * 4
    assert all(n == len(gate) for n in counts.values())
    assert {r["seed"] for r in records} == {"2026", "2027", "2028"}
    assert {r["image_id"] for r in records} == ids
    assert len({(r["seed"], r["method"], r["image_id"], r["target_rms_8bit"]) for r in records}) == len(records)
    for r in records:
        for key in ("delta_psnr", "delta_ssim", "runtime_ms", "physics_loss_before", "physics_loss_after"):
            assert math.isfinite(float(r[key])), (key, r["image_id"])
        assert float(r["parameter_delta_l2"]) <= .500001
        if r["valid"] == "True":
            target = float(r["target_rms_8bit"])
            assert abs(float(r["achieved_rms_8bit"]) - target) <= .05 * target + 1e-6
        else:
            assert float(r["delta_psnr"]) == 0 and float(r["achieved_rms_8bit"]) == 0
        if float(r["target_rms_8bit"]) == .25:
            attempts = json.loads(r["search_attempts"])
            assert attempts or r["failure_reason"] in ("zero_or_nonfinite_direction", "parameter_cap")
    run = json.loads((root / "gate/run.json").read_text())
    assert run["full_state_restore_exact"] is True
    result = dict(passed=True, gate_images=len(gate), gate_groups=len(group_ids), methods=methods,
                  gate_rows=len(records), models=len(lock["estimator_hashes"]), epochs_per_model=60,
                  config_sha256=lock["config_sha256"], gate_sha256=lock["manifest_sha256"])
    (root / "quality_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
