"""Audit a completed LSUI checkpoint on the fixed source Tune subset."""

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from datasets.manifest import read_csv, write_csv
from datasets.paired import PairedImages
from experiments.e1_report import deliver
from prtta.candidate import Candidate, apply_candidate, physics_directions, random_direction, snapshot
from prtta.metrics import SSIM, psnr
from prtta.physics import estimate, linear_to_srgb, loss
from prtta.run import (
    checkpoint_path, configure, load_config, load_static, run_info, sha_file, stable_seed, tensors,
    write_json,
)


ARMS = ("static", "physical", "random", "wrong_transmission", "wrong_background")


def fixed_subset(config, count):
    dataset = PairedImages(config, "tune")
    dataset.rows.sort(key=lambda row: (stable_seed(config["seed"], "e1_sample", row["image_id"]), row["image_id"]))
    dataset.rows = dataset.rows[:count]
    return dataset


def prepare_backgrounds(dataset, device, config):
    backgrounds = {}
    for index, row in enumerate(dataset.rows):
        inp = dataset[index][0][None]
        image, _ = tensors((inp, inp), device)
        backgrounds[row["image_id"]] = estimate(image, config["physics"]).background
    return backgrounds


def donor_id(row, rows, seed):
    options = [other["image_id"] for other in rows if other["group_id"] != row["group_id"]]
    if not options:
        raise ValueError("Wrong-background control needs at least two distinct Tune groups")
    return options[stable_seed(seed, "e1_wrong_background", row["image_id"]) % len(options)]


def output_gradients(image, reference, static, hypothesis, epsilon):
    recovery = static.detach().requires_grad_(True)
    g_phy, = torch.autograd.grad(loss(image, recovery, hypothesis, epsilon), recovery)
    g_ref, = torch.autograd.grad((recovery - reference).square().mean(), recovery)
    phy_norm = torch.linalg.vector_norm(g_phy.double())
    ref_norm = torch.linalg.vector_norm(g_ref.double())
    cosine = (g_phy.double() * g_ref.double()).sum() / (phy_norm * ref_norm + 1e-24)
    return g_phy, float(cosine), float(phy_norm)


def audit_one(model, image, reference, image_id, group, background_id, backgrounds, ssim, config):
    start = time.perf_counter()
    settings = config["physics"]
    epsilon = settings["charbonnier_eps"]
    hypothesis = estimate(image, settings)
    wrong_t = replace(hypothesis, transmission=(1 - hypothesis.transmission).clamp(*settings["t_clip"]))
    wrong_b = replace(hypothesis, background=backgrounds[background_id])
    eligible = float(hypothesis.confidence.mean()) >= settings["min_confidence_mean"]
    with torch.no_grad():
        static = model(image)
    g_phy, cosine, output_norm = output_gradients(image, reference, static, hypothesis, epsilon)
    parameters = tuple(model.adapter_parameters())
    initial = snapshot(parameters)
    g_main, g_t, g_b = physics_directions(model, image, (hypothesis, wrong_t, wrong_b), epsilon)
    gradients = {
        "physical": g_main, "random": random_direction(g_main, stable_seed(config["seed"], "e1_random", image_id)),
        "wrong_transmission": g_t, "wrong_background": g_b,
    }
    candidates = {"static": Candidate(static, 0.0, 0.0, 0.0, 0.0, "static")}
    runtimes = {"static": 0.0}
    for arm, direction in gradients.items():
        candidate_start = time.perf_counter()
        candidates[arm] = apply_candidate(
            model, image, static, parameters, initial, direction, config["candidate"], eligible,
        )
        runtimes[arm] = (time.perf_counter() - candidate_start) * 1000
    hypotheses = dict(static=hypothesis, physical=hypothesis, random=hypothesis, wrong_transmission=wrong_t, wrong_background=wrong_b)
    with torch.no_grad():
        pixel_steps = config["e1"]["output_probe_pixel_steps"]
        normalized = g_phy / (g_phy.square().mean().sqrt() + 1e-12)
        probes = [(static - (step / 255) * normalized).clamp(0, 1) if eligible else static for step in pixel_steps]
        predictions = torch.cat([*(candidates[arm].output for arm in ARMS), *probes], dim=0)
        reference = reference.expand_as(predictions)
        p, s = psnr(predictions, reference).tolist(), ssim(predictions, reference).tolist()
        canonical_before = float(loss(image, static, hypothesis, epsilon))
        t = hypothesis.transmission
        t_mean = float(t.mean())
        low_t = float((t < config["e1"]["low_transmission_threshold"]).float().mean())
        t_shift = float((t - wrong_t.transmission).abs().mean())
        b_shift = float((hypothesis.background - wrong_b.background).abs().mean())
        records = []
        for i, arm in enumerate(ARMS):
            candidate = candidates[arm]
            phy = hypotheses[arm]
            records.append(dict(
                image_id=image_id, group_id=group, condition=arm,
                psnr_static=p[0], psnr_candidate=p[i], delta_psnr=p[i] - p[0],
                ssim_static=s[0], ssim_candidate=s[i], delta_ssim=s[i] - s[0],
                gradient_cosine=cosine, physics_loss_before=canonical_before,
                physics_loss_after=float(loss(image, candidate.output, hypothesis, epsilon)),
                condition_loss_before=float(loss(image, static, phy, epsilon)),
                condition_loss_after=float(loss(image, candidate.output, phy, epsilon)),
                transmission_mean=t_mean, transmission_low_ratio=low_t,
                confidence_mean=float(hypothesis.confidence.mean()),
                gradient_norm=candidate.gradient_norm, output_gradient_norm=output_norm,
                adapter_step_size=candidate.step, adapter_update_l2=candidate.step * candidate.direction_norm,
                output_drift=candidate.drift, candidate_valid=candidate.valid,
                status=candidate.status, candidate_application_ms=runtimes[arm],
                wrong_transmission_abs_shift=t_shift, wrong_background_abs_shift=b_shift, background_donor=background_id,
            ))
        probe_rows = [
            dict(
                image_id=image_id, group_id=group, step_px=step, eta=step / 255,
                psnr_static=p[0], psnr_probe=pp, delta_psnr=pp - p[0],
                ssim_static=s[0], ssim_probe=ss, delta_ssim=ss - s[0],
                gradient_cosine=cosine, eligible=eligible,
            )
            for step, pp, ss in zip(pixel_steps, p[len(ARMS):], s[len(ARMS):])
        ]
        example = (image, static, candidates["physical"].output, reference[:1], t, hypothesis.confidence,
                   linear_to_srgb(hypothesis.background).expand_as(image))
    elapsed = (time.perf_counter() - start) * 1000
    for record in records:
        record["runtime_ms"] = elapsed
    return records, probe_rows, example


def require_completed(config):
    latest = torch.load(checkpoint_path(config, "_latest"), map_location="cpu", weights_only=True)
    total = config["static"]["train"]["epochs"]
    if latest["epoch"] != total or len(latest["history"]) != total:
        raise RuntimeError(f"Formal E1 requires all {total} training epochs")
    return max(latest["history"], key=lambda row: row["psnr_tune"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v1.yaml")
    parser.add_argument("--split", choices=["tune"], default="tune")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--freeze-subset", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, args.seed)
    settings = config["e1"]
    count = settings["max_images"] if args.max_images is None else args.max_images
    if not 2 <= count <= settings["max_images"]:
        parser.error(f"--max-images must be within 2..{settings['max_images']}")
    dataset = fixed_subset(config, count)
    output = Path(settings["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if count != settings["max_images"]:
        output = output.parent / "exploratory" / f"n{count}" / output.name
    subset_file = output.with_name("e1_fixed_subset.csv")
    subset = [
        dict(image_id=row["image_id"], group_id=row["group_id"],
             wrong_background_donor=donor_id(row, dataset.rows, config["seed"]))
        for row in dataset.rows
    ]
    if len(subset) < count:
        print(f"Tune contains only {len(subset)} images; proceeding with the available subset.")
    if subset_file.is_file():
        if read_csv(subset_file) != subset:
            raise RuntimeError(f"Frozen subset differs from the selected data: {subset_file}")
    else:
        write_csv(subset_file, subset, list(subset[0]))
    print(f"Fixed Tune subset: {len(subset)} images at {subset_file}", flush=True)
    if args.freeze_subset:
        return
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = settings["cublas_workspace_config"]
    torch.use_deterministic_algorithms(settings["deterministic_algorithms"])
    device = configure(config)
    expected = require_completed(config)
    model, state = load_static(config, device)
    if state["epoch"] != expected["epoch"] or state["best_psnr"] != expected["psnr_tune"]:
        raise RuntimeError("Static checkpoint is not the maximum completed Tune PSNR checkpoint")
    model.freeze_for_adaptation()
    original_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    ssim = SSIM().to(device, memory_format=torch.channels_last)
    dataset.prepare_cache(config["data"]["manifest_workers"])
    backgrounds = prepare_backgrounds(dataset, device, config)
    rows, probes, failures = [], [], []
    replay_output = None
    start = time.perf_counter()
    for index, selection in enumerate(subset):
        inp, target, image_id, group = dataset[index]
        image, reference = tensors((inp[None], target[None]), device)
        records, probe_rows, example = audit_one(
            model, image, reference, image_id, group, selection["wrong_background_donor"],
            backgrounds, ssim, config,
        )
        rows.extend(records)
        probes.extend(probe_rows)
        if index == 0:
            replay_output = example[2].cpu().clone()
        physical = records[1]
        delta = physical["delta_psnr"]
        if delta < 0 and (len(failures) < settings["failure_examples"] or delta < failures[-1][0]):
            failures.append((delta, image_id, [value.squeeze(0).cpu() for value in example]))
            failures.sort(key=lambda item: item[0])
            del failures[settings["failure_examples"]:]
        if (index + 1) % 25 == 0 or index + 1 == len(subset):
            print(f"Audited {index + 1}/{len(subset)} images", flush=True)
    inp, target, image_id, group = dataset[0]
    image, reference = tensors((inp[None], target[None]), device)
    repeated, _, replay_images = audit_one(
        model, image, reference, image_id, group, subset[0]["wrong_background_donor"],
        backgrounds, ssim, config,
    )
    if not torch.equal(replay_output, replay_images[2].cpu()):
        raise RuntimeError("The first real image did not replay exactly after the full audit")
    excluded = {"runtime_ms", "candidate_application_ms"}
    for before, after in zip(rows[:len(ARMS)], repeated):
        if any(before[key] != after[key] for key in before.keys() - excluded):
            raise RuntimeError("The first real image's candidate statistics changed during replay")
    if any(not torch.equal(value, model.state_dict()[name].cpu()) for name, value in original_state.items()):
        raise RuntimeError("The model state was not restored exactly after the audit")
    write_csv(output, rows, list(rows[0]))
    write_csv(output.with_name("e1_output_probes.csv"), probes, list(probes[0]))
    info = dict(
        run_info(config, device), checkpoint_sha256=sha_file(checkpoint_path(config)),
        fixed_subset_sha256=sha_file(subset_file), checkpoint_epoch=state["epoch"],
        tune_psnr=state["best_psnr"], elapsed_seconds=time.perf_counter() - start,
        peak_vram_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        training_epochs=config["static"]["train"]["epochs"],
        real_image_replay_exact=True, full_model_restoration_exact=True,
    )
    report = Path(settings["report"]) if count == settings["max_images"] else output.parent / "E1_findings.md"
    deliver(output.parent, report, rows, probes, failures, info)
    write_json(output.with_name("e1_run.json"), info)
    print(f"Completed E1: {output} and {report}", flush=True)


if __name__ == "__main__":
    main()
