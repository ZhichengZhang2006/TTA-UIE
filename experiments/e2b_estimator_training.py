"""Train source-only operator estimators for all preregistered seeds."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F

from datasets.manifest import read_csv, write_csv
from datasets.paired import PairedImages
from prtta.metrics import SSIM
from prtta.operator_estimator import OperatorEstimator, estimation_loss, reproject, synthesize_batch
from prtta.physics import estimate, srgb_to_linear
from prtta.run import atomic_save, configure, load_config, load_static, loader, run_info, sha_file, stable_seed, tensors, write_json
from prtta.synthetic import DualPhysics, synthesize


def operator_path(mode, seed):
    return Path("checkpoints/e2b") / f"{mode}_{seed}.pt"


def make_estimator(config, mode, seed, device):
    net = OperatorEstimator(mode, config["estimator_channels"], config["estimator_resolution"]).to(device)
    state = torch.load(operator_path(mode, seed), map_location="cpu", weights_only=True)
    assert state["config_sha256"] == sha_file("configs/e2b_operator_study.yaml")
    net.load_state_dict(state["model"])
    return net.eval().requires_grad_(False)


def fit_effective(image, reference, options, physics):
    """Source-label teacher only, with bounded 8x8 fields and explicit priors."""
    prior = estimate(image, physics)
    size = options["grid"]
    prior_t = prior.transmission.expand(-1, 3, -1, -1)
    initial = F.interpolate(prior_t, (size, size), mode="area")
    logit = torch.logit(((initial - 0.05) / 0.93).clamp(1e-4, 1-1e-4))
    fields = torch.cat((logit, logit), 1).detach().requires_grad_(True)
    background = torch.logit(prior.background.clamp(1e-4, 1-1e-4)).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([fields, background], lr=options["fit_lr"])
    for _ in range(options["fit_steps"]):
        value = 0.05 + 0.93 * F.interpolate(fields, image.shape[-2:], mode="bilinear", align_corners=False).sigmoid()
        h = DualPhysics(value[:, :3], value[:, 3:], background.sigmoid(), prior.confidence)
        tv = (value[..., 1:, :] - value[..., :-1, :]).abs().mean() + (value[..., 1:] - value[..., :-1]).abs().mean()
        loss = ((reproject(reference, h) - srgb_to_linear(image)).abs().mean()
                + options["prior_weight"] * (value - prior_t.repeat(1, 2, 1, 1)).abs().mean()
                + options["tv_weight"] * tv
                + options["background_prior_weight"] * (h.background - prior.background).square().mean())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return fields.detach(), background.detach()


def prepare_effective_targets(config, base, device):
    path = Path("checkpoints/e2b/effective_teacher.pt")
    if path.exists():
        return path
    data = PairedImages(base, "train")
    data.prepare_cache(base["data"]["manifest_workers"])
    fields, backgrounds, ids = [], [], []
    for batch in loader(base, data, device, config["batch_size"], workers=4):
        image, reference = tensors(batch, device)
        field, background = fit_effective(image, reference, config["effective"], base["physics"])
        fields.append(field.cpu()); backgrounds.append(background.cpu()); ids.extend(batch[2])
        if len(ids) % 160 == 0:
            print(f"Effective teacher {len(ids)}/{len(data)}", flush=True)
    atomic_save(path, dict(fields=torch.cat(fields), backgrounds=torch.cat(backgrounds), ids=ids,
                           config_sha256=sha_file("configs/e2b_operator_study.yaml")))
    return path


@torch.no_grad()
def validation(net, data, base, config, device, mode, backbone=None):
    values = []
    for index, row in enumerate(data.rows):
        inp, ref, *_ = data[index]
        image, reference = tensors((inp[None], ref[None]), device)
        if mode in ("shared", "dual"):
            image, truth, _ = synthesize(reference, base["diagnostic"]["e2"], stable_seed(2026, row["image_id"], "sspe_val"), mode == "dual")
            predicted = net(image)
            values.append(float(estimation_loss(predicted, truth, reference, image, config["loss_weights"])))
        else:
            static = backbone(image)
            output = net(image, static)
            if mode == "residual":
                values.append(float((output - reference).abs().mean()))
            else:
                values.append(float((reproject(reference, output) - srgb_to_linear(image)).abs().mean()))
    return sum(values) / len(values)


def train_one(config, base, device, mode, seed, teacher=None):
    torch.manual_seed(seed)
    base["seed"] = seed
    net = OperatorEstimator(mode, config["estimator_channels"], config["estimator_resolution"]).to(device)
    parameters = sum(p.numel() for p in net.parameters())
    assert parameters < 2_000_000
    optimizer = torch.optim.AdamW(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    train = PairedImages(base, "train")
    train.prepare_cache(base["data"]["manifest_workers"])
    development = PairedImages(base, "utility_fit")
    development.rows = read_csv(config["development_manifest"])
    development.prepare_cache(base["data"]["manifest_workers"])
    backbone = load_static(base, device, Path(config["checkpoint"]))[0].requires_grad_(False) if mode in ("effective", "residual") else None
    id_index = {key: i for i, key in enumerate(teacher["ids"])} if teacher else None
    path = operator_path(mode, seed)
    latest = path.with_name(path.stem + "_latest.pt")
    config_hash = sha_file("configs/e2b_operator_study.yaml")
    first, best, history = 0, float("inf"), []
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=True)
        assert state["config_sha256"] == config_hash
        net.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        first, best, history = state["epoch"], state["best"], state["history"]
    root = Path("results/e2b/training")
    root.mkdir(parents=True, exist_ok=True)
    amp = device.type == "cuda" and config["amp"] == "bf16"
    metric = SSIM().to(device)
    static_cache, cache_checks, cache_enabled = {}, 0, True
    with (root / f"{mode}_{seed}_generation.jsonl").open("a", encoding="utf-8") as generation:
        for epoch in range(first, config["epochs"]):
            began = time.perf_counter()
            net.train()
            losses = []
            batches = loader(base, train, device, config["batch_size"], True, stable_seed(seed, mode, epoch), workers=4)
            for index, batch in enumerate(batches):
                image, reference = tensors(batch, device)
                if mode in ("shared", "dual"):
                    generation_seed = stable_seed(seed, mode, epoch, index, "training_generation")
                    image, truth, metadata = synthesize_batch(reference, base["diagnostic"]["e2"], generation_seed, mode == "dual")
                    generation.write(json.dumps(dict(epoch=epoch, batch=index, image_ids=list(batch[2]), group_ids=list(batch[3]), **metadata)) + "\n")
                    static = None
                else:
                    with torch.no_grad():
                        keys = [(len(batch[2]), key) for key in batch[2]]
                        hit = cache_enabled and all(key in static_cache for key in keys)
                        if hit:
                            static = torch.cat([static_cache[key] for key in keys]).to(device, memory_format=torch.channels_last)
                            if cache_checks < 2:
                                fresh = backbone(image)
                                if not torch.equal(static, fresh):
                                    print("Static cache disabled: bitwise equivalence failed", flush=True)
                                    cache_enabled = False
                                    static_cache.clear()
                                    static = fresh
                                cache_checks += 1
                        else:
                            static = backbone(image)
                            if cache_enabled:
                                cached = static.cpu()
                                for key, value in zip(keys, cached):
                                    static_cache[key] = value.unsqueeze(0).clone()
                    if mode == "effective":
                        indices = [id_index[key] for key in batch[2]]
                        field = teacher["fields"][indices].to(device)
                        fields = 0.05 + 0.93 * F.interpolate(field, image.shape[-2:], mode="bilinear", align_corners=False).sigmoid()
                        truth = DualPhysics(fields[:, :3], fields[:, 3:], teacher["backgrounds"][indices].to(device).sigmoid(),
                                            torch.ones_like(fields[:, :1]))
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    predicted = net(image, static)
                    if mode == "residual":
                        loss = (predicted - reference).abs().mean() + 0.2 * (1 - metric(predicted, reference).mean())
                    else:
                        loss = estimation_loss(predicted, truth, reference, image, config["loss_weights"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
            score = validation(net.eval(), development, base, config, device, mode, backbone)
            record = dict(epoch=epoch+1, train_loss=sum(losses)/len(losses), validation=score, seconds=time.perf_counter()-began)
            history.append(record)
            improved = score < best
            best = min(best, score)
            state = dict(model=net.state_dict(), epoch=epoch+1, best=best, config_sha256=config_hash,
                         mode=mode, seed=seed, parameters=parameters)
            if improved:
                atomic_save(path, state)
            atomic_save(latest, dict(state, optimizer=optimizer.state_dict(), history=history))
            write_csv(root / f"{mode}_{seed}.csv", history, list(record))
            print(mode, seed, record, flush=True)
    write_json(root / f"{mode}_{seed}_run.json", dict(run_info(base, device), estimator_parameters=parameters,
        checkpoint_sha256=sha_file(path), config_sha256=config_hash, selection="development loss only",
        static_cache_enabled=cache_enabled if backbone is not None else False, static_cache_bitwise_checks=cache_checks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effective", action="store_true")
    args = parser.parse_args()
    from experiments.e2b_factorial_error import setup
    config, base, device, _ = setup()
    teacher = None
    modes = ("shared", "dual")
    if args.effective:
        teacher = torch.load(prepare_effective_targets(config, base, device), map_location="cpu", weights_only=True)
        modes = ("effective", "residual")
    for mode in modes:
        for seed in config["seeds"]:
            train_one(config, base, device, mode, seed, teacher)


if __name__ == "__main__":
    main()
