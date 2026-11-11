import argparse
import math
import time
from pathlib import Path

import torch

from datasets.manifest import write_csv
from prtta.metrics import SSIM, psnr
from prtta.model import build_model
from prtta.run import (
    atomic_save, checkpoint_path, configure, load_config, loader, make_dataset,
    run_info, stable_seed, tensors, write_json,
)


def objective(output, reference, ssim, options):
    return (
        options["loss_l1"] * (output.float() - reference).abs().mean()
        + options["loss_ssim"] * (1 - ssim(output, reference).mean())
    )


def amp_enabled(options, device):
    return options["amp"] == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported()


def learning_rate(step, total, warm, options):
    if step < warm:
        return options["lr"] * (step + 1) / warm
    progress = (step - warm) / max(1, total - warm - 1)
    return options["min_lr"] + 0.5 * (options["lr"] - options["min_lr"]) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def score(model, batches, device, ssim, amp=False):
    model.eval()
    scores = []
    for batch in batches:
        inp, ref = tensors(batch, device)
        with torch.autocast(device.type, enabled=amp, dtype=torch.bfloat16):
            out = model(inp)
        scores.append(torch.stack((psnr(out, ref), ssim(out, ref)), dim=1))
    return torch.cat(scores).mean(0).tolist()


def smoke(config, device, model, optimizer, ssim, options, info):
    settings = config["smoke"]
    dataset = make_dataset(config, "train", limit=settings["batch_size"])
    batch = next(iter(loader(config, dataset, device, settings["batch_size"], workers=0)))
    inp, ref = tensors(batch, device)
    amp = amp_enabled(options, device)
    losses, gradients = [], []
    for _ in range(settings["steps"]):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=amp, dtype=torch.bfloat16):
            loss = objective(model(inp), ref, ssim, options)
        loss.backward()
        gradients.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), options["grad_clip"], error_if_nonfinite=True)))
        losses.append(float(loss.detach()))
        optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(inp)
        final_loss = float(objective(prediction, ref, ssim, options))
        path = checkpoint_path(config, "_smoke")
        atomic_save(path, {"model": model.state_dict(), "config": config, "manifest_sha256": info["manifest_sha256"]})
        restored = build_model(config, device).eval()
        restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
        exact = torch.equal(prediction, restored(inp))
    if not (final_loss < losses[0] and min(gradients) > 0 and exact):
        raise RuntimeError(f"Smoke failed: losses={losses}, final={final_loss}, gradients={gradients}, exact={exact}")
    result = dict(
        info, batch_ids=list(batch[2]), bf16=amp, training_losses=losses,
        final_fp32_loss=final_loss, gradient_norms=gradients, reload_exact=exact,
        peak_vram_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    )
    write_json(Path(config["paths"]["results"]) / "smoke.json", result)
    print(f"Smoke passed: loss {losses[0]:.6f} -> {final_loss:.6f}; exact reload={exact}", flush=True)


def fit(config, device, model, optimizer, ssim, options, info, resume):
    train = make_dataset(config, "train", augment=True)
    tune = make_dataset(config, "tune")
    tune_batches = loader(config, tune, device, options["batch_size"])
    accum = options["grad_accum"]
    batches_per_epoch = math.ceil(len(train) / options["batch_size"])
    steps_per_epoch = math.ceil(batches_per_epoch / accum)
    total_steps = steps_per_epoch * options["epochs"]
    warm_steps = steps_per_epoch * options["warmup_epochs"]
    latest = checkpoint_path(config, "_latest")
    best_path = checkpoint_path(config)
    best, first_epoch, history = -math.inf, 0, []
    if resume:
        state = torch.load(latest, map_location="cpu", weights_only=True)
        for key in ("config", "manifest_sha256", "training_sources_sha256"):
            if state[key] != info[key]:
                raise ValueError(f"Cannot resume because {key} changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        best, first_epoch, history = state["best_psnr"], state["epoch"], state["history"]
    elif latest.exists() or best_path.exists():
        raise FileExistsError(f"Existing seed {config['seed']} run. Use --resume to continue.")
    amp = amp_enabled(options, device)
    if options["amp"] == "bf16" and device.type == "cuda" and not amp:
        raise RuntimeError("Configured BF16 is unavailable on the selected GPU")
    results = Path(config["paths"]["results"])
    name = f"source_seed{config['seed']}"
    write_json(results / f"{name}_run.json", dict(info, bf16=amp))
    if history:
        write_csv(results / f"{name}_epochs.csv", history, list(history[0]))
    print(f"Train {len(train)} / tune {len(tune)}; BF16={amp}; {steps_per_epoch} updates per epoch", flush=True)
    for epoch in range(first_epoch, options["epochs"]):
        start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = torch.zeros((), device=device)
        batches = loader(
            config, train, device, options["batch_size"], shuffle=True,
            seed=stable_seed(config["seed"], "train", epoch),
        )
        for i, batch in enumerate(batches):
            inp, ref = tensors(batch, device)
            with torch.autocast(device.type, enabled=amp, dtype=torch.bfloat16):
                loss = objective(model(inp), ref, ssim, options)
            window_first = i - i % accum
            remaining_samples = len(train) - window_first * options["batch_size"]
            window_samples = min(accum * options["batch_size"], remaining_samples)
            (loss * (inp.shape[0] / window_samples)).backward()
            epoch_loss += loss.detach() * inp.shape[0]
            if (i + 1) % accum == 0 or i + 1 == len(batches):
                step = epoch * steps_per_epoch + i // accum
                lr = learning_rate(step, total_steps, warm_steps, options)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), options["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if epoch == first_epoch and (i + 1) % 50 == 0:
                print(f"Epoch {epoch + 1} batch {i + 1}/{len(batches)} in {time.perf_counter() - start:.1f}s", flush=True)
        mean_psnr, mean_ssim = score(model, tune_batches, device, ssim)
        if not (math.isfinite(mean_psnr) and math.isfinite(mean_ssim)):
            raise FloatingPointError("Non-finite tune metrics")
        improved = mean_psnr > best
        if improved:
            best = mean_psnr
        record = dict(
            epoch=epoch + 1, loss=float(epoch_loss / len(train)), psnr_tune=mean_psnr,
            ssim_tune=mean_ssim, lr=lr, is_best=improved, seconds=time.perf_counter() - start,
        )
        history.append(record)
        state = {
            "model": model.state_dict(), "config": config, "epoch": epoch + 1,
            "best_psnr": best, "manifest_sha256": info["manifest_sha256"],
            "training_sources_sha256": info["training_sources_sha256"],
        }
        if improved:
            atomic_save(best_path, state)
        atomic_save(latest, dict(state, optimizer=optimizer.state_dict(), history=history))
        write_csv(results / f"{name}_epochs.csv", history, list(record))
        print(record, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train the LSUI source-only static backbone.")
    parser.add_argument("--config", default="configs/v1.yaml")
    parser.add_argument("--split", choices=["train"], default="train")
    parser.add_argument("--seed", type=int)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--smoke", action="store_true")
    actions.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, args.seed)
    device = configure(config)
    model = build_model(config, device)
    ssim = SSIM().to(device, memory_format=torch.channels_last)
    options = config["static"]["train"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["lr"], weight_decay=options["weight_decay"])
    info = run_info(config, device)
    # Hash the modules used by training independently of the downstream audit.
    import hashlib
    from prtta.run import sha_file
    root = Path(__file__).resolve().parent
    files = ["train_source.py", "prtta/run.py", "prtta/model.py", "prtta/metrics.py", "datasets/paired.py", "datasets/manifest.py"]
    info["training_sources_sha256"] = hashlib.sha256(
        "".join(sha_file(root / filename) for filename in files).encode()
    ).hexdigest()
    if args.smoke:
        smoke(config, device, model, optimizer, ssim, options, info)
    else:
        fit(config, device, model, optimizer, ssim, options, info, args.resume)


if __name__ == "__main__":
    main()
