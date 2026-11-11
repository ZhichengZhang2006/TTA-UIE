import argparse
from pathlib import Path

import torch

from datasets.manifest import write_csv
from prtta.metrics import SSIM, psnr
from prtta.run import configure, load_config, load_static, loader, make_dataset, tensors, write_json


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Score a static LSUI model in FP32.")
    parser.add_argument("--config", default="configs/v1.yaml")
    parser.add_argument("--split", choices=["train", "tune", "utility_fit", "risk_calib"], default="tune")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    config = load_config(args.config, args.seed)
    device = configure(config)
    model, state = load_static(config, device, args.checkpoint)
    dataset = make_dataset(config, args.split)
    batches = loader(config, dataset, device, config["static"]["train"]["batch_size"])
    ssim = SSIM().to(device, memory_format=torch.channels_last)
    rows = []
    for batch in batches:
        image, reference = tensors(batch, device)
        output = model(image)
        for image_id, group, p, s in zip(batch[2], batch[3], psnr(output, reference).tolist(), ssim(output, reference).tolist()):
            rows.append(dict(image_id=image_id, group_id=group, psnr=p, ssim=s))
    path = Path(config["paths"]["results"]) / f"static_{args.split}_seed{config['seed']}.csv"
    write_csv(path, rows, list(rows[0]))
    summary = dict(
        split=args.split, seed=config["seed"], checkpoint_epoch=state["epoch"], count=len(rows),
        psnr=sum(row["psnr"] for row in rows) / len(rows),
        ssim=sum(row["ssim"] for row in rows) / len(rows),
    )
    write_json(path.with_suffix(".json"), summary)
    print(summary)


if __name__ == "__main__":
    main()
