"""Prepare an explicitly partial visual review of cross-dataset match candidates."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from datasets.manifest import image_index, read_csv, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="manifests/cross_dataset_matches.csv")
    parser.add_argument("--source", default="manifests/manifest.csv")
    parser.add_argument("--dataset-root", default="/home/user/UnderWaterDataset")
    parser.add_argument("--out", default="results/duplicates")
    args = parser.parse_args()
    source = {r["image_id"]: r for r in read_csv(args.source)}
    matches = read_csv(args.input)
    candidates = [r for r in matches if r["target_original_split"] == "test"]
    unique = {r["target_id"]: r for r in candidates}
    candidates = list(unique.values())
    selected = [candidates[i] for i in np.linspace(0, len(candidates) - 1, min(24, len(candidates)), dtype=int)]
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    cache, review = {}, []
    for page in range(0, len(selected), 8):
        fig, axes = plt.subplots(8, 4, figsize=(8, 16), constrained_layout=True)
        for i, row in enumerate(selected[page:page + 8]):
            _, split, key = row["target_id"].split(":")
            images = [source[row["source_id"]][field] for field in ("input_path", "reference_path")]
            for role in ("input", "GT"):
                directory = Path(args.dataset_root) / row["target_dataset"] / split / role
                if str(directory) not in cache:
                    cache[str(directory)] = image_index(directory)
                images.append(cache[str(directory)][key.casefold()])
            for j, path in enumerate(images):
                with Image.open(path) as image:
                    axes[i, j].imshow(image.convert("RGB"))
                axes[i, j].set(xticks=[], yticks=[])
                if i == 0:
                    axes[i, j].set_title(["LSUI input", "LSUI reference", "Target input", "Target reference"][j])
            axes[i, 0].set_ylabel(f"{page+i+1}\n{row['source_id'].split(':')[-1]}")
            review.append(dict(
                review_index=page+i+1, source_id=row["source_id"], target_id=row["target_id"],
                reviewer="pending", status="unconfirmed", evidence=f"contact_{page//8+1}.png row {i+1}",
            ))
        fig.savefig(output / f"contact_{page//8+1}.png", dpi=150)
        plt.close(fig)
    write_csv(output / "visual_review.csv", review, list(review[0]))


if __name__ == "__main__":
    main()
