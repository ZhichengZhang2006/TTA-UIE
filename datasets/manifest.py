"""Verify paired data and keep connected duplicate groups in one split."""

import csv
import hashlib
import io
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
FIELDS = [
    "image_id", "input_path", "reference_path", "dataset", "split",
    "group_id", "sha256_input", "sha256_ref", "width", "height", "valid_pair",
    "original_split", "phash_input", "phash_ref", "dhash_input", "dhash_ref",
    "error",
]
AXIS = np.arange(32) + 0.5
DCT = np.cos(np.pi / 32 * np.arange(8)[:, None] * AXIS)
DCT[0] /= np.sqrt(2)
RESAMPLE = getattr(Image, "Resampling", Image)


def bitstring(bits):
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def image_info(path):
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        gray = image.convert("L")
        pixels = np.asarray(gray.resize((32, 32), RESAMPLE.LANCZOS), dtype=float)
        frequency = (DCT @ pixels @ DCT.T).ravel()
        bits = frequency > np.median(frequency[1:])
        bits[0] = False
        edges = np.asarray(gray.resize((9, 8), RESAMPLE.LANCZOS))
        return (
            hashlib.sha256(data).hexdigest(), image.size,
            bitstring(bits), bitstring(edges[:, 1:] > edges[:, :-1]),
        )


def image_index(directory):
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            key = path.relative_to(directory).with_suffix("").as_posix().casefold()
            if key in files:
                raise ValueError(f"Ambiguous pair stem {key} under {directory}")
            files[key] = path
    return files


def paired_directories(root):
    for input_name, reference_name in (
        ("input", "GT"), ("INPUT", "GT"), ("raw", "reference"),
    ):
        if (root / input_name).is_dir() and (root / reference_name).is_dir():
            yield root / input_name, root / reference_name
            return
    for name in ("train", "val", "test"):
        if (root / name).is_dir():
            yield from paired_directories(root / name)


def inspect_pair(job):
    dataset, original, key, inp, ref = job
    row = dict(
        image_id=f"{dataset}:{original}:{key}", input_path=str(inp or ""),
        reference_path=str(ref or ""), dataset=dataset, original_split=original,
        split="", group_id="", valid_pair=False, error="",
    )
    try:
        if inp is None or ref is None:
            raise ValueError("Missing paired file")
        sha_in, size, p_in, d_in = image_info(inp)
        sha_ref, ref_size, p_ref, d_ref = image_info(ref)
        row.update(
            sha256_input=sha_in, sha256_ref=sha_ref, width=size[0], height=size[1],
            phash_input=f"{p_in:016x}", phash_ref=f"{p_ref:016x}",
            dhash_input=f"{d_in:016x}", dhash_ref=f"{d_ref:016x}",
        )
        if size != ref_size:
            raise ValueError(f"Spatial mismatch {size} versus {ref_size}")
        row["valid_pair"] = True
    except (ValueError, OSError, Image.DecompressionBombError) as error:
        row["error"] = str(error)
    return row


def scan(root, dataset, workers):
    jobs = []
    for inp, ref in paired_directories(root):
        original = inp.parent.relative_to(root).as_posix()
        inputs, references = image_index(inp), image_index(ref)
        jobs.extend(
            (dataset, original, key, inputs.get(key), references.get(key))
            for key in sorted(inputs.keys() | references.keys())
        )
    if not jobs:
        raise ValueError(f"No paired directories found under {root}")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(inspect_pair, jobs))


def hamming(left, right):
    return bin(left ^ right).count("1")


def duplicate_links(rows):
    """Find exact matches and corroborated pHash matches across both roles."""
    exact = {}
    bands = defaultdict(list)
    adjacent = defaultdict(list)
    links = set()
    for i, row in enumerate(rows):
        stem = Path(row["input_path"]).stem
        numeric = re.fullmatch(r"(.*?)(\d+)", stem)
        for role in ("input", "ref"):
            sha = row[f"sha256_{role}"]
            p = int(row[f"phash_{role}"], 16)
            d = int(row[f"dhash_{role}"], 16)
            if sha in exact and exact[sha] != i:
                links.add((exact[sha], i, "sha256"))
            exact[sha] = i
            for band in range(5):
                key = band, (p >> (13 * band)) & 8191
                for j, other_p, other_d in bands[key]:
                    if i != j and hamming(p, other_p) <= 4 and hamming(d, other_d) <= 6:
                        links.add((j, i, "phash"))
                bands[key].append((i, p, d))
            if numeric:
                prefix, number = numeric.groups()
                n = int(number)
                key = row["dataset"], prefix.casefold(), role
                for k in (n - 1, n + 1):
                    for j, other_p, other_d in adjacent[key, k]:
                        if i != j and hamming(p, other_p) <= 8 and hamming(d, other_d) <= 8:
                            links.add((j, i, "sequence_phash"))
                adjacent[key, n].append((i, p, d))
    return sorted(links)


def assign_splits(rows, targets, seed):
    valid = [row for row in rows if row["valid_pair"]]
    parents = list(range(len(valid)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    links = duplicate_links(valid)
    for i, j, _ in links:
        parents[max(find(i), find(j))] = min(find(i), find(j))
    grouped = defaultdict(list)
    for i, row in enumerate(valid):
        grouped[find(i)].append(row)
    groups = list(grouped.values())
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)
    counts = Counter()
    total_target = sum(targets.values())
    for group in groups:
        split = max(
            targets,
            key=lambda k: (targets[k] * len(valid) / total_target - counts[k])
            / targets[k],
        )
        group_id = min(row["image_id"] for row in group)
        for row in group:
            row.update(split=split, group_id=group_id)
        counts[split] += len(group)
    return counts, groups, links


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
