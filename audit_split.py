"""Check that image content and acquisition groups stay in one split."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = {"train", "validation", "test"}


def pixel_digest(path):
    """Hash the dimensions and decoded RGBA grid of a single-frame image."""
    from PIL import Image

    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"Pixel audit requires a single-frame image: {path.name}")
        with image.convert("RGBA") as rgba:
            digest = hashlib.sha256(f"{rgba.width}x{rgba.height}\0".encode("ascii"))
            digest.update(rgba.tobytes())
            return digest.hexdigest()


def audit(manifest, decoded_pixels=False):
    by_group, by_digest = defaultdict(set), defaultdict(set)
    by_pixels = defaultdict(set)
    counts = Counter()
    seen = set()
    with manifest.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not {"path", "split", "group"}.issubset(reader.fieldnames or ()):
            raise ValueError("Manifest requires path, split and group columns.")
        for row in reader:
            image_path, split, group = ((row[key] or "").strip() for key in ("path", "split", "group"))
            if split not in SPLITS or not group or not image_path:
                raise ValueError(f"Invalid manifest row {reader.line_num}")
            path = (manifest.parent / image_path).resolve()
            if (path, split) in seen:
                raise ValueError(f"Repeated path in split at row {reader.line_num}")
            seen.add((path, split))
            with path.open("rb") as image:
                digest = hashlib.file_digest(image, "sha256").hexdigest()
            counts[split] += 1
            by_digest[digest].add(split)
            by_group[group].add(split)
            if decoded_pixels:
                by_pixels[pixel_digest(path)].add(split)
    if not counts["train"] or not counts["test"]:
        raise ValueError("Both train and test must contain images.")
    report = {
        "rows_by_split": dict(counts),
        "cross_split_groups": {k: sorted(v) for k, v in by_group.items() if len(v) > 1},
        "cross_split_image_hashes": {k: sorted(v) for k, v in by_digest.items() if len(v) > 1},
    }
    if decoded_pixels:
        from PIL import __version__

        report["pixel_hash_format"] = "dimensions-and-RGBA-v1"
        report["pillow_version"] = __version__
        report["cross_split_pixel_hashes"] = {k: sorted(v) for k, v in by_pixels.items() if len(v) > 1}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--decoded-pixels", action="store_true", help="Also check decoded image pixels (requires Pillow)")
    args = parser.parse_args()
    try:
        result = audit(args.manifest, decoded_pixels=args.decoded_pixels)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(2, f"Audit failed: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(bool(result["cross_split_groups"] or result["cross_split_image_hashes"]
                    or result.get("cross_split_pixel_hashes")))


if __name__ == "__main__":
    raise SystemExit(main())
