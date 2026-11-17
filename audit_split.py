"""Check that image content and acquisition groups stay in one split."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = {"train", "validation", "test"}


def audit(manifest):
    by_group, by_digest = defaultdict(set), defaultdict(set)
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
    if not counts["train"] or not counts["test"]:
        raise ValueError("Both train and test must contain images.")
    return {
        "rows_by_split": dict(counts),
        "cross_split_groups": {k: sorted(v) for k, v in by_group.items() if len(v) > 1},
        "cross_split_image_hashes": {k: sorted(v) for k, v in by_digest.items() if len(v) > 1},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    result = audit(args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(bool(result["cross_split_groups"] or result["cross_split_image_hashes"]))


if __name__ == "__main__":
    raise SystemExit(main())
