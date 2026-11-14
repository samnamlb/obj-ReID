"""Recover only metadata-referenced regular image files from a local tar archive.

Does not overwrite metadata or existing files, and does not invent a new split.
"""
import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reid_data import read_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="images.tar.gz")
    parser.add_argument("--dataset_root", default="datasets/dataset_a")
    parser.add_argument("--output", default="results/dataset_a_recovery.json")
    args = parser.parse_args()
    root = Path(args.dataset_root).resolve()
    train = read_metadata(root, "train.parquet")
    test = read_metadata(root, "test.parquet")
    wanted = set(train.image_path) | set(test.image_path)
    found, extracted, existing, total_bytes = set(), 0, 0, 0
    with tarfile.open(args.archive, "r|gz") as archive:
        for member in archive:
            name = str(PurePosixPath(member.name))
            if name not in wanted:
                continue
            if not member.isfile():
                raise ValueError(f"Referenced archive entry is not a regular file: {name}")
            target = root / name
            if not target.resolve().is_relative_to(root):
                raise ValueError(f"Unsafe archive path: {name}")
            if name in found:
                raise ValueError(f"Duplicate referenced archive entry: {name}")
            found.add(name)
            total_bytes += member.size
            if target.exists():
                if not target.is_file() or target.stat().st_size != member.size:
                    raise ValueError(f"Existing path conflicts with archive: {target}")
                existing += 1
                continue
            if shutil.disk_usage(root).free < member.size + 2 * 1024 ** 3:
                raise OSError("Less than 2 GiB disk space would remain; stopping extraction")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            with source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination)
            extracted += 1
            if extracted % 2000 == 0:
                print(f"Extracted {extracted:,} referenced images", flush=True)
    missing = sorted(wanted - found)
    result = {
        "archive": str(Path(args.archive).resolve()), "dataset_root": str(root),
        "train_rows": len(train), "test_rows": len(test),
        "train_identities": int(train.identity.nunique()),
        "test_identities": int(test.identity.nunique()),
        "overlapping_train_test_paths": len(set(train.image_path) & set(test.image_path)),
        "overlapping_train_test_identities": len(set(train.identity) & set(test.identity)),
        "referenced_unique_images": len(wanted), "archive_matched_images": len(found),
        "extracted_images": extracted, "existing_images": existing,
        "referenced_image_bytes": total_bytes, "missing_image_count": len(missing),
        "missing_image_examples": missing[:20],
        "warning": "Original split preserved. Overlapping training/test images cannot support independent held-out accuracy claims. Checkpoints do not record the original training image paths.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
