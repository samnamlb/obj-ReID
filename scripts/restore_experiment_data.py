"""Copy saved split manifests and their images into an isolated runnable dataset."""
import argparse
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reid_data import read_metadata, validate_images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata_root", default="experiments/dataset_a_run/split_metadata")
    parser.add_argument("--image_root", default="datasets/dataset_a")
    parser.add_argument("--output_root", default="runs/reproduction/dataset_a")
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["test"])
    args = parser.parse_args()
    source = Path(args.image_root).resolve()
    output = Path(args.output_root).resolve()
    if source == output or output.is_relative_to(source):
        raise ValueError("Use a separate output folder to preserve original data")
    frames = {split: read_metadata(args.metadata_root, f"{split}.parquet") for split in args.splits}
    for frame in frames.values():
        validate_images(source, frame.image_path)
    output.mkdir(parents=True, exist_ok=True)
    for split, frame in frames.items():
        for relative in frame.image_path:
            original, target = source / relative, output / relative
            if not target.resolve().is_relative_to(output):
                raise ValueError(f"Output image escapes dataset root: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.stat().st_size != original.stat().st_size:
                    raise FileExistsError(f"Conflicting existing image: {target}")
            else:
                shutil.copy2(original, target)
        metadata = Path(args.metadata_root) / f"{split}.parquet"
        target = output / metadata.name
        if target.exists() and target.read_bytes() != metadata.read_bytes():
            raise FileExistsError(f"Refusing to replace existing metadata: {target}")
        shutil.copy2(metadata, target)
        print(f"Restored {split}: {len(frame)} images to {output}")


if __name__ == "__main__":
    main()
