import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Usage:
#python scripts\check_images_exist.py `
#>>   --parquet_path datasets\dataset_a\train.parquet `    
#>>   --dataset_root datasets\dataset_a

def main(parquet_path, dataset_root, max_errors=20):
    df = pd.read_parquet(parquet_path)

    missing = []
    root = Path(dataset_root)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        img_path = root / row["image_path"]
        if not img_path.exists():
            missing.append(str(img_path))
            if len(missing) >= max_errors:
                break

    if missing:
        print("\nMissing images detected!")
        for p in missing:
            print("  ", p)
        print(f"\nShowing first {len(missing)} missing paths.")
        print("Fix dataset_root or regenerate Parquet.")
        raise SystemExit(1)
    else:
        print(f"All {len(df)} images exist. Safe to train.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", required=True)
    parser.add_argument("--dataset_root", required=True)
    args = parser.parse_args()

    main(args.parquet_path, args.dataset_root)