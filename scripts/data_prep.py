"""
data_prep.py

Turns a raw dataset laid out as:

    data/raw/
    ├── cattle/<breed>/*.jpg
    ├── buffalo/<breed>/*.jpg
    └── other/*.jpg

into two sets of stratified train/val/test ImageFolder-style directories:

    data/splits/species/{train,val,test}/{cattle,buffalo,other}/...
    data/splits/breed_cattle/{train,val,test}/<breed>/...
    data/splits/breed_buffalo/{train,val,test}/<breed>/...

Images are copied (not moved) so the original raw dataset is left untouched.
"""

import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path):
    return [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]


def stratified_split(files, train_ratio, val_ratio, seed):
    rng = random.Random(seed)
    files = list(files)
    rng.shuffle(files)
    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }


def copy_split(split_files, dest_root: Path, class_name: str):
    for split_name, files in split_files.items():
        dest_dir = dest_root / split_name / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, dest_dir / f.name)


def build_species_split(raw_dir: Path, out_dir: Path, train_ratio, val_ratio, seed):
    species_out = out_dir / "species"
    counts = {}

    # cattle + buffalo: gather all images across all breed subfolders
    for species in ["cattle", "buffalo"]:
        species_dir = raw_dir / species
        if not species_dir.exists():
            print(f"[warn] {species_dir} not found, skipping")
            continue
        all_imgs = []
        for breed_dir in species_dir.iterdir():
            if breed_dir.is_dir():
                all_imgs.extend(list_images(breed_dir))
        split = stratified_split(all_imgs, train_ratio, val_ratio, seed)
        copy_split(split, species_out, species)
        counts[species] = len(all_imgs)

    # other: flat folder of negative examples
    other_dir = raw_dir / "other"
    if other_dir.exists():
        other_imgs = list_images(other_dir)
        split = stratified_split(other_imgs, train_ratio, val_ratio, seed)
        copy_split(split, species_out, "other")
        counts["other"] = len(other_imgs)
    else:
        print("[warn] data/raw/other not found — Stage 1 will not learn to reject "
              "non-cattle/buffalo images without negative examples")

    print("Species image counts:", counts)


def build_breed_split(raw_dir: Path, out_dir: Path, species: str, train_ratio, val_ratio, seed):
    species_dir = raw_dir / species
    if not species_dir.exists():
        print(f"[warn] {species_dir} not found, skipping breed split for {species}")
        return
    breed_out = out_dir / f"breed_{species}"
    counts = {}
    for breed_dir in species_dir.iterdir():
        if not breed_dir.is_dir():
            continue
        imgs = list_images(breed_dir)
        if len(imgs) < 5:
            print(f"[warn] {breed_dir} has only {len(imgs)} images — too few to split reliably")
        split = stratified_split(imgs, train_ratio, val_ratio, seed)
        copy_split(split, breed_out, breed_dir.name)
        counts[breed_dir.name] = len(imgs)
    print(f"{species} breed image counts:", counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--out_dir", default="data/splits")
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    build_species_split(raw_dir, out_dir, args.train_ratio, args.val_ratio, args.seed)
    build_breed_split(raw_dir, out_dir, "cattle", args.train_ratio, args.val_ratio, args.seed)
    build_breed_split(raw_dir, out_dir, "buffalo", args.train_ratio, args.val_ratio, args.seed)

    print(f"\nDone. Splits written to {out_dir}/species, {out_dir}/breed_cattle, {out_dir}/breed_buffalo")


if __name__ == "__main__":
    main()
