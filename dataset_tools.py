"""Dataset prep for YOLO fire training.

Collect your photos with ONE SUBFOLDER PER SCENE/SESSION:

  raw/
    firepit_day/    a01.jpg  a01.txt  a02.jpg  a02.txt ...
    candle_dark/    b01.jpg  b01.txt ...
    empty_room/     c01.jpg  c02.jpg          <- no .txt file = no fire (a negative)

Labels are YOLO format, one line per box:  class cx cy w h   (all 0-1, relative to image)
with class 0 = fire, 1 = smoke. Roboflow, CVAT and LabelImg can all export this.

Commands:
  python dataset_tools.py split raw dataset --val 0.2 --add-gray
  python dataset_tools.py check dataset

Whole scenes go to train OR val, never both. If neighboring photos from one scene landed
on both sides, the model would just memorize them and the val score would be fake.

--add-gray adds a grayscale copy of every TRAIN image so the model also learns what fire
looks like in IR night mode (gray), not only daytime color.
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2

CLASSES = ["fire", "smoke"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_scenes(raw: Path) -> dict[str, list[Path]]:
    loose = [f for f in raw.iterdir() if f.suffix.lower() in IMG_EXTS]
    if loose:
        sys.exit(f"{len(loose)} images sit directly in {raw}. Put them in scene subfolders "
                 "(one per session) so train/val can be split without leakage.")
    scenes = {}
    for d in sorted(p for p in raw.iterdir() if p.is_dir()):
        imgs = sorted(f for f in d.iterdir() if f.suffix.lower() in IMG_EXTS)
        if imgs:
            scenes[d.name] = imgs
    return scenes


def split_scene_names(names: list[str], val_frac: float, seed: int) -> tuple[list[str], list[str]]:
    if len(names) < 2:
        raise ValueError("need at least 2 scenes (one for train, one for val)")
    names = sorted(names)
    random.Random(seed).shuffle(names)
    n_val = min(max(1, round(len(names) * val_frac)), len(names) - 1)
    return sorted(names[n_val:]), sorted(names[:n_val])


def _write_label(src_img: Path, dst_label: Path) -> None:
    src = src_img.with_suffix(".txt")
    if src.exists():
        shutil.copyfile(src, dst_label)
    else:
        dst_label.write_text("")            # empty label file = negative image


def make_split(raw: Path, out: Path, val_frac: float = 0.2, seed: int = 0,
               add_gray: bool = False) -> dict:
    scenes = find_scenes(raw)
    train_scenes, val_scenes = split_scene_names(list(scenes), val_frac, seed)
    counts = {"train": 0, "val": 0, "train_gray": 0}
    for split, names in (("train", train_scenes), ("val", val_scenes)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for scene in names:
            for img in scenes[scene]:
                stem = f"{scene}__{img.stem}"
                shutil.copyfile(img, out / "images" / split / f"{stem}{img.suffix.lower()}")
                _write_label(img, out / "labels" / split / f"{stem}.txt")
                counts[split] += 1
                if add_gray and split == "train":
                    frame = cv2.imread(str(img))
                    if frame is None:
                        print(f"warning: unreadable image skipped for gray copy: {img}")
                        continue
                    gray = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                        cv2.COLOR_GRAY2BGR)
                    cv2.imwrite(str(out / "images" / split / f"{stem}__gray.jpg"), gray)
                    _write_label(img, out / "labels" / split / f"{stem}__gray.txt")
                    counts["train_gray"] += 1

    names_yaml = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names_yaml}\n")
    return {"train_scenes": train_scenes, "val_scenes": val_scenes, **counts}


def check_dataset(root: Path) -> tuple[dict, list[str], list[str]]:
    """Returns (stats, problems, warnings). Problems will break training; warnings won't."""
    problems, warnings, stats = [], [], {}
    for split in ("train", "val"):
        img_dir, lab_dir = root / "images" / split, root / "labels" / split
        imgs = sorted(f for f in img_dir.glob("*") if f.suffix.lower() in IMG_EXTS) \
            if img_dir.exists() else []
        s = {"images": len(imgs), "negatives": 0, "boxes": [0] * len(CLASSES)}
        for img in imgs:
            lab = lab_dir / f"{img.stem}.txt"
            if not lab.exists():
                problems.append(f"{split}: no label file for {img.name}")
                continue
            lines = [l for l in lab.read_text().splitlines() if l.strip()]
            if not lines:
                s["negatives"] += 1
            for n, line in enumerate(lines, 1):
                parts = line.split()
                try:
                    cls, vals = int(parts[0]), [float(v) for v in parts[1:]]
                except ValueError:
                    problems.append(f"{lab.name}:{n}: not numbers: {line!r}")
                    continue
                if len(parts) != 5:
                    problems.append(f"{lab.name}:{n}: need 5 values, got {len(parts)}")
                elif not 0 <= cls < len(CLASSES):
                    problems.append(f"{lab.name}:{n}: class {cls} is not 0..{len(CLASSES) - 1}")
                elif any(not 0 <= v <= 1 for v in vals):
                    problems.append(f"{lab.name}:{n}: values must be between 0 and 1")
                else:
                    s["boxes"][cls] += 1
        stats[split] = s

    if stats["train"]["images"] == 0 or stats["val"]["images"] == 0:
        problems.append("train or val has no images")
    else:
        if stats["val"]["boxes"][0] == 0:
            warnings.append("val has no fire boxes, so the score will mean nothing")
        if stats["train"]["negatives"] == 0:
            warnings.append("train has no negative images (no fire); expect false alarms")
        if stats["train"]["images"] < 200:
            warnings.append("under 200 train images: fine for a demo, thin for real use")
    return stats, problems, warnings


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("split", help="build the YOLO train/val folders")
    sp.add_argument("raw", type=Path)
    sp.add_argument("out", type=Path)
    sp.add_argument("--val", type=float, default=0.2, help="fraction of scenes for validation")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--add-gray", action="store_true", help="add gray copies of train images")
    ck = sub.add_parser("check", help="sanity-check a built dataset")
    ck.add_argument("root", type=Path)
    args = p.parse_args()

    if args.cmd == "split":
        if not args.raw.is_dir():
            sys.exit(f"not a folder: {args.raw}")
        info = make_split(args.raw, args.out, args.val, args.seed, args.add_gray)
        print(f"train scenes: {info['train_scenes']}\nval scenes:   {info['val_scenes']}")
        print(f"train images: {info['train']} (+{info['train_gray']} gray copies)   "
              f"val images: {info['val']}\nwrote {args.out / 'data.yaml'}")
    else:
        stats, problems, warnings = check_dataset(args.root)
        for split, s in stats.items():
            print(f"{split}: {s['images']} images, {s['negatives']} negatives, "
                  f"boxes {dict(zip(CLASSES, s['boxes']))}")
        for w in warnings:
            print("warning:", w)
        for pr in problems:
            print("PROBLEM:", pr)
        if problems:
            sys.exit(1)
        print("dataset looks OK")


if __name__ == "__main__":
    main()
