"""Run with:  python -m unittest test_dataset_tools -v"""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from dataset_tools import check_dataset, make_split, split_scene_names


def make_raw(root: Path):
    """5 scenes x 4 images. 'empty*' scenes are negatives (no .txt files)."""
    for scene in ["fire_a", "fire_b", "fire_c", "empty_a", "empty_b"]:
        d = root / scene
        d.mkdir(parents=True)
        for i in range(4):
            img = np.zeros((120, 160, 3), np.uint8)
            cv2.rectangle(img, (40, 40), (90, 90), (0, 140, 255), -1)
            cv2.imwrite(str(d / f"im{i}.jpg"), img)
            if scene.startswith("fire"):
                (d / f"im{i}.txt").write_text("0 0.4 0.5 0.3 0.4\n")


class TestSplit(unittest.TestCase):
    def test_scene_split_has_no_overlap_and_is_repeatable(self):
        names = [f"s{i}" for i in range(10)]
        tr, va = split_scene_names(names, 0.2, seed=1)
        self.assertEqual(len(va), 2)
        self.assertFalse(set(tr) & set(va))
        self.assertEqual((tr, va), split_scene_names(names, 0.2, seed=1))

    def test_needs_two_scenes(self):
        with self.assertRaises(ValueError):
            split_scene_names(["only"], 0.2, 0)

    def test_full_build_and_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, out = Path(tmp) / "raw", Path(tmp) / "out"
            make_raw(raw)
            info = make_split(raw, out, val_frac=0.4, seed=3, add_gray=True)
            self.assertEqual(info["train"] + info["val"], 20)
            self.assertEqual(info["train_gray"], info["train"])
            self.assertTrue((out / "data.yaml").exists())
            # no scene appears on both sides
            tr = {f.name.split("__")[0] for f in (out / "images" / "train").iterdir()}
            va = {f.name.split("__")[0] for f in (out / "images" / "val").iterdir()}
            self.assertFalse(tr & va)
            # gray copies are really gray and only in train
            grays = list((out / "images" / "train").glob("*__gray.jpg"))
            self.assertTrue(grays)
            px = cv2.imread(str(grays[0])).astype(int)
            self.assertLess(np.abs(px[:, :, 0] - px[:, :, 2]).max(), 3)
            self.assertFalse(list((out / "images" / "val").glob("*__gray*")))
            # every image has a label file; negatives are empty
            stats, problems, _ = check_dataset(out)
            self.assertEqual(problems, [])
            self.assertEqual(stats["train"]["images"] + stats["val"]["images"],
                             20 + info["train_gray"])

    def test_check_catches_bad_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, out = Path(tmp) / "raw", Path(tmp) / "out"
            make_raw(raw)
            make_split(raw, out, 0.4, 3)
            lab = next((out / "labels" / "train").glob("fire*.txt"))
            lab.write_text("0 1.5 0.5 0.3 0.4\n7 0.5 0.5 0.1 0.1\n0 0.5 0.5\nabc\n")
            _, problems, _ = check_dataset(out)
            self.assertEqual(len(problems), 4)

    def test_check_catches_missing_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, out = Path(tmp) / "raw", Path(tmp) / "out"
            make_raw(raw)
            make_split(raw, out, 0.4, 3)
            next((out / "labels" / "val").glob("*.txt")).unlink()
            _, problems, _ = check_dataset(out)
            self.assertTrue(any("no label file" in p for p in problems))

    def test_loose_images_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            make_raw(raw)
            cv2.imwrite(str(raw / "loose.jpg"), np.zeros((10, 10, 3), np.uint8))
            with self.assertRaises(SystemExit):
                make_split(raw, Path(tmp) / "out")


if __name__ == "__main__":
    unittest.main()
