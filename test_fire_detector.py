"""Run with:  python -m unittest test_fire_detector -v"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from fire_detector import (AlertFilter, ColorDetector, Config, FirePipeline,
                           FrameResult, estimate_area_m2)

ORANGE_BGR = (0, 140, 255)   # bright orange in OpenCV's BGR order


def blank(w=640, h=480):
    return np.zeros((h, w, 3), np.uint8)


def with_fire(side, w=640, h=480):
    img = blank(w, h)
    cv2.rectangle(img, (100, 100), (100 + side, 100 + side), ORANGE_BGR, -1)
    return img


def pipeline():
    return FirePipeline(ColorDetector(Config()))


class TestDetection(unittest.TestCase):
    def test_black_frame_is_none(self):
        r = pipeline().process(blank())
        self.assertEqual(r.level, "NONE")
        self.assertEqual(r.score, 0.0)

    def test_orange_blob_detected(self):
        r = pipeline().process(with_fire(120))
        self.assertNotEqual(r.level, "NONE")
        self.assertEqual(r.n_fires, 1)
        self.assertGreater(r.fire_fraction, 0.03)

    def test_blue_and_green_ignored(self):
        img = blank()
        cv2.rectangle(img, (50, 50), (250, 250), (255, 0, 0), -1)
        cv2.rectangle(img, (300, 50), (500, 250), (0, 255, 0), -1)
        self.assertEqual(pipeline().process(img).level, "NONE")

    def test_dim_orange_ignored(self):
        img = blank()
        cv2.rectangle(img, (100, 100), (250, 250), (0, 50, 100), -1)   # dark brown
        self.assertEqual(pipeline().process(img).level, "NONE")

    def test_tiny_speck_ignored(self):
        self.assertEqual(pipeline().process(with_fire(6)).level, "NONE")

    def test_bigger_fire_scores_higher(self):
        small = pipeline().process(with_fire(40))
        big = pipeline().process(with_fire(200))
        self.assertGreater(big.score, small.score)

    def test_large_fire_is_high(self):
        self.assertEqual(pipeline().process(with_fire(250)).level, "HIGH")

    def test_two_separate_fires(self):
        img = blank()
        cv2.rectangle(img, (50, 50), (130, 130), ORANGE_BGR, -1)
        cv2.rectangle(img, (400, 300), (480, 380), ORANGE_BGR, -1)
        self.assertEqual(pipeline().process(img).n_fires, 2)

    def test_large_frame_is_downscaled_but_still_works(self):
        img = blank(1920, 1080)
        cv2.rectangle(img, (400, 300), (900, 700), ORANGE_BGR, -1)
        r = pipeline().process(img)
        self.assertNotEqual(r.level, "NONE")
        self.assertLessEqual(r.annotated.shape[1], 640)

    def test_empty_frame_raises(self):
        with self.assertRaises(ValueError):
            pipeline().process(np.zeros((0, 0, 3), np.uint8))


class TestAreaEstimate(unittest.TestCase):
    def test_known_numbers(self):
        # 10 m up, 90 deg fov -> ground width 20 m; 640x480 -> height 15 m -> 300 m2
        self.assertAlmostEqual(estimate_area_m2(1.0, 10, 90, 640, 480), 300.0, places=3)
        self.assertAlmostEqual(estimate_area_m2(0.1, 10, 90, 640, 480), 30.0, places=3)

    def test_pipeline_reports_area_only_when_given_inputs(self):
        img = with_fire(120)
        self.assertIsNone(pipeline().process(img).area_m2)
        p = FirePipeline(ColorDetector(Config()), altitude_m=40, hfov_deg=80)
        self.assertGreater(p.process(img).area_m2, 0)


class TestNightMode(unittest.TestCase):
    def gray_fire(self):
        img = blank()
        cv2.rectangle(img, (100, 100), (260, 260), (255, 255, 255), -1)   # bright white blob
        return img

    def test_gray_frame_is_night_and_color_detector_finds_nothing(self):
        r = pipeline().process(self.gray_fire())
        self.assertEqual(r.mode, "night")
        self.assertEqual(r.level, "NONE")

    def test_color_frame_is_day(self):
        self.assertEqual(pipeline().process(with_fire(120)).mode, "day")

    def test_fully_dark_frame_is_night(self):
        self.assertEqual(pipeline().process(blank()).mode, "night")

    def test_gray_copy_of_color_scene_is_night(self):
        gray = cv2.cvtColor(cv2.cvtColor(with_fire(150), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        self.assertEqual(pipeline().process(gray).mode, "night")

    def test_single_channel_frame_does_not_crash(self):
        gray = cv2.cvtColor(self.gray_fire(), cv2.COLOR_BGR2GRAY)
        from fire_detector import colorfulness
        self.assertEqual(colorfulness(gray), 0.0)


def fake(level):
    return FrameResult(level, 0.5 if level != "NONE" else 0.0, 0, 0, 0, 0, None)


class TestAlertFilter(unittest.TestCase):
    def test_needs_k_of_n(self):
        f = AlertFilter(window=5, required=3)
        self.assertFalse(f.update(fake("HIGH")))
        self.assertFalse(f.update(fake("HIGH")))
        self.assertTrue(f.update(fake("HIGH")))

    def test_single_glitch_never_confirms(self):
        f = AlertFilter(window=5, required=3)
        results = [f.update(fake(l)) for l in ["NONE", "HIGH", "NONE", "NONE", "NONE", "NONE"]]
        self.assertFalse(any(results))

    def test_old_hits_age_out(self):
        f = AlertFilter(window=3, required=2)
        for l in ["HIGH", "HIGH"]:
            f.update(fake(l))
        for l in ["NONE", "NONE"]:
            confirmed = f.update(fake(l))
        self.assertFalse(confirmed)

    def test_bad_args(self):
        with self.assertRaises(ValueError):
            AlertFilter(window=3, required=5)


class TestCommandLine(unittest.TestCase):
    def test_image_and_video_end_to_end(self):
        here = Path(__file__).parent
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cv2.imwrite(str(tmp / "fire.png"), with_fire(150))
            out = subprocess.run(
                [sys.executable, str(here / "fire_detector.py"), str(tmp / "fire.png"),
                 "--out", str(tmp / "o")], capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn('"level"', out.stdout)
            self.assertTrue((tmp / "o" / "annotated_fire.jpg").exists())

            vid = tmp / "clip.avi"
            vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 480))
            for _ in range(20):
                vw.write(with_fire(150))
            vw.release()
            out = subprocess.run(
                [sys.executable, str(here / "fire_detector.py"), str(vid), "--fps", "5",
                 "--out", str(tmp / "o")], capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn('"confirmed": true', out.stdout)
            self.assertTrue(any((tmp / "o").glob("alert_*.jpg")))

    def test_missing_file_exits_cleanly(self):
        here = Path(__file__).parent
        out = subprocess.run([sys.executable, str(here / "fire_detector.py"), "nope.jpg"],
                             capture_output=True, text=True)
        self.assertNotEqual(out.returncode, 0)
        self.assertNotIn("Traceback", out.stderr)


if __name__ == "__main__":
    unittest.main()
