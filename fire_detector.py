"""Fire detection + severity scoring for the drone's Raspberry Pi 4.

Pipeline:  frame -> Detector -> list[Detection] -> assess() -> FrameResult
           -> AlertFilter (needs fire in K of the last N frames before alerting)

Two interchangeable detectors:
  ColorDetector  HSV color rules. No training needed. Baseline, prone to false positives.
                 Daytime color frames only: in IR night mode (gray) it returns nothing.
  YoloDetector   Trained YOLO model (ultralytics). Much better once you have weights.

Usage:
  python fire_detector.py forest.jpg
  python fire_detector.py flight.mp4 --fps 5
  python fire_detector.py 0 --detector yolo --model fire.pt --altitude 40 --hfov 80

Install:  pip install opencv-python-headless numpy      (+ ultralytics for YOLO)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def colorfulness(frame: np.ndarray) -> float:
    """Median color saturation (0-255) of the lit pixels in the frame.

    The camera's day/night switch makes night frames gray, so this is near 0 at night
    and clearly higher in daylight. Only lit pixels count, so a dark frame with one small
    orange fire still reads as 'day'. No lit pixels at all counts as 0.
    """
    if frame.ndim == 2:
        return 0.0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lit = hsv[:, :, 2] > 40
    if not lit.any():
        return 0.0
    return float(np.median(hsv[:, :, 1][lit]))


# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class Config:
    # OpenCV HSV: H 0-179, S/V 0-255. Fire = red/orange/yellow, saturated, very bright.
    hsv_low: tuple = (0, 120, 200)
    hsv_high: tuple = (35, 255, 255)
    min_blob_px: int = 150          # ignore specks (measured at process_width)
    night_sat_thresh: float = 30.0  # colorfulness below this = IR night mode (gray frames)
    process_width: int = 640        # frames are downscaled to this width first (speed)
    area_saturation: float = 0.10   # fire covering 10% of frame maxes out the area term
    level_low: float = 0.25         # score < this -> LOW
    level_high: float = 0.55        # score >= this -> HIGH, in between -> MODERATE


# --------------------------------------------------------------------------- data
@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    label: str = "fire"             # "fire" or "smoke"
    confidence: float = 1.0
    area_px: float = 0.0            # contour area for color, box area for YOLO


@dataclass
class FrameResult:
    level: str
    score: float
    fire_fraction: float            # share of the frame that is fire, 0..1
    smoke_fraction: float
    intensity: float                # mean brightness inside fire boxes, 0..1
    n_fires: int
    area_m2: float | None           # only if altitude + fov were given
    detections: list[Detection] = field(default_factory=list)
    annotated: np.ndarray | None = None
    mode: str = "day"               # "day" (color) or "night" (IR, gray)
    color_sat: float = 0.0          # colorfulness; use it to tune night_sat_thresh

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "color_sat": round(self.color_sat, 1),
            "level": self.level,
            "score": round(self.score, 3),
            "fire_pct": round(self.fire_fraction * 100, 2),
            "smoke_pct": round(self.smoke_fraction * 100, 2),
            "intensity": round(self.intensity, 3),
            "n_fires": self.n_fires,
            "area_m2": None if self.area_m2 is None else round(self.area_m2, 1),
        }


# --------------------------------------------------------------------------- detectors
class ColorDetector:
    """Baseline: threshold fire-colored pixels, clean up, keep big blobs."""

    def __init__(self, cfg: Config = Config()):
        self.cfg = cfg
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if colorfulness(frame) < self.cfg.night_sat_thresh:
            return []        # gray night frame: there is no orange to find, use YOLO
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(self.cfg.hsv_low), np.array(self.cfg.hsv_high))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)    # remove specks
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)   # fill holes
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.cfg.min_blob_px:
                continue
            x, y, w, h = cv2.boundingRect(c)
            out.append(Detection(x, y, w, h, "fire", 1.0, area))
        return out


class YoloDetector:
    """Trained YOLO model. Class names containing 'fire' or 'smoke' are used."""

    def __init__(self, weights: str, conf: float = 0.4):
        try:
            from ultralytics import YOLO
        except ImportError:
            sys.exit("YOLO detector needs:  pip install ultralytics")
        self.model = YOLO(weights)
        self.conf = conf

    def detect(self, frame: np.ndarray) -> list[Detection]:
        res = self.model(frame, conf=self.conf, verbose=False)[0]
        out = []
        for box in res.boxes:
            name = str(res.names[int(box.cls)]).lower()
            label = "fire" if "fire" in name else "smoke" if "smoke" in name else None
            if label is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            w, h = x2 - x1, y2 - y1
            out.append(Detection(x1, y1, w, h, label, float(box.conf), float(w * h)))
        return out


# --------------------------------------------------------------------------- scoring
def estimate_area_m2(fire_fraction: float, altitude_m: float, hfov_deg: float,
                     width: int, height: int) -> float:
    """Ground area on fire, assuming the camera points straight down.

    ground width = 2 * altitude * tan(hfov / 2); height scales by the aspect ratio.
    Inaccurate if the camera is tilted.
    """
    ground_w = 2 * altitude_m * np.tan(np.radians(hfov_deg) / 2)
    ground_h = ground_w * height / width
    return float(fire_fraction * ground_w * ground_h)


def _intensity(frame: np.ndarray, dets: list[Detection]) -> float:
    if not dets:
        return 0.0
    v = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
    mask = np.zeros(v.shape, np.uint8)
    for d in dets:
        cv2.rectangle(mask, (d.x, d.y), (d.x + d.w - 1, d.y + d.h - 1), 255, -1)
    return float(cv2.mean(v, mask=mask)[0]) / 255.0


def assess(frame: np.ndarray, dets: list[Detection], cfg: Config,
           altitude_m: float | None = None, hfov_deg: float | None = None) -> FrameResult:
    h, w = frame.shape[:2]
    total = float(h * w)
    fires = [d for d in dets if d.label == "fire"]
    smokes = [d for d in dets if d.label == "smoke"]
    fire_fraction = min(sum(d.area_px for d in fires) / total, 1.0)
    smoke_fraction = min(sum(d.area_px for d in smokes) / total, 1.0)
    intensity = _intensity(frame, fires)

    if fires:
        score = 0.7 * min(fire_fraction / cfg.area_saturation, 1.0) + 0.3 * intensity
        level = "LOW" if score < cfg.level_low else "MODERATE" if score < cfg.level_high else "HIGH"
    else:
        score, level = 0.0, "NONE"

    area = None
    if altitude_m is not None and hfov_deg is not None:
        area = estimate_area_m2(fire_fraction, altitude_m, hfov_deg, w, h)

    return FrameResult(level, score, fire_fraction, smoke_fraction, intensity,
                       len(fires), area, dets)


def annotate(frame: np.ndarray, r: FrameResult) -> np.ndarray:
    img = frame.copy()
    for d in r.detections:
        color = (0, 0, 255) if d.label == "fire" else (200, 200, 200)
        cv2.rectangle(img, (d.x, d.y), (d.x + d.w, d.y + d.h), color, 2)
        cv2.putText(img, f"{d.label} {d.confidence:.2f}", (d.x, max(d.y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    text = f"{r.level} [{r.mode}]  fire {r.fire_fraction * 100:.1f}%  score {r.score:.2f}"
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return img


# --------------------------------------------------------------------------- pipeline
class FirePipeline:
    def __init__(self, detector, cfg: Config = Config(),
                 altitude_m: float | None = None, hfov_deg: float | None = None):
        self.detector, self.cfg = detector, cfg
        self.altitude_m, self.hfov_deg = altitude_m, hfov_deg

    def process(self, frame: np.ndarray) -> FrameResult:
        if frame is None or frame.size == 0:
            raise ValueError("empty frame")
        h, w = frame.shape[:2]
        if w > self.cfg.process_width:                      # downscale for speed
            scale = self.cfg.process_width / w
            frame = cv2.resize(frame, (self.cfg.process_width, int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        dets = self.detector.detect(frame)
        result = assess(frame, dets, self.cfg, self.altitude_m, self.hfov_deg)
        result.color_sat = colorfulness(frame)
        result.mode = "night" if result.color_sat < self.cfg.night_sat_thresh else "day"
        result.annotated = annotate(frame, result)
        return result


class AlertFilter:
    """Confirm a fire only if it appears in `required` of the last `window` frames.

    Kills one-frame false positives (sun glare, a red tarp flashing by).
    """

    def __init__(self, window: int = 5, required: int = 3):
        if not 1 <= required <= window:
            raise ValueError("need 1 <= required <= window")
        self.required = required
        self.hits: deque[bool] = deque(maxlen=window)
        self.scores: deque[float] = deque(maxlen=window)

    def update(self, r: FrameResult) -> bool:
        self.hits.append(r.level != "NONE")
        self.scores.append(r.score)
        return sum(self.hits) >= self.required

    @property
    def smoothed_score(self) -> float:
        return float(np.mean(self.scores)) if self.scores else 0.0


# --------------------------------------------------------------------------- CLI
def _log(record: dict) -> None:
    print(json.dumps(record), flush=True)


def run_image(path: Path, pipeline: FirePipeline, out_dir: Path) -> None:
    frame = cv2.imread(str(path))
    if frame is None:
        sys.exit(f"Could not read image: {path}")
    r = pipeline.process(frame)
    _log({"source": path.name, **r.to_dict()})
    cv2.imwrite(str(out_dir / f"annotated_{path.stem}.jpg"), r.annotated)


def run_stream(src, pipeline: FirePipeline, out_dir: Path, args) -> None:
    is_live = isinstance(src, int)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"Could not open video source: {src}")
    if is_live:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)                 # always analyze the newest frame

    stride = 1 if is_live else max(1, round((cap.get(cv2.CAP_PROP_FPS) or 30) / args.fps))
    alert = AlertFilter(args.window, args.required)
    frame_no = analyzed = failures = 0
    warned = False
    last_t = last_save = 0.0
    try:
        while args.max_frames is None or analyzed < args.max_frames:
            if not cap.grab():
                failures += 1
                if not is_live or failures > 30:            # end of file, or camera is gone
                    break
                time.sleep(0.1)
                continue
            failures = 0
            frame_no += 1
            if is_live:                                     # analyze at most --fps times/sec
                now = time.monotonic()
                if now - last_t < 1.0 / args.fps:
                    continue
                last_t = now
            elif frame_no % stride:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                continue

            r = pipeline.process(frame)
            analyzed += 1
            if r.mode == "night" and args.detector == "color" and not warned:
                print("WARNING: night-mode (gray) frames; the color detector cannot see "
                      "fire in these. Use --detector yolo.", file=sys.stderr)
                warned = True
            confirmed = alert.update(r)
            _log({"t": round(time.time(), 2), "frame": frame_no, "confirmed": confirmed,
                  "smoothed_score": round(alert.smoothed_score, 3), **r.to_dict()})

            if confirmed and time.monotonic() - last_save > 2.0:   # at most one image / 2 s
                cv2.imwrite(str(out_dir / f"alert_{frame_no:06d}.jpg"), r.annotated)
                last_save = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


def main() -> None:
    p = argparse.ArgumentParser(description="Fire detection + severity")
    p.add_argument("source", help="image file, video file, or camera index (e.g. 0)")
    p.add_argument("--detector", choices=["color", "yolo"], default="color")
    p.add_argument("--model", help="YOLO weights file (.pt), required for --detector yolo")
    p.add_argument("--conf", type=float, default=0.4, help="YOLO confidence threshold")
    p.add_argument("--fps", type=float, default=5.0, help="analysis rate")
    p.add_argument("--window", type=int, default=5, help="alert filter window (frames)")
    p.add_argument("--required", type=int, default=3, help="detections needed within window")
    p.add_argument("--altitude", type=float, help="drone height in meters (for area estimate)")
    p.add_argument("--hfov", type=float, help="camera horizontal field of view in degrees")
    p.add_argument("--out", default="out", help="folder for annotated images")
    p.add_argument("--max-frames", type=int, help="stop after this many analyzed frames")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    if args.detector == "yolo":
        if not args.model:
            sys.exit("--detector yolo needs --model weights.pt")
        detector = YoloDetector(args.model, args.conf)
    else:
        detector = ColorDetector(cfg)
    pipeline = FirePipeline(detector, cfg, args.altitude, args.hfov)

    path = Path(args.source)
    if path.suffix.lower() in IMAGE_EXTS:
        run_image(path, pipeline, out_dir)
    else:
        run_stream(int(args.source) if args.source.isdigit() else args.source,
                   pipeline, out_dir, args)


if __name__ == "__main__":
    main()
